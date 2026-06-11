import math
import torch
import torch.nn.functional as F
from torch import nn, einsum
import torch.cuda as cuda
from beartype import beartype
from typing import Tuple

from einops import rearrange, repeat

# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def leaky_relu(p = 0.1):
    return nn.LeakyReLU(p)

def l2norm(t):
    return F.normalize(t, dim = -1)

# 改进的 LayerNorm，参考 Swin V2 的 residual-post-norm 方法
class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer("beta", torch.zeros(dim))
        self.eps = eps

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta, eps=self.eps)

# feedforward

class GEGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim = -1)
        return F.gelu(gate) * x

def FeedForward(dim, mult = 4, dropout = 0.):
    inner_dim = int(mult * (2 / 3) * dim)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim * 2, bias = False),
        GEGLU(),
        nn.Dropout(dropout),
        nn.Linear(inner_dim, dim, bias = False)
    )

# PEG - position generating module

class PEG(nn.Module):
    def __init__(self, dim, causal = False):
        super().__init__()
        self.causal = causal
        self.dsconv = nn.Conv3d(dim, dim, 3, groups = dim, padding=0)
        # 确保参数正确初始化
        self._init_weights()

    def _init_weights(self):
        """确保权重正确初始化"""
        for module in self.modules():
            if isinstance(module, nn.Conv3d):
                nn.init.normal_(module.weight, 0., 0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    @beartype
    def forward(self, x, shape: Tuple[int, int, int, int] = None):
        needs_shape = x.ndim == 3
        assert not (needs_shape and not exists(shape))

        orig_shape = x.shape

        if needs_shape:
            x = x.reshape(*shape, -1)

        x = rearrange(x, 'b ... d -> b d ...')

        frame_padding = (2, 0) if self.causal else (1, 1)

        x = F.pad(x, (1, 1, 1, 1, *frame_padding), value = 0.)
        x = self.dsconv(x)

        x = rearrange(x, 'b d ... -> b ... d')

        if needs_shape:
            x = rearrange(x, 'b ... d -> b (...) d')

        return x.reshape(orig_shape)

# attention
def audit_logits(name, tensor):
    if not torch.isfinite(tensor).all():
        maxv = tensor[tensor==tensor].max().item()    # 过滤 nan
        minv = tensor[tensor==tensor].min().item()
        print(f"[{name}] logits not finite  min={minv:.1f}  max={maxv:.1f}")
        torch.save(tensor.float().cpu(), f"{name}_bad_logits.pt")
        raise RuntimeError(f"NaN/Inf in {name}. Check {name}_bad_logits.pt")

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        dim_context = None,
        dim_head = 64,
        heads = 8,
        causal = False,
        num_null_kv = 0,
        norm_context = True,
        dropout = 0.,
        scale = None
    ):
        super().__init__()
        self.heads = heads
        self.causal = causal
        self.dim_head = dim_head
        
        if scale is None:
            scale = dim_head ** -0.5
        self.scale = scale
        
        inner_dim = dim_head * heads
        dim_context = default(dim_context, dim)
        
        if causal:
            self.rel_pos_bias = AlibiPositionalBias(heads = heads)

        self.attn_dropout = nn.Dropout(dropout)

        self.norm = LayerNorm(dim)
        self.context_norm = LayerNorm(dim_context) if norm_context else nn.Identity()

        self.num_null_kv = num_null_kv
        # 确保 null_kv 参数正确初始化
        if num_null_kv > 0:
            self.null_kv = nn.Parameter(torch.randn(heads, 2 * num_null_kv, dim_head))
            nn.init.normal_(self.null_kv, 0., 0.02)
        else:
            self.register_parameter('null_kv', None)

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim_context, inner_dim * 2, bias = False)

        # 参考 Swin V2，添加可学习的缩放参数
        self.q_scale = nn.Parameter(torch.ones(dim_head))
        self.k_scale = nn.Parameter(torch.ones(dim_head))

        self.to_out = nn.Linear(inner_dim, dim, bias = False)
        
        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """正确初始化所有参数"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0., 0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        
        # 初始化缩放参数
        nn.init.ones_(self.q_scale)
        nn.init.ones_(self.k_scale)

    def forward(
        self,
        x,
        mask = None,
        context = None,
        attn_bias = None
    ):
        batch, device, dtype = x.shape[0], x.device, x.dtype
        
        if exists(context):
            context = self.context_norm(context)

        kv_input = default(context, x)

        x = self.norm(x)

        q, k, v = self.to_q(x), *self.to_kv(kv_input).chunk(2, dim = -1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), (q, k, v))

        # 只有在 num_null_kv > 0 时才添加 null key-value
        if self.num_null_kv > 0 and self.null_kv is not None:
            nk, nv = repeat(self.null_kv, 'h (n r) d -> b h n r d', b = batch, r = 2).unbind(dim = -2)
            k = torch.cat((nk, k), dim = -2)
            v = torch.cat((nv, v), dim = -2)

        q, k = map(l2norm, (q, k))
        q = q * self.q_scale
        k = k * self.k_scale
        
        # 数值稳定性检查
        if torch.isnan(q).any() or torch.isinf(q).any():
            print(f"Warning: q contains NaN or inf values, shape: {q.shape}")
            q = torch.nan_to_num(q, nan=0.0, posinf=1e4, neginf=-1e4)
            
        if torch.isnan(k).any() or torch.isinf(k).any():
            print(f"Warning: k contains NaN or inf values, shape: {k.shape}")
            k = torch.nan_to_num(k, nan=0.0, posinf=1e4, neginf=-1e4)

        # 确保张量连续性
        if not q.is_contiguous():
            q = q.contiguous()
        if not k.is_contiguous():
            k = k.contiguous()
            
        # 使用数值稳定的缩放
        scale = torch.tensor(self.scale, dtype=q.dtype, device=device)
        sim = einsum('b h i d, b h j d -> b h i j', q, k) * scale
        
        # 数值稳定性检查
        if not torch.isfinite(sim).all():
            print(f"Warning: sim contains NaN or inf values")
            sim = torch.nan_to_num(sim, nan=0.0, posinf=1e4, neginf=-1e4)

        i, j = sim.shape[-2:]

        if exists(attn_bias):
            attn_bias = F.pad(attn_bias, (self.num_null_kv, 0), value = 0.)
            sim = sim + attn_bias
        
        if exists(mask):
            mask = F.pad(mask, (self.num_null_kv, 0), value = True)
            mask = rearrange(mask, 'b j -> b 1 1 j')
            sim = sim.masked_fill(~mask, -torch.finfo(sim.dtype).max)

        if self.causal:
            sim = sim + self.rel_pos_bias(sim)
            causal_mask = torch.ones((i, j), device = device, dtype = torch.bool).triu(j - i + 1)
            sim = sim.masked_fill(causal_mask, -torch.finfo(sim.dtype).max)

        # 数值稳定的 softmax
        attn = F.softmax(sim, dim=-1, dtype=torch.float32).type_as(sim)
        attn = self.attn_dropout(attn)
        
        # 检查注意力权重的数值稳定性
        if not torch.isfinite(attn).all():
            print(f"Warning: attn contains NaN or inf values, shape: {attn.shape}")
            attn = torch.nan_to_num(attn, nan=0.0, posinf=0.0, neginf=0.0)
            # 重新归一化
            attn = attn / (attn.sum(dim=-1, keepdim=True) + 1e-8)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

# alibi positional bias for extrapolation

class AlibiPositionalBias(nn.Module):
    def __init__(self, heads):
        super().__init__()
        self.heads = heads
        slopes = torch.Tensor(self._get_slopes(heads))
        slopes = rearrange(slopes, 'h -> h 1 1')
        self.register_buffer('slopes', slopes, persistent = False)
        self.register_buffer('bias', None, persistent = False)

    def get_bias(self, i, j, device):
        i_arange = torch.arange(j - i, j, device = device)
        j_arange = torch.arange(j, device = device)
        bias = -torch.abs(rearrange(j_arange, 'j -> 1 1 j') - rearrange(i_arange, 'i -> 1 i 1'))
        return bias

    @staticmethod
    def _get_slopes(heads):
        def get_slopes_power_of_2(n):
            start = (2**(-2**-(math.log2(n)-3)))
            ratio = start
            return [start*ratio**i for i in range(n)]

        if math.log2(heads).is_integer():
            return get_slopes_power_of_2(heads)

        closest_power_of_2 = 2 ** math.floor(math.log2(heads))
        return get_slopes_power_of_2(closest_power_of_2) + get_slopes_power_of_2(2 * closest_power_of_2)[0::2][:heads-closest_power_of_2]

    def forward(self, sim):
        h, i, j, device = *sim.shape[-3:], sim.device

        if exists(self.bias) and self.bias.shape[-1] >= j:
            return self.bias[..., :i, :j]

        bias = self.get_bias(i, j, device)
        bias = bias * self.slopes

        num_heads_unalibied = h - bias.shape[0]
        bias = F.pad(bias, (0, 0, 0, 0, 0, num_heads_unalibied))
        self.register_buffer('bias', bias, persistent = False)

        return self.bias

class ContinuousPositionBias(nn.Module):
    """ from https://arxiv.org/abs/2111.09883 """

    def __init__(
        self,
        *,
        dim,
        heads,
        num_dims = 2, # 2 for images, 3 for video
        layers = 2,
        log_dist = True,
        cache_rel_pos = False
    ):
        super().__init__()
        self.num_dims = num_dims
        self.log_dist = log_dist

        self.net = nn.ModuleList([])
        self.net.append(nn.Sequential(nn.Linear(self.num_dims, dim), leaky_relu()))

        for _ in range(layers - 1):
            self.net.append(nn.Sequential(nn.Linear(dim, dim), leaky_relu()))

        self.net.append(nn.Linear(dim, heads))

        self.cache_rel_pos = cache_rel_pos
        self.register_buffer('rel_pos', None, persistent = False)
        
        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """正确初始化所有参数"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0., 0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, *dimensions, device = None):
        if device is None:
            device = next(self.parameters()).device

        if not exists(self.rel_pos) or not self.cache_rel_pos:
            positions = [torch.arange(d, device = device) for d in dimensions]
            grid = torch.stack(torch.meshgrid(*positions, indexing = 'ij'))
            grid = rearrange(grid, 'c ... -> (...) c')
            rel_pos = rearrange(grid, 'i c -> i 1 c') - rearrange(grid, 'j c -> 1 j c')

            if self.log_dist:
                rel_pos = torch.sign(rel_pos) * torch.log(rel_pos.abs() + 1)

            self.register_buffer('rel_pos', rel_pos, persistent = False)

        rel_pos = self.rel_pos.to(torch.float32)

        for layer in self.net:
            target_dtype = next(layer.parameters()).dtype
            rel_pos = layer(rel_pos.to(target_dtype))

        return rearrange(rel_pos, 'i j h -> h i j')

# transformer

class Transformer(nn.Module):
    def __init__(
        self,
        dim,
        *,
        depth,
        dim_context = None,
        causal = False,
        dim_head = 64,
        heads = 8,
        ff_mult = 4,
        peg = False,
        peg_causal = False,
        attn_num_null_kv = 2,
        has_cross_attn = False,
        attn_dropout = 0.,
        ff_dropout = 0.
    ):
        super().__init__()
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PEG(dim = dim, causal = peg_causal) if peg else None,
                Attention(dim = dim, dim_head = dim_head, heads = heads, causal = causal, dropout = attn_dropout),
                Attention(dim = dim, dim_head = dim_head, dim_context = dim_context, heads = heads, causal = False, num_null_kv = attn_num_null_kv, dropout = attn_dropout) if has_cross_attn else None,
                FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout)
            ]))

        self.norm_out = LayerNorm(dim)
        
        # 应用权重初始化
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """统一的权重初始化方法"""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, 0., 0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    @beartype
    def forward(
        self,
        x,
        video_shape: Tuple[int, int, int, int] = None,
        attn_bias = None,
        context = None,
        self_attn_mask = None,
        cross_attn_context_mask = None
    ):
        for peg, self_attn, cross_attn, ff in self.layers:
            if exists(peg):
                x = peg(x, shape = video_shape) + x

            x = self_attn(x, attn_bias = attn_bias, mask = self_attn_mask) + x

            if exists(cross_attn) and exists(context):
                x = cross_attn(x, context = context, mask = cross_attn_context_mask) + x
            
            # 数值稳定性检查
            if torch.isnan(x).any() or torch.isinf(x).any():
                print(f"Warning: x contains NaN or inf values, shape: {x.shape}, dtype: {x.dtype}, device: {x.device}")
                x = torch.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4)
                
            x = ff(x) + x

        return self.norm_out(x)

class TransformerCLS(nn.Module):
    def __init__(
        self,
        dim,
        *,
        depth,
        dim_context = None,
        causal = False,
        dim_head = 64,
        heads = 8,
        ff_mult = 4,
        peg = False,
        peg_causal = False,
        attn_num_null_kv = 2,
        has_cross_attn = False,
        attn_dropout = 0.,
        ff_dropout = 0.
    ):
        super().__init__()
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PEG(dim = dim, causal = peg_causal) if peg else None,
                Attention(dim = dim, dim_head = dim_head, heads = heads, causal = causal, dropout = attn_dropout),
                Attention(dim = dim, dim_head = dim_head, dim_context = dim_context, heads = heads, causal = False, num_null_kv = attn_num_null_kv, dropout = attn_dropout) if has_cross_attn else None,
                FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout)
            ]))

        self.norm_out = LayerNorm(dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        
        # 应用权重初始化
        self.apply(self._init_weights)
        # 单独初始化 CLS token
        nn.init.normal_(self.cls_token, 0., 0.02)

    def _init_weights(self, module):
        """统一的权重初始化方法"""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, 0., 0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
    
    @beartype
    def forward(
        self,
        x,
        video_shape: Tuple[int, int, int, int] = None,
        attn_bias = None,
        context = None,
        self_attn_mask = None,
        cross_attn_context_mask = None
    ):
        B = x.shape[0]
        # 添加CLS token
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        
        for peg, self_attn, cross_attn, ff in self.layers:
            if exists(peg):
                x[:,1:] = peg(x[:,1:], shape = video_shape) + x[:,1:]

            x = self_attn(x, attn_bias = attn_bias, mask = self_attn_mask) + x

            if exists(cross_attn) and exists(context):
                x = cross_attn(x, context = context, mask = cross_attn_context_mask) + x
            
            # 数值稳定性检查
            if torch.isnan(x).any() or torch.isinf(x).any():
                print(f"Warning: x contains NaN or inf values, shape: {x.shape}, dtype: {x.dtype}, device: {x.device}")
                x = torch.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4)
                
            x = ff(x) + x

        return self.norm_out(x)