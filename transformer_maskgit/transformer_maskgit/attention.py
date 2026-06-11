import math
import torch
import torch.nn.functional as F
from torch import nn, einsum
import torch.cuda as cuda
from beartype import beartype
from typing import Tuple
import torch.utils.checkpoint as checkpoint
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

# bias-less layernorm, being used in more recent T5s, PaLM, also in @borisdayma 's experiments shared with me
# greater stability

class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer("beta", torch.zeros(dim))

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)

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

class MoEFeedForward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0., num_experts=4):
        super().__init__()
        self.num_experts = num_experts
        inner_dim = int(mult * (2 / 3) * dim)

        def _build_expert():
            return nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, inner_dim * 2, bias=False),
                GEGLU(),
                nn.Dropout(dropout),
                nn.Linear(inner_dim, dim, bias=False)
            )

        self.experts = nn.ModuleList([_build_expert() for _ in range(num_experts)])

    def forward(self, x: torch.Tensor, expert_indices: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq, dim)
        expert_indices: (batch,) int tensor, values in [0, num_experts - 1]
        """
        if expert_indices.ndim != 1:
            raise ValueError(f"`expert_indices` must be 1D (batch,), but got shape {expert_indices.shape}")

        if expert_indices.shape[0] != x.shape[0]:
            raise ValueError(
                f"`expert_indices` batch dimension ({expert_indices.shape[0]}) "
                f"must match x batch dimension ({x.shape[0]})"
            )

        expert_indices = expert_indices.to(device=x.device)
        if expert_indices.dtype != torch.long:
            expert_indices = expert_indices.long()

        out = torch.zeros_like(x)

        for expert_id, expert in enumerate(self.experts):
            batch_mask = expert_indices == expert_id
            if not batch_mask.any():
                continue

            expert_input = x[batch_mask]
            
            expert_output = expert(expert_input)
            # out[batch_mask] = expert_output
            # 显式转换数据类型
            out[batch_mask] = expert_output.to(out.dtype)

        return out
# PEG - position generating module

class PEG(nn.Module):
    def __init__(self, dim, causal = False):
        super().__init__()
        self.causal = causal
        self.dsconv = nn.Conv3d(dim, dim, 3, groups = dim)

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
        num_null_kv = 1,
        norm_context = True,
        dropout = 0.,
        scale = None
    ):
        super().__init__()
        self.heads = heads
        self.causal = causal
        if scale is None:
            scale = dim_head ** -0.5
        print('scale:', scale)
        self.scale = scale
        inner_dim = dim_head * heads
        dim_context = default(dim_context, dim)
        if causal:
            self.rel_pos_bias = AlibiPositionalBias(heads = heads)

        self.attn_dropout = nn.Dropout(dropout)

        self.norm = LayerNorm(dim)
        self.context_norm = LayerNorm(dim_context) if norm_context else nn.Identity()

        self.num_null_kv = num_null_kv
        self.null_kv = nn.Parameter(torch.randn(heads, 2 * num_null_kv, dim_head))

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim_context, inner_dim * 2, bias = False)

        self.q_scale = nn.Parameter(torch.ones(dim_head))
        self.k_scale = nn.Parameter(torch.ones(dim_head))

        self.to_out = nn.Linear(inner_dim, dim, bias = False)

    def forward(
        self,
        x,
        mask = None,
        context = None,
        attn_bias = None
    ):
        batch, device, dtype = x.shape[0], x.device, x.dtype
        device=torch.device('cuda')
        if exists(context):
            context = self.context_norm(context)

        kv_input = default(context, x)

        x = self.norm(x)

        q, k, v = self.to_q(x), *self.to_kv(kv_input).chunk(2, dim = -1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), (q, k, v))

        nk, nv = repeat(self.null_kv, 'h (n r) d -> b h n r d', b = batch, r = 2).unbind(dim = -2)

        k = torch.cat((nk, k), dim = -2)
        v = torch.cat((nv, v), dim = -2)

        q, k = map(l2norm, (q, k))
        q = q * self.q_scale
        k = k * self.k_scale

        # before einsum
        if not q.is_contiguous():
            q = q.contiguous()
        if not k.is_contiguous():
            k = k.contiguous()

        # keep dtype consistent with scale
        
        scale = torch.tensor(self.scale, dtype=q.dtype, device=q.device)
        sim = einsum('b h i d, b h j d -> b h i j', q, k) * scale
        audit_logits("sim", sim)

        i, j = sim.shape[-2:]

        if exists(attn_bias):
            attn_bias = F.pad(attn_bias, (self.num_null_kv, 0), value = 0.)
            sim = sim + attn_bias
            audit_logits("attn_bias", attn_bias)
        audit_logits("sim after attn_bias", sim)
        
        if exists(mask):
            mask = F.pad(mask, (self.num_null_kv, 0), value = True)
            mask = rearrange(mask, 'b j -> b 1 1 j')
            sim = sim.masked_fill(~mask, -torch.finfo(sim.dtype).max)

        if self.causal:
            sim = sim + self.rel_pos_bias(sim)
            device=torch.device('cuda')
            causal_mask = torch.ones((i, j), device = device, dtype = torch.bool).triu(j - i + 1)
            sim = sim.masked_fill(causal_mask, -torch.finfo(sim.dtype).max)

        attn = sim.softmax(dim = -1)
        attn = self.attn_dropout(attn)


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
        device=torch.device('cuda')
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
        device=torch.device('cuda')
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

    def forward(self, *dimensions, device = torch.device('cpu')):

        if not exists(self.rel_pos) or not self.cache_rel_pos:
            device=torch.device('cuda')
            positions = [torch.arange(d, device = device) for d in dimensions]
            grid = torch.stack(torch.meshgrid(*positions, indexing = 'ij'))
            grid = rearrange(grid, 'c ... -> (...) c')
            rel_pos = rearrange(grid, 'i c -> i 1 c') - rearrange(grid, 'j c -> 1 j c')

            if self.log_dist:
                rel_pos = torch.sign(rel_pos) * torch.log(rel_pos.abs() + 1)

            self.register_buffer('rel_pos', rel_pos, persistent = False)

        rel_pos = self.rel_pos.to(torch.float32)

        for layer in self.net:
            rel_pos = layer(rel_pos.float())

        return rearrange(rel_pos, 'i j h -> h i j')

# transformer

# class Transformer(nn.Module):
#     def __init__(
#         self,
#         dim,
#         *,
#         depth,
#         dim_context = None,
#         causal = False,
#         dim_head = 64,
#         heads = 8,
#         ff_mult = 4,
#         peg = False,
#         peg_causal = False,
#         attn_num_null_kv = 2,
#         has_cross_attn = False,
#         attn_dropout = 0.,
#         ff_dropout = 0.
#     ):
#         super().__init__()
#         self.layers = nn.ModuleList([])

#         for _ in range(depth):
#             self.layers.append(nn.ModuleList([
#                 PEG(dim = dim, causal = peg_causal) if peg else None,
#                 Attention(dim = dim, dim_head = dim_head, heads = heads, causal = causal, dropout = attn_dropout),
#                 Attention(dim = dim, dim_head = dim_head, dim_context = dim_context, heads = heads, causal = False, num_null_kv = attn_num_null_kv, dropout = attn_dropout) if has_cross_attn else None,
#                 MoEFeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout, num_experts=5)
#             ]))

#         self.norm_out = LayerNorm(dim)

#     @beartype
#     def forward(
#         self,
#         x,
#         video_shape: Tuple[int, int, int, int] = None,
#         attn_bias = None,
#         context = None,
#         self_attn_mask = None,
#         cross_attn_context_mask = None,
#         expert_indices = None
#     ):
#         if expert_indices is None:
#             raise ValueError("`expert_indices` must be provided and contain the expert id (0-3) for each batch element.")
#         for peg, self_attn, cross_attn, ff in self.layers:
#             if exists(peg):
#                 x = peg(x, shape = video_shape) + x

#             x = self_attn(x, attn_bias = attn_bias, mask = self_attn_mask) + x

#             if exists(cross_attn) and exists(context):
#                 x = cross_attn(x, context = context, mask = cross_attn_context_mask) + x
#             # 检查x的形状和数据类型以及数据范围，是否有空值无限值
#             # print("[DEBUG] Transormer forward", x.shape, x.dtype, x.device, flush=True)
#             if torch.isnan(x).any() or torch.isinf(x).any():
#                 print(f"Warning: x contains NaN or inf values, shape: {x.shape}, dtype: {x.dtype}, device: {x.device}")
#             x = ff(x,expert_indices=expert_indices) + x

#         return self.norm_out(x)


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
        ff_dropout = 0.,
        use_checkpoint = True  # 新增参数
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.use_checkpoint = use_checkpoint  # 存储检查点标志

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PEG(dim = dim, causal = peg_causal) if peg else None,
                Attention(dim = dim, dim_head = dim_head, heads = heads, causal = causal, dropout = attn_dropout),
                Attention(dim = dim, dim_head = dim_head, dim_context = dim_context, heads = heads, causal = False, num_null_kv = attn_num_null_kv, dropout = attn_dropout) if has_cross_attn else None,
                MoEFeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout, num_experts=5)
            ]))

        self.norm_out = LayerNorm(dim)

    def _forward_layer(self, x, peg, self_attn, cross_attn, ff, video_shape, attn_bias, context, self_attn_mask, cross_attn_context_mask, expert_indices):
        """单层的前向传播，用于梯度检查点"""
        if exists(peg):
            x = peg(x, shape = video_shape) + x

        x = self_attn(x, attn_bias = attn_bias, mask = self_attn_mask) + x

        if exists(cross_attn) and exists(context):
            x = cross_attn(x, context = context, mask = cross_attn_context_mask) + x

        if torch.isnan(x).any() or torch.isinf(x).any():
            print(f"Warning: x contains NaN or inf values, shape: {x.shape}, dtype: {x.dtype}, device: {x.device}")
        
        x = ff(x, expert_indices=expert_indices) + x
        
        return x

    @beartype
    def forward(
        self,
        x,
        video_shape: Tuple[int, int, int, int] = None,
        attn_bias = None,
        context = None,
        self_attn_mask = None,
        cross_attn_context_mask = None,
        expert_indices = None
    ):
        if expert_indices is None:
            raise ValueError("`expert_indices` must be provided and contain the expert id (0-3) for each batch element.")
        
        for peg, self_attn, cross_attn, ff in self.layers:
            if self.use_checkpoint and self.training:
                # 使用梯度检查点
                x = checkpoint.checkpoint(
                    self._forward_layer,
                    x, peg, self_attn, cross_attn, ff,
                    video_shape, attn_bias, context,
                    self_attn_mask, cross_attn_context_mask, expert_indices,
                    use_reentrant=False  # 推荐使用新的检查点API
                )
            else:
                # 正常前向传播
                x = self._forward_layer(
                    x, peg, self_attn, cross_attn, ff,
                    video_shape, attn_bias, context,
                    self_attn_mask, cross_attn_context_mask, expert_indices
                )

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
                MoEFeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout, num_experts=5)
            ]))

        self.norm_out = LayerNorm(dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))  # 添加CLS token
    @beartype
    def forward(
        self,
        x,
        video_shape: Tuple[int, int, int, int] = None,
        attn_bias = None,
        context = None,
        self_attn_mask = None,
        cross_attn_context_mask = None,
        expert_indices = None
    ):
        if expert_indices is None:
            raise ValueError("`expert_indices` must be provided and contain the expert id (0-3) for each batch element.")
        B = x.shape[0]
        # 添加CLS token
        cls_token = self.cls_token.expand(B, -1, -1)  # 扩展到batch size
        x = torch.cat((cls_token, x), dim=1)  # 在序列的开头添加CLS token
        for peg, self_attn, cross_attn, ff in self.layers:
            if exists(peg):
                x[:,1:] = peg(x[:,1:], shape = video_shape) + x[:,1:]

            x = self_attn(x, attn_bias = attn_bias, mask = self_attn_mask) + x

            if exists(cross_attn) and exists(context):
                x = cross_attn(x, context = context, mask = cross_attn_context_mask) + x
            # 检查x的形状和数据类型以及数据范围，是否有空值无限值
            # print("[DEBUG] Transormer forward", x.shape, x.dtype, x.device, flush=True)
            if torch.isnan(x).any() or torch.isinf(x).any():
                print(f"Warning: x contains NaN or inf values, shape: {x.shape}, dtype: {x.dtype}, device: {x.device}")
            x = ff(x,expert_indices=expert_indices) + x

        return self.norm_out(x)
