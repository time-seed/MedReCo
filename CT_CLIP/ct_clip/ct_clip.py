import math
import copy
from contextlib import contextmanager
from functools import partial, wraps
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn, einsum
from torch.utils.checkpoint import checkpoint
from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange, Reduce
import numpy as np
# from ct_clip.mlm import MLM
# from ct_clip.visual_ssl import SimSiam, SimCLR
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

from transformers import BertTokenizer, BertModel

# NOTE: Import a Cross-Attn Module and PE Module
from positional_encodings.torch_encodings import PositionalEncoding3D, PositionalEncoding1D, PositionalEncoding2D
from ct_clip.transformer_decoder import TransformerDecoder, TransformerDecoderLayer
from ct_clip.sample_hard_neg import generate_rerank_training_data,generate_rerank_training_data_uniform, generate_rerank_listwise_data_ultimate
# from transformer_decoder import TransformerDecoder, TransformerDecoderLayer
from .moe_reranker import MoECrossAttentionReranker,MoEListwiseCrossAttentionReranker
# NOTE

# helper functions

def identity(t, *args, **kwargs):
    return t

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

@contextmanager
def null_context():
    yield

def max_neg_value(dtype):
    return -torch.finfo(dtype).max

def cast_tuple(t):
    return t if isinstance(t, (tuple, list)) else (t,)

def masked_mean(t, mask, dim = 1, eps = 1e-6):
    t = t.masked_fill(~mask, 0.)
    numer = t.sum(dim = dim)
    denom = mask.sum(dim = dim).clamp(min = eps)
    return numer / denom

def log(t, eps = 1e-20):
    return torch.log(t + eps)

def l2norm(t):
    
    return F.normalize(t, dim = -1)

def matrix_diag(t):
    device = t.device
    i, j = t.shape[-2:]
    num_diag_el = min(i, j)
    i_range = torch.arange(i, device = device)
    j_range = torch.arange(j, device = device)
    diag_mask = rearrange(i_range, 'i -> i 1') == rearrange(j_range, 'j -> 1 j')
    diag_el = t.masked_select(diag_mask)
    return rearrange(diag_el, '(b d) -> b d', d = num_diag_el)

# checkpointing helper function

def make_checkpointable(fn):
    @wraps(fn)
    def inner(*args):
        input_needs_grad = any([isinstance(el, torch.Tensor) and el.requires_grad for el in args])

        if not input_needs_grad:
            return fn(*args)

        return checkpoint(fn, *args)

    return inner

# keyword argument helpers

def pick_and_pop(keys, d):
    values = list(map(lambda key: d.pop(key), keys))
    return dict(zip(keys, values))

def group_dict_by_key(cond, d):
    return_val = [dict(),dict()]
    for key in d.keys():
        match = bool(cond(key))
        ind = int(not match)
        return_val[ind][key] = d[key]
    return (*return_val,)

def string_begins_with(prefix, str):
    return str.startswith(prefix)

def group_by_key_prefix(prefix, d):
    return group_dict_by_key(partial(string_begins_with, prefix), d)

def groupby_prefix_and_trim(prefix, d):
    kwargs_with_prefix, kwargs = group_dict_by_key(partial(string_begins_with, prefix), d)
    kwargs_without_prefix = dict(map(lambda x: (x[0][len(prefix):], x[1]), tuple(kwargs_with_prefix.items())))
    return kwargs_without_prefix, kwargs

# helper classes

class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim = -1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = -1, keepdim = True)
        return (x - mean) * (var + eps).rsqrt() * self.g

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = LayerNorm(dim)
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(self.norm(x), *args, **kwargs)

# patch dropout

class PatchDropout(nn.Module):
    def __init__(self, prob):
        super().__init__()
        assert 0 <= prob < 1.
        self.prob = prob

    def forward(self, x, force_keep_all = False):
        if not self.training or self.prob == 0. or force_keep_all:
            return x

        b, n, _, device = *x.shape, x.device

        batch_indices = torch.arange(b, device = device)
        batch_indices = rearrange(batch_indices, '... -> ... 1')
        num_patches_keep = max(1, int(n * (1 - self.prob)))
        patch_indices_keep = torch.randn(b, n, device = device).topk(num_patches_keep, dim = -1).indices

        return x[batch_indices, patch_indices_keep]

# rotary positional embedding

class RotaryEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        inv_freq = 1. / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, seq_len, device):
        inv_freq = self.inv_freq
        t = torch.arange(seq_len, device = device).type_as(inv_freq)
        freqs = torch.einsum('i , j -> i j', t, inv_freq)
        return torch.cat((freqs, freqs), dim = -1)

def rotate_half(x):
    x = rearrange(x, '... (j d) -> ... j d', j = 2)
    x1, x2 = x.unbind(dim = -2)
    return torch.cat((-x2, x1), dim = -1)

def apply_rotary_pos_emb(freqs, t):
    rot_dim = freqs.shape[-1]
    t, t_pass = t[..., :rot_dim], t[..., rot_dim:]
    t = (t * freqs.cos()) + (rotate_half(t) * freqs.sin())
    return torch.cat((t, t_pass), dim = -1)

# transformer

class GEGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim = -1)
        return x * F.gelu(gate)

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4, dropout = 0.):
        super().__init__()
        inner_dim = int(dim * mult)

        self.net = nn.Sequential(
            nn.Linear(dim, inner_dim * 2, bias = False),
            GEGLU(),
            LayerNorm(inner_dim),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim, bias = False)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, dim_head = 64, heads = 8, causal = False, dropout = 0.):
        super().__init__()
        self.heads = heads
        self.causal = causal
        self.scale = dim_head ** -0.5
        inner_dim = dim_head * heads

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim, bias = False), LayerNorm(dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask = None, rotary_pos_emb = None):
        h, device, scale = self.heads, x.device, self.scale

        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        q = q * self.scale

        if exists(rotary_pos_emb):
            apply_rotary = partial(apply_rotary_pos_emb, rotary_pos_emb)
            q, k, v = map(apply_rotary, (q, k, v))

        sim = einsum('b h i d, b h j d -> b h i j', q, k)

        mask_value = -torch.finfo(sim.dtype).max

        if exists(mask):
            mask = rearrange(mask, 'b j -> b 1 1 j')
            sim = sim.masked_fill(~mask, mask_value)

        if self.causal:
            i, j = sim.shape[-2:]
            causal_mask = torch.ones((i, j), dtype = torch.bool, device = device).triu(j - i + 1)
            sim = sim.masked_fill(causal_mask, mask_value)

        attn = sim.softmax(dim = -1, dtype = torch.float32)
        attn = attn.type(sim.dtype)

        attn = self.dropout(attn)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(
            self,
            dim,
            *,
            depth,
            dim_head = 64,
            heads = 8,
            causal = False,
            attn_dropout = 0.,
            ff_dropout = 0.,
            ff_mult = 4,
            checkpoint_during_training = False
    ):
        super().__init__()
        self.checkpoint_during_training = checkpoint_during_training

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim = dim, dim_head = dim_head, heads = heads, causal = causal, dropout = attn_dropout)),
                PreNorm(dim, FeedForward(dim = dim, mult = ff_mult)),
            ]))

        self.norm_in = LayerNorm(dim)
        self.norm_out = LayerNorm(dim)

    def forward(
            self,
            x,
            rotary_pos_emb = None,
            mask = None
    ):
        can_checkpoint = self.training and self.checkpoint_during_training
        checkpoint_fn = make_checkpointable if can_checkpoint else identity

        x = self.norm_in(x)

        for attn, ff in self.layers:
            attn, ff = map(checkpoint_fn, (attn, ff))

            x = attn(x, mask, rotary_pos_emb) + x
            x = ff(x) + x

        return self.norm_out(x)

# text and vision transformers

class TextTransformer(nn.Module):
    def __init__(
            self,
            dim,
            *,
            num_tokens,
            max_seq_len,
            dim_head,
            rotary_pos_emb = None,
            causal = False,
            **kwargs
    ):
        super().__init__()
        self.token_emb = nn.Embedding(num_tokens, dim)

        self.abs_pos_emb = nn.Embedding(max_seq_len, dim) if not rotary_pos_emb else None
        self.rotary_pos_emb = RotaryEmbedding(min(dim_head, 32)) if rotary_pos_emb else None

        self.cls_token = nn.Parameter(torch.randn(dim)) if not causal else None

        self.transformer = Transformer(dim, dim_head = dim_head, causal = causal, **kwargs)

    def forward(self, x, mask = None):
        b, n, device = *x.shape, x.device

        x = self.token_emb(x)

        if exists(self.abs_pos_emb):
            pos_emb = self.abs_pos_emb(torch.arange(n, device = device))
            x = x + rearrange(pos_emb, 'n d -> 1 n d')

        rotary_pos_emb = None
        if exists(self.rotary_pos_emb):
            rotary_pos_emb = self.rotary_pos_emb(n + 1, device = device)

        if exists(self.cls_token):
            cls_tokens = repeat(self.cls_token, 'd -> b 1 d', b = b)
            x = torch.cat((cls_tokens, x), dim = 1)

            if exists(mask):
                mask = F.pad(mask, (1, 0), value = True)

        out = self.transformer(x, mask = mask, rotary_pos_emb = rotary_pos_emb)
        return out

class VisionTransformer(nn.Module):
    def __init__(
            self,
            dim,
            *,
            image_size,
            patch_size,
            channels,
            patch_dropout = 0.5,
            **kwargs
    ):
        super().__init__()
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2

        self.to_tokens = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
            nn.Linear(patch_dim, dim)
        )

        self.pos_emb = nn.Embedding(num_patches, dim)
        self.patch_dropout = PatchDropout(patch_dropout)

        self.transformer = Transformer(dim, **kwargs)

        self.to_cls_tokens = nn.Sequential(
            Reduce('b n d -> b d', 'mean'),
            nn.Linear(dim, dim, bias = False),
            Rearrange('b d -> b 1 d')
        )

    def forward(
            self,
            x,
            keep_all_patches = False
    ):
        device = x.device

        x = self.to_tokens(x)
        b, n, _ = x.shape

        pos_emb = self.pos_emb(torch.arange(n, device = device))
        x = x + rearrange(pos_emb, 'n d -> 1 n d')

        x = self.patch_dropout(x, force_keep_all = keep_all_patches)

        out = self.transformer(x)

        cls_tokens = self.to_cls_tokens(out)
        return torch.cat((cls_tokens, out), dim = 1)

# contrastive learning functions

def model_forward_with_context(
        *,
        fn,
        args,
        freeze,
):
    encoding_context = null_context if not freeze else torch.no_grad

    with encoding_context():
        enc = fn(*args)

        if freeze:
            enc.detach_()

    return enc

# main clip class
def select_top_k_by_attention(attn_weights, image_tokens, pos_embeddings, top_k):
    """
    Args:
        attn_weights: (B, Heads, 1, N) 或 (B, 1, N) 来自 Fusion 最后一层
        image_tokens: (B, N, D) 原始图像 patch features (无位置编码)
        pos_embeddings: (B, N, D) 对应的位置编码
        top_k: int
    Returns:
        selected_features: (B, K, D) 也就是 Feature + Pos
    """
    # 1. 处理多头注意力，取平均得到关注度图
    if attn_weights.dim() == 4:
        attn_score = attn_weights.mean(dim=1) # (B, 1, N)
    else:
        attn_score = attn_weights

    # 2. 获取 Top-K 索引 (无需梯度)
    # 我们只利用这个索引去“查表”，不需要对索引本身求导
    with torch.no_grad():
        _, indices = attn_score.topk(top_k, dim=-1) # (B, 1, K)
    
    # 3. 准备 Gather
    B, _, K = indices.shape
    D = image_tokens.shape[-1]
    # 扩展索引维度以匹配 gather: (B, 1, K) -> (B, K, D)
    indices_expanded = indices.transpose(1, 2).expand(-1, -1, D)
    
    # 4. 提取 Token 和 Position
    # 这一步是关键！Token 必须加上位置信息，否则 Rerank 模型不知道它们在哪
    sel_tokens = torch.gather(image_tokens, 1, indices_expanded)
    sel_pos = torch.gather(pos_embeddings, 1, indices_expanded)
    
    # 融合特征与位置
    return sel_tokens + sel_pos

class CrossAttentionReranker(nn.Module):        # TODO: 改进成，MOE架构，每个模态单独训练一个rerank结果
    def __init__(self, dim, num_heads, depth, top_k, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.top_k = top_k
        
        # 2. 特殊 Token 和 Embedding
        self.sep_token = nn.Parameter(torch.randn(1, 1, dim))
        # Segment Embedding: 0=Condition/Img1, 1=Img2
        self.segment_emb = nn.Embedding(2, dim)
        self.cls_emb = nn.Parameter(torch.randn(1, 1, dim))  # 用于替代 Condition Embedding，直接作为 [CLS] 输入
        # 3. Transformer Encoder (Cross-Encoder)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=num_heads, dim_feedforward=dim*4, 
            dropout=dropout, batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        
        # 4. 二分类头
        self.classifier = nn.Sequential(
            nn.Linear(dim, dim),
            # nn.Tanh(),
            nn.GELU(),  # 👈 强力推荐
            nn.Dropout(0.1),
            nn.Linear(dim, 1) # 输出 logit
        )

    def forward(self,tokens1, tokens2):
        """
        Args:
            tokens1: (B, 1 + K, D) 筛选后带位置信息的 Img1 Tokens
            tokens2: (B, 1 + K, D) 筛选后带位置信息的 Img2 Tokens
        """
        B = tokens1.shape[0]
        
        # A. 准备头部 [CLS]
        cls_token = self.cls_emb.expand(B, -1, -1)  # (B, 1, D)
        
        # B. 准备分割符 [SEP]
        sep = repeat(self.sep_token, '1 1 d -> b 1 d', b=B)
        
        # C. 构建长序列
        tokens1_with_seg = tokens1 + self.segment_emb(torch.zeros(B, self.top_k + 1, dtype=torch.long, device=tokens1.device))
        tokens2_with_seg = tokens2 + self.segment_emb(torch.ones(B, self.top_k + 1, dtype=torch.long, device=tokens2.device))
    
        # Sequence: [CLS/Cond] + [Img1] + [SEP] + [Img2]
        sequence = torch.cat([cls_token, tokens1_with_seg, sep, tokens2_with_seg], dim=1)
        
        # E. Cross-Attention 交互
        output = self.transformer(sequence)
        
        # F. 预测 (使用第一个 token 即 Condition Token 的输出)
        logits = self.classifier(output[:, 0, :])
        return logits

class CTCLIP(nn.Module):
    def __init__(
            self,
            *,
            use_triplet_loss = 1,
            use_infoNCE_loss = 1,
            use_abnormal_exist_loss = 1,
            use_rerank_loss = 1,
            use_uncon_triplet_loss = 1,
            use_uncon_infoNCE_loss = 1,
            use_image2image_loss = 1,
            infonce_temp = False,
            triplet_loss_margin = 0.1,
            positive_distance_threshold = 0.2,
            positive_threshold = 0.7,   # ON default, >0.7 can be positive
            negative_threshold = 1, # ON default, eveyone can be negative
            image_encoder = None,
            text_encoder = None,
            tokenizer = None,
            fusion_module = None,   # NOTE: Pass in a Transformer Decoder Module
            dim_text = 768,
            dim_image = 512,
            dim_latent = 512,
            num_text_tokens = 28897,
            text_enc_depth = 6,
            text_seq_len = 256,
            text_heads = 8,
            text_dim_head = 64,
            text_has_cls_token = False,
            text_pad_id = 0,
            text_rotary_pos_emb = False,
            text_causal_mask = False,
            text_eos_id = None,
            text_encode_without_mask = False,
            visual_enc_depth = 6,
            visual_heads = 8,
            visual_dim_head = 64,
            visual_image_size = 256,
            visual_patch_size = 32,
            visual_patch_dropout = 0.5,
            visual_has_cls_token = False,
            channels = 3,
            use_all_token_embeds = False,
            downsample_image_embeds = False,
            decoupled_contrastive_learning = False,
            extra_latent_projection = False,
            use_mlm = False,
            text_ssl_loss_weight = 0.05,
            use_visual_ssl = False,
            visual_ssl = None,
            visual_ssl_type = 'simsiam',
            visual_ssl_hidden_layer = -1,
            simclr_temperature = 0.1,
            image_ssl_loss_weight = 0.05,
            multiview_loss_weight = 0.1,
            checkpoint_during_training = False,
            rerank_topk=128,
            **kwargs
    ):
        super().__init__()
        self.smooth_ap_tau = 0.1   # 可以选择外部修改掉这部分
        self.infonce_temp = infonce_temp
        self.log_temperature = nn.Parameter(torch.tensor(np.log(0.07)))
        self.condition_embeddings =  nn.Embedding(400, dim_latent)
        self.use_rerank_loss = use_rerank_loss
        self.use_abnormal_exist_loss = use_abnormal_exist_loss
        self.use_triplet_loss = use_triplet_loss
        self.triplet_loss_margin = triplet_loss_margin
        self.positive_distance_threshold = positive_distance_threshold  # triplet中确定ij>ik的阈值
        
        self.use_infoNCE_loss = use_infoNCE_loss
        self.positive_threshold = positive_threshold    # infoNCE中找到false negative
        self.negative_threshold = negative_threshold
        self.use_image2image_loss = use_image2image_loss
        self.use_uncon_triplet_loss = use_uncon_triplet_loss
        self.use_uncon_infoNCE_loss = use_uncon_infoNCE_loss
        #assert use_all_token_embeds or (visual_has_cls_token or text_has_cls_token), 'CLS token must be included on both vision and text transformers if you are not using fine-grained contrastive learning loss'
        self.dtype=torch.float32
        # store some parameters for access

        self.dim_text = dim_text
        self.dim_image = dim_image
        self.dim_latent = dim_latent

        self.image_channels = channels
        self.image_size = visual_image_size

        # instantiate text transformer
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim_image))
        self.text_pad_id = text_pad_id
        self.text_has_cls_token = text_has_cls_token
        self.text_seq_len = text_seq_len
        self.rerank_topk = rerank_topk
        self.text_encode_without_mask = text_encode_without_mask # whether to pass in text mask to text encoder

        self.text_causal_mask = text_causal_mask
        self.text_eos_id = text_eos_id

        assert not (text_causal_mask and not exists(text_eos_id)), 'text EOS token id must be given if using causal mask in text transformer'

        if exists(text_encoder):
            self.text_transformer = text_encoder
        else:
            self.text_transformer = TextTransformer(
                dim = dim_text,
                num_tokens = num_text_tokens + (1 if use_mlm else 0),
                max_seq_len = text_seq_len,
                depth = text_enc_depth,
                heads = text_heads,
                causal = text_causal_mask,
                dim_head = text_dim_head,
                rotary_pos_emb = text_rotary_pos_emb,
                checkpoint_during_training = checkpoint_during_training
            )

        # instantiate image transformer

        self.visual_has_cls_token = visual_has_cls_token

        if exists(image_encoder):
            self.visual_transformer = image_encoder
        else:
            self.visual_transformer = VisionTransformer(
                dim = dim_image,
                image_size = visual_image_size,
                patch_size = visual_patch_size,
                channels = channels,
                depth = visual_enc_depth,
                heads = visual_heads,
                dim_head = visual_dim_head,
                patch_dropout = visual_patch_dropout,
                checkpoint_during_training = checkpoint_during_training
            )
        
        # NOTE: Pass in a Transformer Decoder Module and a Positional-Encoding Module
        if exists(fusion_module):
            self.fusion_module = fusion_module
        else:
            decoder_layer = TransformerDecoderLayer(d_model=dim_latent, nhead=8, normalize_before=True,num_experts=5)
            decoder_norm = nn.LayerNorm(dim_latent)
            self.fusion_module = TransformerDecoder(decoder_layer=decoder_layer, num_layers=3, norm=decoder_norm)   # 3层的数据量，训练和补充结果
        # FIXME: 这里的设定是输入B只会是1，一个设备处理一个条件。
        
        self.rank_module = MoECrossAttentionReranker(
            dim=dim_latent,
            num_heads=8,
            # depth=2,
            depth=3, 
            top_k=self.rerank_topk,
            dropout=0.1
        )
        # self.rank_module = MoEListwiseCrossAttentionReranker(
        #     dim = dim_latent,
        #     num_heads = 8,
        #     depth = 3,
        #     top_k_query=self.rerank_topk,
        #     top_k_cand=self.rerank_topk,
        #     dropout = 0.1
        # )
        self.rank_classifier = nn.Sequential(
            # nn.Linear(dim_latent*4, dim_latent),
            nn.Linear(dim_latent, dim_latent),
            nn.GELU(),  # 👈 强力推荐
            nn.Dropout(0.1),
            nn.Linear(dim_latent, 1)
        )
        
        self.abnormal_exist_classifier = nn.Linear(dim_latent, 1)  # 用于判断是否存在异常,这个越简单越好
        # self.pos_embedding = PositionalEncoding3D(dim_latent)(torch.zeros(1, 24, 24, 24, dim_latent)) # 1 H W D Dim
        # self.pos_embedding = PositionalEncoding1D(dim_latent)(torch.zeros(1, 24, dim_latent)) # 1 D Dim
        self.pos_embedding = PositionalEncoding2D(dim_latent)(torch.zeros(1, 24, 24, dim_latent)) # 1 H W Dim
        # 修改形状为1 (HW) Dim
        self.pos_embedding = rearrange(self.pos_embedding, 'b h w d -> b (h w) d')
        
        # NOTE

        # text ssl

        self.use_mlm = use_mlm
        self.text_ssl_loss_weight = text_ssl_loss_weight if use_mlm else 0

        # image ssl

        self.use_visual_ssl = use_visual_ssl or exists(visual_ssl)
        self.image_ssl_loss_weight = image_ssl_loss_weight if use_visual_ssl else 0

        # text latent projection

        self.to_text_latent = nn.Linear(dim_text, dim_latent, bias = False)

        # image latent projection

        # if downsample_image_embeds:
        #     #assert use_all_token_embeds, 'must be using all token embeds for contrastive learning in order to downsampling'
        #     dim_conv=512
        #     self.to_visual_latent = nn.Sequential(
        #         RearrangeImage(),
        #         nn.Conv3d(dim_conv, dim_conv, 4, stride = 2, padding = 1, bias = False, groups = dim_conv),
        #         nn.Conv3d(dim_conv, dim_latent, 1),
        #         Rearrange('b c h w z -> b (h w z c)'),
        #         nn.Linear(dim_image, dim_latent, bias = False)
        #     )
        # else:
        # self.to_visual_latent = nn.Linear(dim_image, dim_latent, bias = False)
        self.to_fused_latent = nn.Linear(dim_latent, dim_latent, bias = False)
        # 增加了一个视觉编码器于融合层输入的对齐部分
        self.to_visual_latent = nn.Sequential(
            # 第一层：线性变换
            nn.Linear(dim_image, dim_latent*2),
            # LayerNorm: 你的模型没有BN，这里必须加LN来稳定分布，防止梯度消失/爆炸
            nn.LayerNorm(dim_latent*2),
            # 激活函数: 引入非线性，这是对齐特征空间的关键。GELU 优于 ReLU
            nn.GELU(),
            # 第二层：映射到最终维度
            nn.Linear(dim_latent*2, dim_latent)
        )

        # 【重要】初始化建议：
        # 在模型初始化最后，手动初始化这个Projector，打破对称性
        self.apply(self._init_weights)

        # temperature

        self.temperature = nn.Parameter(torch.tensor(1.))

        # from https://arxiv.org/abs/2111.07783 (FILIP paper)
        self.use_all_token_embeds = use_all_token_embeds

        # proposed in https://arxiv.org/abs/2110.06848 (DCL) and https://arxiv.org/abs/2110.11316 (CLOOB)
        self.decoupled_contrastive_learning = decoupled_contrastive_learning

        self.multiview_loss_weight = multiview_loss_weight

        if tokenizer != None:
            self.tokenizer=tokenizer
        else:
            raise ValueError('Tokenzier is not defined.')
        
    def state_dict(self, *args, **kwargs):
        return super().state_dict(*args, **kwargs)

    def load_state_dict(self, *args, **kwargs):
        return super().load_state_dict(*args, **kwargs)

    def _init_weights(self, m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
    
    def load(self, path):
        path = Path(path)
        assert path.exists()
        pt = torch.load(str(path))
        self.load_state_dict(pt)
        
    def freeze_text_encoder(self):
        for param in self.text_transformer.parameters():
            param.requires_grad = False
        for param in self.to_text_latent.parameters():
            param.requires_grad = False
            
    def open_partial_text_encoder(self):
        for name, param in self.text_transformer.named_parameters():
            if 'pooler' in name or 'encoder.layer.11' in name or 'encoder.layer.10' in name or 'encoder.layer.9' in name or 'encoder.layer.8' in name: # or 'encoder.layer.7' in name or 'encoder.layer.6' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        for param in self.to_text_latent.parameters():
            param.requires_grad = True
            
    # def open_text_encoder(self):
    #     for param in self.text_transformer.parameters():
    #         param.requires_grad = True
    #     for param in self.to_text_latent.parameters():
    #         param.requires_grad = True
    def open_text_encoder(self, unfreeze_layers=None):
        if unfreeze_layers is None:
            unfreeze_layers = [8, 9, 10, 11]  # 默认解冻最后4层
        
        for name, param in self.text_transformer.named_parameters():
            # 检查是否在要解冻的层中
            should_unfreeze = False
            for layer_num in unfreeze_layers:
                if f'encoder.layer.{layer_num}.' in name:
                    should_unfreeze = True
                    break
            
            if should_unfreeze:
                param.requires_grad = True
            elif 'embeddings' in name:  # 解冻嵌入层
                param.requires_grad = True
            elif 'relative_attention_bias' in name or 'pooler' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        
        for param in self.to_text_latent.parameters():
            param.requires_grad = True

    def freeze_vision_encoder(self):
        for param in self.visual_transformer.parameters():
            param.requires_grad = False
        for param in self.to_visual_latent.parameters():  # 这里实际变成了对齐模块，用于对齐融合层的输入
            param.requires_grad = False
        self.cls_token.requires_grad = False
    # 打开最后两层训练
    def open_partial_vision_encoder(self):
        for name, param in self.visual_transformer.named_parameters():
            if 'enc_spatial_transformer.layers.6' in name or 'enc_spatial_transformer.layers.7' in name or 'enc_temporal_transformer' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        for param in self.to_visual_latent.parameters():
            param.requires_grad = True
            
    def open_vision_encoder(self):
        for param in self.visual_transformer.parameters():
            param.requires_grad = True
        for param in self.to_visual_latent.parameters():
            param.requires_grad = True
    
    # def open_vision_encoder(self):
    #     for name,param in self.visual_transformer.named_parameters():
    #         if ('experts.3' in name) or ('experts.4' in name):
    #             param.requires_grad = True
    #         else:
    #             param.requires_grad = False
    #     for param in self.to_visual_latent.parameters():
    #         param.requires_grad = False
    
    # def open_vision_encoder(self):
    #     for name,param in self.visual_transformer.named_parameters():
    #         if ('experts' in name) :
    #             param.requires_grad = True
    #         else:
    #             param.requires_grad = False
    #     for param in self.to_visual_latent.parameters():
    #         param.requires_grad = False
    
    def open_fusion_module(self):
        for param in self.fusion_module.parameters():
            param.requires_grad = True
        for param in self.to_fused_latent.parameters():
            param.requires_grad = True
        # for param in self.to_visual_latent.parameters():
        #     param.requires_grad = True
            
    def freeze_fusion_module(self):
        for param in self.fusion_module.parameters():
            param.requires_grad = False
        for param in self.to_fused_latent.parameters():
            param.requires_grad = False
        for param in self.condition_embeddings.parameters():
            param.requires_grad = False
        for param in self.abnormal_exist_classifier.parameters():
            param.requires_grad = False
        
    def open_rank_module(self):
        for param in self.rank_module.parameters():
            param.requires_grad = True
        for param in self.rank_classifier.parameters():
            param.requires_grad = True
            
    def freeze_rank_module(self):
        for param in self.rank_module.parameters():
            param.requires_grad = False
        for param in self.rank_classifier.parameters():
            param.requires_grad = False

    def tokenize(self, prompt):
        text_tokens=self.tokenizer(prompt, return_tensors="pt", padding="max_length", truncation=True, max_length=512).to(torch.cuda)
        return text_tokens
    
    def token_embedding(self,input_ids):
        input_shape = input_ids.size()
        batch_size, seq_length = input_shape
        if hasattr(self.text_transformer.embeddings, "token_type_ids"):
            print("hahatrue")

        buffered_token_type_ids = self.text_transformer.embeddings.token_type_ids[:, :seq_length]
        buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
        token_type_ids = buffered_token_type_ids_expanded
        text_embeddings = self.text_transformer.embeddings(input_ids = input_ids, token_type_ids = token_type_ids)
        return text_embeddings
    
    def multi_positive_infoNCE_loss(self, pred_sim, true_sim, positive_threshold=0.7):
        """
        infoNCE改进版本，分母去掉positive的元素（除了对角线）
        
        参数：
            pred_sim (torch.Tensor): 预测的相似度矩阵，形状为(b, n, m)，b为batchsize，n为query数量，m为每个query的样本数。
            true_sim (torch.Tensor): 真实的相似度矩阵，形状与pred_sim相同。
            positive_threshold (float): 定义正样本
        """
        true_sim[true_sim>=positive_threshold] = 1
        true_sim[true_sim<positive_threshold] = 0
        
        diag_mask = torch.eye(true_sim.shape[1]).to(true_sim.device)
        diag_mask = repeat(diag_mask, 'n m -> b n m', b=true_sim.shape[0])
        undiag_mask = torch.zeros_like(true_sim)
        undiag_mask[true_sim==0] = 1
        mask = undiag_mask + diag_mask  # b n m

        logits_sum = torch.exp(pred_sim).mul(mask).sum(2)   # b n
        logits_norm = torch.exp(torch.diagonal(pred_sim, dim1=1, dim2=2)) / logits_sum  # b n
        loss = -1 * torch.log(logits_norm)
        loss = loss.mean(dim=1) # b 保留了每个anatomy的loss
        
        return loss

    def multi_positive_infoNCE_loss_temp(self, pred_sim, true_sim, positive_threshold=0.7, tau=0.1):
        """
        infoNCE改进版本,分母去掉positive的元素(除了对角线)
        输入形状: pred_sim [B, local_B, local_B], true_sim [B, local_B, local_B]
        返回形状: [B] - 每个样本的损失值
        """
        B, local_B, _ = pred_sim.shape
        
        # 二值化true_sim
        true_sim_binary = true_sim.clone()
        true_sim_binary[true_sim_binary >= positive_threshold] = 1
        true_sim_binary[true_sim_binary < positive_threshold] = 0
        
        # 创建mask: 对角线 + 非正样本位置
        diag_mask = torch.eye(local_B).unsqueeze(0).expand(B, -1, -1).to(true_sim.device)
        undiag_mask = (true_sim_binary == 0).float()
        mask = undiag_mask + diag_mask   # [B, local_B, local_B]
        
        # 温度缩放
        pred_sim_scaled = pred_sim / tau  # [B, local_B, local_B]
        
        # 提取对角线元素
        diag_elements = torch.diagonal(pred_sim_scaled, dim1=1, dim2=2)  # [B, local_B]
        
        # 计算分母: exp(pred_sim) * mask 的行求和
        logits_sum = (torch.exp(pred_sim_scaled) * mask).sum(dim=2)  # [B, local_B]
        
        # 计算归一化后的logits
        logits_norm = torch.exp(diag_elements) / logits_sum  # [B, local_B]
        
        # 计算损失并对local_B维度求均值
        loss = -torch.log(logits_norm).mean(dim=1)  # [B]
        
        return loss
    
    def multi_positive_supcon_loss(self, pred_sim, true_sim, positive_threshold=0.7, tau=0.1):
        """
        Supervised Contrastive Loss 风格 (修正维度版)。
        返回形状: [B]
        """
        B, local_B, _ = pred_sim.shape
        
        # 1. 定义正样本 Mask
        pos_mask = (true_sim >= positive_threshold).float()
        
        # 2. 数值稳定性处理 (LogSumExp trick)
        scores = pred_sim / tau
        max_score = torch.max(scores, dim=2, keepdim=True)[0].detach()
        scores_sub = scores - max_score
        exp_scores = torch.exp(scores_sub)
        
        # 3. 分母：所有样本的 exp 之和
        denominator = exp_scores.sum(dim=2, keepdim=True) # [B, local_B, 1]
        
        # 4. 计算 Log Probability
        log_probs = scores_sub - torch.log(denominator + 1e-10) # [B, local_B, local_B]
        
        # 5. 计算 Loss
        # 避免除以0
        pos_count = pos_mask.sum(dim=2) # [B, local_B]
        pos_count = torch.clamp(pos_count, min=1.0)
        
        # 遮罩求和：只保留正样本位置的 log_prob
        log_probs_sum = (log_probs * pos_mask).sum(dim=2) # [B, local_B]
        
        # 每个样本的平均 loss
        loss_per_sample = - (log_probs_sum / pos_count) # [B, local_B]
        
        # 6. 修正返回值：在 local_B 维度求均值，保留 B 维度
        return loss_per_sample.mean(dim=1) # [B]
    # 不需要分开吧，还是只用1就行
    def binary_cross_entropy_loss(self, predictions, targets): # 输入都是B localB维度的结果,分开计算不同的loss情况
        criterion = nn.BCEWithLogitsLoss(reduction='none')  # 使用BCEWithLogitsLoss以提高数值稳定性
        loss = criterion(predictions, targets)
        return loss.mean(dim=1)  # 返回每个样本的平均损失，形状为 [B]
    
    def smooth_ap_loss(self, pred_sim, true_sim, positive_threshold=0.9, tau=0.01):
        """
        SmoothAP: Differentiable approximation of Average Precision (fully vectorized).
        对每个 query，用 sigmoid 近似排名位置，使 AP 可微分，直接优化检索排序质量。

        参数:
            pred_sim: [B, N, N] 预测的 cosine similarity
            true_sim: [B, N, N] GT similarity (BioLORD, 0-1)
            positive_threshold: GT >= 此阈值视为正样本 (应与评估对齐=0.9)
            tau: sigmoid 温度，越小近似越精确但梯度越陡 (推荐 0.01-0.05)
        返回: [B] 每个 batch 的 loss
        """
        B, N, _ = pred_sim.shape
        diag_mask = torch.eye(N, device=pred_sim.device).unsqueeze(0).expand(B, -1, -1)
        pos_mask = (true_sim >= positive_threshold).float() * (1.0 - diag_mask)  # [B, N, N]
        valid_mask = 1.0 - diag_mask  # [B, N, N]

        # diff[b, i, j, k] = pred_sim[b, i, j] - pred_sim[b, i, k]
        # 表示对 query i，item j 与 item k 的相似度差
        diff = pred_sim.unsqueeze(3) - pred_sim.unsqueeze(2)  # [B, N, N, N]

        # sigmoid 近似: item k 排在 item j 前面的概率，排除自身(对角线)
        approx_ranks = torch.sigmoid(-diff / tau) * valid_mask.unsqueeze(2)  # [B, N, N, N]

        # 排在 j 前面的正样本数
        rank_pos = (approx_ranks * pos_mask.unsqueeze(2)).sum(dim=3)  # [B, N, N]
        # 排在 j 前面的总样本数
        rank_all = approx_ranks.sum(dim=3)  # [B, N, N]

        # 修正 k=j 自身比较: σ(0)=0.5 应为 1（标准 SmoothAP 排除 k=j 后 +1）
        # 等价于 +0.5 补偿差值，仅对 gallery 内的 j (j≠i) 生效
        rank_all = rank_all + 0.5 * valid_mask   # [B, N, N]
        rank_pos = rank_pos + 0.5 * pos_mask     # [B, N, N]

        # precision at position j for query i
        precision_at_j = rank_pos / (rank_all + 1e-10)  # [B, N, N]

        # 每个 query 的正样本数 & AP
        n_pos = pos_mask.sum(dim=2)  # [B, N]
        has_pos = (n_pos > 0).float()  # [B, N]
        ap = (precision_at_j * pos_mask).sum(dim=2) / (n_pos + 1e-10)  # [B, N]

        # Loss: 1 - AP，仅对有正样本的 query 计算
        losses = ((1.0 - ap) * has_pos).sum(dim=1)  # [B]

        # 对所有有正样本的 query 平均
        valid_query_count = has_pos.sum(dim=1)  # [B]
        losses = losses / (valid_query_count + 1e-10)
        return losses  # [B]
    
    # 针对 rerank 阶段的 SmoothAP 版本，输入是 [B, N, K] 的局部相似度矩阵和预测，输出是 [B] 的 loss
    def smooth_ap_loss_rerank(self, pred_sim, true_sim, positive_threshold=0.9, tau=0.01):
        """
        SmoothAP loss for top-K retrieval setting.

        参数:
            pred_sim: [B, N, K] 预测的 cosine similarity，每个 query 对应 K 个候选
            true_sim: [B, N, K] GT similarity，每个 query 对应的 K 个候选的真实相似度
            positive_threshold: GT >= 此阈值视为正样本
            tau: sigmoid 温度
        返回: [B] 每个 batch 的 loss
        """
        B, N, K = pred_sim.shape

        # 正样本 mask，不需要对角线 mask（候选集不包含 query 自身，或即使包含也视为普通候选）
        pos_mask = (true_sim >= positive_threshold).float()  # [B, N, K]

        # diff[b, i, j, k] = pred_sim[b, i, j] - pred_sim[b, i, k]
        # 对 query i，候选 j 与候选 k 的预测相似度差
        diff = pred_sim.unsqueeze(3) - pred_sim.unsqueeze(2)  # [B, N, K, K]

        # sigmoid 近似: 候选 k 排在候选 j 前面的概率
        # 对角线 (k==j) 时 diff=0, sigmoid=0.5，后面会修正
        approx_ranks = torch.sigmoid(-diff / tau)  # [B, N, K, K]

        # 排在 j 前面的正样本数（沿 k 维度求和）
        rank_pos = (approx_ranks * pos_mask.unsqueeze(2)).sum(dim=3)  # [B, N, K]
        # 排在 j 前面的总样本数
        rank_all = approx_ranks.sum(dim=3)  # [B, N, K]

        # 修正 k==j 自身比较: σ(0)=0.5，应计为 1（自身一定排在自身位置）
        # rank_all 对每个 j 多加 0.5（因为 k=j 时贡献了 0.5，应该是 1）
        # rank_pos 仅当 j 本身是正样本时才补 0.5
        rank_all = rank_all + 0.5               # [B, N, K]
        rank_pos = rank_pos + 0.5 * pos_mask    # [B, N, K]

        # precision at position j for query i
        precision_at_j = rank_pos / (rank_all + 1e-10)  # [B, N, K]

        # 每个 query 的正样本数 & AP
        n_pos = pos_mask.sum(dim=2)       # [B, N]
        has_pos = (n_pos > 0).float()     # [B, N]
        ap = (precision_at_j * pos_mask).sum(dim=2) / (n_pos + 1e-10)  # [B, N]

        # Loss: 1 - AP，仅对有正样本的 query 计算
        losses = ((1.0 - ap) * has_pos).sum(dim=1)  # [B]
        valid_query_count = has_pos.sum(dim=1)       # [B]
        losses = losses / (valid_query_count + 1e-10)
        return losses  # [B]

    def infoNCE_loss(self, pred_sim, temperature=0.07):
        """
        标准 InfoNCE Loss
        - 正样本：对角线元素
        - 分母：每行所有元素（包括正样本自身）
        """
        # 温度系数缩放
        pred_sim = pred_sim / temperature
        
        # 分母：每行所有 exp 之和
        logits_sum = torch.exp(pred_sim).sum(dim=1)
        
        # 分子：对角线元素（正样本对）
        logits_pos = torch.exp(torch.diag(pred_sim))
        
        # loss = -log(正样本 / 所有样本)
        loss = -torch.log(logits_pos / logits_sum)
        
        return loss.mean()
    
    def multi_positive_supcon_loss_no_diag(self, pred_sim, true_sim, positive_threshold=0.7, tau=0.1):
        """
        Supervised Contrastive Loss (排除对角线自身版本)
        适用于 I2I 任务，防止 Self-Similarity 主导梯度。
        """
        B, local_B, _ = pred_sim.shape
        
        # 1. 生成对角线 Mask (1 代表是对角线)
        # [B, local_B, local_B]
        diag_mask = torch.eye(local_B, device=pred_sim.device).unsqueeze(0).expand(B, -1, -1)
        
        # 2. 定义正样本 Mask
        # 逻辑：相似度 > 阈值 AND 不是对角线自身
        # 这样分子就不会包含“自己拉近自己”
        pos_mask = (true_sim >= positive_threshold).float() * (1.0 - diag_mask)
        
        # 3. 数值稳定性处理
        scores = pred_sim / tau
        max_score = torch.max(scores, dim=2, keepdim=True)[0].detach()
        scores_sub = scores - max_score
        exp_scores = torch.exp(scores_sub)
        
        # 4. 分母计算：所有样本的 exp 之和，但是减去对角线元素
        # 逻辑：Sum_All - (Exp_Diag * 1)
        # 注意：因为是对角线，直接乘 diag_mask 就能提取出来
        denominator_all = exp_scores.sum(dim=2, keepdim=True) # 包含对角线
        denominator_diag = (exp_scores * diag_mask).sum(dim=2, keepdim=True) # 仅对角线
        
        # 真正的分母 = 总和 - 对角线
        denominator = denominator_all - denominator_diag
        
        # 5. 计算 Log Probability
        # log_prob = score - log(denominator)
        log_probs = scores_sub - torch.log(denominator + 1e-10)
        
        # 6. 计算 Loss
        # 计算每行有多少个“非自身”的正样本
        pos_count = pos_mask.sum(dim=2) # [B, local_B]
        
        # 处理没有正样本的情况（除了自己没有别的相似图）
        # 这种情况下 loss 应该为 0，或者这就成了一个只有负样本的行（SupCon通常会mask掉这种情况）
        # 这里为了简单，如果没有正样本，就设为1避免除零，然后通过mask让分子为0
        has_pos_mask = (pos_count > 0).float()
        pos_count = torch.clamp(pos_count, min=1.0)
        
        # 遮罩求和
        log_probs_sum = (log_probs * pos_mask).sum(dim=2) # [B, local_B]
        
        # 计算均值
        loss_per_sample = - (log_probs_sum / pos_count)
        
        # 如果某一行没有正样本，该行的 loss 设为 0
        loss_per_sample = loss_per_sample * has_pos_mask
        
        return loss_per_sample.mean(dim=1) # 返回 [B]

    def create_block_diagonal_gt(self, local_gt_sim):
        """
        辅助函数：将 [B, local_B, local_B] 的局部GT矩阵
        构建成 [B*local_B, B*local_B] 的全局块对角矩阵。
        跨 Condition 的区域自动填充为 0 (视为负样本)。
        """
        # local_gt_sim: [B, N, N]
        B, N, N = local_gt_sim.shape
        # 将其转为 list of tensors: [tensor(N,N), tensor(N,N), ...]
        tensor_list = [local_gt_sim[i] for i in range(B)]
        # 构建块对角矩阵 [B*N, B*N]
        global_gt = torch.block_diag(*tensor_list)
        return global_gt

    def uni_multi_positive_supcon_loss(self, embeddings_1, embeddings_2, local_gt_sim, positive_threshold=0.7, tau=0.1):
        """
        I2T Loss (或者非对称 I2I): 
        输入为 Embedding [B, local_B, D]
        计算全局 InfoNCE Loss。
        """
        # 1. 维度展平 [B, local_B, D] -> [B*local_B, D]
        # 确保输入已经 Normalize 过了，否则这里需要 F.normalize
        B, local_B, D = embeddings_1.shape
        flat_emb_1 = embeddings_1.reshape(-1, D) # [Total_N, D]
        flat_emb_2 = embeddings_2.reshape(-1, D) # [Total_N, D]
        
        # 2. 计算全局相似度矩阵 [Total_N, Total_N]
        # 这一步让模型看到了跨 Condition 的负样本
        pred_sim = torch.matmul(flat_emb_1, flat_emb_2.T) 
        
        # 3. 构建全局 GT 矩阵 [Total_N, Total_N]
        # 只有同 Condition 内的部分位置是 1 (或大于阈值)，其余全是 0
        true_sim = self.create_block_diagonal_gt(local_gt_sim)
        
        # --- 以下逻辑与原始 InfoNCE 类似，但维度变成了全局 Total_N ---
        
        # 4. 定义正样本 Mask
        pos_mask = (true_sim >= positive_threshold).float()
        
        # 5. 数值稳定性处理
        scores = pred_sim / tau
        max_score = torch.max(scores, dim=1, keepdim=True)[0].detach()
        scores_sub = scores - max_score
        exp_scores = torch.exp(scores_sub)
        
        # 6. 分母：所有样本的 exp 之和 (包含同一 Condition 的负样本 + 其他 Condition 的负样本)
        denominator = exp_scores.sum(dim=1, keepdim=True) # [Total_N, 1]
        
        # 7. 计算 Log Probability
        log_probs = scores_sub - torch.log(denominator + 1e-10) # [Total_N, Total_N]
        
        # 8. 计算 Loss
        pos_count = pos_mask.sum(dim=1) # [Total_N]
        pos_count = torch.clamp(pos_count, min=1.0)
        
        # 遮罩求和：只保留正样本位置的 log_prob
        log_probs_sum = (log_probs * pos_mask).sum(dim=1) # [Total_N]
        
        # 每个样本的 loss
        loss_per_sample = - (log_probs_sum / pos_count)
        
        # 返回标量 mean loss
        return loss_per_sample.mean()

    def uni_multi_positive_supcon_loss_no_diag(self, embeddings, local_gt_sim, positive_threshold=0.7, tau=0.1):
        """
        I2I Loss (排除自身):
        输入为 Embedding [B, local_B, D]
        计算全局 I2I Contrastive Loss，排除对角线自身。
        """
        # 1. 维度展平 [B, local_B, D] -> [Total_N, D]
        B, local_B, D = embeddings.shape
        Total_N = B * local_B
        
        flat_emb = embeddings.reshape(-1, D)
        
        # 2. 计算全局相似度矩阵 [Total_N, Total_N]
        pred_sim = torch.matmul(flat_emb, flat_emb.T)
        
        # 3. 构建全局 GT 矩阵
        true_sim = self.create_block_diagonal_gt(local_gt_sim)
        
        # 4. 生成全局对角线 Mask (1 代表是对角线)
        diag_mask = torch.eye(Total_N, device=embeddings.device)
        
        # 5. 定义正样本 Mask
        # 必须排除对角线 (自己和自己不是需要拉近的正样本，是无效对)
        pos_mask = (true_sim >= positive_threshold).float() * (1.0 - diag_mask)
        
        # 6. 数值稳定性处理
        scores = pred_sim / tau
        max_score = torch.max(scores, dim=1, keepdim=True)[0].detach()
        scores_sub = scores - max_score
        exp_scores = torch.exp(scores_sub)
        
        # 7. 分母计算：Sum_All - Exp_Diag
        # 这里的 Sum_All 包含了跨 Condition 的所有负样本
        denominator_all = exp_scores.sum(dim=1, keepdim=True)
        denominator_diag = (exp_scores * diag_mask).sum(dim=1, keepdim=True)
        denominator = denominator_all - denominator_diag
        
        # 8. 计算 Log Probability
        log_probs = scores_sub - torch.log(denominator + 1e-10)
        
        # 9. 计算 Loss
        pos_count = pos_mask.sum(dim=1)
        
        # 处理可能的无正样本情况 (除了自己没有别的同类)
        has_pos_mask = (pos_count > 0).float()
        pos_count = torch.clamp(pos_count, min=1.0)
        
        log_probs_sum = (log_probs * pos_mask).sum(dim=1)
        
        loss_per_sample = - (log_probs_sum / pos_count)
        loss_per_sample = loss_per_sample * has_pos_mask
        
        return loss_per_sample.mean()
    # triplet loss无需修改
    def triplet_loss(self, pred_sim, true_sim, margin=0.2, gt_threshold=0.3, positive_threshold=0.7, negative_threshold=0.4, temp=0.07):
        """
        计算Triplet Loss，用于排序模型的训练。

        参数：
            pred_sim (torch.Tensor): 预测的相似度矩阵，形状为(b, n, m)，b为batchsize，n为query数量，m为每个query的样本数。
            true_sim (torch.Tensor): 真实的相似度矩阵，形状与pred_sim相同。
            margin (float): 正负样本之间的最小间隔。
            gt_threshold (float): 定义正负样本

        返回：
            torch.Tensor: 计算得到的Triplet Loss标量值。
        """
        # 确保输入张量的形状一致
        assert pred_sim.shape == true_sim.shape, "预测矩阵和真实矩阵形状不一致"
        b, n, m = pred_sim.shape
        # 生成三维掩码矩阵，标记所有满足 true_sim[i] > true_sim[j] 的位置
        # true_sim.unsqueeze(3) 形状为(b, n, m, 1), true_sim.unsqueeze(2) 形状为(b, n, 1, m)
        mask = (true_sim.unsqueeze(3) - true_sim.unsqueeze(2)) > gt_threshold  # 形状(n, m, m),还要满足这个差值部分
        abs_mask =(true_sim > positive_threshold).unsqueeze(3) # 同时满足ij>0.7
        mask = mask & abs_mask
        abs_mask =(true_sim < negative_threshold).unsqueeze(2) # 同时满足ik<0.4
        mask = mask & abs_mask
        # 计算预测相似度的差值矩阵 pred_sim[j] - pred_sim[i]
        diff = - pred_sim.unsqueeze(3) + pred_sim.unsqueeze(2)  # 形状(b, n, m, m)  # 逼近-margion # pred_sim.unsqueeze(3)-pred_sim.unsqueeze(2)涨到0.2
        magnify = true_sim.unsqueeze(3) - true_sim.unsqueeze(2)  # 形状(b, n, m, m)
        # 计算Triplet Loss各项：max(0, diff + margin)
        losses = torch.clamp_min(diff + margin, 1e-8)
        # softplus : log(1+exp(.../temp))
        losses = torch.log(1 + torch.exp((losses * magnify) / temp))    # true_sim.unsqueeze(3)-true_sim.unsqueeze(2)越大，weight越大
        # 应用掩码，仅保留有效正负对
        masked_losses = losses * mask.float()
        # 计算每个query的总损失和有效对数量
        sum_per_query = masked_losses.sum(dim=(2, 3))  # 形状(b, n)
        num_pairs_per_query = mask.sum(dim=(2, 3))     # 形状(b, n)
        # 处理无有效对的情况，避免除零
        avg_loss_per_query = sum_per_query / (num_pairs_per_query + 1e-10)   # 避免分母为0（全部满足）
        avg_loss_per_query[num_pairs_per_query == 0] = 0  # 无有效对的query损失置零（好像是多余的）
        # 返回batch平均损失
        valid_mask = num_pairs_per_query != 0
        valid_count = valid_mask.float().sum(dim=1) # b # n个sample中有几个loss不为0
        # avg_valid_loss = torch.where(
        #     valid_count > 0,
        #     (avg_loss_per_query * valid_mask.float()).sum(dim=1) / valid_count,
        #     torch.zeros_like(valid_count)
        # )
        # 7.28修改avg_valid_loss的计算方式，避免除0
        epsilon = 1e-8
        safe_valid_count = valid_count + epsilon  # 防止0
        avg_valid_loss = ((avg_loss_per_query * valid_mask.float()).sum(dim=1) / safe_valid_count) * (valid_count > 0)

        return avg_valid_loss, torch.sum(mask).item()

    def multi_positive_infoNCE_loss_uncon(self, pred_sim, true_sim, positive_threshold=0.7):
        """
        infoNCE改进版本，分母去掉positive的元素（除了对角线）
        """
        true_sim[true_sim>=positive_threshold] = 1
        true_sim[true_sim<positive_threshold] = 0
        
        diag_mask = torch.eye(true_sim.shape[0]).to(true_sim.device)
        undiag_mask = torch.zeros_like(true_sim)
        undiag_mask[true_sim==0] = 1
        mask = undiag_mask + diag_mask

        logits_sum = torch.exp(pred_sim).mul(mask).sum(1)
        logits_norm = torch.exp(torch.diag(pred_sim)) / logits_sum
        loss = -1 * torch.log(logits_norm)
        loss = loss.mean()
        
        return loss
    # 修改了infonce loss的部分
    def multi_positive_infoNCE_loss_uncon_temp(self, pred_sim, true_sim, positive_threshold=0.7,tau = 0.1):
        """
        infoNCE改进版本，分母去掉positive的元素（除了对角线）
        """
        true_sim[true_sim>=positive_threshold] = 1
        true_sim[true_sim<positive_threshold] = 0
        
        diag_mask = torch.eye(true_sim.shape[0]).to(true_sim.device)
        undiag_mask = torch.zeros_like(true_sim)
        undiag_mask[true_sim==0] = 1
        mask = undiag_mask + diag_mask
        pred_sim = pred_sim / tau  # 温度缩放
        logits_sum = torch.exp(pred_sim).mul(mask).sum(1)
        logits_norm = torch.exp(torch.diag(pred_sim)) / logits_sum
        loss = -1 * torch.log(logits_norm)
        loss = loss.mean()
        
        return loss
    
    def triplet_loss_uncon(self, pred_sim, true_sim, margin=0.2, gt_threshold=0.3, positive_threshold=0.7, negative_threshold=0.4, temp=0.07):
        """
        计算Triplet Loss，用于排序模型的训练。

        参数：
            pred_sim (torch.Tensor): 预测的相似度矩阵，形状为(n, m)，n为query数量，m为每个query的样本数。
            true_sim (torch.Tensor): 真实的相似度矩阵，形状与pred_sim相同。
            margin (float): 正负样本之间的最小间隔。
            gt_threshold (float): 定义正负样本

        返回：
            torch.Tensor: 计算得到的Triplet Loss标量值。
        """
        # 确保输入张量的形状一致
        assert pred_sim.shape == true_sim.shape, "预测矩阵和真实矩阵形状不一致"
        n, m = pred_sim.shape
        # 生成三维掩码矩阵，标记所有满足 true_sim[i]-true_sim[j] > true_sim[i] - true_sim[k] 的位置
        # true_sim.unsqueeze(2) 形状为(n, m, 1), true_sim.unsqueeze(1) 形状为(n, 1, m)
        mask = (true_sim.unsqueeze(2) - true_sim.unsqueeze(1)) > gt_threshold  # 形状(n, m, m)
        abs_mask =(true_sim > positive_threshold).unsqueeze(2) # 同时满足ij>0.7
        mask = mask & abs_mask
        abs_mask =(true_sim < negative_threshold).unsqueeze(1) # 同时满足ik<0.4
        mask = mask & abs_mask
        # 计算预测相似度的差值矩阵 pred_sim[j] - pred_sim[i]
        diff = - pred_sim.unsqueeze(2) + pred_sim.unsqueeze(1)  # 形状(n, m, m)
        # 计算Triplet Loss各项：max(0, diff + margin)
        losses = torch.clamp_min(diff + margin, 1e-8)
        # softplus : log(1+exp(.../temp))
        losses = F.softplus(losses / temp) # torch.log(1 + torch.exp(losses / temp))
        # 应用掩码，仅保留有效正负对
        masked_losses = losses * mask.float()
        # 计算每个query的总损失和有效对数量
        sum_per_query = masked_losses.sum(dim=(1, 2))  # 形状(n,)
        num_pairs_per_query = mask.sum(dim=(1, 2)) + 1e-14     # 形状(n,)   # 避免分母为0（全部满足）
        # 处理无有效对的情况，避免除零
        avg_loss_per_query = sum_per_query / num_pairs_per_query.clamp(min=1)
        avg_loss_per_query[num_pairs_per_query == 0] = 0  # 无有效对的query损失置零
        # 返回batch平均损失
        return avg_loss_per_query.mean()
    
    def forward(
            self,
            text,   # B
            image,  # B local_B 1 H W D
            device,
            local_text=None, # (B*local_B)     # 对应的是一个一维的结果，后面再调整即可
            gt_similarity_matrix=None,  # B local_B local_B
            return_loss = False,
            return_latents = False,
            return_latents_all = False,
            is_condition = True,
            modal_embedding = False,
            modal_indexs = None,
            abnormal_exist_labels = None,   # 大小是 B local_B
    ):
        if is_condition:
            temperature = self.log_temperature.exp()
            temperature = torch.clamp(temperature, min=0.05, max=0.5)
            condition_index = text
            B, device = condition_index.shape[0], device
            
            if return_loss:
                assert gt_similarity_matrix is not None, 'calculate loss but ground truth similarity matrix is not given'

            condition_latents = self.condition_embeddings(condition_index)
            
            # NOTE: need to flatten B local_B ahead
            B, local_B, _, D, H, W = image.shape
            print('image_shape',image.shape)
            image = rearrange(image, 'B N O D H W -> (B N) O D H W')
            # NOTE
            # modal_index是B大小的，要变成(B * local_B)大小
            modal_indexs = modal_indexs.unsqueeze(1).repeat(1, local_B).view(-1)  # (B * local_B) 重复之后，condition就能和uncon的情况一样了
            enc_image = self.visual_transformer(image, modal_embedding=modal_embedding,modal_indexs=modal_indexs)   # B*N (1 24 24) 512(vis_dim)
            # enc_image = self.to_visual_latent(enc_image)  # B*N (1 24 24) 512 这里被重新关闭了, 不使用这个额外的部分做visual的映射
            # 增加了这里的一部分，前面的都重新训练即可
            image_embeds = rearrange(enc_image,'bn hw dim -> hw bn dim')  # (HWD) B*N Dim，这里增加了一个rearange部分，给数据进行了切换shape的部分.
            
            # 这里修改了condition embedding的尺寸用于和原始的部分对齐
            condition_latents =  repeat(condition_latents, 'b dim -> one (b n) dim', one=1 , n=local_B)  # one (B*N) 512
            image_tokens = image_embeds
            t = image_tokens.shape[0]
            temp_pos_embedding = self.pos_embedding[:,:t, :]  # 1 t 512
            pos = repeat(temp_pos_embedding, 'one d dim -> d (one bn) dim', bn=image_tokens.shape[1]).to(device)  # 24*24, B*N, 512
            fused_latents_ori, atten_layers = self.fusion_module(condition_latents, image_tokens, pos=pos,expert_indices=modal_indexs) # 1 B*N 512
            # 上面的代码会返回attention的信息
            last_attn_map = atten_layers[-1]  # 取最后一层的attention map
            # 修改infonce loss的结果，使其能够对齐结果
            fused_latents = fused_latents_ori[0]
            print('fused',fused_latents.shape)
            fused_latents = self.to_fused_latent(fused_latents) # B*N 512
                
            image_latents = fused_latents
            
            abnormal_exist_pred = self.abnormal_exist_classifier(image_latents)  # B*N 1
            
            abnormal_exist_pred = abnormal_exist_pred.view(B, local_B)
            
            fused_latents = rearrange(fused_latents, '(b n) dim -> b n dim', n=local_B)
            if local_text is not None:
                text_embeddings = self.text_transformer(**local_text)  # (B*local_B) L D
                text_embeddings = mean_pooling(text_embeddings, local_text['attention_mask'])
                text_embeddings = F.normalize(text_embeddings, p=2, dim=1)
                text_latents = self.to_text_latent(text_embeddings)
                # NOTE
                text_latents = rearrange(text_latents, '(b n) dim -> b n dim', n=local_B)
                text_latents = F.normalize(text_latents, dim = -1)
            else:
                text_latents = None
            image_latents = fused_latents   # 这里的iamge_latents是融合之后的结果
            image_latents = F.normalize(image_latents, dim = -1)
            if return_latents_all:
                # [修改点]: 这里不仅返回原来的，还需要返回 pos 和 last_attn_map 用于 rerank
                # pos 原来形状是 (24*24, B*N, 512)，为了方便存储，转置为 (B*N, 24*24, 512)
                abnormal_probs = torch.sigmoid(abnormal_exist_pred) # 还要使用sigmod的结果
                abnormal_preds = (abnormal_probs > 0.5).float() # 返回预测的结果
                pos_out = pos.transpose(0, 1) 
                if text_latents == None:
                    return None, image_latents.squeeze(), fused_latents_ori.squeeze(), enc_image, atten_layers, pos_out, last_attn_map,abnormal_preds
                else:
                    return text_latents.squeeze(), image_latents.squeeze(), fused_latents_ori.squeeze(), enc_image, atten_layers, pos_out, last_attn_map,abnormal_preds
            if return_latents:
                abnormal_probs = torch.sigmoid(abnormal_exist_pred) # 还要使用sigmod的结果
                abnormal_preds = (abnormal_probs > 0.5).float() # 返回预测的结果
                return text_latents, image_latents, abnormal_preds, self.temperature.exp()
            
            image_to_image = einsum('b n d, b m d -> b n m', (image_latents, image_latents))   # B local_B local_B
            # 选择对，然后交互的重点是什么？
            rerank_loss = torch.zeros(len(image_to_image)).to(device)   # B
            if self.use_rerank_loss:
                # # pairs_tensor, labels_tensor = generate_rerank_training_data(gt_similarity_matrix[0],image_to_image[0])
                pairs_tensor, labels_tensor = generate_rerank_training_data_uniform(gt_similarity_matrix[0],image_to_image[0],top_k_pool=60,num_neg=1,positivate=self.positive_threshold)
                pairs_num = pairs_tensor.shape[0]
                expert_indices = modal_indexs[0].expand(pairs_num)  # num_pairs
                idx1 = pairs_tensor[:,0]
                idx2 = pairs_tensor[:,1]
                pos_emb_batch = pos.transpose(0,1)
                image_tokens_batch = image_tokens.transpose(0,1)
                image1_condition_cls = fused_latents_ori.transpose(0,1)[idx1]   # num_pairs 1 D
                image2_condition_cls = fused_latents_ori.transpose(0,1)[idx2]   # num_pairs 1 D
                
                tokens1_topk = select_top_k_by_attention(attn_weights=last_attn_map[idx1],image_tokens=image_tokens_batch[idx1],pos_embeddings=pos_emb_batch[idx1], top_k=self.rerank_topk)  # num_pairs k D
                tokens2_topk = select_top_k_by_attention(attn_weights=last_attn_map[idx2],image_tokens=image_tokens_batch[idx2],pos_embeddings=pos_emb_batch[idx2], top_k=self.rerank_topk)  # num_pairs k D
                
                tokens1_topk = torch.cat([image1_condition_cls, tokens1_topk], dim=1)  # num_pairs (1+k) D
                tokens2_topk = torch.cat([image2_condition_cls, tokens2_topk], dim=1)  # num_pairs (1+k) D
                rank_logits = self.rank_module(tokens1_topk, tokens2_topk,expert_indices).squeeze(1) # num_pairs 1, 内部自带一个MOE分类器
                print('debug rerank logits: ', rank_logits.shape, labels_tensor.shape)
                rerank_loss = self.binary_cross_entropy_loss(rank_logits.unsqueeze(0), labels_tensor.unsqueeze(0).to(device))

            image_to_text = einsum('b n d, b m d -> b n m', image_latents, text_latents)
            text_to_image = image_to_text.transpose(-2, -1)  # 直接转置，避免重复计算
            
            
            if not (self.use_infoNCE_loss > 0 or self.use_triplet_loss > 0 or self.use_abnormal_exist_loss >0 or self.use_rerank_loss>0):   # 必须有一种loss才能计算
                raise ValueError("To Calculate Loss, use_triplet_loss and use_infoNCE_loss")

            image_to_image_triplet_loss = torch.zeros(len(image_to_image)).to(device)
            # if self.use_triplet_loss > 0:
            #     image_to_image_triplet_loss,valid_triplet_count = self.triplet_loss(image_to_image, gt_similarity_matrix, margin=self.triplet_loss_margin, gt_threshold=self.positive_distance_threshold, positive_threshold=self.positive_threshold, negative_threshold=self.negative_threshold)
            # else:
            #     valid_triplet_count = 1
            if self.use_triplet_loss >0:
                # 
                # image_to_image_triplet_loss = self.multi_positive_supcon_loss_no_diag(image_to_image, gt_similarity_matrix, positive_threshold=self.positive_threshold,tau=temperature)
                print('smooth_ap_tau: ', self.smooth_ap_tau)
                image_to_image_triplet_loss = self.smooth_ap_loss(image_to_image, gt_similarity_matrix, positive_threshold=self.positive_threshold, tau=self.smooth_ap_tau)
                valid_triplet_count = 1
                # uni_image_to_image_triplet_loss = self.uni_multi_positive_supcon_loss_no_diag(image_latents, gt_similarity_matrix, positive_threshold=self.positive_threshold,tau=0.07)
            else:
                valid_triplet_count = 1
            
            text_to_image_infoNCE_loss = torch.zeros(len(image_to_image)).to(device)
            image_to_text_infoNCE_loss = torch.zeros(len(image_to_image)).to(device)
            # if self.use_infoNCE_loss > 0:
            #     if self.infonce_temp == True:
            #         text_to_image_infoNCE_loss = self.multi_positive_infoNCE_loss_temp(text_to_image, gt_similarity_matrix, positive_threshold=self.positive_threshold)
            #         image_to_text_infoNCE_loss = self.multi_positive_infoNCE_loss_temp(image_to_text, gt_similarity_matrix, positive_threshold=self.positive_threshold)
            #     else:
            #         raise ValueError("Currently only support infonce_temp=True")
            if self.use_infoNCE_loss > 0:
                if self.infonce_temp == True:
                    text_to_image_infoNCE_loss = self.multi_positive_supcon_loss(text_to_image, gt_similarity_matrix, positive_threshold=self.positive_threshold,tau=temperature)
                    image_to_text_infoNCE_loss = self.multi_positive_supcon_loss(image_to_text, gt_similarity_matrix, positive_threshold=self.positive_threshold,tau=temperature)
                    # uni_text_to_image_infoNCE_loss = self.uni_multi_positive_supcon_loss(image_latents, text_latents, gt_similarity_matrix, positive_threshold=self.positive_threshold,tau=0.07)
                    # uni_image_to_text_infoNCE_loss = self.uni_multi_positive_supcon_loss(text_latents, image_latents, gt_similarity_matrix, positive_threshold=self.positive_threshold,tau=0.07)
                else:
                    raise ValueError("Currently only support infonce_temp=True")
            image_binary_loss = torch.zeros(len(image_to_image)).to(device)
            # 增加二分类的数据信息
            if self.use_abnormal_exist_loss > 0:
                image_binary_loss = self.binary_cross_entropy_loss(abnormal_exist_pred, abnormal_exist_labels)
            image_text_infoNCE_loss = (text_to_image_infoNCE_loss + image_to_text_infoNCE_loss) / 2
            # uni_image_text_infoNCE_loss = (uni_text_to_image_infoNCE_loss + uni_image_to_text_infoNCE_loss) / 2
            # calculate CL loss
            
            cl_losses = torch.zeros(1).to(device)
            # print('debug: ', self.use_triplet_loss, self.use_infoNCE_loss)
            # print('debug: image_to_image_triplet_loss ', image_to_image_triplet_loss)
            # print('debug: image_text_infoNCE_loss ', image_text_infoNCE_loss)
            cl_losses += self.use_triplet_loss * image_to_image_triplet_loss.mean()
            cl_losses += self.use_infoNCE_loss * image_text_infoNCE_loss.mean()
            cl_losses += self.use_abnormal_exist_loss * image_binary_loss.mean()
            cl_losses += self.use_rerank_loss * rerank_loss.mean()
            # cl_losses += self.use_triplet_loss * uni_image_to_image_triplet_loss.mean()
            # cl_losses += self.use_infoNCE_loss * uni_image_text_infoNCE_loss.mean()
            
            
            return cl_losses, image_text_infoNCE_loss, image_to_image_triplet_loss,image_binary_loss,rerank_loss,valid_triplet_count
        
        
        else:
            b, device = text.input_ids.shape[0], device
                # derive text mask

            if return_loss:
                assert gt_similarity_matrix is not None, 'calculate loss but ground truth similarity matrix is not given'

            # concat augmented texts and images and do some asserts

            num_batch_texts = num_batch_images = 1

            # text_embeddings = self.text_transformer(text.input_ids, attention_mask = text.attention_mask )
            text_embeddings = self.text_transformer(**text)
            text_embeddings = mean_pooling(text_embeddings, text['attention_mask'])
        # Normalize embeddings
            text_embeddings = F.normalize(text_embeddings, p=2, dim=1)
            text_embeds = text_embeddings  # B 768
            
            enc_image= self.visual_transformer(image,modal_embedding=modal_embedding,modal_indexs = modal_indexs)   # B 24 24 24 512(vis_dim)
            # 返回的结果应该是 (B,(24*24),512)
        
            enc_image = rearrange(enc_image,'bn hw dim -> hw bn dim')
            
            cls_token = repeat(self.cls_token, '1 1 dim -> 1 bn dim', bn=enc_image.shape[1])  # 1 B*N 512
               
            t = enc_image.shape[0] # 对应的是序列长度了
            temp_pos_embedding = self.pos_embedding[:,:t, :]  # 1 t 512

            pos = repeat(temp_pos_embedding, 'one d dim -> d (one bn) dim', bn=enc_image.shape[1]).to(device)  # 24*24, B*N, 512
            # h_r, w_r, z_r = enc_image.shape[1], enc_image.shape[2], enc_image.shape[3]
            fused_latents, _ = self.fusion_module(cls_token, enc_image, pos=pos,expert_indices=modal_indexs) # B 1 512
            fused_latents = fused_latents[0]
            fused_latents = self.to_fused_latent(fused_latents) # B 512
            
            image_latents = fused_latents

            text_latents = self.to_text_latent(text_embeds) # B 512

            # image_latents = self.to_visual_latent(image_embeds) # B 512 # 有一个就足够了哈

            text_latents, image_latents = map(l2norm, (text_latents, image_latents))

            # whether to early return latents

            if return_latents:
                return text_latents, image_latents, self.temperature.exp()

            # get temperature

            temp = self.temperature.exp()
            text_latents = rearrange(text_latents, '(m b) ... -> m b ...', m = num_batch_texts) # 1 B 512
            image_latents = rearrange(image_latents, '(m b) ... -> m b ...', m = num_batch_images) # 1 B 512
            
            text_to_image = einsum('m t d, n i d -> m n t i', text_latents, image_latents) * temp   # 1 1 B B
            image_to_text = rearrange(text_to_image, '... t i -> ... i t')
            
            if self.use_image2image_loss:
                image_to_image = einsum('m t d, n i d -> m n t i', image_latents, image_latents) * temp   # 1 1 B B
                image_to_image = image_to_image.squeeze()
                
            # calculate loss

            text_to_image = rearrange(text_to_image, 'm n ... -> (m n) ...').squeeze()    # 1 B B -> B B
            image_to_text = rearrange(image_to_text, 'm n ... -> (m n) ...').squeeze()    # 1 B B -> B B
            
            # NOTE: Official loss calculation only treat diagonal as positive samples (1), others as negative (0)
            # NOTE: Our implementation with soft similarity between each sample pair
            
            # gt_similarity_matrix = F.normalize(gt_similarity_matrix, dim=0) # B B
            
            if not (self.use_uncon_infoNCE_loss > 0 or self.use_uncon_triplet_loss > 0 or self.use_image2image_loss > 0):
                raise ValueError("To Calculate Loss, use_triplet_loss and use_infoNCE_loss and use_image2image_loss cannot all be False")
            
            text_to_image_triplet_loss = torch.zeros(1).to(device)
            image_to_text_triplet_loss = torch.zeros(1).to(device)
            if self.use_uncon_triplet_loss > 0:
                text_to_image_triplet_loss = self.triplet_loss_uncon(text_to_image, gt_similarity_matrix, margin=self.triplet_loss_margin, gt_threshold=self.positive_distance_threshold, positive_threshold=self.positive_threshold, negative_threshold=self.negative_threshold)
                image_to_text_triplet_loss = self.triplet_loss_uncon(image_to_text, gt_similarity_matrix, margin=self.triplet_loss_margin, gt_threshold=self.positive_distance_threshold, positive_threshold=self.positive_threshold, negative_threshold=self.negative_threshold)
            image_text_triplet_loss = (text_to_image_triplet_loss + image_to_text_triplet_loss) / 2
            
            image_to_image_triplet_loss = torch.zeros(1).to(device)
            if self.use_image2image_loss > 0:
                image_to_image_triplet_loss = self.triplet_loss_uncon(image_to_image, gt_similarity_matrix, margin=self.triplet_loss_margin, gt_threshold=self.positive_distance_threshold, positive_threshold=self.positive_threshold, negative_threshold=self.negative_threshold)
            
            text_to_image_infoNCE_loss = torch.zeros(1).to(device)
            image_to_text_infoNCE_loss = torch.zeros(1).to(device)
            if self.use_uncon_infoNCE_loss > 0:
                if self.infonce_temp == False:
                    text_to_image_infoNCE_loss = self.multi_positive_infoNCE_loss_uncon(text_to_image, gt_similarity_matrix, positive_threshold=self.positive_threshold)
                    image_to_text_infoNCE_loss = self.multi_positive_infoNCE_loss_uncon(image_to_text, gt_similarity_matrix, positive_threshold=self.positive_threshold)
                else:
                    # text_to_image_infoNCE_loss = self.multi_positive_infoNCE_loss_uncon_temp(text_to_image, gt_similarity_matrix, positive_threshold=self.positive_threshold)
                    # image_to_text_infoNCE_loss = self.multi_positive_infoNCE_loss_uncon_temp(image_to_text, gt_similarity_matrix, positive_threshold=self.positive_threshold)
                    text_to_image_infoNCE_loss = self.infoNCE_loss(text_to_image)
                    image_to_text_infoNCE_loss = self.infoNCE_loss(image_to_text)
            image_text_infoNCE_loss = (text_to_image_infoNCE_loss + image_to_text_infoNCE_loss) / 2

            # calculate CL loss
            
            cl_losses = torch.zeros(1).to(device)
            # cl_losses += self.use_uncon_triplet_loss * image_text_triplet_loss  # 为什么这里不使用image2text loss 了,加一个实验试一下
            cl_losses += self.use_image2image_loss * image_to_image_triplet_loss
            cl_losses += self.use_uncon_infoNCE_loss * image_text_infoNCE_loss

            return cl_losses, image_text_triplet_loss, image_text_infoNCE_loss, image_to_image_triplet_loss

if __name__ == '__main__':
    
    from transformer_maskgit import CTViT
    
    device = torch.device('cuda:0')
    
    tokenizer = BertTokenizer.from_pretrained('microsoft/BiomedVLP-CXR-BERT-specialized',do_lower_case=True)
    text_encoder = BertModel.from_pretrained("microsoft/BiomedVLP-CXR-BERT-specialized")


    image_encoder = CTViT(
        dim = 512,
        codebook_size = 8192,
        image_size = 480,
        patch_size = 20,
        temporal_patch_size = 10,
        spatial_depth = 8,
        temporal_depth = 6,
        cls_depth = 4,
        dim_head = 32,
        heads = 8
    )

    clip = CTCLIP(
        tokenizer=tokenizer,
        image_encoder = image_encoder,
        text_encoder = text_encoder,
        dim_text = 768,
        dim_image = 512,
        dim_latent = 512,
        extra_latent_projection = False,         # whether to use separate projections for text-to-image vs image-to-text comparisons (CLOOB)
        use_mlm=False,
        downsample_image_embeds = False,
        use_all_token_embeds = False
    ).to(device)
    
    
    def analyze_visual_model(model):
        """分析 PyTorch 模型的视觉部分，打印键和参数数量"""
        # 获取模型的 state_dict
        state_dict = model.state_dict()
        
        # 视觉部分通常以 'visual' 开头或包含 'vision' 
        visual_keys = [key for key in state_dict.keys() if 'visual' in key or 'vision' in key]
        
        # 打印视觉部分的键
        print("模型视觉部分的键:")
        for key in visual_keys:
            print(f"- {key}")
        
        # 计算视觉部分的参数数量
        visual_params = sum(state_dict[key].numel() for key in visual_keys)
        print(f"\n视觉部分总参数数量: {visual_params:,}")
        
        # 打印每个视觉层的参数数量
        count = 0
        print("\n视觉层参数详情:")
        for name, param in model.named_parameters():
            if ('visual' in name or 'vision' in name) and 'enc_temporal_transformer' not in name:
                print(f"{name}: {param.shape}, 参数数量: {param.numel():,}")
                count += param.numel()
        print(f"\n视觉层总参数数量: {count:,}")
    analyze_visual_model(clip)
