import torch.nn as nn
import torch.distributed as dist
from einops import rearrange
import copy
from pathlib import Path
from einops.layers.torch import Rearrange

from pathlib import Path
import copy
from functools import wraps

import torch
import torch.nn.functional as F
from torch import nn


from .attention import Attention, Transformer, ContinuousPositionBias,TransformerCLS

# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def divisible_by(numer, denom):
    return (numer % denom) == 0

def remove_vgg(fn):
    @wraps(fn)
    def inner(self, *args, **kwargs):
        has_vgg = hasattr(self, 'vgg')
        if has_vgg:
            vgg = self.vgg
            delattr(self, 'vgg')

        out = fn(self, *args, **kwargs)

        if has_vgg:
            self.vgg = vgg

        return out
    return inner

def pair(val):
    ret = (val, val) if not isinstance(val, tuple) else val
    assert len(ret) == 2
    return ret

def l2norm(t):
    return F.normalize(t, dim = -1)


class CTViT(nn.Module):
    def __init__(
        self,
        *,
        dim,
        final_dim,
        image_size,
        patch_size=20,
        temporal_patch_size=10,
        spatial_depth,
        temporal_depth,
        cls_depth,
        dim_head = 64,
        heads = 8,
        channels = 1,
        attn_dropout = 0.,
        ff_dropout = 0.
    ):
        """
        DeepSpeed-compatible Vision Transformer for multi-card training
        
        einstein notations:
        b - batch
        c - channels
        t - time
        d - feature dimension
        p1, p2, pt - image patch sizes and then temporal patch size
        """

        super().__init__()

        self.image_size = pair(image_size)
        self.patch_size = pair(patch_size)
        patch_height, patch_width = self.patch_size

        self.temporal_patch_size = temporal_patch_size
        self.dim = dim
        self.final_dim = final_dim

        # 初始化相对位置偏置（确保它能正确处理设备）
        self.spatial_rel_pos_bias = ContinuousPositionBias(dim = dim, heads = heads)

        image_height, image_width = self.image_size
        assert (image_height % patch_height) == 0 and (image_width % patch_width) == 0

        # Patch embedding layers
        self.to_patch_emb_first_frame = nn.Sequential(
            Rearrange('b c 1 (h p1) (w p2) -> b 1 h w (c p1 p2)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(channels * patch_width * patch_height),
            nn.Linear(channels * patch_width * patch_height, dim),
            nn.LayerNorm(dim)
        )

        self.to_patch_emb = nn.Sequential(
            Rearrange('b c (t pt) (h p1) (w p2) -> b t h w (c pt p1 p2)', p1 = patch_height, p2 = patch_width, pt = temporal_patch_size),
            nn.LayerNorm(channels * patch_width * patch_height * temporal_patch_size),
            nn.Linear(channels * patch_width * patch_height * temporal_patch_size, dim),
            nn.LayerNorm(dim)
        )
        
        # Modal embeddings - 确保正确初始化
        self.modal_embeddings = nn.Embedding(10, dim)
        
        # Transformer configurations
        transformer_kwargs = dict(
            dim = dim,
            dim_head = dim_head,
            heads = heads,
            attn_dropout = attn_dropout,
            ff_dropout = ff_dropout,
            peg = True,
            peg_causal = False,
        )
        
        self.enc_spatial_transformer = Transformer(depth = spatial_depth, **transformer_kwargs)
        self.enc_temporal_transformer = Transformer(depth = temporal_depth, **transformer_kwargs)
        self.enc_cls_transformer = TransformerCLS(depth = cls_depth, **transformer_kwargs)
        self.to_visual_latent = nn.Linear(dim, final_dim)
        
        # 注册缓冲区用于设备管理
        self.register_buffer('_dummy', torch.empty(0))
                # 初始化权重
    #     self._init_weights()

    # def _init_weights(self):
    #     """初始化模型权重，遵循标准 Vision Transformer 初始化策略"""
    #     for module in self.modules():
    #         if isinstance(module, nn.Linear):
    #             # 使用 Xavier uniform 初始化线性层
    #             nn.init.xavier_uniform_(module.weight)
    #             if module.bias is not None:
    #                 nn.init.zeros_(module.bias)
    #         elif isinstance(module, nn.LayerNorm):
    #             # 标准化层的权重设为1，偏置设为0
    #             nn.init.ones_(module.weight)
    #             nn.init.zeros_(module.bias)
    #         elif isinstance(module, nn.Embedding):
    #             # 嵌入层使用正态分布初始化
    #             nn.init.normal_(module.weight, std=0.02)
    @property
    def device(self):
        """动态获取模型当前设备"""
        return self._dummy.device
    
    @property
    def image_num_tokens(self):
        return int(self.image_size[0] / self.patch_size[0]) * int(self.image_size[1] / self.patch_size[1])

    def num_tokens_per_frames(self, num_frames, include_first_frame = True):
        image_num_tokens = self.image_num_tokens

        total_tokens = 0

        if include_first_frame:
            num_frames -= 1
            total_tokens += image_num_tokens

        assert (num_frames % self.temporal_patch_size) == 0

        return total_tokens + int(num_frames / self.temporal_patch_size) * image_num_tokens

    def copy_for_eval(self):
        """创建用于评估的模型副本，兼容DeepSpeed"""
        device = self.device
        vae_copy = copy.deepcopy(self.cpu())

        # 安全地删除可能不存在的属性
        if hasattr(vae_copy, 'use_vgg_and_gan') and vae_copy.use_vgg_and_gan:
            if hasattr(vae_copy, 'discr'):
                del vae_copy.discr
            if hasattr(vae_copy, 'vgg'):
                del vae_copy.vgg

        vae_copy.eval()
        return vae_copy.to(device)

    def state_dict(self, *args, **kwargs):
        """重写state_dict以确保兼容性"""
        return super().state_dict(*args, **kwargs)

    def load_state_dict(self, *args, **kwargs):
        """重写load_state_dict以确保兼容性"""
        return super().load_state_dict(*args, **kwargs)

    def load(self, path):
        """加载模型权重，支持多卡环境"""
        path = Path(path)
        assert path.exists()
        
        # 使用map_location确保正确的设备加载
        pt = torch.load(str(path), map_location=self.device)
        self.load_state_dict(pt)

    @property
    def patch_height_width(self):
        return self.image_size[0] // self.patch_size[0], self.image_size[1] // self.patch_size[1]

    def encode(self, tokens):
        """编码函数，确保设备一致性"""
        b = tokens.shape[0]
        h, w = self.patch_height_width
        device = self.device

        video_shape = tuple(tokens.shape[:-1])
        tokens = rearrange(tokens, 'b t h w d -> (b t) (h w) d')
        
        # 确保tokens在正确的设备上
        tokens = tokens.to(device)
        
        # 空间编码
        attn_bias = self.spatial_rel_pos_bias(h, w, device=device)
        tokens = self.enc_spatial_transformer(tokens, attn_bias=attn_bias, video_shape=video_shape)

        # 清理注意力偏置（在DeepSpeed中更安全的内存管理）
        del attn_bias
        if torch.cuda.is_available() and device.type == 'cuda':
            torch.cuda.empty_cache()
        
        tokens = rearrange(tokens, '(b t) (h w) d -> b t h w d', b=b, h=h, w=w)
 
        # 时间编码
        if tokens.shape[1] > 1:
            tokens = rearrange(tokens, 'b t h w d -> (b h w) t d')
            tokens = self.enc_temporal_transformer(tokens, video_shape=video_shape)
            tokens = rearrange(tokens, '(b h w) t d -> b t h w d', b=b, h=h, w=w)
                   
        # 平均池化和重排列
        tokens = torch.mean(tokens, dim=1)
        tokens = rearrange(tokens, 'b h w d -> b (h w) d')
        
        # 更新video_shape用于CLS transformer
        video_shape = list(video_shape)
        video_shape[1] = 1
        video_shape = tuple(video_shape)
        
        tokens = self.enc_cls_transformer(tokens, video_shape=video_shape)
        tokens = tokens[:, 1:]  # 移除CLS token
        
        return tokens

    def forward(
        self,
        video,
        mask=None,
        modal_embedding=True,
        modal_indexs=None,
        *args,
        **kwargs
    ):
        """前向传播，确保DeepSpeed兼容性"""
        
        assert video.ndim in {4, 5}
        device = self.device

        is_image = video.ndim == 4

        if is_image:
            video = rearrange(video, 'b c h w -> b c 1 h w')
            assert not exists(mask)

        b, c, f, *image_dims = video.shape
        
        # 确保输入在正确设备上
        video = video.to(device)
        
        assert tuple(image_dims) == self.image_size
        assert not exists(mask) or mask.shape[-1] == f
        
        # 自动确定模态索引
        if modal_indexs is None:
            if video.shape[2] > 1:
                modal_indexs = torch.tensor([1] * b, device=device, dtype=torch.long)
            else:
                modal_indexs = torch.tensor([0] * b, device=device, dtype=torch.long)
        else:
            modal_indexs = modal_indexs.to(device)
        
        # Patch embedding
        if video.shape[2] > 1:
            tokens = self.to_patch_emb(video)
        else:
            tokens = self.to_patch_emb_first_frame(video)

        # 添加模态嵌入
        if modal_embedding:
            modality_emb = self.modal_embeddings(modal_indexs)
            modality_emb = modality_emb.view(-1, 1, 1, 1, modality_emb.size(-1))
            tokens = tokens + modality_emb
        
        # 编码
        tokens = self.encode(tokens)
        
        # 输出投影
        tokens = self.to_visual_latent(tokens)
        tokens = tokens.flatten(0, 1)
        
        return tokens

    def get_device_info(self):
        """获取设备信息，用于调试"""
        if dist.is_initialized():
            return {
                'rank': dist.get_rank(),
                'world_size': dist.get_world_size(),
                'device': self.device,
                'local_rank': torch.cuda.current_device() if torch.cuda.is_available() else None
            }
        else:
            return {
                'device': self.device,
                'cuda_available': torch.cuda.is_available()
            }


# 示例：如何在DeepSpeed中使用这个模型
def create_model_for_deepspeed():
    """创建适用于DeepSpeed的模型实例"""
    model = CTViT(
        dim=512,
        final_dim=768,
        image_size=480,
        patch_size=20,
        temporal_patch_size=10,
        spatial_depth=6,
        temporal_depth=6,
        cls_depth=2,
        dim_head=64,
        heads=8,
        channels=1,
        attn_dropout=0.1,
        ff_dropout=0.1
    )
    
    return model


# DeepSpeed配置示例
def get_deepspeed_config():
    """返回DeepSpeed配置"""
    return {
        "train_batch_size": 32,
        "train_micro_batch_size_per_gpu": 4,
        "gradient_accumulation_steps": 2,
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": 1e-4,
                "betas": [0.9, 0.999],
                "eps": 1e-8,
                "weight_decay": 0.01
            }
        },
        "scheduler": {
            "type": "WarmupLR",
            "params": {
                "warmup_min_lr": 0,
                "warmup_max_lr": 1e-4,
                "warmup_num_steps": 1000
            }
        },
        "zero_optimization": {
            "stage": 2,
            "offload_optimizer": {
                "device": "cpu",
                "pin_memory": True
            },
            "allgather_partitions": True,
            "allgather_bucket_size": 2e8,
            "overlap_comm": True,
            "reduce_scatter": True,
            "reduce_bucket_size": 2e8,
            "contiguous_gradients": True
        },
        "gradient_clipping": 1.0,
        "fp16": {
            "enabled": True,
            "auto_cast": False,
            "loss_scale": 0,
            "initial_scale_power": 16,
            "loss_scale_window": 1000,
            "hysteresis": 2,
            "min_loss_scale": 1
        },
        "activation_checkpointing": {
            "partition_activations": True,
            "cpu_checkpointing": True,
            "contiguous_memory_optimization": False,
            "number_checkpoints": 4,
            "synchronize_checkpoint_boundary": False,
            "profile": False
        }
    }