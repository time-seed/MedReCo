# """
# Code modified from DETR tranformer:
# https://github.com/facebookresearch/detr
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# """

# import copy
# from typing import Optional, List
# import pickle as cp

# import torch
# import torch.nn.functional as F
# from torch import nn, Tensor

# class GEGLU(nn.Module):
#     def forward(self, x):
#         x, gate = x.chunk(2, dim = -1)
#         return x * F.gelu(gate)

# class MoEFeedForward(nn.Module):
#     def __init__(self, dim, mult=4, dropout=0., num_experts=5, include_norm=False):
#         super().__init__()
#         self.num_experts = num_experts
#         inner_dim = int(mult * (2 / 3) * dim)

#         def _build_expert():
#             layers = []
#             if include_norm:
#                 layers.append(nn.LayerNorm(dim))
#             layers.extend([
#                 nn.Linear(dim, inner_dim * 2, bias=False),
#                 GEGLU(),
#                 nn.Dropout(dropout),
#                 nn.Linear(inner_dim, dim, bias=False)
#             ])
#             return nn.Sequential(*layers)

#         self.experts = nn.ModuleList([_build_expert() for _ in range(num_experts)])

#     def forward(self, x: torch.Tensor, expert_indices: torch.Tensor) -> torch.Tensor:
#         """
#         x: (batch, seq, dim) or (seq, batch, dim)
#         expert_indices: (batch,) int tensor, values in [0, num_experts - 1]
#         """
#         # 处理不同的输入形状 (seq, batch, dim) 或 (batch, seq, dim)
#         original_shape = x.shape
#         need_transpose_back = False
        
#         if x.ndim == 3:
#             # # 假设输入是 (seq, batch, dim),转换为 (batch, seq, dim)
#             # if (x.shape[0] == expert_indices.shape[0]) and (x.shape[1] == expert_indices.shape[0]):
#             #     raise ValueError(
#             #         f"Cannot determine batch dimension from x shape {x.shape} "
#             #         f"and expert_indices shape {expert_indices.shape}"
#             #     )
#             if x.shape[1] == expert_indices.shape[0]:
#                 x = x.transpose(0, 1)  # (seq, batch, dim) -> (batch, seq, dim)
#                 need_transpose_back = True
                

#         if expert_indices.ndim != 1:
#             raise ValueError(f"`expert_indices` must be 1D (batch,), but got shape {expert_indices.shape}")

#         if expert_indices.shape[0] != x.shape[0]:
#             raise ValueError(
#                 f"`expert_indices` batch dimension ({expert_indices.shape[0]}) "
#                 f"must match x batch dimension ({x.shape[0]})"
#             )

#         expert_indices = expert_indices.to(device=x.device)
#         if expert_indices.dtype != torch.long:
#             expert_indices = expert_indices.long()

#         out = torch.zeros_like(x)

#         for expert_id, expert in enumerate(self.experts):
#             batch_mask = expert_indices == expert_id
#             if not batch_mask.any():
#                 continue

#             expert_input = x[batch_mask]
#             expert_output = expert(expert_input)
#             out[batch_mask] = expert_output.to(out.dtype)

#         # 如果需要,转换回原始形状
#         if need_transpose_back:
#             out = out.transpose(0, 1)  # (batch, seq, dim) -> (seq, batch, dim)

#         return out

# class TransformerDecoder(nn.Module):

#     def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
#         super().__init__()
#         self.layers = _get_clones(decoder_layer, num_layers)
#         self.num_layers = num_layers
#         self.norm = norm
#         self.return_intermediate = return_intermediate

#     def forward(self, tgt, memory,
#                 tgt_mask: Optional[Tensor] = None,
#                 memory_mask: Optional[Tensor] = None,
#                 tgt_key_padding_mask: Optional[Tensor] = None,
#                 memory_key_padding_mask: Optional[Tensor] = None,
#                 pos: Optional[Tensor] = None,
#                 query_pos: Optional[Tensor] = None,
#                 expert_indices: Optional[Tensor] = None):
        
#         if expert_indices is None:
#             raise ValueError("`expert_indices` must be provided for MoE decoder")
        
#         output = tgt
#         T,B,C = memory.shape
#         intermediate = []
#         atten_layers = []
#         for n,layer in enumerate(self.layers):
             
#             residual=True
#             output,ws = layer(output, memory, tgt_mask=tgt_mask,
#                            memory_mask=memory_mask,
#                            tgt_key_padding_mask=tgt_key_padding_mask,
#                            memory_key_padding_mask=memory_key_padding_mask,
#                            pos=pos, query_pos=query_pos, residual=residual,
#                            expert_indices=expert_indices)
#             atten_layers.append(ws)
#             if self.return_intermediate:
#                 intermediate.append(self.norm(output))
                
#         if self.norm is not None:
#             output = self.norm(output)
#             if self.return_intermediate:
#                 intermediate.pop()
#                 intermediate.append(output)

#         if self.return_intermediate:
#             return torch.stack(intermediate), atten_layers
#         return output,atten_layers
# # FIXME:修改默认的激活函数由relu为geglu。然后测试一下能否正常运行

# # class TransformerDecoderLayer(nn.Module):

# #     def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
# #                  activation="relu", normalize_before=False):
# #         super().__init__()
# #         # self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
# #         self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
# #         # Implementation of Feedforward model
# #         self.linear1 = nn.Linear(d_model, dim_feedforward)
# #         self.dropout = nn.Dropout(dropout)
# #         self.linear2 = nn.Linear(dim_feedforward, d_model)

# #         self.norm1 = nn.LayerNorm(d_model)
# #         self.norm2 = nn.LayerNorm(d_model)
# #         self.norm3 = nn.LayerNorm(d_model)
# #         self.dropout1 = nn.Dropout(dropout)
# #         self.dropout2 = nn.Dropout(dropout)
# #         self.dropout3 = nn.Dropout(dropout)

# #         self.activation = _get_activation_fn(activation)
# #         self.normalize_before = normalize_before
# #     def with_pos_embed(self, tensor, pos: Optional[Tensor]):
# #         return tensor if pos is None else tensor + pos

# #     def forward_post(self, tgt, memory,
# #                      tgt_mask: Optional[Tensor] = None,
# #                      memory_mask: Optional[Tensor] = None,
# #                      tgt_key_padding_mask: Optional[Tensor] = None,
# #                      memory_key_padding_mask: Optional[Tensor] = None,
# #                      pos: Optional[Tensor] = None,
# #                      query_pos: Optional[Tensor] = None,
# #                      residual=True):
# #         q = k = self.with_pos_embed(tgt, query_pos)
# #         tgt2,ws = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
# #                               key_padding_mask=tgt_key_padding_mask)
# #         tgt = self.norm1(tgt)
# #         tgt2,ws = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
# #                                    key=self.with_pos_embed(memory, pos),
# #                                    value=memory, attn_mask=memory_mask,
# #                                    key_padding_mask=memory_key_padding_mask)

# #         # attn_weights [B,NUM_Q,T]
# #         tgt = tgt + self.dropout2(tgt2)
# #         tgt = self.norm2(tgt)
# #         # FIXME:由于tgt2已经是经历过norm的，因此FFN中不需要再norm一次
# #         tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
# #         tgt = tgt + self.dropout3(tgt2)
# #         tgt = self.norm3(tgt)
# #         return tgt,ws

# #     def forward_pre(self, tgt, memory,
# #                     tgt_mask: Optional[Tensor] = None,
# #                     memory_mask: Optional[Tensor] = None,
# #                     tgt_key_padding_mask: Optional[Tensor] = None,
# #                     memory_key_padding_mask: Optional[Tensor] = None,
# #                     pos: Optional[Tensor] = None,
# #                     query_pos: Optional[Tensor] = None):

# #         tgt2 = self.norm2(tgt)
# #         tgt2,attn_weights = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
# #                                    key=self.with_pos_embed(memory, pos),
# #                                    value=memory, attn_mask=memory_mask,
# #                                    key_padding_mask=memory_key_padding_mask)
# #         tgt = tgt + self.dropout2(tgt2)
# #         tgt2 = self.norm3(tgt)
# #         tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
# #         tgt = tgt + self.dropout3(tgt2)
# #         return tgt,attn_weights

# #     def forward(self, tgt, memory,
# #                 tgt_mask: Optional[Tensor] = None,
# #                 memory_mask: Optional[Tensor] = None,
# #                 tgt_key_padding_mask: Optional[Tensor] = None,
# #                 memory_key_padding_mask: Optional[Tensor] = None,
# #                 pos: Optional[Tensor] = None,
# #                 query_pos: Optional[Tensor] = None,
# #                 residual=True):
# #         if self.normalize_before:
# #             return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
# #                                     tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
# #         return self.forward_post(tgt, memory, tgt_mask, memory_mask,
# #                                  tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos,residual)

# class TransformerDecoderLayer(nn.Module):

#     def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.2,
#                  activation="relu", normalize_before=False, num_experts=5):
#         super().__init__()
#         self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        
#         # 使用MoE替代原始的FFN
#         self.moe_ffn = MoEFeedForward(
#             dim=d_model, 
#             mult=dim_feedforward / d_model, 
#             dropout=dropout, 
#             num_experts=num_experts
#         )

#         self.norm1 = nn.LayerNorm(d_model)
#         self.norm2 = nn.LayerNorm(d_model)
#         self.norm3 = nn.LayerNorm(d_model)
#         self.dropout1 = nn.Dropout(dropout)
#         self.dropout2 = nn.Dropout(dropout)
#         self.dropout3 = nn.Dropout(dropout)

#         self.activation = _get_activation_fn(activation)
#         self.normalize_before = normalize_before
        
#     def with_pos_embed(self, tensor, pos: Optional[Tensor]):
#         return tensor if pos is None else tensor + pos

#     def forward_post(self, tgt, memory,
#                      tgt_mask: Optional[Tensor] = None,
#                      memory_mask: Optional[Tensor] = None,
#                      tgt_key_padding_mask: Optional[Tensor] = None,
#                      memory_key_padding_mask: Optional[Tensor] = None,
#                      pos: Optional[Tensor] = None,
#                      query_pos: Optional[Tensor] = None,
#                      residual=True,
#                      expert_indices: Optional[Tensor] = None):
        
#         tgt = self.norm1(tgt)
#         tgt2, ws = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
#                                    key=self.with_pos_embed(memory, pos),
#                                    value=memory, attn_mask=memory_mask,
#                                    key_padding_mask=memory_key_padding_mask)

#         tgt = tgt + self.dropout2(tgt2)
#         tgt = self.norm2(tgt)
        
#         # 使用MoE FFN
#         tgt2 = self.moe_ffn(tgt, expert_indices=expert_indices)
#         tgt = tgt + self.dropout3(tgt2)
#         tgt = self.norm3(tgt)
        
#         return tgt, ws

#     def forward_pre(self, tgt, memory,
#                     tgt_mask: Optional[Tensor] = None,
#                     memory_mask: Optional[Tensor] = None,
#                     tgt_key_padding_mask: Optional[Tensor] = None,
#                     memory_key_padding_mask: Optional[Tensor] = None,
#                     pos: Optional[Tensor] = None,
#                     query_pos: Optional[Tensor] = None,
#                     expert_indices: Optional[Tensor] = None):

#         tgt2 = self.norm2(tgt)
#         # print("In decoder layer forward_pre")
#         # print("tgt2 shape:", tgt2.shape)
#         # print("memory shape:", memory.shape)
#         # print('pos shape:', pos.shape if pos is not None else None)
        
#         tgt2, attn_weights = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
#                                    key=self.with_pos_embed(memory, pos),
#                                    value=memory, attn_mask=memory_mask,
#                                    key_padding_mask=memory_key_padding_mask)
#         tgt = tgt + self.dropout2(tgt2)
        
#         tgt2 = self.norm3(tgt)
#         # 使用MoE FFN
#         tgt2 = self.moe_ffn(tgt2, expert_indices=expert_indices)
#         tgt = tgt + self.dropout3(tgt2)
        
#         return tgt, attn_weights

#     def forward(self, tgt, memory,
#                 tgt_mask: Optional[Tensor] = None,
#                 memory_mask: Optional[Tensor] = None,
#                 tgt_key_padding_mask: Optional[Tensor] = None,
#                 memory_key_padding_mask: Optional[Tensor] = None,
#                 pos: Optional[Tensor] = None,
#                 query_pos: Optional[Tensor] = None,
#                 residual=True,
#                 expert_indices: Optional[Tensor] = None):
        
#         if self.normalize_before:
#             return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
#                                     tgt_key_padding_mask, memory_key_padding_mask, 
#                                     pos, query_pos, expert_indices=expert_indices)
#         return self.forward_post(tgt, memory, tgt_mask, memory_mask,
#                                  tgt_key_padding_mask, memory_key_padding_mask, 
#                                  pos, query_pos, residual, expert_indices=expert_indices)

# def _get_clones(module, N):
#     return nn.ModuleList([copy.deepcopy(module) for i in range(N)])



# def _get_activation_fn(activation):
#     """Return an activation function given a string"""
#     if activation == "relu":
#         return F.relu
#     if activation == "gelu":
#         return F.gelu
#     if activation == "glu":
#         return F.glu
#     raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


"""
Code modified from DETR tranformer:
https://github.com/facebookresearch/detr
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""

import copy
from typing import Optional, List
import pickle as cp

import torch
import torch.nn.functional as F
from torch import nn, Tensor
def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same implementation as the one in timm.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    # handle tensors with different dimensions, not just 4D tensors
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample."""
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)
    
class GEGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim = -1)
        return x * F.gelu(gate)

class MoEFeedForward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0., num_experts=5, include_norm=False):
        super().__init__()
        self.num_experts = num_experts
        inner_dim = int(mult * (2 / 3) * dim)

        def _build_expert():
            layers = []
            if include_norm:
                layers.append(nn.LayerNorm(dim))
            layers.extend([
                nn.Linear(dim, inner_dim * 2, bias=False),
                GEGLU(),
                nn.Dropout(dropout),
                nn.Linear(inner_dim, dim, bias=False)
            ])
            return nn.Sequential(*layers)

        self.experts = nn.ModuleList([_build_expert() for _ in range(num_experts)])

    def forward(self, x: torch.Tensor, expert_indices: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq, dim) or (seq, batch, dim)
        expert_indices: (batch,) int tensor, values in [0, num_experts - 1]
        """
        # 处理不同的输入形状 (seq, batch, dim) 或 (batch, seq, dim)
        original_shape = x.shape
        need_transpose_back = False
        
        if x.ndim == 3:
            # # 假设输入是 (seq, batch, dim),转换为 (batch, seq, dim)
            # if (x.shape[0] == expert_indices.shape[0]) and (x.shape[1] == expert_indices.shape[0]):
            #     raise ValueError(
            #         f"Cannot determine batch dimension from x shape {x.shape} "
            #         f"and expert_indices shape {expert_indices.shape}"
            #     )
            if x.shape[1] == expert_indices.shape[0]:
                x = x.transpose(0, 1)  # (seq, batch, dim) -> (batch, seq, dim)
                need_transpose_back = True
                

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
            out[batch_mask] = expert_output.to(out.dtype)

        # 如果需要,转换回原始形状
        if need_transpose_back:
            out = out.transpose(0, 1)  # (batch, seq, dim) -> (seq, batch, dim)

        return out

class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False, 
                 drop_path_rate=0.1): # <--- 新增参数
        super().__init__()
        
        # <--- 修改开始: 实现 Stochastic Depth (线性衰减 DropPath)
        # 生成每一层的 drop rate, 从 0 线性增长到 drop_path_rate
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, num_layers)]
        
        layers = []
        for i in range(num_layers):
            # 深拷贝原始 layer
            layer = copy.deepcopy(decoder_layer)
            # 如果 layer 中有 drop_path 模块，更新其 drop_prob
            if hasattr(layer, 'drop_path') and isinstance(layer.drop_path, DropPath):
                layer.drop_path.drop_prob = dpr[i]
            layers.append(layer)
        
        self.layers = nn.ModuleList(layers)
        # <--- 修改结束
        
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    # forward 函数不需要修改，逻辑保持不变
    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None,
                expert_indices: Optional[Tensor] = None):
        
        if expert_indices is None:
            raise ValueError("`expert_indices` must be provided for MoE decoder")
        
        output = tgt
        # T, B, C = memory.shape # memory shape may vary depending on implementation, ensuring robustness
        intermediate = []
        atten_layers = []
        
        for n, layer in enumerate(self.layers):
            residual = True
            output, ws = layer(output, memory, tgt_mask=tgt_mask,
                               memory_mask=memory_mask,
                               tgt_key_padding_mask=tgt_key_padding_mask,
                               memory_key_padding_mask=memory_key_padding_mask,
                               pos=pos, query_pos=query_pos, residual=residual,
                               expert_indices=expert_indices)
            atten_layers.append(ws)
            if self.return_intermediate:
                intermediate.append(self.norm(output) if self.norm else output) # Safety check for norm
                
        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate), atten_layers
        return output, atten_layers
    
class TransformerDecoderLayer(nn.Module):
    # 两个dropout的值是一致的
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.25,
                 activation="relu", normalize_before=False, num_experts=5, 
                 drop_path=0.): # <--- 新增参数
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        
        # 使用MoE替代原始的FFN
        self.moe_ffn = MoEFeedForward(
            dim=d_model, 
            mult=dim_feedforward / d_model, 
            dropout=dropout, 
            num_experts=num_experts
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        # <--- 新增 DropPath 初始化
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        
    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None,
                     residual=True,
                     expert_indices: Optional[Tensor] = None):
        
        tgt = self.norm1(tgt)
        tgt2, ws = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                       key=self.with_pos_embed(memory, pos),
                                       value=memory, attn_mask=memory_mask,
                                       key_padding_mask=memory_key_padding_mask)

        # <--- 修改: 应用 DropPath
        tgt = tgt + self.drop_path(self.dropout2(tgt2))
        tgt = self.norm2(tgt)
        
        # 使用MoE FFN
        tgt2 = self.moe_ffn(tgt, expert_indices=expert_indices)
        
        # <--- 修改: 应用 DropPath
        tgt = tgt + self.drop_path(self.dropout3(tgt2))
        tgt = self.norm3(tgt)
        
        return tgt, ws

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None,
                    expert_indices: Optional[Tensor] = None):

        tgt2 = self.norm2(tgt)
        
        tgt2, attn_weights = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                                 key=self.with_pos_embed(memory, pos),
                                                 value=memory, attn_mask=memory_mask,
                                                 key_padding_mask=memory_key_padding_mask)
        # <--- 修改: 应用 DropPath
        tgt = tgt + self.drop_path(self.dropout2(tgt2))
        
        tgt2 = self.norm3(tgt)
        # 使用MoE FFN
        tgt2 = self.moe_ffn(tgt2, expert_indices=expert_indices)
        
        # <--- 修改: 应用 DropPath
        tgt = tgt + self.drop_path(self.dropout3(tgt2))
        
        return tgt, attn_weights

    # forward 函数保持不变，直接调用 pre 或 post
    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None,
                residual=True,
                expert_indices: Optional[Tensor] = None):
        
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, 
                                    pos, query_pos, expert_indices=expert_indices)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, 
                                 pos, query_pos, residual, expert_indices=expert_indices)
        
def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])



def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")