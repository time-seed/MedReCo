import torch
import torch.nn as nn
from einops import repeat,rearrange
import torch.nn.functional as F
# 假设你已经有了 GEGLU 的实现，这里仅作占位示意
# class GEGLU(nn.Module): ... 
class GEGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim = -1)
        return x * F.gelu(gate)
# ---------------------------------------------------------
# 1. 你提供的 MoEFeedForward (保持不变)
# ---------------------------------------------------------
# ---------------------------------------------------------
# 1. 修复后的 MoEFeedForward
# ---------------------------------------------------------
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
                # 👈 修复 1：直接使用 inner_dim，删去错误的 hasattr 判断
                nn.Linear(inner_dim, dim, bias=False) 
            ])
            return nn.Sequential(*layers)

        self.experts = nn.ModuleList([_build_expert() for _ in range(num_experts)])

    def forward(self, x: torch.Tensor, expert_indices: torch.Tensor) -> torch.Tensor:
        # 👈 修复 3：删除了危险的 transpose 逻辑，直接假设 batch_first=True
        if expert_indices.ndim != 1:
            raise ValueError(f"`expert_indices` must be 1D (batch,), but got shape {expert_indices.shape}")
        if expert_indices.shape[0] != x.shape[0]:
            raise ValueError(f"Batch dimension mismatch: x has {x.shape[0]}, indices have {expert_indices.shape[0]}")

        expert_indices = expert_indices.to(device=x.device, dtype=torch.long)
        out = torch.zeros_like(x)

        for expert_id, expert in enumerate(self.experts):
            batch_mask = (expert_indices == expert_id)
            if not batch_mask.any():
                continue
            # x[batch_mask] 的 shape 会变成 (被选中的batch数, seq_len, dim)
            expert_input = x[batch_mask] 
            out[batch_mask] = expert(expert_input).to(out.dtype)

        return out

# ---------------------------------------------------------
# 2. 自定义 MoE Transformer Encoder Layer
# ---------------------------------------------------------
class MoETransformerEncoderLayer(nn.Module):
    def __init__(self, dim, num_heads, num_experts, dropout=0.1):
        super().__init__()
        # 自注意力机制
        self.self_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        
        # 将原生 FFN 替换为你手写的 MoE FFN
        self.moe_ffn = MoEFeedForward(dim=dim, mult=4, dropout=dropout, num_experts=num_experts)
        
        # Norm First 架构的 LayerNorm
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, expert_indices):
        # 1. Norm + Self-Attention + Add
        src_norm = self.norm1(src)
        attn_output, _ = self.self_attn(src_norm, src_norm, src_norm)
        src = src + self.dropout1(attn_output)
        
        # 2. Norm + MoE FFN + Add (需要传入 expert_indices)
        src_norm2 = self.norm2(src)
        ffn_output = self.moe_ffn(src_norm2, expert_indices)
        src = src + self.dropout2(ffn_output)
        
        return src


# ---------------------------------------------------------
# 3. 改进后的 MoE Cross-Attention Reranker
# ---------------------------------------------------------
class MoECrossAttentionReranker(nn.Module):        
    def __init__(self, dim, num_heads, depth, top_k, num_experts=5, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.top_k = top_k
        self.num_experts = num_experts
        
        # 特殊 Token 和 Embedding
        self.sep_token = nn.Parameter(torch.randn(1, 1, dim))
        self.segment_emb = nn.Embedding(2, dim)
        self.cls_emb = nn.Parameter(torch.randn(1, 1, dim))  
        
        # 替换原生的 TransformerEncoder，使用 ModuleList 包装自定义的 MoE 层
        self.layers = nn.ModuleList([
            MoETransformerEncoderLayer(dim, num_heads, num_experts, dropout)
            for _ in range(depth)
        ])
        
        # 最后的二分类头也改成 MoE 架构，每个模态单独输出 rerank 结果！
        # 相当于用专属于该模态的非线性变换来降维
        self.classifier_moe = MoEFeedForward(dim, mult=2, dropout=0.1, num_experts=num_experts)
        self.classifier_out = nn.Linear(dim, 1) # 输出 logit

    def forward(self, tokens1, tokens2, expert_indices):
        """
        Args:
            tokens1: (B, 1 + K, D)
            tokens2: (B, 1 + K, D)
            expert_indices: (B,) 传入的手动路由模态索引，例如文本=0，图像=1，视频=2...
        """
        B = tokens1.shape[0]
        
        # A & B. 准备 [CLS] 和 [SEP]
        cls_token = self.cls_emb.expand(B, -1, -1)  # (B, 1, D)
        sep = repeat(self.sep_token, '1 1 d -> b 1 d', b=B)
        
        # C. 构建长序列
        tokens1_with_seg = tokens1 + self.segment_emb(torch.zeros(B, self.top_k + 1, dtype=torch.long, device=tokens1.device))
        tokens2_with_seg = tokens2 + self.segment_emb(torch.ones(B, self.top_k + 1, dtype=torch.long, device=tokens2.device))
    
        # Sequence: [CLS/Cond] + [Img1] + [SEP] + [Img2]
        sequence = torch.cat([cls_token, tokens1_with_seg, sep, tokens2_with_seg], dim=1)
        
        # E. Cross-Attention + MoE 交互
        for layer in self.layers:
            sequence = layer(sequence, expert_indices)
            
        # F. 提取 Condition Token ([CLS])
        cls_output = sequence[:, 0, :]  # (B, D)
        
        # 扩充维度以适配 MoEFeedForward 的三维输入要求 (B, 1, D)
        cls_output = cls_output.unsqueeze(1) 
        
        # G. 使用 MoE 分类头进行独立模态预测
        moe_cls_features = self.classifier_moe(cls_output, expert_indices) # (B, 1, D)
        logits = self.classifier_out(moe_cls_features.squeeze(1))          # (B, 1)
        
        return logits
    

# ---------------------------------------------------------
# 4. Listwise MoE Reranker — 所有候选在同一序列中竞争
# ---------------------------------------------------------
class MoEListwiseCrossAttentionReranker(nn.Module):
    """
    Listwise Reranker：query 与多个候选在同一序列中做全局 self-attention，
    然后提取每个候选的 CLS token 经 MoE 分类头输出相关性分数。
 
    序列结构:
        [query_tokens] [cand_1_tokens] [cand_2_tokens] ... [cand_N_tokens]
 
    区分机制:
        - segment_id = 0  →  query tokens
        - segment_id = 1  →  cand_1 tokens
        - segment_id = 2  →  cand_2 tokens
        - ...
        使用正弦位置编码生成 segment embedding，支持任意数量候选。
    """
 
    def __init__(
        self,
        dim: int,
        num_heads: int,
        depth: int,
        top_k_query: int,
        top_k_cand: int,
        max_num_cands: int = 64,
        num_experts: int = 5,
        dropout: float = 0.1,
    ):
        """
        Args:
            dim:            特征维度 D
            num_heads:      多头注意力头数
            depth:          Transformer 层数
            top_k_query:    query 的 patch token 数 (不含 CLS), 即 K_q
            top_k_cand:     每个候选的 patch token 数 (不含 CLS), 即 K_c
            max_num_cands:  支持的最大候选数量 (用于 segment embedding 表大小)
            num_experts:    MoE 专家数
            dropout:        dropout 概率
        """
        super().__init__()
        self.dim = dim
        self.top_k_query = top_k_query
        self.top_k_cand = top_k_cand
        self.num_experts = num_experts
 
        # --- Segment Embedding ---
        # segment 0 = query, segment 1..max_num_cands = 各候选
        # 使用可学习的 embedding，大小为 max_num_cands + 1
        self.segment_emb = nn.Embedding(max_num_cands + 1, dim)
 
        # --- CLS Type Embedding ---
        # 显式标记 CLS token，使模型区分 CLS (聚合锚点) 与普通 patch token
        # query 和 candidate 使用不同的 CLS embedding，兼顾角色区分
        self.query_cls_emb = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.cand_cls_emb = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
 
        # --- Transformer Encoder Layers (MoE) ---
        self.layers = nn.ModuleList([
            MoETransformerEncoderLayer(dim, num_heads, num_experts, dropout)
            for _ in range(depth)
        ])
        self.final_norm = nn.LayerNorm(dim)
 
        # --- MoE 分类头 ---
        # 对每个候选的 CLS token 做 MoE 变换后降维为标量 score
        self.classifier_moe = MoEFeedForward(
            dim, mult=2, dropout=dropout, num_experts=num_experts
        )
        self.classifier_out = nn.Linear(dim, 1)
 
    def forward(
        self,
        query_tokens: torch.Tensor,
        candidate_tokens_list: torch.Tensor,
        expert_indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            query_tokens:          (M, 1+K_q, D) — CLS + top-K query patch tokens
            candidate_tokens_list: (M, num_cands, 1+K_c, D) — 每个候选的 CLS + top-K patch tokens
            expert_indices:        (M,) — 模态路由索引
 
        Returns:
            scores: (M, num_cands) — 每个候选的相关性得分
        """
        M, num_cands, tokens_per_cand, D = candidate_tokens_list.shape
        tokens_per_query = query_tokens.shape[1]  # 1 + K_q
 
        device = query_tokens.device
 
        # =============================================================
        # 1. 构建 segment ids
        # =============================================================
        # query 部分: segment_id = 0, shape (M, 1+K_q)
        query_seg_ids = torch.zeros(M, tokens_per_query, dtype=torch.long, device=device)
 
        # 候选部分: 第 i 个候选 segment_id = i+1, shape (M, num_cands * tokens_per_cand)
        # 先构建 (num_cands, tokens_per_cand) 再 expand 到 batch
        cand_seg_ids = torch.arange(1, num_cands + 1, device=device)          # (num_cands,)
        cand_seg_ids = cand_seg_ids.unsqueeze(1).expand(-1, tokens_per_cand)  # (num_cands, tokens_per_cand)
        cand_seg_ids = cand_seg_ids.reshape(1, -1).expand(M, -1)              # (M, num_cands * tokens_per_cand)
 
        all_seg_ids = torch.cat([query_seg_ids, cand_seg_ids], dim=1)  # (M, L_total)
 
        # =============================================================
        # 2. 注入 CLS Type Embedding (在拼接前分别处理)
        # =============================================================
        # 给 query 的第 0 个位置 (CLS) 加上 query_cls_emb
        query_tokens = query_tokens.clone()
        query_tokens[:, 0:1, :] = query_tokens[:, 0:1, :] + self.query_cls_emb
 
        # 给每个候选的第 0 个位置 (CLS) 加上 cand_cls_emb
        candidate_tokens_list = candidate_tokens_list.clone()
        candidate_tokens_list[:, :, 0:1, :] = candidate_tokens_list[:, :, 0:1, :] + self.cand_cls_emb
 
        # =============================================================
        # 3. 拼接序列: [query_tokens] [cand_1] [cand_2] ... [cand_N]
        # =============================================================
        # candidate_tokens_list: (M, num_cands, 1+K_c, D) → (M, num_cands*(1+K_c), D)
        flat_cands = rearrange(candidate_tokens_list, 'm n t d -> m (n t) d')
        sequence = torch.cat([query_tokens, flat_cands], dim=1)  # (M, L_total, D)
 
        # =============================================================
        # 4. 加上 segment embedding
        # =============================================================
        sequence = sequence + self.segment_emb(all_seg_ids)
 
        # =============================================================
        # 5. Transformer 编码 (带 MoE)
        # =============================================================
        for layer in self.layers:
            sequence = layer(sequence, expert_indices)
 
        sequence = self.final_norm(sequence)
 
        # =============================================================
        # 6. 提取每个候选的 CLS token
        # =============================================================
        # 候选 i 的 CLS token 位置 = tokens_per_query + i * tokens_per_cand
        cls_positions = tokens_per_query + torch.arange(num_cands, device=device) * tokens_per_cand  # (num_cands,)
 
        # 用 index_select 提取: (M, num_cands, D)
        cls_tokens = sequence[:, cls_positions, :]  # (M, num_cands, D)
 
        # =============================================================
        # 7. MoE 分类头: 对 CLS tokens 打分
        # =============================================================
        # classifier_moe 需要 (batch, seq, dim) 输入，这里 batch=M, seq=num_cands
        moe_features = self.classifier_moe(cls_tokens, expert_indices)  # (M, num_cands, D)
        scores = self.classifier_out(moe_features).squeeze(-1)          # (M, num_cands)
 
        return scores