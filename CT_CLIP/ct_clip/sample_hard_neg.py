import torch
import torch.nn.functional as F


@torch.no_grad()
def generate_rerank_training_data(gt_matrix, pred_matrix, top_k_pool=30, num_neg=3,postivate=0.9):
    B = gt_matrix.shape[0]
    device = gt_matrix.device
    
    # --- 1. 基础Mask ---
    eye_mask = torch.eye(B, dtype=torch.bool, device=device)
    pos_mask = (gt_matrix >= postivate) & (~eye_mask)
    
    # --- 2. 初步筛选：有正样本的行 ---
    row_pos_counts = pos_mask.sum(dim=1)
    
    # --- [新增] 二次筛选：负样本足够的行 ---
    # 负样本数 = 总数B - 正样本数 - 1(自身)
    # 我们需要至少 num_neg 个负样本
    row_neg_counts = B - row_pos_counts - 1
    
    # 同时满足：有正样本 AND 负样本够多
    valid_row_mask = (row_pos_counts > 0) & (row_neg_counts >= num_neg)
    
    valid_row_indices = torch.nonzero(valid_row_mask).squeeze(-1)
    
    M = valid_row_indices.numel()
    if M == 0:
        return torch.empty((0, 2), dtype=torch.long, device=device), torch.empty((0,), dtype=torch.float, device=device)
    
    # 提取子数据
    sub_gt_pos_mask = pos_mask[valid_row_indices]
    sub_pred = pred_matrix[valid_row_indices]
    
    # --- 3. 采样正样本 ---
    pos_weights = sub_gt_pos_mask.float()
    pos_col_indices = torch.multinomial(pos_weights, num_samples=1)
    
    # --- 4. 采样负样本 ---
    sub_eye_mask = eye_mask[valid_row_indices]
    masked_pred = sub_pred.clone()
    invalid_mask = sub_gt_pos_mask | sub_eye_mask
    masked_pred.masked_fill_(invalid_mask, -float('inf'))
    
    # 动态计算 K，此时已保证所有行至少有 num_neg 个负样本
    # 取当前这就这批数据中，最少的负样本数作为上限
    min_neg_available = (~invalid_mask).sum(dim=1).min().item()
    actual_k = min(top_k_pool, min_neg_available) 
    
    # topk
    _, candidates_indices = torch.topk(masked_pred, k=actual_k, dim=1)
    
    # 权重采样
    sample_weights = torch.linspace(actual_k, 1, steps=actual_k, device=device)
    sample_weights = sample_weights.expand(M, -1)
    
    selected_rank_indices = torch.multinomial(sample_weights, num_samples=num_neg, replacement=False)
    neg_col_indices = torch.gather(candidates_indices, 1, selected_rank_indices)
    
    # --- 5. 组装 ---
    all_cols = torch.cat([pos_col_indices, neg_col_indices], dim=1)
    all_rows = valid_row_indices.unsqueeze(1).expand(-1, 1 + num_neg)
    
    pairs_tensor = torch.stack([all_rows.reshape(-1), all_cols.reshape(-1)], dim=1)
    
    # --- 6. Labels ---
    labels = torch.zeros((M, 1 + num_neg), dtype=torch.float, device=device)
    labels[:, 0] = 1.0
    labels_tensor = labels.reshape(-1)
    
#     return pairs_tensor, labels_tensor

import torch

@torch.no_grad()
def generate_rerank_training_data_uniform(gt_matrix, pred_matrix, top_k_pool=30, num_neg=3,positivate=0.9):
    """
    修改版：负样本在 Top-K pool 中进行【均匀随机】采样，而非加权采样。
    """
    B = gt_matrix.shape[0]
    device = gt_matrix.device
    
    # --- 1. 基础Mask ---
    eye_mask = torch.eye(B, dtype=torch.bool, device=device)
    pos_mask = (gt_matrix >= positivate) & (~eye_mask)
    
    # --- 2. 筛选有效行 ---
    row_pos_counts = pos_mask.sum(dim=1)
    row_neg_counts = B - row_pos_counts - 1
    
    # 确保既有正样本，且负样本数量至少够 num_neg 个
    valid_row_mask = (row_pos_counts > 0) & (row_neg_counts >= num_neg)
    valid_row_indices = torch.nonzero(valid_row_mask).squeeze(-1)
    
    M = valid_row_indices.numel()
    if M == 0:
        return torch.empty((0, 2), dtype=torch.long, device=device), torch.empty((0,), dtype=torch.float, device=device)
    
    sub_gt_pos_mask = pos_mask[valid_row_indices]
    sub_pred = pred_matrix[valid_row_indices]
    
    # --- 3. 采样正样本 (保持不变) ---
    pos_weights = sub_gt_pos_mask.float()
    pos_col_indices = torch.multinomial(pos_weights, num_samples=1)
    
    # --- 4. 采样负样本 (修改核心) ---
    sub_eye_mask = eye_mask[valid_row_indices]
    masked_pred = sub_pred.clone()
    
    # 屏蔽正样本和自身
    invalid_mask = sub_gt_pos_mask | sub_eye_mask
    masked_pred.masked_fill_(invalid_mask, -float('inf'))
    
    # 计算实际可用的 K (防止越界)
    min_neg_available = (~invalid_mask).sum(dim=1).min().item()
    
    # 注意：这里需要确保实际K不小于我们要采样的数量 num_neg
    # 如果 top_k_pool 设置得比 num_neg 小，这里强制取 max 可能会导致重复采样或报错
    # 建议调用时保证 top_k_pool >= num_neg
    actual_k = min(top_k_pool, min_neg_available)
    
    # 选出 Top-K 的候选集索引
    _, candidates_indices = torch.topk(masked_pred, k=actual_k, dim=1)
    
    # ================= 修改开始 =================
    # 原逻辑：权重随排名递减 (Hard Negative Mining 强度大)
    # sample_weights = torch.linspace(actual_k, 1, steps=actual_k, device=device).expand(M, -1)
    
    # 新逻辑：权重全为 1 (Top-K 内均匀分布/完全随机)
    # 既然都在 Top-K 里了，大家被选中的概率平等
    sample_weights = torch.ones((M, actual_k), device=device)
    # ================= 修改结束 =================
    
    # 从 candidates 中随机选出 num_neg 个索引
    # replacement=False 保证不会选出重复的负样本
    selected_rank_indices = torch.multinomial(sample_weights, num_samples=num_neg, replacement=False)
    
    # 映射回原始列索引
    neg_col_indices = torch.gather(candidates_indices, 1, selected_rank_indices)
    
    # --- 5. 组装 ---
    all_cols = torch.cat([pos_col_indices, neg_col_indices], dim=1)
    # 扩展行索引
    all_rows = valid_row_indices.unsqueeze(1).expand(-1, 1 + num_neg)
    
    pairs_tensor = torch.stack([all_rows.reshape(-1), all_cols.reshape(-1)], dim=1)
    
    # --- 6. Labels ---
    labels = torch.zeros((M, 1 + num_neg), dtype=torch.float, device=device)
    labels[:, 0] = 1.0
    labels_tensor = labels.reshape(-1)
    
    return pairs_tensor, labels_tensor

# @torch.no_grad()
# def generate_rerank_listwise_data_ultimate(
#     gt_matrix, 
#     pred_matrix, 
#     top_k_rerank=10, 
#     positive_threshold=0.9,
#     candidate_pool=30,
#     sampling_strategy="uniform", # "linear" 或 "uniform"
#     num_easy_negatives=2
# ):
#     """
#     生成用于重排序 (Reranking) 模型的 Listwise 训练数据。
    
#     采样策略包含三个部分，确保每个 Query 的列表中至少包含 1 个正样本，
#     同时结合了模型当前的高分预测结果和全局的随机探索：
    
#     1. 保底正样本 (Anchor Positive): 从 Ground Truth 中随机抽取 1 个正样本。
#     2. 高分候选采样 (Top-N Pool Candidates): 从预测分数 (pred_matrix) 最高的 Top-N 中
#        采样 K_pool 个样本。这些样本是模型认为相关的候选，可能包含 Hard Negatives 
#        (高分负样本) 以及额外的 Positives (其他正样本)。
#     3. 长尾探索采样 (Easy Negatives/Tail Candidates): 从除了上述两步之外的剩余大盘样本中，
#        均匀采样 K_easy 个样本。这些通常是模型打分较低的 Easy Negatives，用于维持全局视野。

#     Args:
#         gt_matrix: (B, B) 真实相关性矩阵
#         pred_matrix: (B, B) 模型预测的相似度得分矩阵
#         top_k_rerank: int, 最终每个 Query 生成的列表长度 K
#         positive_threshold: float, gt_matrix 中 >= 该值被认为是正样本
#         candidate_pool: int, 高分候选池 Top-N 的大小
#         sampling_strategy: str, 高分候选池的采样策略 ("linear" 线性衰减倾向更高分，或 "uniform" 均匀采样)
#         num_easy_negatives: int, 长尾探索采样的数量 K_easy

#     Returns:
#         query_indices: (M,) 有效 query 在 batch 中的索引, 如果没有包含正样本的 query 则返回 None
#         candidate_indices: (M, K) 每个 query 对应的 K 个候选样本索引
#         relevance_labels: (M, K) 候选样本的真实二值标签 (1.0 为正，0.0 为负)
#         initial_scores: (M, K) 候选样本的初始预测分数
#     """
#     B = gt_matrix.shape[0]
#     device = gt_matrix.device

#     # 1. 屏蔽对角线 (排除 Query 自身)
#     pred_masked = pred_matrix.clone()
#     pred_masked.fill_diagonal_(-float('inf'))
    
#     # 2. 识别全局正样本并筛选出至少有 1 个正样本的有效 Query
#     gt_masked = gt_matrix.clone()
#     gt_masked.fill_diagonal_(0.0) 
#     is_positive_mask = gt_masked >= positive_threshold  # (B, B)
#     has_pos = is_positive_mask.sum(dim=1) > 0           # (B,)
#     query_indices = torch.nonzero(has_pos).squeeze(-1)  # (M,) 
    
#     if query_indices.numel() == 0:
#         return None, None, None, None
        
#     M = query_indices.size(0)

#     # ---------------- 阶段一：选定 1 个保底正样本 ----------------
#     pos_weights = is_positive_mask[query_indices].float()
#     sampled_pos_indices = torch.multinomial(pos_weights, num_samples=1) # (M, 1)

#     # 计算高分候选池和长尾探索的名额分配
#     max_available = B - 1
#     K = min(top_k_rerank, max_available)
#     if K <= 1: 
#         # 如果只需要 1 个候选，直接返回选定的正样本即可
#         relevance_labels = torch.ones(M, 1, device=device)
#         initial_scores = torch.gather(pred_masked[query_indices], 1, sampled_pos_indices)
#         return query_indices, sampled_pos_indices, relevance_labels, initial_scores

#     K_easy = min(num_easy_negatives, K - 1)
#     K_pool = K - 1 - K_easy  # 留给 Top-N 高分候选池的数量

#     # ---------------- 阶段二：从 Top-N 候选池中采样 (包含困难负样本和其他正样本) ----------------
#     # 排除刚刚选中的保底正样本 (设为 -inf，防止重复采样)
#     pred_for_pool = pred_masked[query_indices].clone()
#     pred_for_pool.scatter_(1, sampled_pos_indices, -float('inf'))
    
#     # 确定候选池大小 N 并获取 Top-N
#     N = min(candidate_pool, B - 2) if candidate_pool > K_pool else K_pool
#     pool_scores, pool_indices = torch.topk(pred_for_pool, k=N, dim=1) # (M, N)

#     if K_pool > 0:
#         if candidate_pool > K_pool:
#             if sampling_strategy == "linear":
#                 sample_weights = torch.linspace(N, 1, steps=N, device=device).unsqueeze(0).expand(M, -1)
#             elif sampling_strategy == "uniform":
#                 sample_weights = torch.ones(M, N, device=device)
#             else:
#                 raise ValueError("不支持的 sampling_strategy")
                
#             sampled_pool_rank_indices = torch.multinomial(sample_weights, num_samples=K_pool, replacement=False) 
#             sampled_pool_indices = torch.gather(pool_indices, 1, sampled_pool_rank_indices) # (M, K_pool)
#         else:
#             sampled_pool_indices = pool_indices[:, :K_pool] # (M, K_pool)
#     else:
#         sampled_pool_indices = torch.empty((M, 0), dtype=torch.long, device=device)

#     # ---------------- 阶段三：长尾探索采样 (通常为 Easy Negatives) ----------------
#     if K_easy > 0:
#         pred_for_easy = pred_for_pool.clone() 
        
#         # 将 Top-N 候选池也排除掉 (设为 -inf)
#         pred_for_easy.scatter_(1, pool_indices, -float('inf'))
        
#         # 剩下的非 -inf 元素进行均匀随机采样
#         easy_weights = (pred_for_easy != -float('inf')).float()
#         sampled_easy_indices = torch.multinomial(easy_weights, num_samples=K_easy, replacement=False) # (M, K_easy)
#     else:
#         sampled_easy_indices = torch.empty((M, 0), dtype=torch.long, device=device)

#     # ---------------- 阶段四：组装最终的 Listwise 数据 ----------------
#     # 按 [保底正样本, 高分候选样本, 长尾探索样本] 的顺序拼接
#     candidate_indices = torch.cat([sampled_pos_indices, sampled_pool_indices, sampled_easy_indices], dim=1) # (M, K)
    
#     # 重新用 gt_matrix 获取这 K 个样本的真实 relevance labels
#     # 因为 Top-N 候选和长尾采样的样本中，可能恰好也包含了其他的正样本，所以必须重新校验打标
#     gt_at_candidates = torch.gather(gt_matrix[query_indices], 1, candidate_indices)
#     relevance_labels = (gt_at_candidates >= positive_threshold).float()
    
#     # 获取这 K 个候选在模型中的初始预测分数
#     initial_scores = torch.gather(pred_masked[query_indices], 1, candidate_indices)
#     # (M,) (M,K) (M,K) (M,K)
#     return query_indices, candidate_indices, relevance_labels, initial_scores

@torch.no_grad()
def generate_rerank_listwise_data_ultimate(
    gt_matrix, 
    pred_matrix, 
    top_k_rerank=10, 
    positive_threshold=0.9,
    candidate_pool=20,
    sampling_strategy="uniform", # "linear" 或 "uniform"
    num_easy_negatives=1
):
    """
    生成用于重排序 (Reranking) 模型的 Listwise 训练数据。
    
    采样策略包含三个部分，确保每个 Query 的列表中至少包含 1 个正样本，
    同时结合了模型当前的高分预测结果和全局的随机探索：
    
    1. 保底正样本 (Anchor Positive): 从 Ground Truth 中随机抽取 1 个正样本。
    2. 高分候选采样 (Top-N Pool Candidates): 从预测分数 (pred_matrix) 最高的 Top-N 中
       采样 K_pool 个样本。这些样本是模型认为相关的候选，可能包含 Hard Negatives 
       (高分负样本) 以及额外的 Positives (其他正样本)。
    3. 长尾探索采样 (Easy Negatives/Tail Candidates): 从除了上述两步之外的剩余大盘样本中，
       均匀采样 K_easy 个样本。这些通常是模型打分较低的 Easy Negatives，用于维持全局视野。

    Args:
        gt_matrix: (B, B) 真实相关性矩阵
        pred_matrix: (B, B) 模型预测的相似度得分矩阵
        top_k_rerank: int, 最终每个 Query 生成的列表长度 K
        positive_threshold: float, gt_matrix 中 >= 该值被认为是正样本
        candidate_pool: int, 高分候选池 Top-N 的大小
        sampling_strategy: str, 高分候选池的采样策略 ("linear" 线性衰减倾向更高分，或 "uniform" 均匀采样)
        num_easy_negatives: int, 长尾探索采样的数量 K_easy

    Returns:
        query_indices: (M,) 有效 query 在 batch 中的索引, 如果没有包含正样本的 query 则返回 None
        candidate_indices: (M, K) 每个 query 对应的 K 个候选样本索引
        relevance_labels: (M, K) 候选样本的真实二值标签 (1.0 为正，0.0 为负)
        initial_scores: (M, K) 候选样本的初始预测分数
    """
    B = gt_matrix.shape[0]
    # print('gt_matrix', gt_matrix)
    # print('pred_matrix', pred_matrix)
    device = gt_matrix.device

    # 1. 屏蔽对角线 (排除 Query 自身)
    pred_masked = pred_matrix.clone()
    pred_masked.fill_diagonal_(-float('inf'))
    
    # 2. 识别全局正样本并筛选出至少有 1 个正样本的有效 Query
    gt_masked = gt_matrix.clone()
    gt_masked.fill_diagonal_(0.0) 
    is_positive_mask = gt_masked >= positive_threshold  # (B, B)
    has_pos = is_positive_mask.sum(dim=1) > 0           # (B,)
    query_indices = torch.nonzero(has_pos).squeeze(-1)  # (M,) 
    
    if query_indices.numel() == 0:
        return None, None, None, None
        
    M = query_indices.size(0)

    # ---------------- 阶段一：选定 1 个保底正样本 ----------------
    pos_weights = is_positive_mask[query_indices].float()
    sampled_pos_indices = torch.multinomial(pos_weights, num_samples=1) # (M, 1)

    # 计算高分候选池和长尾探索的名额分配
    max_available = B - 1
    K = min(top_k_rerank, max_available)
    if K <= 1: 
        # 如果只需要 1 个候选，直接返回选定的正样本即可
        relevance_labels = torch.ones(M, 1, device=device)
        initial_scores = torch.gather(pred_masked[query_indices], 1, sampled_pos_indices)
        return query_indices, sampled_pos_indices, relevance_labels, initial_scores

    K_easy = min(num_easy_negatives, K - 1)
    K_pool = K - 1 - K_easy  # 留给 Top-N 高分候选池的数量

    # ---------------- 阶段二：从 Top-N 候选池中采样 (包含困难负样本和其他正样本) ----------------
    # 排除刚刚选中的保底正样本 (设为 -inf，防止重复采样)
    pred_for_pool = pred_masked[query_indices].clone()
    pred_for_pool.scatter_(1, sampled_pos_indices, -float('inf'))
    
    # 确定候选池大小 N 并获取 Top-N
    # N = min(candidate_pool, B - 2) if candidate_pool > K_pool else K_pool
    N = max(K_pool, min(candidate_pool, B - 2 - K_easy))
    pool_scores, pool_indices = torch.topk(pred_for_pool, k=N, dim=1) # (M, N)

    if K_pool > 0:
        if candidate_pool > K_pool:
            if sampling_strategy == "linear":
                sample_weights = torch.linspace(N, 1, steps=N, device=device).unsqueeze(0).expand(M, -1)
            elif sampling_strategy == "uniform":
                sample_weights = torch.ones(M, N, device=device)
            else:
                raise ValueError("不支持的 sampling_strategy")
                
            sampled_pool_rank_indices = torch.multinomial(sample_weights, num_samples=K_pool, replacement=False) 
            sampled_pool_indices = torch.gather(pool_indices, 1, sampled_pool_rank_indices) # (M, K_pool)
        else:
            sampled_pool_indices = pool_indices[:, :K_pool] # (M, K_pool)
    else:
        sampled_pool_indices = torch.empty((M, 0), dtype=torch.long, device=device)

    # ---------------- 阶段三：长尾探索采样 (通常为 Easy Negatives) ----------------
    if K_easy > 0:
        pred_for_easy = pred_for_pool.clone() 
        
        # 将 Top-N 候选池也排除掉 (设为 -inf)
        pred_for_easy.scatter_(1, pool_indices, -float('inf'))
        
        # 剩下的非 -inf 元素进行均匀随机采样
        easy_weights = (pred_for_easy != -float('inf')).float()
        # print('k_pool', K_pool)
        # print('candidate_pool', candidate_pool)
        # print('easy_weights', easy_weights)
        sampled_easy_indices = torch.multinomial(easy_weights, num_samples=K_easy, replacement=False) # (M, K_easy)
    else:
        sampled_easy_indices = torch.empty((M, 0), dtype=torch.long, device=device)

    # ---------------- 阶段四：组装最终的 Listwise 数据 ----------------
    # 按 [保底正样本, 高分候选样本, 长尾探索样本] 的顺序拼接
    candidate_indices = torch.cat([sampled_pos_indices, sampled_pool_indices, sampled_easy_indices], dim=1) # (M, K)
    # Shuffle：打乱每个 query 的候选列表顺序，避免模型学到位置偏置
    shuffle_perm = torch.argsort(torch.rand(M, K, device=device), dim=1)  # (M, K)
    candidate_indices = torch.gather(candidate_indices, 1, shuffle_perm)
    
    # 重新用 gt_matrix 获取这 K 个样本的真实 relevance labels
    # 因为 Top-N 候选和长尾采样的样本中，可能恰好也包含了其他的正样本，所以必须重新校验打标
    gt_at_candidates = torch.gather(gt_matrix[query_indices], 1, candidate_indices)
    relevance_labels = (gt_at_candidates >= positive_threshold).float()
    
    # 获取这 K 个候选在模型中的初始预测分数
    initial_scores = torch.gather(pred_masked[query_indices], 1, candidate_indices)
    # (M,) (M,K) (M,K) (M,K)
    return query_indices, candidate_indices, relevance_labels, initial_scores

# # # --- 测试代码 ---
# B = 40
# M_approx = 80
# # 模拟数据
# sim_matrix = torch.randn(B, B) # 预测的相似度
# # 模拟 label，稀疏一点
# # label_matrix = (torch.rand(B, B) >= 0.9).long() # 01矩阵
# label_matrix = torch.rand(B, B)
# label_matrix.fill_diagonal_(1)

# print('label_matrix', label_matrix)
# pairs, labels = generate_rerank_training_data(label_matrix,sim_matrix)

# print(f"Pairs shape: {pairs.shape}")   # 应该是 (4*M_actual, 2)
# print(f"Labels shape: {labels.shape}") # 应该是 (4*M_actual,)
# # 检查一下前5个
# print("Sample pairs:", pairs)
# print("Sample labels:", labels)

# import torch

# M, N = 2, 5

# # 创建示例数据
# indices = torch.randint(0, N, (M, 2))  # (M, 2)
# values = torch.randn(N,3, 5)  # (N, 512)

# # 提取结果
# result_0 = values[indices[:, 0]]  # (M, 512)
# result_1 = values[indices[:, 1]]  # (M, 512)

# print(f"indices shape: {indices}")
# print(f"values shape: {values}")
# print(f"result_0 shape: {result_0}")
# print(f"result_1 shape: {result_1}")