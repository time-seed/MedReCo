from conditional_dataset_evaluate import Conditional_CTReportDataset_Eval
import torch
from torch import nn, einsum
from torch.utils.data import Dataset, DataLoader
import numpy as np
import tqdm
from ct_clip import CTCLIP
from transformers import AutoTokenizer, AutoModel
import json
from transformer_maskgit import CTViT
import gc
import os

# --- 环境设置 ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
torch.cuda.empty_cache()
gc.collect()

# --- 模型定义 ---
image_encoder = CTViT(
    dim=512,
    codebook_size=8192,
    image_size=480,
    patch_size=20,
    temporal_patch_size=10,
    spatial_depth=8,
    temporal_depth=4,
    dim_head=32,
    heads=8
)
tokenizer = AutoTokenizer.from_pretrained('FremyCompany/BioLORD-2023')
text_encoder = AutoModel.from_pretrained('FremyCompany/BioLORD-2023')
clip = CTCLIP(
    image_encoder=image_encoder,
    text_encoder=text_encoder,
    tokenizer=tokenizer,
    dim_text=768,
    dim_image=512,
    dim_latent=512,
    extra_latent_projection=False,
    use_mlm=False,
    downsample_image_embeds=False,
    use_all_token_embeds=False
).to(device)

# --- 加载权重 ---
# check_point = '/mnt/petrelfs/zhangtengfei/RadIR_log/CT-CLIP_v7/aug_log_extend_test/205_CXR_stage2_train_1e-4_anatomy_only_rerank/CTClip.4000.pt'
check_point = '/mnt/petrelfs/zhangtengfei/RadIR_log/CT-CLIP_v7/aug_log_extend_test/205_CXR_stage2_train_1e-4_anatomy_only_rerank/CTClip.10000.pt'

pkg = torch.load(check_point, map_location=device, weights_only=False)
state_dict = pkg['model'] if 'model' in pkg else pkg
clip.load_state_dict(state_dict, strict=False)
print('Loaded checkpoint from {}'.format(check_point))
clip.eval()

# --- 数据集设置 ---
datset_name = 'MIMIC-CXR'
modality_type = '2D-CXR'
jsonl_file = '/mnt/petrelfs/zhangtengfei/RadIR/dataset/CXR/MIMIC_CXR/test.jsonl'
csv_file = '/mnt/petrelfs/zhangtengfei/public_dataset/extend_test/CXR_anatomy/test_anatomy_csv_balanced'
npy_file = '/mnt/petrelfs/zhangtengfei/public_dataset/extend_test/CXR_anatomy/ratescore/test_npy'

anatomy_filter = ["Pulmonary Parenchyma", "Bronchi (Lower Respiratory Tract)", "Pulmonary Interstitium",
                  "Trachea_Tracheal Lumen", "Cardiac Chambers", "Cardiac Valves", "Pulmonary Vasculature",
                  "Mediastinum", "Pleura", "Diaphragm", "Stomach", "Ribs", "Sternum", "Aorta"]

valid_ds_i = Conditional_CTReportDataset_Eval(
    modality=modality_type,
    jsonl_file=jsonl_file,
    csv_file_dir=csv_file,
    npy_file_dir=npy_file,
    anatomy_filter=anatomy_filter,
    modal_embedding=True,
    max_samples=5000 
)
valid_dl_i = DataLoader(
    valid_ds_i,
    num_workers=4,
    batch_size=100,
    shuffle=False,
    pin_memory=True,
)

with open('/mnt/petrelfs/zhangtengfei/RadIR/CT-CLIP/CT-UNI-Image-Retrieval-CXR_CT_change_layer_condition_moe_cross_v5_abnormal_bool/scripts/all_condition_index.json', 'r', encoding='utf-8') as f:
    condition_index = json.load(f)

anatomy2id_ls = valid_ds_i.get_anatomy2id_ls()

# 仅存储 Attention Map，其他特征对于本次统计不是必须的
attn_map_all = {anatomy: [] for anatomy in anatomy_filter}

print("Start Inference for Attention Analysis...")
for video, anatomy_ls, sample_id_ls, modal_indexs, local_text_ls, abnormal_gt in tqdm.tqdm(valid_dl_i):
    video = video.unsqueeze(1).to(device)
    anatomy_ls = list(anatomy_ls)
    condition_indexs = [condition_index[datset_name][condition] for condition in anatomy_ls]
    condition_indexs = torch.tensor(condition_indexs).to(device)
    local_text_tokens = tokenizer(local_text_ls, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)

    with torch.no_grad():
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            # 只需要提取 last_attn_map
            # 假设 outputs 返回顺序为: local, fused, cond, img_tok, _, pos, last_attn_map
            outputs = clip(
                condition_indexs, video, return_latents_all=True, device=device,
                is_condition=True, modal_indexs=modal_indexs.to(device),
                modal_embedding=True, local_text=local_text_tokens
            )
            # 获取 Attention Map (最后一项)
            last_attn_map = outputs[-2]
            # print(f"Batch Attention Map Shape: {last_attn_map.shape}")
            # print(last_attn_map)
    # 按 anatomy 归档
    for i, anatomy in enumerate(anatomy_ls):
        # 将 attn map 转到 CPU 并转为 float32 以便后续精确计算 sum
        attn_map_all[anatomy].append(last_attn_map[i].cpu().float())

# 整理数据格式
for anatomy in anatomy_filter:
    if len(attn_map_all[anatomy]) > 0:
        attn_map_all[anatomy] = torch.stack(attn_map_all[anatomy])

# --- 核心统计逻辑: 计算 attn_weight > 0.8 需要多少个 token ---

print("\n" + "=" * 80)
print(f"{'Anatomy Name':<40} | {'Mean Top-K (0.8)':<20} | {'Max Needed':<10}")
print("=" * 80)

global_k_values = []

for anatomy in anatomy_filter:
    attn_maps = attn_map_all[anatomy] # Shape 预期: (B, Heads, 1, N) 或 (B, 1, N) 或 (B, N, N)
    
    if len(attn_maps) == 0:
        print(f"{anatomy:<40} | No Samples")
        continue

    # 1. 维度处理与 Head 平均
    # 目标 Shape: (B, N) -> N 是 patch 数量 (如 24*24=576)
    if attn_maps.dim() == 4:
        # (B, Heads, 1, N) -> Average Heads -> (B, 1, N)
        attn_avg = attn_maps.mean(dim=1)
        attn_avg = attn_avg.squeeze(1) # (B, N)
    elif attn_maps.dim() == 3:
        # (B, 1, N)
        attn_avg = attn_maps.squeeze(1)
    else:
        attn_avg = attn_maps

    B, N = attn_avg.shape

    # 2. 排序 (Descending)
    # 对每个样本的 attention map 进行降序排列
    sorted_weights, _ = torch.sort(attn_avg, descending=True, dim=-1)

    # 3. 计算累加和 (Cumulative Sum)
    cumsum_weights = torch.cumsum(sorted_weights, dim=-1)

    # 4. 找到阈值切点 (0.8)
    # 找到第一个大于等于 0.8 的索引
    # (B, N) 的布尔矩阵
    threshold_mask = cumsum_weights >= 0.8
    
    # argmax 会返回第一个 True 的索引。如果没有 True (理论上 attn sum=1，但可能 float 误差)，
    # 我们可以 clamp 或者默认取最后一个。
    # indices 是 0-based，所以数量是 index + 1
    k_needed = threshold_mask.to(torch.int).argmax(dim=-1) + 1
    
    # 处理特殊情况：如果某行所有 cumsum 都没到 0.8 (极少见)，argmax 会返回 0 (如果全 False)。
    # 这种情况下通常意味着需要所有 token，或者 attn 没归一化。
    # 简单的修正：如果最后一个元素的 cumsum 小于 0.8，则设为 N。
    mask_check = cumsum_weights[:, -1] < 0.8
    k_needed[mask_check] = N

    k_needed = k_needed.float()

    # 5. 统计
    mean_k = k_needed.mean().item()
    max_k = k_needed.max().item()
    
    # 记录到全局
    global_k_values.extend(k_needed.tolist())

    print(f"{anatomy:<40} | {mean_k:.4f} (pixels)       | {int(max_k)}")

print("-" * 80)
if len(global_k_values) > 0:
    grand_mean = np.mean(global_k_values)
    grand_median = np.median(global_k_values)
    print(f"{'OVERALL AVERAGE':<40} | {grand_mean:.4f}")
    print(f"{'OVERALL MEDIAN':<40} | {grand_median:.4f}")
else:
    print("No valid data found.")
print("=" * 80)

print("\nSuggested Logic for Top-K:")
print(f"If Overall Mean is e.g. 50, it means on average focusing on the top 50 patches covers 90% of the attention mass.")
print("Typically, you might want to set your Rerank Top-K to Mean + 1*StdDev or cover 95% of cases (Max needed might be outliers).")