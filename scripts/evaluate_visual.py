# from conditional_dataset_evaluate import Conditional_CTReportDataset_Eval
# import torch
# from torch import nn, einsum
# from torch.utils.data import Dataset, DataLoader, random_split
# from torch.utils.data.distributed import DistributedSampler
# from torch.utils.tensorboard import SummaryWriter
# from torch import cuda
# import numpy as np
# import pandas as pd
# import tqdm
# from ct_clip import CTCLIP
# from transformers import AutoTokenizer, AutoModel
# from retrieval_metric import compute_ndcg, hard_retrieval, hard_retrieval_exclude_self, soft_retrieval,compute_ndcg_exclude_self, soft_retrieval_exclude_self, RateScore_retrieval, compute_ndcg_uncon,soft_retrieval_uncon
# from datetime import datetime
# import json
# from transformer_maskgit import CTViT
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# # 清理显存
# import gc
# import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# torch.cuda.empty_cache()
# gc.collect()
# image_encoder = CTViT(
#             dim = 512,
#             codebook_size = 8192,
#             image_size = 480,
#             patch_size = 20,
#             temporal_patch_size = 10,
#             spatial_depth = 8,
#             temporal_depth = 4,
#             dim_head = 32,
#             heads = 8
#         )
# tokenizer = AutoTokenizer.from_pretrained('FremyCompany/BioLORD-2023')
# text_encoder = AutoModel.from_pretrained('FremyCompany/BioLORD-2023')
# clip = CTCLIP(  # Cross-Attn as feature fusion module will be initialized in CTCLIP 
#             image_encoder = image_encoder,
#             text_encoder = text_encoder,
#             tokenizer = tokenizer,
#             dim_text = 768,
#             dim_image = 512,
#             dim_latent = 512,
#             extra_latent_projection = False,         # whether to use separate projections for text-to-image vs image-to-text comparisons (CLOOB)
#             use_mlm=False,
#             downsample_image_embeds = False,
#             use_all_token_embeds = False
#         ).to(device)

# # check_point = '/mnt/petrelfs/zhangtengfei/RadIR_log/CT-CLIP_v6/aug_log_extend_test/128_CXR_stage2_train_1e-4_anatomy_ratescore_rerank/CTClip.8000.pt'
# check_point = '/mnt/petrelfs/zhangtengfei/RadIR_log/CT-CLIP_v6/aug_log_extend_test/130_CXR_stage2_train_1e-4_anatomy_ratescore_rerank_only_rerank/CTClip.6000.pt'

# pkg = torch.load(check_point, map_location=device, weights_only=False)
# state_dict = pkg['model'] if 'model' in pkg else pkg
# clip.load_state_dict(state_dict, strict=False)
# print('Loaded checkpoint from {}'.format(check_point))
# clip.eval()
# datset_name = 'MIMIC-CXR'
# modality_type= '2D-CXR'  # '3D-CT-Chest'  '2D-CXR' '3D-Brain-MRI' '3D-CT-abdomen' '2D-Ultrasound'
# # 对应的是玩真更多结果啊
# # jsonl_file = '/mnt/petrelfs/zhangtengfei/RadIR/dataset/CXR/MIMIC_CXR/train.jsonl'
# # # 处理数据会花一段时间？
# # csv_file = '/mnt/petrelfs/zhangtengfei/public_dataset/extend_test/CXR_anatomy/train_anatomy_csv_balanced'
# # npy_file = '/mnt/petrelfs/zhangtengfei/public_dataset/extend_test/CXR_anatomy/ratescore/train_npy'
# jsonl_file = '/mnt/petrelfs/zhangtengfei/RadIR/dataset/CXR/MIMIC_CXR/test.jsonl'
# csv_file = '/mnt/petrelfs/zhangtengfei/public_dataset/extend_test/CXR_anatomy/test_anatomy_csv_balanced'
# npy_file = '/mnt/petrelfs/zhangtengfei/public_dataset/extend_test/CXR_anatomy/ratescore/test_npy'
# anatomy_filter = ["Pulmonary Parenchyma","Bronchi (Lower Respiratory Tract)","Pulmonary Interstitium","Trachea_Tracheal Lumen","Cardiac Chambers","Cardiac Valves","Pulmonary Vasculature","Mediastinum","Pleura","Diaphragm","Stomach","Ribs","Sternum","Aorta"]
# valid_ds_i = Conditional_CTReportDataset_Eval(
#                 modality=modality_type,
#                 jsonl_file=jsonl_file, 
#                 csv_file_dir=csv_file, 
#                 npy_file_dir=npy_file, 
#                 anatomy_filter=anatomy_filter,
#                 modal_embedding=True,
#                 max_samples=5000
#                 )
# valid_dl_i = DataLoader(
#                 valid_ds_i,
#                 num_workers = 4,
#                 batch_size = 100,
#                 shuffle = False,
#                 pin_memory = True,
#             )

# # B*512的结果来看，损失了很多精度的情况下，能否再涨一点呢？
# with open('/mnt/petrelfs/zhangtengfei/RadIR/CT-CLIP/CT-UNI-Image-Retrieval-CXR_CT_change_layer_condition_moe_cross_v5_abnormal_bool/scripts/all_condition_index.json', 'r', encoding='utf-8') as f:
#     condition_index = json.load(f)
    
# anatomy2id_ls = valid_ds_i.get_anatomy2id_ls()
# fused_latents_all = {anatomy:torch.zeros(len(anatomy2id_ls[anatomy]), 512, dtype=torch.float16) for anatomy in anatomy_filter} 
# local_latents_all = {anatomy:torch.zeros(len(anatomy2id_ls[anatomy]), 512, dtype=torch.float16) for anatomy in anatomy_filter}
# condition_latents_all = {anatomy:torch.zeros(len(anatomy2id_ls[anatomy]), 512, dtype=torch.float16) for anatomy in anatomy_filter}

# for video, anatomy_ls, sample_id_ls,modal_indexs,local_text_ls,abnormal_gt  in tqdm.tqdm(valid_dl_i):
#     # video: B 1 d h w
#     video = video.unsqueeze(1).to(device)  # B 1 1 d h w (凑出local_B)
#     anatomy_ls = list(anatomy_ls)
#     sample_id_ls = list(sample_id_ls)
#     condition_indexs = [condition_index[datset_name][condition] for condition in anatomy_ls]
#     condition_indexs = torch.tensor(condition_indexs).to(device)
#     local_text_tokens = tokenizer(local_text_ls, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
#     # 会不会是测试的时候影响到我的结果了，这个调整一下
#     with torch.no_grad():
#         with torch.autocast(device_type='cuda', dtype=torch.float16):
#             try:
#                 local_latents, fused_latents,condition_latents,image_tokens,atten_layers ,_ = clip(
#                     condition_indexs, video, return_latents=True, device=device,is_condition=True,modal_indexs=modal_indexs.to(device),modal_embedding=True,local_text=local_text_tokens
#                 )  # B 1 Dim .local_B一定是1
#             except RuntimeError as e:
#                 print(f"Error details: {e}")
#                 print('condition_indexs', condition_indexs)
#                 print('video shape', video.shape)
#                 print('modal_indexs', modal_indexs)
#                 print('local_text_tokens', local_text_tokens)
#                 raise
#     for i, (sample_id, anatomy) in enumerate(zip(sample_id_ls, anatomy_ls)):
#         idx = anatomy2id_ls[anatomy].index(sample_id)
#         fused_latents_all[anatomy][idx] = fused_latents[i].to(torch.float16)
#         local_latents_all[anatomy][idx] = local_latents[i].to(torch.float16)
#         condition_latents_all[anatomy][idx] = condition_latents[i].to(torch.float16)


import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import json
import tqdm
from torch.utils.data import DataLoader
from conditional_dataset_evaluate import Conditional_CTReportDataset_Eval
from ct_clip import CTCLIP
from transformer_maskgit import CTViT
from transformers import AutoTokenizer, AutoModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 初始化模型
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

# 加载checkpoint
check_point = '/mnt/petrelfs/zhangtengfei/RadIR_log/CT-CLIP_v5/aug_log_extend_test/120_CXR_stage2_train_1e-4_anatomy_ratescore_open_all/CTClip.4000.pt'
pkg = torch.load(check_point, map_location=device, weights_only=False)
state_dict = pkg['model'] if 'model' in pkg else pkg
clip.load_state_dict(state_dict, strict=False)
print(f'Loaded checkpoint from {check_point}')
clip.eval()

# 数据集配置
datset_name = 'MIMIC-CXR'
modality_type = '2D-CXR'
jsonl_file = '/mnt/petrelfs/zhangtengfei/RadIR/dataset/CXR/MIMIC_CXR/test.jsonl'
csv_file = '/mnt/petrelfs/zhangtengfei/public_dataset/extend_test/CXR_anatomy/test_anatomy_csv_balanced'
npy_file = '/mnt/petrelfs/zhangtengfei/public_dataset/extend_test/CXR_anatomy/ratescore/test_npy'
anatomy_filter = ["Pulmonary Parenchyma", "Bronchi (Lower Respiratory Tract)", 
                  "Pulmonary Interstitium", "Trachea_Tracheal Lumen", 
                  "Cardiac Chambers", "Cardiac Valves", 
                  "Pulmonary Vasculature", "Mediastinum", 
                  "Pleura", "Diaphragm", "Stomach", "Ribs", "Sternum", "Aorta"]

# 加载数据集
valid_ds_i = Conditional_CTReportDataset_Eval(
    modality=modality_type,
    jsonl_file=jsonl_file,
    csv_file_dir=csv_file,
    npy_file_dir=npy_file,
    anatomy_filter=anatomy_filter,
    modal_embedding=True,
    max_samples=5000
)

# 加载condition index
with open('/mnt/petrelfs/zhangtengfei/RadIR/CT-CLIP/CT-UNI-Image-Retrieval-CXR_CT_change_layer_condition_moe_cross_v5_abnormal_bool/scripts/all_condition_index.json', 'r', encoding='utf-8') as f:
    condition_index = json.load(f)

# 创建输出文件夹
output_dir = '/mnt/petrelfs/zhangtengfei/RadIR/CT-CLIP/visualizations/attention_maps'
os.makedirs(output_dir, exist_ok=True)

def visualize_attention(image, attention_weights, anatomy, sample_id, layer_idx, output_path):
    """
    可视化attention map
    
    Args:
        image: 原始图像 tensor [1, H, W]
        attention_weights: attention权重 [1, num_patches]
        anatomy: anatomy名称
        sample_id: 样本ID
        layer_idx: decoder层索引
        output_path: 输出路径
    """
    # 转换图像为numpy
    img_np = image.squeeze().cpu().numpy()
    
    # 重塑attention map为2D (假设是24x24的patch)
    attn_map = attention_weights.squeeze().cpu().numpy()
    attn_2d = attn_map.reshape(24, 24)
    
    # 上采样attention map到原始图像大小
    from scipy.ndimage import zoom
    zoom_factor = img_np.shape[0] / attn_2d.shape[0]
    attn_resized = zoom(attn_2d, zoom_factor, order=1)
    
    # 归一化
    attn_resized = (attn_resized - attn_resized.min()) / (attn_resized.max() - attn_resized.min() + 1e-8)
    
    # 创建图形
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 原始图像
    axes[0].imshow(img_np, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Attention map
    axes[1].imshow(attn_resized, cmap='jet')
    axes[1].set_title(f'Attention Map (Layer {layer_idx})')
    axes[1].axis('off')
    
    # 叠加图像
    axes[2].imshow(img_np, cmap='gray')
    axes[2].imshow(attn_resized, cmap='jet', alpha=0.5)
    axes[2].set_title('Overlay')
    axes[2].axis('off')
    
    plt.suptitle(f'Anatomy: {anatomy}\nSample: {sample_id}')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

# 为每个anatomy选择10张图片
anatomy2id_ls = valid_ds_i.get_anatomy2id_ls()

for anatomy in anatomy_filter:
    print(f'Processing anatomy: {anatomy}')
    
    # 创建anatomy文件夹
    anatomy_dir = os.path.join(output_dir, anatomy.replace('/', '_'))
    os.makedirs(anatomy_dir, exist_ok=True)
    
    # 选择前10个样本
    sample_ids = anatomy2id_ls[anatomy][:10]
    
    for idx, sample_id in enumerate(tqdm.tqdm(sample_ids)):
        # 获取样本数据
        sample_idx = valid_ds_i.id_antomy_ls.index([anatomy, sample_id])
        video, anatomy_name, sid, modal_index, local_text, abnormal_gt = valid_ds_i[sample_idx]
        
        # 准备输入
        video = video.unsqueeze(0).unsqueeze(0).to(device)  # [1, 1, 1, H, W]
        condition_idx = torch.tensor([condition_index[datset_name][anatomy]]).to(device)
        modal_indexs = torch.tensor([modal_index]).to(device)
        local_text_tokens = tokenizer([local_text], return_tensors="pt", padding=True, 
                                      truncation=True, max_length=512).to(device)
        
        # 前向传播获取attention
        with torch.no_grad():
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                _, _, _, _, atten_layers, _ = clip(
                    condition_idx, video, 
                    return_latents_all=True, 
                    device=device,
                    is_condition=True,
                    modal_indexs=modal_indexs,
                    modal_embedding=True,
                    local_text=local_text_tokens
                )
        
        # 可视化每一层的attention
        for layer_idx, attn_weights in enumerate(atten_layers):
            # attn_weights shape: [1, num_query_tokens, num_key_tokens]
            # 取第一个query token的attention (condition token)
            attn_map = attn_weights[0, 0, :]  # [num_key_tokens]
            
            output_path = os.path.join(anatomy_dir, 
                                       f'sample_{idx:02d}_layer_{layer_idx}_attn.png')
            
            visualize_attention(video[0, 0, 0], attn_map.unsqueeze(0), 
                              anatomy, sample_id, layer_idx, output_path)
        
        print(f'  Saved visualizations for sample {idx+1}/10: {sample_id}')

print(f'All visualizations saved to {output_dir}')