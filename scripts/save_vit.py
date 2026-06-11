import argparse
import datetime
import os
import yaml
from pathlib import Path
import torch

from transformer_maskgit import CTViT, CTViT_Text_Fusion, CTViT_Temporal_Fusion, CTViT_Temporal_Fusion_butCLS
from transformers import BertTokenizer, BertModel
from ct_clip import CTCLIP, CTCLIP_NO_IMG_LATENT, CTCLIP_Tengfei_CrossAttn, CTCLIP_Earyly_Fusion, CTCLIP_Temporal_Fusion, CTCLIP_Temporal_Fusion_butCLS
# from CTCLIPTrainer import CTClipTrainer

from CTCLIPTrainer_CXR_CT import CTClipTrainer

image_encoder = CTViT(
        dim = 512,
        codebook_size = 8192,
        image_size = 480,
        patch_size = 20,
        temporal_patch_size = 10,
        spatial_depth = 8,
        temporal_depth = 6,
        cls_depth = 2,
        dim_head = 32,
        heads = 8
    )
tokenizer = BertTokenizer.from_pretrained('microsoft/BiomedVLP-CXR-BERT-specialized',do_lower_case=True)
text_encoder = BertModel.from_pretrained("microsoft/BiomedVLP-CXR-BERT-specialized")

clip = CTCLIP(  # Cross-Attn as feature fusion module will be initialized in CTCLIP 
            image_encoder = image_encoder,
            text_encoder = text_encoder,
            tokenizer = tokenizer,
            dim_text = 768,
            dim_image = 512,
            dim_latent = 512,
            extra_latent_projection = False,         # whether to use separate projections for text-to-image vs image-to-text comparisons (CLOOB)
            use_mlm=False,
            downsample_image_embeds = False,
            use_all_token_embeds = False
        )

path = '/mnt/petrelfs/zhangtengfei/RadIR/CT-CLIP/log/CXR_CT_Uni_CXRCT_1028_moe_expert_train_/CTClip.1600.pt'
print(f'** MODEL ** Load Checkpoint from {path}')
    
path = Path(path)
assert path.exists()
pkg = torch.load(path, map_location='cpu', weights_only=False)

CTClip = clip

model_dict = CTClip.state_dict()
if 'model' in pkg:
    pkg_state_dict = pkg['model']
else:
    pkg_state_dict = pkg

# 处理MOE专家权重的复制逻辑
processed_state_dict = {}
expert_copy_map = {3: 0, 4: 1}  # 第4个专家复制第1个，第5个专家复制第2个

for key, value in pkg_state_dict.items():
    # 检查是否是MOE专家相关的权重
    if '.experts.' in key:
        # 解析专家编号
        parts = key.split('.')
        for i, part in enumerate(parts):
            if part == 'experts' and i + 1 < len(parts) and parts[i+1].isdigit():
                expert_idx = int(parts[i+1])
                # 如果需要复制权重
                if expert_idx in expert_copy_map:
                    source_expert = expert_copy_map[expert_idx]
                    # 构建源权重的key
                    source_parts = parts.copy()
                    source_parts[i+1] = str(source_expert)
                    source_key = '.'.join(source_parts)
                    
                    # 如果源权重存在，则使用源权重
                    if source_key in pkg_state_dict:
                        processed_state_dict[key] = pkg_state_dict[source_key]
                        print(f'** MODEL ** Copy expert weight: {source_key} -> {key}')
                        continue
    # 其他权重正常处理
    processed_state_dict[key] = value

# 检查差异
unexpected_state_dict = [k for k in processed_state_dict.keys() if k not in model_dict.keys()]
missing_state_dict = [k for k in model_dict.keys() if k not in processed_state_dict.keys()]
unmatchd_state_dict = [k for k, v in processed_state_dict.items() if k in model_dict.keys() and v.shape != model_dict[k].shape]

# 加载部分参数
state_dict = {k: v for k, v in processed_state_dict.items() if k in model_dict.keys() and v.shape == model_dict[k].shape}
model_dict.update(state_dict)
CTClip.load_state_dict(model_dict)

print('** MODEL ** The following parameters are UNEXPECTED in checkpoint:\n')
print(unexpected_state_dict)
print('** MODEL ** The following parameters are MISSING in checkpoint:\n')
print(missing_state_dict)
print('** MODEL ** The following parameters have DIFFERENT SHAPES in checkpoint:\n')
print(unmatchd_state_dict)
print('** MODEL ** The following parameters are LOADED in:\n')
print(state_dict.keys())
        
torch.save(CTClip.visual_transformer.state_dict(), '/mnt/petrelfs/zhangtengfei/RadIR/CT-CLIP/CT-UNI-Image-Retrieval-CXR_CT_change_layer_condition_moe_cross/ctclip_vit_moe_cross.pth')