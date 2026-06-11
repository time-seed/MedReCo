import torch
import json
from pathlib import Path
from tqdm import tqdm
import numpy as np
from transformer_maskgit import CTViT
from transformers import BertTokenizer, BertModel
from ct_clip import CTCLIP
import torch.nn.functional as F
import nibabel as nib
from PIL import Image
import random
from transformers import AutoTokenizer, AutoModel

# Setup device
model_path = '/mnt/petrelfs/zhangtengfei/RadIR_log/CT-CLIP_v9/aug_log_extend_test/331_3modality_train_1e-4_with_text_US_ori_infonce/CTClip.4800.pt'
# model_path =  "/mnt/petrelfs/zhangtengfei/RadIR_log/CT-CLIP_v7/aug_log_extend_test/304_3modality_train_1e-4_with_text_US/CTClip.1200.pt"
# model_path =  "/mnt/petrelfs/zhangtengfei/RadIR_log/CT-CLIP/new_log/1124_3modality_train_1e-4_with_text/CTClip.2400_condition_embeddings.pt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize models
print("Loading models...")
tokenizer = AutoTokenizer.from_pretrained('FremyCompany/BioLORD-2023')
text_encoder = AutoModel.from_pretrained('FremyCompany/BioLORD-2023')
text_encoder.resize_token_embeddings(len(tokenizer))

image_encoder = CTViT(
        dim = 512,
        codebook_size = 8192,
        image_size = 480,
        patch_size = 20,
        temporal_patch_size = 10,
        spatial_depth = 8,
        temporal_depth = 4,
        dim_head = 32,
        heads = 8
    ).to(device)

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
    ).to(device)

# clip.load(model_path)
pkg = torch.load(model_path, map_location='cpu', weights_only=False)
model_dict =clip.state_dict()
if 'model' in pkg:
    pkg_state_dict = pkg['model']  # 假设 pkg['model'] 是模型的状态字典
else:
    pkg_state_dict = pkg
# 检查差异
unexpected_state_dict = [k for k in pkg_state_dict.keys() if k not in model_dict.keys()]
missing_state_dict = [k for k in model_dict.keys() if k not in pkg_state_dict.keys()]
unmatchd_state_dict = [k for k, v in pkg_state_dict.items() if k in model_dict.keys() and v.shape != model_dict[k].shape]

# 加载部分参数
state_dict = {k: v for k, v in pkg_state_dict.items() if k in model_dict.keys() and v.shape == model_dict[k].shape}
model_dict.update(state_dict)
clip.load_state_dict(model_dict)

print('** MODEL ** The following parameters are UNEXPECTED in checkpoint:\n')
print(unexpected_state_dict)
print('** MODEL ** The following parameters are MISSING in checkpoint:\n')
print(missing_state_dict)
print('** MODEL ** The following parameters have DIFFERENT SHAPES in checkpoint:\n')
print(unmatchd_state_dict)
print('** MODEL ** The following parameters are LOADED in:\n')
print(state_dict.keys())

condition_embeddings = torch.load('/mnt/petrelfs/zhangtengfei/RadIR/CT-CLIP/CT-UNI-Image-Retrieval-CXR_CT_change_layer_condition_moe_cross_v9_abnormal_bool/scripts/all_condition_embeds.pt')
clip.condition_embeddings.weight.data = condition_embeddings
pkg = dict(model=clip.state_dict())
torch.save(pkg, '/mnt/petrelfs/zhangtengfei/RadIR_log/CT-CLIP_v9/aug_log_extend_test/331_3modality_train_1e-4_with_text_US_ori_infonce/CTClip.4800_condition_embeddings_.pt')

# torch.save(pkg, '/mnt/petrelfs/zhangtengfei/RadIR_log/CT-CLIP_v7/aug_log_extend_test/304_3modality_train_1e-4_with_text_US/CTClip.1200_condition_embeddings_.pt')