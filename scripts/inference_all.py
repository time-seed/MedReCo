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
mean = -534.953680
std = 483.647344

def resize_array(array, current_spacing, target_spacing):
    """
    Resize the array to match the target spacing.

    Args:
    array (torch.Tensor): Input array to be resized.
    current_spacing (tuple): Current voxel spacing (z_spacing, xy_spacing, xy_spacing).
    target_spacing (tuple): Target voxel spacing (target_z_spacing, target_x_spacing, target_y_spacing).

    Returns:
    np.ndarray: Resized array.
    """
    # Calculate new dimensions
    original_shape = array.shape[2:]
    scaling_factors = [
        current_spacing[i] / target_spacing[i] for i in range(len(original_shape))
    ]
    new_shape = [
        int(original_shape[i] * scaling_factors[i]) for i in range(len(original_shape))
    ]
    # Resize the array
    resized_array = F.interpolate(array, size=new_shape, mode='trilinear', align_corners=False).cpu().numpy()
    return resized_array


def MLS_nii_img_to_tensor( path):

        nii_img = nib.load(str(path))
        img_data = nii_img.get_fdata()
        img_data = np.flip(img_data, axis=0)
        img_data = np.flip(img_data, axis=1)

        # WARNING Respacing
        img_data = img_data.transpose(2, 0, 1)
        img_data = np.copy(img_data)
        tensor = torch.tensor(img_data)
        tensor = tensor.unsqueeze(0).unsqueeze(0)
        
        target_x_spacing = 0.75
        target_y_spacing = 0.75
        target_z_spacing = 1.5
        current = (3, 1, 1)   # this is all set to 3 1 1
        target = (target_z_spacing, target_x_spacing, target_y_spacing)
        
        img_data = resize_array(tensor, current, target)
        img_data = img_data[0][0]
        img_data= np.transpose(img_data, (1, 2, 0))

        # WARNING Normalization 2
        hu_min, hu_max = -1000, 1000
        img_data = np.clip(img_data, hu_min, hu_max)
        # img_data = (((img_data ) / 1000)).astype(np.float32)
        img_data = (img_data - mean) / (std + 1e-8)
        

        tensor = torch.tensor(img_data)
        
        # WARNING Padding or Crop
        
        # Get the dimensions of the input tensor
        target_shape = (480,480,240)    # h w d
        
        # Extract dimensions
        h, w, d = tensor.shape
        
        # Calculate cropping/padding values for height, width, and depth
        dh, dw, dd = target_shape
        h_start = max((h - dh) // 2, 0)
        h_end = min(h_start + dh, h)
        w_start = max((w - dw) // 2, 0)
        w_end = min(w_start + dw, w)
        d_start = max((d - dd) // 2, 0)
        d_end = min(d_start + dd, d)

        # Crop or pad the tensor
        tensor = tensor[h_start:h_end, w_start:w_end, d_start:d_end]

        pad_h_before = (dh - tensor.size(0)) // 2
        pad_h_after = dh - tensor.size(0) - pad_h_before

        pad_w_before = (dw - tensor.size(1)) // 2
        pad_w_after = dw - tensor.size(1) - pad_w_before

        pad_d_before = (dd - tensor.size(2)) // 2
        pad_d_after = dd - tensor.size(2) - pad_d_before

        tensor = torch.nn.functional.pad(tensor, (pad_d_before, pad_d_after, pad_w_before, pad_w_after, pad_h_before, pad_h_after), value=-1)

        tensor = tensor.permute(2, 0, 1)    # d h w

        tensor = tensor.unsqueeze(0)    # 1 d h w

        return tensor


def process_jsonl_to_embeddings(
    jsonl_path,
    output_img_path,
    output_text_path,
    model_path,
    batch_size=4,
    device=None
):
    """
    Process JSONL file and generate image and text embeddings.
    
    Args:
        jsonl_path (str): Path to the JSONL file
        output_img_path (str): Path to save image embeddings (.pt file)
        output_text_path (str): Path to save text embeddings (.pt file)
        model_path (str): Path to the pretrained CT-CLIP model
        batch_size (int): Batch size for processing
        device (str or torch.device): Device to use for computation
    """
    
    # Setup device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize models
    print("Loading models...")
    tokenizer = BertTokenizer.from_pretrained('microsoft/BiomedVLP-CXR-BERT-specialized', do_lower_case=True)
    text_encoder = BertModel.from_pretrained("microsoft/BiomedVLP-CXR-BERT-specialized").to(device)
    text_encoder.resize_token_embeddings(len(tokenizer))
    
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
    clip.eval()
    
    # Read JSONL file
    print("Reading JSONL file...")
    data = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    
    total_samples = len(data)
    print(f"Total samples: {total_samples}")
    
    # Initialize lists to store embeddings
    all_img_embeds = []
    all_text_embeds = []
    
    # Process in batches
    num_batches = (total_samples + batch_size - 1) // batch_size
    
    with torch.no_grad():
        for batch_idx in tqdm(range(num_batches), desc="Processing batches"):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, total_samples)
            batch_data = data[start_idx:end_idx]
            
            # Prepare batch
            batch_images = []
            batch_texts = []
            
            for item in batch_data:
                try:
                    # Load image
                    img_tensor = MLS_nii_img_to_tensor(item['img_path'])
                    batch_images.append(img_tensor)
                    
                    # Get text
                    batch_texts.append(item['text'])
                    
                except Exception as e:
                    print(f"\nError processing {item.get('img_path', 'unknown')}: {e}")
                    # Use dummy data for failed samples
                    batch_images.append(torch.zeros(1, 240, 480, 480))
                    batch_texts.append("")
            
            # Stack images
            image_batch = torch.stack(batch_images, dim=0).to(device)  # [batch, 1, 240, 480, 480]
            
            # Tokenize texts
            text_tokens = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=512
            ).to(device)
            
            # Get embeddings
            current_batch_size = image_batch.size(0)
            zero_vector = torch.zeros(current_batch_size, dtype=torch.long, device=device)
            text_latents, image_latents, _ ,_= clip(
                text_tokens,
                image_batch,
                return_latents=True,
                device=device,
                modal_embedding=True,
                is_condition=False,
                modal_indexs=zero_vector
            )
            
            # Store embeddings
            all_text_embeds.append(text_latents.cpu())
            all_img_embeds.append(image_latents.cpu())
            
            # Clear cache
            del image_batch, text_tokens, text_latents, image_latents
            torch.cuda.empty_cache()
    
    # Concatenate all embeddings
    print("Concatenating embeddings...")
    img_embeds = torch.cat(all_img_embeds, dim=0)  # [n, 512]
    text_embeds = torch.cat(all_text_embeds, dim=0)  # [n, 512]
    
    print(f"Image embeddings shape: {img_embeds.shape}")
    print(f"Text embeddings shape: {text_embeds.shape}")
    
    # Save embeddings
    print(f"Saving image embeddings to {output_img_path}...")
    torch.save(img_embeds, output_img_path)
    
    print(f"Saving text embeddings to {output_text_path}...")
    torch.save(text_embeds, output_text_path)
    
    print("Done!")
    
    return img_embeds, text_embeds


if __name__ == "__main__":
    # Configuration
    JSONL_PATH = "/mnt/petrelfs/zhangtengfei/RadIR/dataset/CT_RATE/test_no_split.jsonl"  # TODO: 修改为你的jsonl文件路径
    OUTPUT_IMG_EMBEDS = "/mnt/petrelfs/zhangtengfei/RadIR/baseline/stage1/img_embeds.pt"  # 输出图像embeddings的路径
    OUTPUT_TEXT_EMBEDS = "/mnt/petrelfs/zhangtengfei/RadIR/baseline/stage1/text_embeds.pt"  # 输出文本embeddings的路径
    MODEL_PATH = "/mnt/petrelfs/zhangtengfei/RadIR/CT-CLIP/log/CXR_CT_Uni_CXRCT_1028_moe_expert_train_/CTClip.1600.pt"  # 模型路径
    BATCH_SIZE = 6  # TODO: 根据你的GPU内存调整batch size
    
    # Process
    img_embeds, text_embeds = process_jsonl_to_embeddings(
        jsonl_path=JSONL_PATH,
        output_img_path=OUTPUT_IMG_EMBEDS,
        output_text_path=OUTPUT_TEXT_EMBEDS,
        model_path=MODEL_PATH,
        batch_size=BATCH_SIZE
    )