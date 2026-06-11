import torch
import json
from pathlib import Path
from tqdm import tqdm
import numpy as np
from transformer_maskgit import CTViT
from transformers import BertTokenizer, BertModel
from ct_clip import CTCLIP
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

def mean_pooling(model_output, attention_mask):
    """Mean pooling - take attention mask into account for correct averaging"""
    token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def process_condition_embeddings(
    condition_json_path,
    output_path,
    model_path,
    device=None
):
    """
    Process condition JSON file and generate text embeddings.
    
    Args:
        condition_json_path (str): Path to the all_condition_index.json file
        output_path (str): Path to save condition embeddings (.pt file)
        model_path (str): Path to the pretrained CT-CLIP model
        device (str or torch.device): Device to use for computation
    """
    
    # Setup device
    if device is None:
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
    
    clip = CTCLIP(
        image_encoder = image_encoder,
        text_encoder = text_encoder,
        tokenizer = tokenizer,
        dim_text = 768,
        dim_image = 512,
        dim_latent = 512,
        extra_latent_projection = False,
        use_mlm=False,
        downsample_image_embeds = False,
        use_all_token_embeds = False
    ).to(device)
    
    # Load model
    pkg = torch.load(model_path, map_location='cpu', weights_only=False)
    model_dict = clip.state_dict()
    if 'model' in pkg:
        pkg_state_dict = pkg['model']
    else:
        pkg_state_dict = pkg
    
    state_dict = {k: v for k, v in pkg_state_dict.items() if k in model_dict.keys() and v.shape == model_dict[k].shape}
    model_dict.update(state_dict)
    clip.load_state_dict(model_dict)
    clip.eval()
    
    # Load condition index JSON
    print(f"Loading condition index from {condition_json_path}...")
    with open(condition_json_path, 'r', encoding='utf-8') as f:
        condition_data = json.load(f)
    
    # Initialize embedding tensor [400, 512] with zeros
    max_index = 400
    embedding_dim = 512
    all_condition_embeds = torch.zeros(max_index, embedding_dim)
    
    # Process each modality
    # modalities = ['CXR', 'CT', 'Brain', 'US']
    modalities = [
    "MIMIC-CXR",
    "CT_RATE",
    "Brain-6th",
    "Ultrasound",]
    
    with torch.no_grad():
        for modality in modalities:
            if modality not in condition_data:
                print(f"Modality {modality} not found in JSON, skipping...")
                continue
            
            print(f"\nProcessing modality: {modality}")
            modality_data = condition_data[modality]
            
            # Process each condition in this modality
            for condition_text, index in tqdm(modality_data.items(), desc=f"{modality} conditions"):
                if index >= max_index:
                    print(f"Warning: Index {index} for '{condition_text}' exceeds max_index {max_index}, skipping...")
                    continue
                
                # Tokenize text
                text_tokens = tokenizer(
                    condition_text,
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=512
                ).to(device)
                
                # Get text embeddings
                condition_embeddings = clip.text_transformer(**text_tokens)
                condition_embeddings = mean_pooling(condition_embeddings, text_tokens['attention_mask'])
                
                # Normalize embeddings
                condition_embeddings = F.normalize(condition_embeddings, p=2, dim=1)
                
                # Project to latent space
                condition_latents = clip.to_text_latent(condition_embeddings)  # [1, 512]
                
                # Store in the corresponding index
                all_condition_embeds[index] = condition_latents.squeeze(0).cpu()
                
                # Clear cache
                del text_tokens, condition_embeddings, condition_latents
    
    print(f"\nCondition embeddings shape: {all_condition_embeds.shape}")
    print(f"Non-zero rows: {(all_condition_embeds.sum(dim=1) != 0).sum().item()}/{max_index}")
    
    # Save embeddings
    print(f"Saving condition embeddings to {output_path}...")
    torch.save(all_condition_embeds, output_path)
    
    print("Done!")
    
    return all_condition_embeds


if __name__ == "__main__":
    # Configuration
    # CONDITION_JSON_PATH = "/mnt/petrelfs/zhangtengfei/RadIR/CT-CLIP/CT-UNI-Image-Retrieval-CXR_CT_change_layer_condition_moe_cross_v7_abnormal_bool/scripts/all_condition_index.json"
    # OUTPUT_CONDITION_EMBEDS = "/mnt/petrelfs/zhangtengfei/RadIR/CT-CLIP/CT-UNI-Image-Retrieval-CXR_CT_change_layer_condition_moe_cross_v7_abnormal_bool/scripts/all_condition_embeds.pt"
    # # MODEL_PATH = "/mnt/petrelfs/zhangtengfei/RadIR_log/CT-CLIP_v7/aug_log_extend_test/303_3modality_train_1e-4_with_text_US/CTClip.1800.pt"
    # MODEL_PATH = "/mnt/petrelfs/zhangtengfei/RadIR_log/CT-CLIP_v7/aug_log_extend_test/304_3modality_train_1e-4_with_text_US/CTClip.1200.pt"
    
    # # Process
    # condition_embeds = process_condition_embeddings(
    #     condition_json_path=CONDITION_JSON_PATH,
    #     output_path=OUTPUT_CONDITION_EMBEDS,
    #     model_path=MODEL_PATH
    # )
    
    CONDITION_JSON_PATH = "/mnt/petrelfs/zhangtengfei/RadIR/CT-CLIP/CT-UNI-Image-Retrieval-CXR_CT_change_layer_condition_moe_cross_v9_abnormal_bool/scripts/all_condition_index.json"
    OUTPUT_CONDITION_EMBEDS = "/mnt/petrelfs/zhangtengfei/RadIR/CT-CLIP/CT-UNI-Image-Retrieval-CXR_CT_change_layer_condition_moe_cross_v9_abnormal_bool/scripts/all_condition_embeds_one_expert.pt"
    # MODEL_PATH = "/mnt/petrelfs/zhangtengfei/RadIR_log/CT-CLIP_v7/aug_log_extend_test/303_3modality_train_1e-4_with_text_US/CTClip.1800.pt"
    MODEL_PATH = "/mnt/petrelfs/zhangtengfei/RadIR_log/CT-CLIP_v9/aug_log_extend_test/331_3modality_train_1e-4_with_text_US_ori_infonce/CTClip.4800.pt"
    
    # Process
    condition_embeds = process_condition_embeddings(
        condition_json_path=CONDITION_JSON_PATH,
        output_path=OUTPUT_CONDITION_EMBEDS,
        model_path=MODEL_PATH
    )