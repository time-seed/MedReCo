# import torch
# import numpy as np
# from PIL import Image

# import torch.nn.functional as F
# import nibabel as nib
# import random

# def resize_array(array, current_spacing, target_spacing):
#     """
#     Resize the array to match the target spacing.

#     Args:
#     array (torch.Tensor): Input array to be resized.
#     current_spacing (tuple): Current voxel spacing (z_spacing, xy_spacing, xy_spacing).
#     target_spacing (tuple): Target voxel spacing (target_z_spacing, target_x_spacing, target_y_spacing).

#     Returns:
#     np.ndarray: Resized array.
#     """
#     # Calculate new dimensions
#     original_shape = array.shape[2:]
#     scaling_factors = [
#         current_spacing[i] / target_spacing[i] for i in range(len(original_shape))
#     ]
#     new_shape = [
#         int(original_shape[i] * scaling_factors[i]) for i in range(len(original_shape))
#     ]
#     # Resize the array
#     resized_array = F.interpolate(array, size=new_shape, mode='trilinear', align_corners=False).cpu().numpy()
#     return resized_array

# def MLS_nii_img_to_tensor(path):

#     nii_img = nib.load(str(path))
#     img_data = nii_img.get_fdata()
#     img_data = np.flip(img_data, axis=0)
#     img_data = np.flip(img_data, axis=1)

#     # WARNING Respacing
#     img_data = img_data.transpose(2, 0, 1)
#     img_data = np.copy(img_data)
#     tensor = torch.tensor(img_data)
#     tensor = tensor.unsqueeze(0).unsqueeze(0)
    
#     target_x_spacing = 0.75
#     target_y_spacing = 0.75
#     target_z_spacing = 1.5
#     current = (3, 1, 1)   # this is all set to 3 1 1
#     target = (target_z_spacing, target_x_spacing, target_y_spacing)
    
#     img_data = resize_array(tensor, current, target)
#     img_data = img_data[0][0]
#     img_data= np.transpose(img_data, (1, 2, 0))

#     # WARNING Normalization 2
#     hu_min, hu_max = -1000, 1000
#     img_data = np.clip(img_data, hu_min, hu_max)
#     # img_data = (((img_data ) / 1000)).astype(np.float32)
#     img_data = (img_data - (-534.953680)) / (483.647344 + 1e-8)
#     tensor = torch.tensor(img_data)
    
#     # WARNING Padding or Crop
    
#     # Get the dimensions of the input tensor
#     target_shape = (480,480,240)    # h w d
    
#     # Extract dimensions
#     h, w, d = tensor.shape
    
#     # Calculate cropping/padding values for height, width, and depth
#     dh, dw, dd = target_shape
#     h_start = max((h - dh) // 2, 0)
#     h_end = min(h_start + dh, h)
#     w_start = max((w - dw) // 2, 0)
#     w_end = min(w_start + dw, w)
#     d_start = max((d - dd) // 2, 0)
#     d_end = min(d_start + dd, d)

#     # Crop or pad the tensor
#     tensor = tensor[h_start:h_end, w_start:w_end, d_start:d_end]

#     pad_h_before = (dh - tensor.size(0)) // 2
#     pad_h_after = dh - tensor.size(0) - pad_h_before

#     pad_w_before = (dw - tensor.size(1)) // 2
#     pad_w_after = dw - tensor.size(1) - pad_w_before

#     pad_d_before = (dd - tensor.size(2)) // 2
#     pad_d_after = dd - tensor.size(2) - pad_d_before

#     tensor = torch.nn.functional.pad(tensor, (pad_d_before, pad_d_after, pad_w_before, pad_w_after, pad_h_before, pad_h_after), value=-1)

#     tensor = tensor.permute(2, 0, 1)    # d h w

#     tensor = tensor.unsqueeze(0)    # 1 d h w

#     return tensor

# def load_2d_image_to_tensor(image_path,resize=True):
#     """
#     Load a 2D grayscale image and convert it to a tensor with dimensions [1, 1, 480, 480].
    
#     Args:
#         image_path (str): Path to the image file.
        
#     Returns:
#         tensor (torch.Tensor): Tensor with shape [1, 1, 480, 480].
#     """
#     # Load the image as grayscale
#     image = Image.open(image_path).convert('L')
    
#     # Resize to 480x480 if needed
#     if resize:
#         if image.size != (480, 480):
#             image = image.resize((480, 480), Image.BILINEAR)
#     else:
#         if image.size != (480, 480):
#             # 计算可以开始裁剪的最大左上角坐标
#             width, height = image.size
#             crop_size = 480
#             if width < crop_size or height < crop_size:
#                 raise ValueError(f"Image size {image.size} is smaller than the crop size {crop_size}.")
#             max_left = width - crop_size
#             max_top = height - crop_size
            
#             # 随机选择裁剪的左上角坐标
#             left = random.randint(0, max(0, max_left))
#             top = random.randint(0, max(0, max_top))
            
#             # 裁剪图像
#             image = image.crop((left, top, left + crop_size, top + crop_size))
        

#     # Convert to numpy array and normalize to [0, 1]
#     img_array = (np.array(image) - 0.474719) / (0.303027 + 1e-8)
    
#     # img_array = (np.array(image, dtype=np.float32) / 255.0) * 2 -1
    
#     # Convert to tensor
#     tensor = torch.from_numpy(img_array)
    
#     # Add batch and channel dimensions
#     tensor = tensor.unsqueeze(0).unsqueeze(0)  # [1, 1, 480, 480]
    
#     return tensor

# def load_BrainMRI_image_to_tensor(image_path):
#         nii_img = nib.load(str(image_path))
#         img_data = nii_img.get_fdata()

#         img_data = (img_data - 1366.556376) / (4309.143205 + 1e-8)

#         img_data = img_data.transpose(2, 0, 1)
#         img_data = np.copy(img_data)
#         tensor = torch.tensor(img_data)
#         tensor = tensor.unsqueeze(0)
#         return tensor
    
# def get_image_info_vit(image_path):
#     if '6th_mri' in image_path:
#         try:
#             image_tensor = load_BrainMRI_image_to_tensor(image_path)
#             assert image_tensor.dim() == 4, f"Expected 4D tensor, got {image_tensor.dim()}D tensor for {index}"
#         except Exception as e:
#             print(f"Error processing {image_path} at index: {e}. Skipping this sample.")
#             image_tensor = torch.zeros((1, 30, 480, 480), dtype=torch.float32)
#     if '.nii.gz' not in image_path:
#         try:
#             image_tensor = load_2d_image_to_tensor(image_path, resize=True)
#             image_tensor = image_tensor.squeeze().unsqueeze(0)
            
#             image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension
        
#         except Exception as e:
#             print(f"Error processing {image_path} at index: {e}. Skipping this sample.")
#             image_tensor = torch.zeros((1, 1, 480, 480), dtype=torch.float32)
#     else:
#         try:

#             image_tensor = MLS_nii_img_to_tensor(image_path)
#             assert image_tensor.dim() == 4, f"Expected 4D tensor, got {image_tensor.dim()}D tensor for {index}"

#         except Exception as e:
#             print(f"Error processing {image_path} at index: {e}. Skipping this sample.")
#             image_tensor = torch.zeros((1, 240, 480, 480), dtype=torch.float32)
#     return image_tensor

import torch
import numpy as np
from PIL import Image

import torch.nn.functional as F
import nibabel as nib
import random

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

def MLS_nii_img_to_tensor(path):

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
    # img_data = (img_data - (-534.953680)) / (483.647344 + 1e-8)
    img_data = (img_data - hu_min) / (hu_max - hu_min) # 归一化到0-1之间    
    
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

def load_2d_image_to_tensor(image_path,resize=True):
    """
    Load a 2D grayscale image and convert it to a tensor with dimensions [1, 1, 480, 480].
    
    Args:
        image_path (str): Path to the image file.
        
    Returns:
        tensor (torch.Tensor): Tensor with shape [1, 1, 480, 480].
    """
    # Load the image as grayscale
    image = Image.open(image_path).convert('L')
    
    # Resize to 480x480 if needed
    if resize:
        if image.size != (480, 480):
            image = image.resize((480, 480), Image.BILINEAR)
    else:
        if image.size != (480, 480):
            # 计算可以开始裁剪的最大左上角坐标
            width, height = image.size
            crop_size = 480
            if width < crop_size or height < crop_size:
                raise ValueError(f"Image size {image.size} is smaller than the crop size {crop_size}.")
            max_left = width - crop_size
            max_top = height - crop_size
            
            # 随机选择裁剪的左上角坐标
            left = random.randint(0, max(0, max_left))
            top = random.randint(0, max(0, max_top))
            
            # 裁剪图像
            image = image.crop((left, top, left + crop_size, top + crop_size))
        

    # Convert to numpy array and normalize to [0, 1]
    # img_array = (np.array(image) - 0.474719) / (0.303027 + 1e-8)
    img_array = np.array(image, dtype=np.float32) / 255.0
    
    # Convert to tensor
    tensor = torch.from_numpy(img_array)
    
    # Add batch and channel dimensions
    tensor = tensor.unsqueeze(0).unsqueeze(0)  # [1, 1, 480, 480]
    
    return tensor

def load_BrainMRI_image_to_tensor(image_path):
    nii_img = nib.load(str(image_path))
    img_data = nii_img.get_fdata()

    image = img_data.astype(np.float32)

    # 排除背景0值来计算分位数，防止背景拉低统计值
    # 如果图像全是背景（极端情况），直接返回全0
    if np.max(image) == 0:
        return image
    non_zero_vals = image[image > 0]
    if len(non_zero_vals) == 0:
        return image
    # 计算 1% 和 99% 分位数
    p1 = np.percentile(non_zero_vals, 1)
    p99 = np.percentile(non_zero_vals, 99)
    # 1. Clip 消除极值影响
    image = np.clip(image, p1, p99)
    # 2. Normalize to 0-1
    # 防止分母为0
    if p99 - p1 == 0:
        return np.zeros_like(image)
    image = (image - p1) / (p99 - p1)
    img_data = image
    # img_data = (img_data - self.mean) / (self.std + 1e-8)

    img_data = img_data.transpose(2, 0, 1)
    img_data = np.copy(img_data)
    tensor = torch.tensor(img_data)
    tensor = tensor.unsqueeze(0)
    return tensor
    
def get_image_info_vit(image_path):
    if '6th_mri' in image_path:
        try:
            image_tensor = load_BrainMRI_image_to_tensor(image_path)
            assert image_tensor.dim() == 4, f"Expected 4D tensor, got {image_tensor.dim()}D tensor for {index}"
        except Exception as e:
            print(f"Error processing {image_path} at index: {e}. Skipping this sample.")
            image_tensor = torch.zeros((1, 30, 480, 480), dtype=torch.float32)
    elif '.nii.gz' not in image_path:
        try:
            image_tensor = load_2d_image_to_tensor(image_path, resize=True)
            image_tensor = image_tensor.squeeze().unsqueeze(0)
            
            image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension
        
        except Exception as e:
            print(f"Error processing {image_path} at index: {e}. Skipping this sample.")
            image_tensor = torch.zeros((1, 1, 480, 480), dtype=torch.float32)
    else:
        try:

            image_tensor = MLS_nii_img_to_tensor(image_path)
            assert image_tensor.dim() == 4, f"Expected 4D tensor, got {image_tensor.dim()}D tensor for {index}"

        except Exception as e:
            print(f"Error processing {image_path} at index: {e}. Skipping this sample.")
            image_tensor = torch.zeros((1, 240, 480, 480), dtype=torch.float32)
    return image_tensor