import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

modality_dict = {
    '3D-CT':   0,
    '2D-CXR':        1,
    '3D-Brain-MRI':  2,
    '2D-Ultrasound': 4,
}


def resize_array(array, current_spacing, target_spacing):
    """按 voxel spacing 对 3D 体数据做三线性重采样。"""
    original_shape = array.shape[2:]
    scaling_factors = [current_spacing[i] / target_spacing[i] for i in range(len(original_shape))]
    new_shape = [int(original_shape[i] * scaling_factors[i]) for i in range(len(original_shape))]
    resized = F.interpolate(array, size=new_shape, mode='trilinear', align_corners=False).cpu().numpy()
    return resized


# --------------------------------------------------------------------------
# 3D CT (chest )
# --------------------------------------------------------------------------
def _load_ct(path):
    import nibabel as nib  # 仅 3D 模态需要，延迟导入

    nii_img = nib.load(str(path))
    img_data = nii_img.get_fdata()
    img_data = np.flip(img_data, axis=0)
    img_data = np.flip(img_data, axis=1)

    # Respacing
    img_data = img_data.transpose(2, 0, 1)
    img_data = np.copy(img_data)
    tensor = torch.tensor(img_data).unsqueeze(0).unsqueeze(0)

    target = (1.5, 0.75, 0.75)   # (z, x, y)
    current = (3, 1, 1)
    img_data = resize_array(tensor, current, target)
    img_data = img_data[0][0]
    img_data = np.transpose(img_data, (1, 2, 0))

    # Normalization (HU -> [0,1])
    hu_min, hu_max = -1000, 1000
    img_data = np.clip(img_data, hu_min, hu_max)
    img_data = (img_data - hu_min) / (hu_max - hu_min)
    tensor = torch.tensor(img_data)

    # Crop / Pad -> (480, 480, 240)  (h, w, d)
    target_shape = (480, 480, 240)
    h, w, d = tensor.shape
    dh, dw, dd = target_shape
    h_start = max((h - dh) // 2, 0); h_end = min(h_start + dh, h)
    w_start = max((w - dw) // 2, 0); w_end = min(w_start + dw, w)
    d_start = max((d - dd) // 2, 0); d_end = min(d_start + dd, d)
    tensor = tensor[h_start:h_end, w_start:w_end, d_start:d_end]

    pad_h_before = (dh - tensor.size(0)) // 2; pad_h_after = dh - tensor.size(0) - pad_h_before
    pad_w_before = (dw - tensor.size(1)) // 2; pad_w_after = dw - tensor.size(1) - pad_w_before
    pad_d_before = (dd - tensor.size(2)) // 2; pad_d_after = dd - tensor.size(2) - pad_d_before
    tensor = torch.nn.functional.pad(
        tensor,
        (pad_d_before, pad_d_after, pad_w_before, pad_w_after, pad_h_before, pad_h_after),
        value=-1,
    )

    tensor = tensor.permute(2, 0, 1)   # d h w
    tensor = tensor.unsqueeze(0)       # 1 d h w
    return tensor.float()


# --------------------------------------------------------------------------
# 2D (CXR / Ultrasound)
# --------------------------------------------------------------------------
def _load_2d(path, resize=True):
    image = Image.open(path).convert('L')
    if resize and image.size != (480, 480):
        image = image.resize((480, 480), Image.BILINEAR)

    img_array = np.array(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(img_array)
    tensor = tensor.unsqueeze(0).unsqueeze(0)   # (1, 1, 480, 480)
    return tensor.float()


# --------------------------------------------------------------------------
# 3D Brain MRI
# --------------------------------------------------------------------------
def _load_brain_mri(path):
    import nibabel as nib  # 仅 3D 模态需要，延迟导入

    nii_img = nib.load(str(path))
    image = nii_img.get_fdata().astype(np.float32)

    if np.max(image) == 0:
        out = image
    else:
        non_zero_vals = image[image > 0]
        if len(non_zero_vals) == 0:
            out = image
        else:
            p1 = np.percentile(non_zero_vals, 1)
            p99 = np.percentile(non_zero_vals, 99)
            image = np.clip(image, p1, p99)
            out = np.zeros_like(image) if (p99 - p1) == 0 else (image - p1) / (p99 - p1)

    img_data = out.transpose(2, 0, 1)
    img_data = np.copy(img_data)
    tensor = torch.tensor(img_data).unsqueeze(0)   # (1, D, H, W)
    return tensor.float()


# --------------------------------------------------------------------------
# 统一入口
# --------------------------------------------------------------------------
def load_image_tensor(path, modality):
    """
    根据模态把一张原始影像读成 (1, D, H, W) 的 float tensor。
    与数据集 __getitem__ 返回的 video_tensor 完全一致。
    """
    if modality in ('3D-CT'):
        return _load_ct(path)
    elif modality in ('2D-CXR', '2D-Ultrasound'):
        return _load_2d(path)
    elif modality == '3D-Brain-MRI':
        return _load_brain_mri(path)
    else:
        raise ValueError(f"未知模态: {modality}. 可选: {list(modality_dict.keys())}")