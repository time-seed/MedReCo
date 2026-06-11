import os
import glob
import json
import torch
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from functools import partial
import torch.nn.functional as F
import nibabel as nib
from tqdm import tqdm
import random
import math
from monai.data.meta_tensor import MetaTensor
def resize_array_to_tensor(array, current_spacing, target_spacing):
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
    resized_array = F.interpolate(array, size=new_shape, mode='trilinear', align_corners=False)
    return resized_array

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

# modality_dict = {'3D-CT-Chest':0,'2D-CXR':1,'3D-Brain-MRI':2,'3D-CT-abdomen':3,'2D-Ultrasound':4}
modality_dict = {'3D-CT-Chest':0,'2D-CXR':0,'3D-Brain-MRI':0,'3D-CT-abdomen':0,'2D-Ultrasound':0}

class Conditional_CTReportDataset_Train(Dataset):
    def __init__(self, local_batch_size, jsonl_file, csv_file_dir, npy_file_dir, anatomy_filter, positive_threshold, negative_threshold, max_samples=30000,modality='CT', need_aug=False,modal_embedding=False,is_train=True):
        self.current_anatomy_index = 0 
        self.anatomy_filter = anatomy_filter
        self.is_train = is_train
        self.modal_embedding = modal_embedding
        self.modality = modality
        self.csv_file_dir = csv_file_dir
        self.npy_file_dir = npy_file_dir
        self.max_samples = max_samples
        self.modality = modality
        self.local_batch_size = local_batch_size    # for each anatomy, how many samples to be selected
        
        if positive_threshold == 0 and negative_threshold != 1: # >0 & <0.4
            positive_threshold = negative_threshold
        if positive_threshold != 1 and negative_threshold == 1: # >0.7 & <1
            negative_threshold = positive_threshold
            
        self.positive_threshold = positive_threshold
        self.negative_threshold = negative_threshold
        
        self.id2image_path = self.prepare_image_paths(jsonl_file)
        
        self.prepare_anatomy_data()
        
        self.anatomy_ls = list(self.anatomy2id_ls.keys())
        # 这里是直接使用
        # self.anatomy_weight = [round(math.sqrt(len(self.anatomy2id_ls[anatomy]))) for anatomy in self.anatomy_ls]   # weight 与 sqrt(size) 相关，还是balance
        self.anatomy_weight = [round(math.sqrt(len(self.anatomy2id_ls[anatomy]))) for anatomy in self.anatomy_ls]   # weight 与 sqrt(size) 相关，还是balance
        
        self.anatomy_weight = [npw/sum(self.anatomy_weight) for npw in self.anatomy_weight] # 手动归一化
        # self.anatomy_weight = [w/sum(self.anatomy_weight) for w in self.anatomy_weight]  # NOTE: Balancing between anatomys
        # self.anatomy_weight = [1/len(self.anatomy_ls) for anatomy in self.anatomy_ls]
        

        if int(os.environ.get("RANK", 0)) == 0:
            print(f'** DATA ** Load {len(self.id2image_path)} images.')
            print(f'** DATA ** Load {len(self.anatomy2id_ls)} anatomy.')

    def prepare_image_paths(self, jsonl_file):
        id2image_path = {}
        
        with open(jsonl_file, 'r') as f:
            data = f.readlines()
        data = [json.loads(l) for l in data]
        for d in data:
            # 这里有问题的
            data_id = d['name']   # valid_692_a_1.nii.gz
            id2image_path[data_id] = d['img_path']

        return id2image_path
    
    def prepare_anatomy_data(self):
        self.anatomy2id_ls = {} # 'lung': ['valid_692_a_1.nii.gz', ...]
        self.anatomy2simi_tab = {} # 'lung': a tensor with shape NxN
        self.anatomy2local_text = {}
        self.anatomy2lable = {}
        for csv_file in os.listdir(self.csv_file_dir):   
            anatomy_name = csv_file.replace('.csv', '') # lung.csv -> lung
            
            # DEBUG
            if anatomy_name not in self.anatomy_filter:
                continue
            # DEBUG
            
            self.anatomy2id_ls[anatomy_name] = []
            self.anatomy2local_text[anatomy_name] = []
            self.anatomy2lable[anatomy_name] = []
            df = pd.read_csv(os.path.join(self.csv_file_dir, csv_file))
            print(f"{csv_file} has {len(df)} rows.")
            

            indices = df.index.tolist()
            for _, row in df.iterrows():
                self.anatomy2id_ls[anatomy_name].append(row['File Path'])
                if 'Findings' not in row:
                    raise ValueError(f"'Findings' column missing in {csv_file}")
                self.anatomy2local_text[anatomy_name].append(row['Findings'])
                self.anatomy2lable[anatomy_name].append(float(row['Label']))  # 01格式的结果
            npy_file = csv_file.replace('.csv', '.npy')
            full_simi_tab = np.load(os.path.join(self.npy_file_dir, npy_file))
            # 截取原csv对应的大小
            assert len(df) == full_simi_tab.shape[0] and len(df) == full_simi_tab.shape[1], f"Error in {npy_file}: csv length {len(df)} vs npy shape {full_simi_tab.shape}"
            full_simi_tab = np.clip(full_simi_tab, 0, 1)
            # 索引一定是与确保与csv中样本一一对应
            # sampled_simi_tab = full_simi_tab[np.ix_(indices, indices)] # 例如只保留[0, 3, 5]行/列
            # simi_tab_scaled = np.round(sampled_simi_tab * 100).astype(np.uint8)
            # self.anatomy2simi_tab[anatomy_name] = simi_tab_scaled
            full_simi_tab = np.round(full_simi_tab * 100).astype(np.uint8)
            self.anatomy2simi_tab[anatomy_name] = full_simi_tab
            
            
        self.check_integrity()

    def check_integrity(self):
        id_keys = set(self.anatomy2id_ls.keys())
        simi_keys = set(self.anatomy2simi_tab.keys())
        text_keys = set(self.anatomy2local_text.keys())
        if id_keys != simi_keys or id_keys != text_keys:
            missing_in_simi = id_keys - simi_keys
            missing_in_id = simi_keys - id_keys
            missing_in_text = id_keys - text_keys
            missing_id_in_text = text_keys - id_keys
            raise ValueError(f"Inconsistent keys detected. Missing in similarity table: {missing_in_simi}. Missing in anatomy data: {missing_in_id}. Missing in text data: {missing_in_text}. Missing anatomy in text: {missing_id_in_text}.")

        for anatomy in id_keys:
            if not (len(self.anatomy2id_ls[anatomy]) == self.anatomy2simi_tab[anatomy].shape[0] and 
                len(self.anatomy2id_ls[anatomy]) == self.anatomy2simi_tab[anatomy].shape[1] and
                len(self.anatomy2id_ls[anatomy]) == len(self.anatomy2local_text[anatomy])):
                raise ValueError(f"Length of anatomy data {len(self.anatomy2id_ls[anatomy])}, similarity table {self.anatomy2simi_tab[anatomy].shape}, and local text {len(self.anatomy2local_text[anatomy])} do not match for {anatomy}.")
        
            # Here we check if all samples are readable
            filtered_sample_id_ls = []
            filtered_local_text_ls = []
            valid_indices = []
            for i, sample_id in enumerate(self.anatomy2id_ls[anatomy]):
                # image_file_name = sample_id.replace('.nii.gz', '.npz')
                if sample_id in self.id2image_path and os.path.exists(self.id2image_path[sample_id]):
                    filtered_sample_id_ls.append(sample_id)
                    filtered_local_text_ls.append(self.anatomy2local_text[anatomy][i])
                    valid_indices.append(i)
                    continue
                else:
                    print(f'{sample_id} is missing, removed')
                    # # Remove the invalid sample from self.anatomy2id_ls and update simi_tab accordingly
                    # self.anatomy2simi_tab[anatomy] = np.delete(self.anatomy2simi_tab[anatomy], i, axis=0)
                    # self.anatomy2simi_tab[anatomy] = np.delete(self.anatomy2simi_tab[anatomy], i, axis=1)
            self.anatomy2simi_tab[anatomy] = self.anatomy2simi_tab[anatomy][np.ix_(valid_indices, valid_indices)]  # 保留有效索引对应的子矩阵
            self.anatomy2id_ls[anatomy] = filtered_sample_id_ls
            self.anatomy2local_text[anatomy] = filtered_local_text_ls
            
            assert len(self.anatomy2id_ls[anatomy]) == self.anatomy2simi_tab[anatomy].shape[0] and len(self.anatomy2id_ls[anatomy]) == self.anatomy2simi_tab[anatomy].shape[1] and len(self.anatomy2id_ls[anatomy]) == len(self.anatomy2local_text[anatomy])
            
    def __len__(self):
        return 100000
    
    def MLS_nii_img_to_tensor(self, path):

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
        # img_data = (img_data - self.mean) / (self.std + 1e-8)
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

        tensor = torch.nn.functional.pad(tensor, (pad_d_before, pad_d_after, pad_w_before, pad_w_after, pad_h_before, pad_h_after), value=0)

        tensor = tensor.permute(2, 0, 1)    # d h w

        tensor = tensor.unsqueeze(0)    # 1 d h w

        return tensor

    def nii_img_to_tensor(self, path):

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

        tensor = torch.nn.functional.pad(tensor, (pad_d_before, pad_d_after, pad_w_before, pad_w_after, pad_h_before, pad_h_after), value=0)

        tensor = tensor.permute(2, 0, 1)    # d h w

        tensor = tensor.unsqueeze(0)    # 1 d h w

        return tensor
    
    def load_2d_image_to_tensor(self, image_path,resize=True):
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
            img_array = np.array(image, dtype=np.float32) / 255.0
            # img_array = (np.array(image) - self.mean) / (self.std + 1e-8)
            
            
            # Convert to tensor
            tensor = torch.from_numpy(img_array)
            
            # Add batch and channel dimensions
            tensor = tensor.unsqueeze(0).unsqueeze(0)  # [1, 1, 480, 480]
            
            return tensor

    def load_BrainMRI_image_to_tensor(self, image_path):
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
    def __getitem__(self, index):
        """
        Returns:
            video_tensor (tensor): N, D, H, W
            input_text (List of str): N
        """
        anatomy = np.random.choice(self.anatomy_ls, size=1, p=self.anatomy_weight)[0]
        similarity = self.anatomy2simi_tab[anatomy]
        num_samples = similarity.shape[0]
        
        # ==================== 修改开始：限制最大采样数 ====================
        MAX_SUBSET_SIZE = 3000
        total_samples = num_samples
        if total_samples > MAX_SUBSET_SIZE:
            subset_indices = np.sort(np.random.choice(total_samples, MAX_SUBSET_SIZE, replace=False))
        else:
            subset_indices = np.arange(total_samples)
            
        current_num_samples = len(subset_indices)

        # Get upper-triangular indices (相对于subset)
        rows_sub, cols_sub = np.triu_indices(current_num_samples, k=1)
        
        # 映射回全局索引
        rows = subset_indices[rows_sub]
        cols = subset_indices[cols_sub]
        # ==================== 修改结束 ====================
        
        # ==================== 新采样策略开始 ====================
        # 1. 分配采样数量：label=1 占 2/3，label=0 占 1/3
        label1_count = (self.local_batch_size * 2) // 3
        label0_count = self.local_batch_size - label1_count
        
        # 2. 获取label=1和label=0的索引
        all_labels = self.anatomy2lable[anatomy]
        label1_indices = [i for i in subset_indices if all_labels[i] == 1]
        label0_indices = [i for i in subset_indices if all_labels[i] == 0]
        
        # 3. 对label=1样本使用原有的正负样本对策略
        sampled_indexes_label1 = []
        if len(label1_indices) > 0:
            # 创建label1的索引映射 (全局索引 -> 相对索引)
            label1_global_to_local = {global_idx: local_idx for local_idx, global_idx in enumerate(label1_indices)}
            
            # 过滤出label=1样本之间的配对
            label1_mask = np.isin(rows, label1_indices) & np.isin(cols, label1_indices)
            label1_rows = rows[label1_mask]
            label1_cols = cols[label1_mask]
            label1_similarities = similarity[label1_rows, label1_cols]
            
            # 正负样本对筛选
            pos_mask_label1 = (label1_similarities > self.positive_threshold) & (label1_rows != label1_cols)
            neg_mask_label1 = (label1_similarities < self.negative_threshold) & (label1_rows != label1_cols)
            pos_candidate_pairs = list(zip(label1_rows[pos_mask_label1], label1_cols[pos_mask_label1]))
            neg_candidate_pairs = list(zip(label1_rows[neg_mask_label1], label1_cols[neg_mask_label1]))
            
            # 采样正负样本对
            pos_pair_count = neg_pair_count = label1_count // 4
            if label1_count % 4 > 1:
                pos_pair_count += 1
            
            if len(pos_candidate_pairs) >= pos_pair_count:
                selected_pos = random.sample(pos_candidate_pairs, pos_pair_count)
            else:
                # 如果正样本对不够，从所有label=1配对中采样
                all_label1_pairs = list(zip(label1_rows, label1_cols))
                selected_pos = random.sample(all_label1_pairs, min(pos_pair_count, len(all_label1_pairs)))
            
            if len(neg_candidate_pairs) >= neg_pair_count:
                selected_neg = random.sample(neg_candidate_pairs, neg_pair_count)
            else:
                # 如果负样本对不够，从所有label=1配对中采样
                all_label1_pairs = list(zip(label1_rows, label1_cols))
                selected_neg = random.sample(all_label1_pairs, min(neg_pair_count, len(all_label1_pairs)))
            
            sampled_indexes_label1 = [idx for pair in (selected_pos + selected_neg) for idx in pair]
            
            # 如果label1_count是奇数，补充一个样本
            if label1_count % 2 == 1:
                remaining = list(set(label1_indices) - set(sampled_indexes_label1))
                extra = random.choice(remaining) if remaining else random.choice(label1_indices)
                sampled_indexes_label1.append(extra)
            
            # 确保采样数量正确
            if len(sampled_indexes_label1) > label1_count:
                sampled_indexes_label1 = random.sample(sampled_indexes_label1, label1_count)
            elif len(sampled_indexes_label1) < label1_count:
                # 补齐不足的样本
                additional_needed = label1_count - len(sampled_indexes_label1)
                remaining = list(set(label1_indices) - set(sampled_indexes_label1))
                if len(remaining) >= additional_needed:
                    sampled_indexes_label1.extend(random.sample(remaining, additional_needed))
                else:
                    # 如果仍然不够，允许重复采样
                    sampled_indexes_label1.extend(random.choices(label1_indices, k=additional_needed))
        
        # 4. 对label=0样本直接随机采样
        sampled_indexes_label0 = []
        if len(label0_indices) >= label0_count:
            sampled_indexes_label0 = random.sample(label0_indices, label0_count)
        else:
            # 如果label=0样本不够，允许重复采样
            sampled_indexes_label0 = random.choices(label0_indices, k=label0_count) if label0_indices else []
        
        # 5. 合并两部分的采样索引
        sampled_indexes = sampled_indexes_label1 + sampled_indexes_label0
        
        # 6. 如果总数不足，从subset_indices中补齐
        if len(sampled_indexes) < self.local_batch_size:
            remaining = list(set(subset_indices) - set(sampled_indexes))
            additional_needed = self.local_batch_size - len(sampled_indexes)
            if len(remaining) >= additional_needed:
                sampled_indexes.extend(random.sample(remaining, additional_needed))
            else:
                sampled_indexes.extend(random.choices(list(subset_indices), k=additional_needed))
        
        # 确保最终数量正确
        sampled_indexes = sampled_indexes[:self.local_batch_size]
        # ==================== 新采样策略结束 ====================
        
        assert len(sampled_indexes) == self.local_batch_size
        
        # 获取对应的数据
        sampled_ids = [self.anatomy2id_ls[anatomy][idx] for idx in sampled_indexes]
        sampled_local_texts = [self.anatomy2local_text[anatomy][idx] for idx in sampled_indexes]
        sampled_label = [self.anatomy2lable[anatomy][idx] for idx in sampled_indexes]
        
        similarity_tab = np.zeros(
            (self.local_batch_size, self.local_batch_size), 
            dtype=np.uint8
        )
                
        # 修改赋值逻辑
        for local_i, global_i in enumerate(sampled_indexes):
            for local_j, global_j in enumerate(sampled_indexes):
                raw_value = max(
                    self.anatomy2simi_tab[anatomy][global_i, global_j],
                    self.anatomy2simi_tab[anatomy][global_j, global_i]
                )
                similarity_tab[local_i, local_j] = raw_value
        
        video_tensors = []

        if  ('3D-CT-Chest' == self.modality) or ('3D-CT-abdomen' == self.modality):
            for sampled_id in sampled_ids:  # 逐个处理拖慢了执行速度
                try:
                    # 返回的 video_tensor 已经是 [1, D, H, W]
                    if self.is_train:
                        video_tensor = self.nii_img_to_tensor(self.id2image_path[sampled_id])
                    else:
                        video_tensor = self.MLS_nii_img_to_tensor(self.id2image_path[sampled_id])
                    assert video_tensor.dim() == 4, f"Expected 4D tensor, got {video_tensor.dim()}D tensor for {sampled_id}"
                    
                    video_tensors.append(video_tensor)

                except Exception as e:
                    print(f"Error processing {sampled_id}: {e}. Skipping.")
                    # 如果一个样本失败，为了保持批次大小，可以加载一个全零张量或重复上一个
                    if video_tensors:
                        video_tensors.append(torch.zeros_like(video_tensors[-1]))
                    else: # 如果第一个就失败了
                        video_tensors.append(torch.zeros((1, 240, 480, 480), dtype=torch.float32))
            
            # 此时 video_tensors 是一个列表，包含 N 个 [1, D, H, W] 的张量
            stacked_video_tensor = torch.cat(video_tensors, dim=0) # -> [N, D, H, W]
            

            stacked_video_tensor = stacked_video_tensor.unsqueeze(1) # -> [N, 1, D, H, W]
            
        elif ('2D-CXR' == self.modality) or ('2D-Ultrasound' == self.modality):
            for sampled_id in sampled_ids:
                try:
                    # 返回的 video_tensor 已经是 [1, 1, H, W]
                    video_tensor = self.load_2d_image_to_tensor(self.id2image_path[sampled_id],resize=True)
                    video_tensor = video_tensor.squeeze().unsqueeze(0)  # 确保是 [1, H, W]
                    
                    video_tensor = video_tensor.unsqueeze(0)  # 变成 [1, 1, H, W]
                    video_tensors.append(video_tensor)
                except Exception as e:
                    print(f"Error processing {sampled_id}: {e}. Skipping.")
                    # 如果一个样本失败，为了保持批次大小，可以加载一个全零张量或重复上一个
                    if video_tensors:
                        video_tensors.append(torch.zeros_like(video_tensors[-1]))
                    else:
                        video_tensors.append(torch.zeros((1, 1, 480, 480), dtype=torch.float32))
            # 此时 video_tensors 是一个列表，包含 N 个 [1, 1, H, W] 的张量
            stacked_video_tensor = torch.cat(video_tensors, dim=0) # -> [N, 1, H, W]
            
            # 确保最终形状符合预期
            stacked_video_tensor = stacked_video_tensor.unsqueeze(1) # -> [N, 1, 1, H, W]
        elif '3D-Brain-MRI' == self.modality:
            for sampled_id in sampled_ids:
                try:
                    # 返回的 video_tensor 已经是 [1, D, H, W]
                    video_tensor = self.load_BrainMRI_image_to_tensor(self.id2image_path[sampled_id])
                    assert video_tensor.dim() == 4, f"Expected 4D tensor, got {video_tensor.dim()}D tensor for {sampled_id}"
                    video_tensors.append(video_tensor)

                except Exception as e:
                    print(f"Error processing {sampled_id}: {e}. Skipping.")
                    # 如果一个样本失败，为了保持批次大小，可以加载一个全零张量或重复上一个
                    video_tensors.append(torch.zeros((1, 30, 480, 480), dtype=torch.float32))
            
            # 此时 video_tensors 是一个列表，包含 N 个 [1, D, H, W] 的张量
            stacked_video_tensor = torch.cat(video_tensors, dim=0) # -> [N, D, H, W]
            

            stacked_video_tensor = stacked_video_tensor.unsqueeze(1) # -> [N, 1, D, H, W]
        else:
            raise ValueError(f"Unsupported modality: {self.modality}")
        if isinstance(stacked_video_tensor, MetaTensor):
            stacked_video_tensor = stacked_video_tensor.as_tensor()
        if self.modal_embedding:
            return {'video_tensor': stacked_video_tensor, 'similarity_tab': torch.tensor(similarity_tab), 'anatomy': anatomy, 'sampled_ids':sampled_ids, 'modality':modality_dict[self.modality],'local_text': sampled_local_texts,'label':sampled_label}   # 返回01标签的数据
        else:
            return {'video_tensor': stacked_video_tensor, 'similarity_tab': torch.tensor(similarity_tab), 'anatomy': anatomy, 'sampled_ids':sampled_ids}

    
    # def __getitem__(self, index):
    #     """
    #     Returns:
    #         video_tensor (tensor): N, D, H, W
    #         input_text (List of str): N
    #     """
    #     # anatomy = self.anatomy_ls[self.current_anatomy_index]
    
    #     # # 更新anatomy索引，循环使用，循环的遍历每一个condition
    #     # self.current_anatomy_index = (self.current_anatomy_index + 1) % len(self.anatomy_ls)
        
    #     anatomy = np.random.choice(self.anatomy_ls, size=1, p=self.anatomy_weight)[0]
    #     similarity = self.anatomy2simi_tab[anatomy]
    #     num_samples = similarity.shape[0]
        
    #     # ==================== 修改开始 ====================
    #     # 设定最大采样行/列数，例如 3000
    #     MAX_SUBSET_SIZE = 3000
    #     total_samples = num_samples
    #     if total_samples > MAX_SUBSET_SIZE:
    #         # 如果总数超过限制，随机抽取 3000 个 全局索引 (global indices)
    #         # 使用 replace=False 保证不重复，np.sort 保持顺序（虽然对随机性无影响，但符合索引直觉）
    #         subset_indices = np.sort(np.random.choice(total_samples, MAX_SUBSET_SIZE, replace=False))
    #     else:
    #         # 如果没超过，就使用全部索引
    #         subset_indices = np.arange(total_samples)
            
    #     current_num_samples = len(subset_indices)

    #     # Get upper-triangular indices 
    #     # 注意：这里的 triu_indices 产生的是针对 subset (0 ~ 4999) 的相对索引
    #     rows_sub, cols_sub = np.triu_indices(current_num_samples, k=1)
        
    #     # 关键步骤：将 相对索引 映射回 全局索引
    #     # 这样 rows 和 cols 里存的依然是原始大矩阵中的真实坐标
    #     rows = subset_indices[rows_sub]
    #     cols = subset_indices[cols_sub]
    #     # ==================== 修改结束 ====================
        
    #     # Get upper-triangular indices
        
    #     # rows, cols = np.triu_indices(num_samples, k=1)    # 原始代码
    #     pos_mask = (similarity[rows, cols] > self.positive_threshold) & (rows != cols)
    #     neg_mask = (similarity[rows, cols] < self.negative_threshold) & (rows != cols)
    #     pos_candidate_pairs = list(zip(rows[pos_mask], cols[pos_mask]))
    #     neg_candidate_pairs = list(zip(rows[neg_mask], cols[neg_mask]))
        
    #     # Sampling
    #     pos_pair_count = neg_pair_count = self.local_batch_size // 4
    #     if self.local_batch_size % 4 > 1:
    #         pos_pair_count += 1
    #     if len(pos_candidate_pairs) >= pos_pair_count:
    #         selected_pos = random.sample(pos_candidate_pairs, pos_pair_count)
    #     else:
    #         selected_pos = random.sample(list(zip(rows, cols)), pos_pair_count)
    #     if len(neg_candidate_pairs) >= neg_pair_count:
    #         selected_neg = random.sample(neg_candidate_pairs, neg_pair_count)
    #     else:
    #         selected_neg = random.sample(list(zip(rows, cols)), neg_pair_count)
    #     sampled_indexes = [idx for pair in (selected_pos + selected_neg) for idx in pair]
        
    #     # If local_batch_size is odd, add one more sample
    #     if self.local_batch_size % 2 == 1:
    #         remaining = list(set(range(num_samples)) - set(sampled_indexes))
    #         extra = random.choice(remaining) if remaining else random.choice(list(range(num_samples)))
    #         sampled_indexes.append(extra)
        
    #     assert len(sampled_indexes) == self.local_batch_size
    #     # 不太一样的地方，这里是按照sampled_indexes来获取sample_id以及local_text
    #     sampled_ids = [self.anatomy2id_ls[anatomy][idx] for idx in sampled_indexes]
    #     # sampled_ids = [sample_id for sample_id in sampled_ids if sample_id in self.id2image_path]   # WARNING 假如有一个sample_id的image不存在，那么会导致stacked_video_tensor后面的元素整体向前移动一位，和similarity_tab对不上
    #     # 获取对应的local text
    #     sampled_local_texts = [self.anatomy2local_text[anatomy][idx] for idx in sampled_indexes]
    #     sampled_label = [self.anatomy2lable[anatomy][idx] for idx in sampled_indexes]
    #     similarity_tab = np.zeros(
    #         (self.local_batch_size, self.local_batch_size), 
    #         dtype=np.uint8  # 使用 uint8 类型
    #     )
                
    #     # 修改赋值逻辑（需要反向缩放）
    #     for local_i, global_i in enumerate(sampled_indexes):
    #         for local_j, global_j in enumerate(sampled_indexes):
    #             raw_value = max(
    #                 self.anatomy2simi_tab[anatomy][global_i, global_j],
    #                 self.anatomy2simi_tab[anatomy][global_j, global_i]
    #             )
    #             # 直接使用存储的 uint8 值（无需转换）
    #             similarity_tab[local_i, local_j] = raw_value  # 已经是 0~100 的整数
    #     # # 修改
    #     video_tensors = []

    #     if  ('3D-CT-Chest' == self.modality) or ('3D-CT-abdomen' == self.modality):
    #         for sampled_id in sampled_ids:  # 逐个处理拖慢了执行速度
    #             try:
    #                 # 返回的 video_tensor 已经是 [1, D, H, W]
    #                 if self.is_train:
    #                     video_tensor = self.nii_img_to_tensor(self.id2image_path[sampled_id])
    #                 else:
    #                     video_tensor = self.MLS_nii_img_to_tensor(self.id2image_path[sampled_id])
    #                 assert video_tensor.dim() == 4, f"Expected 4D tensor, got {video_tensor.dim()}D tensor for {sampled_id}"
                    
    #                 video_tensors.append(video_tensor)

    #             except Exception as e:
    #                 print(f"Error processing {sampled_id}: {e}. Skipping.")
    #                 # 如果一个样本失败，为了保持批次大小，可以加载一个全零张量或重复上一个
    #                 if video_tensors:
    #                     video_tensors.append(torch.zeros_like(video_tensors[-1]))
    #                 else: # 如果第一个就失败了
    #                     video_tensors.append(torch.zeros((1, 240, 480, 480), dtype=torch.float32))
            
    #         # 此时 video_tensors 是一个列表，包含 N 个 [1, D, H, W] 的张量
    #         stacked_video_tensor = torch.cat(video_tensors, dim=0) # -> [N, D, H, W]
            

    #         stacked_video_tensor = stacked_video_tensor.unsqueeze(1) # -> [N, 1, D, H, W]
            
    #     elif ('2D-CXR' == self.modality) or ('2D-Ultrasound' == self.modality):
    #         for sampled_id in sampled_ids:
    #             try:
    #                 # 返回的 video_tensor 已经是 [1, 1, H, W]
    #                 video_tensor = self.load_2d_image_to_tensor(self.id2image_path[sampled_id],resize=True)
    #                 video_tensor = video_tensor.squeeze().unsqueeze(0)  # 确保是 [1, H, W]
                    
    #                 video_tensor = video_tensor.unsqueeze(0)  # 变成 [1, 1, H, W]
    #                 video_tensors.append(video_tensor)
    #             except Exception as e:
    #                 print(f"Error processing {sampled_id}: {e}. Skipping.")
    #                 # 如果一个样本失败，为了保持批次大小，可以加载一个全零张量或重复上一个
    #                 if video_tensors:
    #                     video_tensors.append(torch.zeros_like(video_tensors[-1]))
    #                 else:
    #                     video_tensors.append(torch.zeros((1, 1, 480, 480), dtype=torch.float32))
    #         # 此时 video_tensors 是一个列表，包含 N 个 [1, 1, H, W] 的张量
    #         stacked_video_tensor = torch.cat(video_tensors, dim=0) # -> [N, 1, H, W]
            
    #         # 确保最终形状符合预期
    #         stacked_video_tensor = stacked_video_tensor.unsqueeze(1) # -> [N, 1, 1, H, W]
    #     elif '3D-Brain-MRI' == self.modality:
    #         for sampled_id in sampled_ids:
    #             try:
    #                 # 返回的 video_tensor 已经是 [1, D, H, W]
    #                 video_tensor = self.load_BrainMRI_image_to_tensor(self.id2image_path[sampled_id])
    #                 assert video_tensor.dim() == 4, f"Expected 4D tensor, got {video_tensor.dim()}D tensor for {sampled_id}"
    #                 video_tensors.append(video_tensor)

    #             except Exception as e:
    #                 print(f"Error processing {sampled_id}: {e}. Skipping.")
    #                 # 如果一个样本失败，为了保持批次大小，可以加载一个全零张量或重复上一个
    #                 video_tensors.append(torch.zeros((1, 30, 480, 480), dtype=torch.float32))
            
    #         # 此时 video_tensors 是一个列表，包含 N 个 [1, D, H, W] 的张量
    #         stacked_video_tensor = torch.cat(video_tensors, dim=0) # -> [N, D, H, W]
            

    #         stacked_video_tensor = stacked_video_tensor.unsqueeze(1) # -> [N, 1, D, H, W]
    #     else:
    #         raise ValueError(f"Unsupported modality: {self.modality}")
    #     if isinstance(stacked_video_tensor, MetaTensor):
    #         stacked_video_tensor = stacked_video_tensor.as_tensor()
    #     if self.modal_embedding:
    #         return {'video_tensor': stacked_video_tensor, 'similarity_tab': torch.tensor(similarity_tab), 'anatomy': anatomy, 'sampled_ids':sampled_ids, 'modality':modality_dict[self.modality],'local_text': sampled_local_texts,'label':sampled_label}   # 返回01标签的数据
    #     else:
    #         return {'video_tensor': stacked_video_tensor, 'similarity_tab': torch.tensor(similarity_tab), 'anatomy': anatomy, 'sampled_ids':sampled_ids}

def collate_fn(batch):
    # batch 是一个列表，其中每个元素是 __getitem__ 返回的字典
    
    # 使用列表推导式和 torch.stack 高效地组合批次
    videos = torch.stack([item['video_tensor'] for item in batch], dim=0)
    similarity_tabs = torch.stack([item['similarity_tab'] for item in batch], dim=0)
    
    # 其他非张量数据正常收集
    anatomies = [item['anatomy'] for item in batch] # B大小的列表，其中包含了对应的condition部分
    sampled_ids_list = [item['sampled_ids'] for item in batch]
    local_texts_list = [item['local_text'] for item in batch]  # 二维列表: [[batch1_text1, batch1_text2, ...], [batch2_text1, batch2_text2, ...], ...]
    local_texts_list = [item for sublist in local_texts_list for item in sublist]
    
    if 'modality' in batch[0]:
        modalities = torch.tensor([item['modality'] for item in batch])
        labels = torch.tensor([item['label'] for item in batch])
        return videos, similarity_tabs, anatomies, sampled_ids_list, modalities,local_texts_list, labels
    else:
        return videos, similarity_tabs, anatomies, sampled_ids_list

if __name__ == '__main__':
    from torch.utils.data import Dataset, DataLoader, random_split
    import time
    from tqdm import tqdm
    
    # 创建CXR数据集
    print('** 创建CXR数据集 **')
    cxr_dataset = Conditional_CTReportDataset_Train(
        local_batch_size=10,
        modality='2D-CXR',
        jsonl_file='/mnt/petrelfs/zhangtengfei/RadIR/dataset/CXR/MIMIC_CXR/all.jsonl',
        csv_file_dir='/mnt/petrelfs/zhangtengfei/public_dataset/extend_test/CXR/test_extend_text_deleted_exper',
        npy_file_dir='/mnt/petrelfs/zhangtengfei/public_dataset/extend_test/Biolord/CXR/test_extend_npy',
        positive_threshold=0.75,
        negative_threshold=1,
        anatomy_filter=['Abnormal arterial course in Aorta'],
        need_aug=True,
        modal_embedding=True
    )

    # 测试CXR数据集
    dl_cxr = DataLoader(
        cxr_dataset,
        num_workers=4,
        batch_size=1,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
        collate_fn=collate_fn
    )
    # 读取部分数据，检验dataset代码正确性
    for i, data in enumerate(tqdm(dl_cxr)):
        videos, similarity_tabs, anatomies, sampled_ids_list, modalities, local_texts_list, labels = data
        print(f'videos shape: {videos.shape}')  # Expected: (B, N, 1, H, W)
        print(f'similarity_tabs shape: {similarity_tabs.shape}')  # Expected: (B, N, N)
        print(f'anatomies: {anatomies}')  # List of anatomy names
        print(f'sampled_ids_list: {sampled_ids_list}')  # List of lists of sampled IDs
        print(f'modalities: {modalities}')  # Tensor of modalities
        print(f'local_texts_list length: {len(local_texts_list)}')  # Flattened list of local texts
        print(f'labels: {labels}')  # Tensor of labels
        if i == 2:
            break