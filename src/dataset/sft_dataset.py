import copy
import os
import random
from typing import Dict, Iterator, Optional
import torch
import torch.distributed as dist
import transformers
import ujson as json
from torch.utils.data import Dataset, DistributedSampler
import math
from src.params import DataArguments
import copy
from src.constants import (
    IGNORE_INDEX,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_VIDEO_TOKEN,
    SYSTEM_MESSAGE,
)

from .data_utils import get_image_info, get_video_info, llava_to_openai, pad_sequence
from .data_utils_vit import get_image_info_vit
from .shuffle_options import shuffle_options_and_update_answer_CT, shuffle_options_and_update_answer_CXR

import random
import copy

def shuffle_options_and_update_answer(sample):
    """
    随机重排选项并更新对应的答案
    """
    # 定义选项标签
    option_labels = ['A', 'B', 'C', 'D']
    

    # 提取当前的问题数据
    human_value = sample['conversations'][0]['value']
    gpt_value = sample['conversations'][1]['value']
    
    # 从human_value中提取选项内容（这里需要解析字符串）
    # 由于选项在字符串中，我们需要提取它们
    lines = human_value.split('\n')
    options_start = False
    current_options = {}
    
    for line in lines:
        line = line.strip()
        if '"A":' in line:
            current_options['A'] = line.split('"A":')[1].strip().strip('",')
        elif '"B":' in line:
            current_options['B'] = line.split('"B":')[1].strip().strip('",')
        elif '"C":' in line:
            current_options['C'] = line.split('"C":')[1].strip().strip('",')
        elif '"D":' in line:
            current_options['D'] = line.split('"D":')[1].strip().strip('",')
    
    # 从gpt_value中提取当前答案
    current_answer = None
    if '"answer": "A"' in gpt_value:
        current_answer = 'A'
    elif '"answer": "B"' in gpt_value:
        current_answer = 'B'
    elif '"answer": "C"' in gpt_value:
        current_answer = 'C'
    elif '"answer": "D"' in gpt_value:
        current_answer = 'D'
    
    if not current_answer or len(current_options) != 4:
        return  sample # 跳过无法解析的样本
    
    # 获取正确答案的内容
    correct_option_content = current_options[current_answer]
    
    # 创建选项内容列表
    option_contents = [current_options[label] for label in option_labels]
    
    # 随机打乱选项
    shuffled_contents = option_contents.copy()
    random.shuffle(shuffled_contents)
    
    # 创建新的选项映射
    new_options = {}
    for i, label in enumerate(option_labels):
        new_options[label] = shuffled_contents[i]
    
    # 找到正确答案的新位置
    new_answer = None
    for label, content in new_options.items():
        if content == correct_option_content:
            new_answer = label
            break
        
    # 重构human_value
    # 提取question和condition
    question_line = [line for line in lines if '"question":' in line]
    condition_line = [line for line in lines if '"condition":' in line]
    
    if question_line and condition_line:
        question_content = question_line[0].split('"question":')[1].strip().strip('",')
        condition_content = condition_line[0].split('"condition":')[1].strip().strip('",')
        
        # 重构human_value
        new_human_value = f"""<image>\n<image>\n{question_content}","A": "{new_options['A']}","B": "{new_options['B']}","C": "{new_options['C']}","D": "{new_options['D']}".Your response must strictly follow the format below: "answer": "A/B/C/D", "reason": "Brief explanation of the basis for your judgment"."""
        
        # 提取reason内容
        reason_content = gpt_value.split('"reason":')[1].strip().strip('".').strip('"')
        
        # 重构gpt_value
        new_gpt_value = f""""answer": "{new_answer}", "reason": "{reason_content}"."""
        
        # 更新样本
        sample['conversations'][0]['value'] = new_human_value
        sample['conversations'][1]['value'] = new_gpt_value

class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        data_path: str | list,
        processor: transformers.ProcessorMixin,
        data_args: DataArguments,
        model_id,
        padding=True,
    ):
        super(SupervisedDataset, self).__init__()
        if isinstance(data_path, str):
            list_data_dict = json.load(open(data_path, "r"))
        else:
            list_data_dict = data_path

        self.model_id = model_id
        self.processor = processor
        self.list_data_dict = list_data_dict
        self.data_args = data_args
        self.padding = padding
        self.image_min_pixel = data_args.image_min_pixels
        self.image_max_pixel = data_args.image_max_pixels
        self.video_min_pixel = data_args.video_min_pixels
        self.video_max_pixel = data_args.video_max_pixels
        self.image_resized_w = data_args.image_resized_width
        self.image_resized_h = data_args.image_resized_height
        self.video_resized_w = data_args.video_resized_width
        self.video_resized_h = data_args.video_resized_height
        self.fps = data_args.fps
        self.nframes = data_args.nframes

    def __len__(self):
        return len(self.list_data_dict)
        # FIXME： 过拟合使用
        # return 50000

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        
        sources_ori = self.list_data_dict[i]
        sources = copy.deepcopy(sources_ori)
        shuffle_options_and_update_answer(sources)
        is_video = False
        # print(sources)
        processor = self.processor
        if "image" in sources:
            videos = None
            grid_key = "image_grid_thw"
            pixel_key = "pixel_values"

            image_files = sources["image"]
            image_folder = self.data_args.image_folder

            if isinstance(image_files, str):
                image_files = [image_files]

            images = []

            for image_file in image_files:
                if not os.path.exists(image_file):
                    if not image_file.startswith("http"):
                        image_file = os.path.join(image_folder, image_file)
                # 处理多张图像，逐张处理图像。是一个PIL的列表
                # TODO: 修改image的加载方式
                # 不需要传入这么多的数据
                images.append(get_image_info_vit(image_file))

        elif "video" in sources:
            is_video = True
            images=None
            grid_key = "video_grid_thw"
            pixel_key = "pixel_values_videos"

            video_files = sources["video"]
            video_folder = self.data_args.image_folder

            if isinstance(video_files, str):
                video_files = [video_files]

            videos = []
            for video_file in video_files:
                if not os.path.exists(video_file):
                    if not video_file.startswith("http"):
                        video_file = os.path.join(video_folder, video_file)
                video_input, video_kwargs = get_video_info(video_file, self.video_min_pixel, self.video_max_pixel, self.video_resized_w, self.video_resized_h, self.fps, self.nframes)
                videos.append(video_input)
        else:
            grid_key = None
            pixel_key = None
            images=None
            videos=None

        sources = copy.deepcopy(llava_to_openai(sources['conversations'], is_video=is_video))

        all_input_ids = []
        all_labels = []
        all_pixel_values = []
        all_image_grid_thw = []
        all_second_gird = []

        image_curr_count = 0
        video_curr_count = 0
        # Qwen2-VL uses a default system message so I've added this.
        if len(SYSTEM_MESSAGE) > 0:
            system_message = f"{DEFAULT_IM_START_TOKEN}system\n{SYSTEM_MESSAGE}{DEFAULT_IM_END_TOKEN}\n"
            system_message_input_ids = processor.tokenizer(system_message, add_special_tokens=False, return_tensors='pt')['input_ids']
            system_labels = torch.full_like(system_message_input_ids, IGNORE_INDEX)

            all_input_ids.append(system_message_input_ids.squeeze(0))
            all_labels.append(system_labels.squeeze(0))

        for _, j in enumerate(range(0, len(sources), 2)):
            user_input = sources[j]
            gpt_response = sources[j + 1]

            user_input = f"{DEFAULT_IM_START_TOKEN}{user_input['role']}\n{user_input['content']}{DEFAULT_IM_END_TOKEN}\n{DEFAULT_IM_START_TOKEN}{gpt_response['role']}\n"
            gpt_response = f"{gpt_response['content']}{DEFAULT_IM_END_TOKEN}\n"

            if DEFAULT_IMAGE_TOKEN in user_input:
                num_images = user_input.count(DEFAULT_IMAGE_TOKEN)  # 这一个批次用了多少的图片
                # Slice the images list to get the images for the current turn.
                images_for_this_turn = images[image_curr_count : image_curr_count + num_images]
                inputs = processor(text=[user_input], images=images_for_this_turn, videos=videos, padding=False, do_resize=False, return_tensors='pt')
                # 在这里使用processor处理了数据
                prompt_input_ids = inputs['input_ids']
                all_pixel_values.append(inputs[pixel_key]) # B 1 240 480 480大小的数据
                all_image_grid_thw.append(inputs[grid_key]) # B 3 多张图片导致的
                image_curr_count += num_images

            elif DEFAULT_VIDEO_TOKEN in user_input:
                num_videos = user_input.count(DEFAULT_VIDEO_TOKEN)
                # Slice the videos list to get the videos for the current turn.
                videos_for_this_turn = videos[video_curr_count : video_curr_count + num_videos]
                if "Qwen2.5" in self.model_id:
                    inputs = processor(text=[user_input], images=images, videos=videos_for_this_turn, padding=False, do_resize=False, return_tensors='pt', **video_kwargs)
                    all_second_gird.extend(inputs["second_per_grid_ts"])
                else:
                    inputs = processor(text=[user_input], images=images, videos=videos_for_this_turn, padding=False, do_resize=False, return_tensors='pt')
                prompt_input_ids = inputs['input_ids']
                all_pixel_values.append(inputs[pixel_key])
                all_image_grid_thw.append(inputs[grid_key])
                video_curr_count += num_videos

            else:
                prompt_input_ids = processor.tokenizer(user_input, add_special_tokens=False, padding=False, return_tensors='pt')['input_ids']

            response_input_ids = processor.tokenizer(gpt_response, add_special_tokens=False, padding=False, return_tensors='pt')['input_ids']

            input_ids = torch.cat([prompt_input_ids, response_input_ids], dim=1).squeeze(0)
            labels = torch.cat(
                [
                    torch.tensor([IGNORE_INDEX] * len(prompt_input_ids[0])),
                    response_input_ids.squeeze(0),
                ],
                dim=0,
            )

            all_input_ids.append(input_ids)
            all_labels.append(labels)

        # There is no need for eos or bos tokens in the input_ids
        # Qwen2-VL does not use them
        input_ids = torch.cat(all_input_ids, dim=0).to(torch.long)
        labels = torch.cat(all_labels, dim=0).to(torch.long)

        # eos_token_id = processor.tokenizer.convert_tokens_to_ids(DEFAULT_IM_END_TOKEN)
        # input_ids, labels = truncate_sequence(input_ids, labels, self.max_length, eos_token_id)

        attention_mask = (input_ids > -1000000).to(torch.long)

        data_dict = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        if pixel_key and grid_key:
            pixel_values = torch.cat(all_pixel_values, dim=0) # 合并后BB 1 240 480 480,BB是说B里面可能有小b多个图片的情况
            image_thw = torch.cat(all_image_grid_thw, dim=0)  # BB 3
            data_dict[pixel_key] = pixel_values
            data_dict[grid_key] = image_thw

        if len(all_second_gird) > 0:
            second_gird = all_second_gird
            data_dict["second_per_grid_ts"] = second_gird

        return data_dict

class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, examples):
        batch_input_ids = []
        batch_label_ids = []
        batch_pixel_values = []
        batch_pixel_video_values = []
        batch_video_thw = []
        batch_image_thw = []
        batch_second_per_grid_ts = []

        for example in examples:
            keys = example.keys()
            if "pixel_values_videos" in keys:
                batch_pixel_video_values.append(example["pixel_values_videos"])
                batch_video_thw.append(example["video_grid_thw"])
            elif "pixel_values" in keys:
                batch_pixel_values.append(example["pixel_values"])
                batch_image_thw.append(example["image_grid_thw"])

            batch_input_ids.append(example["input_ids"])
            batch_label_ids.append(example["labels"])

            if "second_per_grid_ts" in keys:
                batch_second_per_grid_ts.extend(example["second_per_grid_ts"])

        input_ids = pad_sequence(
            batch_input_ids, padding_side='right', padding_value=self.pad_token_id
        )

        attention_mask = input_ids != self.pad_token_id
        labels = pad_sequence(batch_label_ids, padding_side='right', padding_value=IGNORE_INDEX)

        data_dict = {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask,
        }

        if len(batch_pixel_values) > 0:
            pixel_values = torch.cat(batch_pixel_values, dim=0) # 在这里合并了数据
            image_thw = torch.cat(batch_image_thw, dim=0)
            data_dict["pixel_values"] = pixel_values
            data_dict["image_grid_thw"] = image_thw

        if len(batch_pixel_video_values) > 0:
            pixel_video_values = torch.cat(batch_pixel_video_values, dim=0)
            video_thw = torch.cat(batch_video_thw, dim=0)
            data_dict["pixel_values_videos"] = pixel_video_values
            data_dict["video_grid_thw"] = video_thw

        if len(batch_second_per_grid_ts) > 0:
            data_dict["second_per_grid_ts"] = batch_second_per_grid_ts
        # === 在这里添加调试代码 ===
        # print("--- Debugging inside Collator ---")
        # for key, value in data_dict.items():
        #     # 必须检查值是不是Tensor，因为像 second_per_grid_ts 可能是列表
        #     if isinstance(value, torch.Tensor):
        #         print(f"Tensor '{key}' is on device: {value.device}")
        #     else:
        #         print(f'key',{key}," is not tensor")
        # print("--------------------------")
        # ==========================
        return data_dict

MODALITY_KEYS = ["us", "ct", "mri", "cxr"]
MODALITY_MAP = {
    "US_data": "us",
    "CT-RATE": "ct",
    "6th_mri": "mri",
    "MIMIC-CXR": "cxr"
}

def _detect_modality(image_paths):
    """
    根据路径中的特定数据集关键字判断模态。
    匹配规则：US_data -> us, CT-RATE -> ct, 6th_mri -> mri, MIMIC-CXR -> cxr
    无匹配则归入 'cxr'。
    """
    for p in image_paths:
        # 建议这里根据实际路径情况决定是否使用 .lower()
        # 如果路径大小写不固定，建议统一转为大写匹配
        p_upper = p.upper() 
        
        for pattern, mod in MODALITY_MAP.items():
            # pattern.upper() 确保匹配不受配置大小写影响
            if pattern.upper() in p_upper:
                return mod
                
    return "cxr"  # 默认兜底


class ModalityAwareSampler(DistributedSampler):
    """
    模态感知分布式采样器：保证每个mini-batch内所有样本模态一致，
    且所有GPU在同一step处理同模态数据。

    支持4种模态：us(超声), ct, mri, cxr(X光)。
    通过文件路径小写后是否包含 'us','ct','mri','cxr' 判断模态。

    采样权重:
    - 若指定 modality_weights（dict），按权重比例决定每种模态的block数量。
    - 若未指定，按各模态原始数据量比例采样（等价于把所有数据都用上）。
    """

    def __init__(
        self,
        dataset,
        batch_size,
        num_replicas=None,
        rank=None,
        shuffle=True,
        seed=0,
        drop_last=False,
        modality_weights=None,
    ):
        super().__init__(
            dataset,
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
            seed=seed,
            drop_last=drop_last,
        )
        self.batch_size = batch_size
        self.modality_weights = modality_weights  # e.g. {"us": 1.0, "ct": 2.0, ...} or None

        # 扫描数据集，按模态分组索引
        self.modality_indices = {mod: [] for mod in MODALITY_KEYS}
        for i, sample in enumerate(dataset.list_data_dict):
            image_paths = sample.get("image", [])
            if isinstance(image_paths, str):
                image_paths = [image_paths]
            mod = _detect_modality(image_paths)
            self.modality_indices[mod].append(i)

        self.global_batch_size = self.batch_size * self.num_replicas

        # 打印各模态统计 & 小模态组warning
        for mod in MODALITY_KEYS:
            n = len(self.modality_indices[mod])
            if n > 0 and n < self.global_batch_size:
                print(f"WARNING: modality '{mod}' has {n} samples < global_batch_size "
                      f"({self.global_batch_size}), all '{mod}' data will be dropped!")
            if n > 0:
                print(f"  [ModalityAwareSampler] modality='{mod}': {n} samples, "
                      f"{n // self.global_batch_size} full blocks")

        # 计算各模态可用的最大block数
        self._max_blocks = {
            mod: len(self.modality_indices[mod]) // self.global_batch_size
            for mod in MODALITY_KEYS
        }

        # 计算实际使用的block数（考虑权重）
        self._effective_blocks = self._compute_effective_blocks()

        total_blocks = sum(self._effective_blocks.values())
        self.num_samples = total_blocks * self.batch_size
        self.total_size = total_blocks * self.global_batch_size

    def _compute_effective_blocks(self):
        """根据权重计算每个模态实际使用的block数。"""
        max_b = self._max_blocks
        active_mods = [m for m in MODALITY_KEYS if max_b[m] > 0]

        if not active_mods:
            return {m: 0 for m in MODALITY_KEYS}

        if self.modality_weights is None:
            # 无权重：每个模态使用全部可用block（按数据量自然比例）
            return dict(max_b)

        # 有权重：按权重比例分配block数，上限为各模态最大block数
        weights = {m: self.modality_weights.get(m, 0.0) for m in active_mods}
        w_sum = sum(weights.values())
        if w_sum == 0:
            return dict(max_b)

        # 归一化权重
        norm_w = {m: weights[m] / w_sum for m in active_mods}

        # 以最大总block数为基准，按权重分配，但不超过各模态可用block数
        # 策略：找到受限最紧的模态，以它为基准缩放
        # ratio[m] = max_b[m] / norm_w[m] 表示该模态能支撑的总block数
        ratios = []
        for m in active_mods:
            if norm_w[m] > 0:
                ratios.append(max_b[m] / norm_w[m])
        if not ratios:
            return dict(max_b)

        total_blocks = min(ratios)  # 受限最紧的模态决定总量

        effective = {m: 0 for m in MODALITY_KEYS}
        for m in active_mods:
            effective[m] = max(1, int(norm_w[m] * total_blocks)) if norm_w[m] > 0 else 0
            effective[m] = min(effective[m], max_b[m])

        print(f"  [ModalityAwareSampler] weighted block allocation: {effective}")
        return effective

    def __iter__(self) -> Iterator:
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        gbs = self.global_batch_size
        all_blocks = []

        for mod in MODALITY_KEYS:
            indices = self.modality_indices[mod]
            n_blocks = self._effective_blocks[mod]
            if n_blocks == 0:
                continue

            # shuffle该模态内部索引
            if self.shuffle:
                perm = torch.randperm(len(indices), generator=g).tolist()
                indices = [indices[i] for i in perm]
            else:
                indices = list(indices)

            # 切成 global_batch_size 大小的块，取前 n_blocks 个
            blocks = [indices[i:i + gbs] for i in range(0, n_blocks * gbs, gbs)]
            all_blocks.extend([(b, mod) for b in blocks])

        # 用相同种子 shuffle 块顺序（所有 GPU 得到相同顺序）
        if self.shuffle:
            block_perm = torch.randperm(len(all_blocks), generator=g).tolist()
            all_blocks = [all_blocks[i] for i in block_perm]

        # 每个 rank 取自己的切片
        rank_indices = []
        bs = self.batch_size
        for block, _ in all_blocks:
            start = self.rank * bs
            end = start + bs
            rank_indices.extend(block[start:end])

        return iter(rank_indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch

# class SynchronizedHalfDatasetSampler(DistributedSampler):
#     """
#     分布式环境下的半数据集采样器，确保所有进程在同一全局批次时
#     都一致地选择前半部分或后半部分数据
#     """
    
#     def __init__(
#         self, 
#         dataset, 
#         num_replicas=None,
#         rank=None,
#         shuffle=True,
#         seed=0,
#         drop_last=False,
#         batch_size=32  # 每个进程的批次大小
#     ):
#         """初始化同步的分布式半数据集采样器"""
#         super().__init__(
#             dataset, 
#             num_replicas=num_replicas,
#             rank=rank,
#             shuffle=shuffle,
#             seed=seed,
#             drop_last=drop_last
#         )
        
#         self.dataset_size = len(dataset)
#         self.half_size = self.dataset_size // 2
#         self.epoch = 0
#         self.batch_size = batch_size
#         self.generator = torch.Generator()
        
#         # 计算每个进程的样本数量
#         if self.drop_last:
#             self.num_samples = math.floor(self.dataset_size / self.num_replicas)
#         else:
#             self.num_samples = math.ceil(self.dataset_size / self.num_replicas)
            
#         # 计算每个进程的批次数
#         if self.drop_last:
#             self.num_batches = self.num_samples // self.batch_size
#         else:
#             self.num_batches = math.ceil(self.num_samples / self.batch_size)
        
#     def _generate_half_choices(self):
#         """为每个全局批次生成前半部分/后半部分的选择"""
#         # 使用固定种子确保所有进程得到相同的选择序列
#         choice_generator = torch.Generator()
#         choice_generator.manual_seed(self.seed + self.epoch)
        
#         choices = []
#         for batch_idx in range(self.num_batches):
#             # 0表示前半部分，1表示后半部分
#             choice = torch.randint(0, 2, (1,), generator=choice_generator).item()
#             choices.append(choice == 0)  # True表示前半部分
            
#         return choices
    
#     def _get_half_indices(self, use_first_half, batch_seed):
#         """获取指定半区的索引并打乱"""
#         if use_first_half:
#             indices = list(range(0, self.half_size))
#         else:
#             indices = list(range(self.half_size, self.dataset_size))
        
#         if self.shuffle:
#             # 使用批次特定的种子打乱该半区的索引
#             shuffle_generator = torch.Generator()
#             shuffle_generator.manual_seed(batch_seed)
#             perm = torch.randperm(len(indices), generator=shuffle_generator)
#             indices = [indices[i] for i in perm]
            
#         return indices
    
#     def __iter__(self):
#         # 生成每个批次的半区选择
#         batch_half_choices = self._generate_half_choices()
        
#         # 为当前进程生成所有批次的索引
#         all_indices = []
        
#         for batch_idx in range(self.num_batches):
#             use_first_half = batch_half_choices[batch_idx]
            
#             # 为当前批次生成种子（所有进程相同）
#             batch_seed = self.seed + self.epoch * 10000 + batch_idx
            
#             # 获取当前半区的所有索引（打乱后）
#             half_indices = self._get_half_indices(use_first_half, batch_seed)
            
#             # 计算当前批次需要多少样本
#             start_idx = batch_idx * self.batch_size
#             end_idx = min(start_idx + self.batch_size, self.num_samples)
#             current_batch_size = end_idx - start_idx
            
#             if current_batch_size <= 0:
#                 break
                
#             # 为当前进程选择样本
#             # 使用rank和batch_idx确保不同进程选择不同的样本
#             rank_offset = self.rank * current_batch_size
            
#             # 确保有足够的样本可选
#             if rank_offset + current_batch_size <= len(half_indices):
#                 batch_indices = half_indices[rank_offset:rank_offset + current_batch_size]
#             else:
#                 # 如果样本不够，进行循环采样
#                 batch_indices = []
#                 for i in range(current_batch_size):
#                     idx = (rank_offset + i) % len(half_indices)
#                     batch_indices.append(half_indices[idx])
            
#             all_indices.extend(batch_indices)
        
#         return iter(all_indices)
    
#     def __len__(self):
#         """返回该进程应该获取的样本数"""
#         return self.num_samples
    
#     def set_epoch(self, epoch):
#         """设置epoch，用于确保每个epoch的随机性"""
#         self.epoch = epoch

# class SynchronizedHalfDatasetSampler(DistributedSampler):
#     """
#     分布式环境下的半数据集采样器，确保所有进程在同一批次时
#     都一致地选择前半部分或后半部分数据
#     """
    
#     def __init__(
#         self, 
#         dataset, 
#         num_replicas=None,
#         rank=None,
#         shuffle=True,
#         seed=0,
#         drop_last=False,
#         batch_size=1  # 需要知道批次大小来确保批次内同步
#     ):
#         """初始化同步的分布式半数据集采样器"""
#         super().__init__(
#             dataset, 
#             num_replicas=num_replicas,
#             rank=rank,
#             shuffle=shuffle,
#             seed=seed,
#             drop_last=drop_last
#         )
        
#         self.half_size = len(dataset) // 2
#         self.epoch = 0
#         self.batch_size = batch_size
#         self.generator = torch.Generator()
        
#     def __iter__(self):
#         # 设置随机种子，确保所有进程使用相同的随机序列
#         self.generator.manual_seed(self.seed + self.epoch)
        
#         # 计算总批次数（向上取整）
#         # 这确保所有进程计算出相同的总批次数
#         total_size = len(self.dataset)
#         num_samples_per_replica = (total_size + self.num_replicas - 1) // self.num_replicas
#         total_batch_num = (num_samples_per_replica + self.batch_size - 1) // self.batch_size
        
#         # 预先为每个批次决定使用前半部分还是后半部分
#         # 所有进程使用相同的种子和算法，因此会得到相同的决策序列
#         batch_half_choices = []
#         for i in range(total_batch_num):
#             # 使用批次索引作为额外的随机种子源
#             local_seed = self.seed + self.epoch * 10000 + i
#             local_generator = torch.Generator()
#             local_generator.manual_seed(local_seed)
            
#             # 确定此批次使用前半部分还是后半部分
#             use_first_half = torch.randint(0, 2, (1,), generator=local_generator).item() == 0
#             batch_half_choices.append(use_first_half)
        
#         # 根据批次选择，为每个批次选择对应的索引范围
#         all_indices = []
        
#         for batch_idx in range(total_batch_num):
#             use_first_half = batch_half_choices[batch_idx]
            
#             if use_first_half:
#                 # 从前半部分中选择索引
#                 batch_indices = list(range(0, self.half_size))
#             else:
#                 # 从后半部分中选择索引
#                 batch_indices = list(range(self.half_size, len(self.dataset)))
            
#             # 为这个批次生成一个特定的随机序列
#             batch_seed = self.seed + self.epoch * 10000 + batch_idx * 100
#             batch_generator = torch.Generator()
#             batch_generator.manual_seed(batch_seed)
            
#             # 在选定的半区内打乱索引顺序
#             if self.shuffle:
#                 shuffle = torch.randperm(len(batch_indices), generator=batch_generator).tolist()
#                 batch_indices = [batch_indices[i] for i in shuffle]
            
#             # 只取出当前批次需要的样本数
#             batch_size = min(self.batch_size, len(batch_indices))
#             batch_indices = batch_indices[:batch_size]
            
#             all_indices.extend(batch_indices)
        
#         # 将所有批次的索引组合起来
#         if self.shuffle:
#             # 打乱批次顺序，但保持批次内的索引连续
#             batches = [all_indices[i:i+self.batch_size] for i in range(0, len(all_indices), self.batch_size)]
#             random.Random(self.seed + self.epoch).shuffle(batches)
#             all_indices = [idx for batch in batches for idx in batch]
        
#         # 分配给当前进程的样本
#         if self.drop_last:
#             # 删除尾部数据以确保均匀分布
#             total_size = (len(all_indices) // self.num_replicas) * self.num_replicas
#             all_indices = all_indices[:total_size]
            
#         # 计算每个进程应获取的样本数量
#         per_rank_samples = len(all_indices) // self.num_replicas
        
#         # 获取当前rank的样本索引
#         rank_indices = all_indices[self.rank * per_rank_samples:(self.rank + 1) * per_rank_samples]
        
#         return iter(rank_indices)
    
#     def __len__(self):
#         # 返回该rank应该获取的样本数
#         if self.drop_last:
#             return len(self.dataset) // self.num_replicas
#         else:
#             return (len(self.dataset) + self.num_replicas - 1) // self.num_replicas
    
#     def set_epoch(self, epoch):
#         """设置epoch，用于确保每个epoch的随机性"""
#         self.epoch = epoch

# def make_supervised_data_module(model_id, processor, data_args,per_device_train_batch_size):
#     """Make dataset and collator for supervised fine-tuning."""
#     # sft_dataset = SupervisedDataset(
#     #     data_path=data_args.data_path, processor=processor, data_args=data_args, model_id=model_id
#     # )
    
#     train_dataset = SupervisedDataset(
#         data_path=data_args.data_path, processor=processor, data_args=data_args, model_id=model_id
#     )
    
#     eval_dataset = None
#     # TODO: 评估部分都没有修改，这里可能会涉及到batchsize，所以大概率要统一训练和评估的batchsize大小了
#     if data_args.eval_path is not None:
#         eval_dataset = SupervisedDataset(
#               data_path=data_args.eval_path,
#               processor=processor,
#               data_args=data_args,
#               model_id=model_id
#           )
        
#     data_collator = DataCollatorForSupervisedDataset(pad_token_id=processor.tokenizer.pad_token_id)
    
#     train_sampler = SynchronizedHalfDatasetSampler(
#         dataset=train_dataset,
#         shuffle=True,
#         seed=data_args.seed if hasattr(data_args, 'seed') else 42,
#         drop_last=True,
#         batch_size=per_device_train_batch_size
#     )
    
#     return dict(train_dataset=train_dataset,
#                 eval_dataset=eval_dataset,
#                 data_collator=data_collator,
#                 train_sampler=train_sampler)

def make_supervised_data_module(model_id, processor, data_args,per_device_train_batch_size):
    """Make dataset and collator for supervised fine-tuning."""
    # sft_dataset = SupervisedDataset(
    #     data_path=data_args.data_path, processor=processor, data_args=data_args, model_id=model_id
    # )
    
    train_dataset = SupervisedDataset(
        data_path=data_args.data_path, processor=processor, data_args=data_args, model_id=model_id
    )
    
    eval_dataset = None
    # TODO: 评估部分都没有修改，这里可能会涉及到batchsize，所以大概率要统一训练和评估的batchsize大小了
    if data_args.eval_path is not None:
        eval_dataset = SupervisedDataset(
              data_path=data_args.eval_path,
              processor=processor,
              data_args=data_args,
              model_id=model_id
          )
        
    data_collator = DataCollatorForSupervisedDataset(pad_token_id=processor.tokenizer.pad_token_id)
    
    ## ======================== 改动2: 替换 Sampler 实例化 ========================
    ## 原: SynchronizedHalfDatasetSampler（按索引前后半分割）
    ## 新: ModalityAwareSampler（按4种模态分组 + 可选采样权重）
    ## =========================================================================
    # 解析模态采样权重，格式: "us:1.0,ct:2.0,mri:1.5,cxr:1.0"
    modality_weights = None
    if hasattr(data_args, 'modality_sample_weights') and data_args.modality_sample_weights:
        modality_weights = {}
        for item in data_args.modality_sample_weights.split(","):
            mod, w = item.strip().split(":")
            modality_weights[mod.strip().lower()] = float(w.strip())
        print(f"  [make_supervised_data_module] parsed modality_weights: {modality_weights}")

    train_sampler = ModalityAwareSampler(
        dataset=train_dataset,
        batch_size=per_device_train_batch_size,
        shuffle=True,
        seed=data_args.seed if hasattr(data_args, 'seed') else 42,
        drop_last=True,
        modality_weights=modality_weights,
    )
    
    return dict(train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                data_collator=data_collator,
                train_sampler=train_sampler)