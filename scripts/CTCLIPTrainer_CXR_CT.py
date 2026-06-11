
from pathlib import Path
import json
from shutil import rmtree
from datetime import timedelta
from collections import defaultdict
import pynvml
from transformer_maskgit.optimizer import get_optimizer
from transformers import BertTokenizer, BertModel
from torchvision.transforms import v2
from eval import evaluate_internal, plot_roc, accuracy, sigmoid, bootstrap, compute_cis
from sklearn.metrics import classification_report, confusion_matrix, multilabel_confusion_matrix, f1_score, accuracy_score

import torch
from torch import nn, einsum
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from torch import cuda

from data import CTReportDataset
from conditional_dataset_train_all import Conditional_CTReportDataset_Train, collate_fn
from conditional_dataset_evaluate import Conditional_CTReportDataset_Eval, custom_data_split

import numpy as np
import pandas as pd
from tqdm import tqdm

from einops import rearrange
import accelerate
from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs
from accelerate.utils import InitProcessGroupKwargs

import math
import torch.optim.lr_scheduler as lr_scheduler
from ct_clip import CTCLIP
import os

from scheduler import cosine_lr

from retrieval_metric import compute_ndcg, hard_retrieval, hard_retrieval_exclude_self, soft_retrieval,compute_ndcg_exclude_self, soft_retrieval_exclude_self, RateScore_retrieval, compute_ndcg_uncon,soft_retrieval_uncon
from datetime import datetime
import heapq

import wandb
from augment import MedicalOnGPUAugmenter

def print_gpu_memory_stats(accelerator,location: str = "Unknown"):
    """
    打印当前GPU的显存统计信息（每个rank独立调用）
    
    Args:
        location: 调用位置的标识符（用于日志）
    """
    # 确保只在使用CUDA时执行
    if not cuda.is_available():
        return

    # 获取当前设备（自动处理多GPU场景）
    device = accelerator.device
    if device.type != "cuda":
        return

    # 同步所有CUDA操作确保准确测量
    cuda.synchronize(device)
    
    # 获取显存统计信息
    stats = cuda.memory_stats(device=device)
    
    # 提取关键指标（单位：字节）
    allocated = cuda.memory_allocated(device)  # 当前已分配显存
    reserved = cuda.memory_reserved(device)    # 当前保留显存
    total = cuda.get_device_properties(device).total_memory  # 总显存
    
    # 转换为MB
    def to_mb(bytes_val):
        return bytes_val / (1024 ** 2)
    
    pynvml.nvmlInit()
    h = pynvml.nvmlDeviceGetHandleByIndex(device.index)
    mem = pynvml.nvmlDeviceGetMemoryInfo(h)
    free_driver_level = mem.free   # 驱动层剩余字节
    
    # 打印信息（包含rank和位置标识）
    print(
        f"[Rank {accelerator.process_index}] {location} - "
        f"Allocated: {to_mb(allocated):.2f} MB | "
        f"Reserved: {to_mb(reserved):.2f} MB | "
        f"Total: {to_mb(total):.2f} MB | "
        f"Driver-level free memory: {free_driver_level/1024**2:.1f} MB"
    )


# helpers
def apply_softmax(array):
    """
    Applies softmax function to a torch array.

    Args:
        array (torch.Tensor): Input tensor array.

    Returns:
        torch.Tensor: Tensor array after applying softmax.
    """
    softmax = torch.nn.Softmax(dim=0)
    softmax_array = softmax(array)
    return softmax_array

def tensor_to_nifti(tensor, path, affine=np.eye(4)):
    """
    Save tensor as a NIfTI file.

    Args:
        tensor (torch.Tensor): The input tensor with shape (D, H, W) or (C, D, H, W).
        path (str): The path to save the NIfTI file.
        affine (np.ndarray, optional): The affine matrix for the NIfTI file. Defaults to np.eye(4).
    """

    tensor = tensor.cpu()

    if tensor.dim() == 4:
        # Assume single channel data if there are multiple channels
        if tensor.size(0) != 1:
            print("Warning: Saving only the first channel of the input tensor")
        tensor = tensor.squeeze(0)
    tensor=tensor.swapaxes(0,2)
    numpy_data = tensor.detach().numpy().astype(np.float32)
    nifti_img = nib.Nifti1Image(numpy_data, affine)
    nib.save(nifti_img, path)

def exists(val):
    return val is not None

def noop(*args, **kwargs):
    pass

def cycle(dl):
    while True:
        for data in dl:
            yield data

def yes_or_no(question):
    answer = input(f'{question} (y/n) ')
    return answer.lower() in ('yes', 'y')

def accum_log(log, new_logs):
    for key, new_value in new_logs.items():
        old_value = log.get(key, 0.)
        log[key] = old_value + new_value
    return log


def get_rows_to_remove(sim_matrix, threshold=0.9, min_count=1):
    """获取需要移除的行索引：行中小于min_count个元素大于threshold的行"""
    rows_to_remove = []
    for i in range(sim_matrix.shape[0]):
        # 统计该行中大于threshold的元素个数（排除对角线元素）
        row = sim_matrix[i]
        # 创建掩码排除对角线元素
        mask = np.ones(len(row), dtype=bool)
        mask[i] = False
        row_without_diag = row[mask]
        
        count_above_threshold = np.sum(row_without_diag > threshold)
        
        if count_above_threshold < min_count:
            rows_to_remove.append(i)
    
    return rows_to_remove

def remove_rows_columns(matrix, indices_to_remove):
    """从矩阵中移除指定的行和列"""
    if len(indices_to_remove) == 0:
        return matrix
    
    # 创建保留索引的掩码
    keep_indices = [i for i in range(matrix.shape[0]) if i not in indices_to_remove]
    
    if len(matrix.shape) == 2:
        # 2D矩阵
        return matrix[np.ix_(keep_indices, keep_indices)]
    else:
        # 1D向量
        return matrix[keep_indices]

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.val = 0

    def update(self, val, n=1): # avg value over samples, and the num of samples, e.g. the avg loss over a batch and batchsize
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class CTClipTrainer(nn.Module):
    def __init__(
        self,
        CTClip: CTCLIP,
        *,
        num_train_steps,
        current_step,
        warmup_steps,
        local_batch_size,       # 也是一个list
        batch_size,             # 也是一个list
        # 带上stage1与stage2分别表示是否做第一二个阶段
        stage1,
        stage2,
        data_train_npy_dir, # 也是一个列表，大小为K
        data_valid_npy_dir, # 也是一个列表，大小为K
        data_train_csv_dir, # 也是一个列表，大小为K
        data_valid_csv_dir, # 也是一个列表，大小为K
        data_train_jsonl, # 也是一个列表，大小为K
        data_valid_jsonl, # 也是一个列表，大小为K
        anatomy_filter, # 二维list，第一个维度是K，表示用了多少个数据集。维度内部就是对应的数据集数量['lungs']之类的
        dataset_names, # 是一个列表，大小是K,内容表示数据集['MIMIC'，'CT_RATE']
        modality, # 是一个列表，大小是K,内容表示是2D还是3D的数据集。['2D','3D']
        uncon_batch_size,       # 也是一个list，大小均为k'
        uncon_batch_size_valid,
        uncon_soft_label,        # 是一个值
        uncon_similarity_lookup_table_train,  # 也是一个list，大小均为k'
        uncon_similarity_lookup_table_valid,  # 也是一个list，大小均为k'
        uncon_data_train_jsonl, # 也是一个list，大小均为k'
        uncon_data_valid_jsonl,  # 也是一个list，大小均为k'
        uncon_dataset_names, # 也是一个list，大小均为k'
        uncon_train_filter, # 二维list，第一个维度是k'，表示用了多少个数据集。维度内部就是对应的数据集数量['train1,train2']之类的，可以避免单个GPU上存储过多的内容
        uncon_valid, # 一维list，表示名字
        uncon_modality, # 一维list，表示2D还是3D的数据集
        positive_threshold,
        negative_threshold,
        tokenizer = None,
        lr = 1.25e-6,
        wd = 0.1,   # 权重已经被设定了
        max_grad_norm = 0.5,
        save_results_every = 1000,
        save_model_every = 1000 ,
        train_max_samples = 30000, # take first N samples for evaluation
        valid_max_samples = 20000,
        shuffle_train_samples_every = 10000, # if train_max_samples < total_samples, shuffle the samples every N steps
        results_folder = '/shares/menze.dqbm.uzh/ihamam/ctclip/',
        uncon_num_workers = 4,
        con_num_workers = 8,
        pin_memory = False,
        accelerate_kwargs: dict = dict(),
        debug = False,
        modal_embedding = False, # 是否选择开启modal_embedding。在image vit中增加modal的信息
        train_no_aug = False,
        train_valid_length_ = 300
    ):
        # 应该为4800，overfit改进了
        # train_valid_length_ = 2400  # 一定要是valid batch_size*gpus的倍数
        self.stage1 = stage1
        self.stage2 = stage2
        self.debug = debug
        print('uncon_num_workers:', uncon_num_workers)
        print('con_num_workers:', con_num_workers)
        self.con_num_workers = con_num_workers
        self.uncon_num_workers = uncon_num_workers
        assert len(data_train_npy_dir) == len(data_valid_npy_dir) == len(data_train_csv_dir) == len(data_valid_csv_dir) == len(data_train_jsonl) == len(data_valid_jsonl) == len(anatomy_filter) == len(dataset_names) == len(modality) == len(local_batch_size) == len(batch_size), "All input lists must have the same length."
        super().__init__()
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=36000))
        # self.accelerator = Accelerator(mixed_precision='fp16',kwargs_handlers=[ddp_kwargs, kwargs], **accelerate_kwargs) # 这里增加了参数，使用混合进度训练
        self.accelerator = Accelerator(mixed_precision='bf16',kwargs_handlers=[ddp_kwargs, kwargs], **accelerate_kwargs) # 这里增加了参数，使用混合进度训练
        
        # self.accelerator = Accelerator(mixed_precision="fp16",kwargs_handlers=[ddp_kwargs, kwargs], **accelerate_kwargs) # 这里增加了参数，使用混合进度训练
        self.CTClip = CTClip
        
        if tokenizer != None:
            self.tokenizer=tokenizer
        else:
            raise ValueError('Tokenzier is not defined.')

        if isinstance(num_train_steps, list):
            self.num_train_steps = num_train_steps[-1]
        else:
            self.num_train_steps = num_train_steps
        self.augmentor = MedicalOnGPUAugmenter()
        self.current_step = current_step
        self.batch_size = batch_size
        self.local_batch_size = local_batch_size
        self.uncon_batch_size = uncon_batch_size
        self.uncon_batch_size_valid = uncon_batch_size_valid
        self.positive_threshold = positive_threshold
        self.negative_threshold = negative_threshold
        self.dataset_names = dataset_names
        self.uncon_dataset_names = uncon_dataset_names
        all_parameters = set(CTClip.parameters())
        self.modal_embedding = modal_embedding
        # with open('/mnt/petrelfs/zhangtengfei/RadIR/CT-CLIP/CT-UNI-Image-Retrieval-CXR_CT_change_layer_condition_moe_cross_v4/scripts/all_condition_index.json', 'r', encoding='utf-8') as f:
        with open('/mnt/petrelfs/zhangtengfei/RadIR/CT-CLIP/CT-UNI-Image-Retrieval-CXR_CT_change_layer_condition_moe_cross_v7_abnormal_bool/scripts/all_condition_index.json', 'r', encoding='utf-8') as f:
        
            self.condition_index = json.load(f)
        if isinstance(lr, list):
            self.optim = get_optimizer(all_parameters, lr=lr[-1], wd=wd)
        else:
            self.optim = get_optimizer(all_parameters, lr=lr, wd=wd)
        self.soft_label = uncon_soft_label
        self.max_grad_norm = max_grad_norm
        local_rank = torch.distributed.get_rank()
        # stage1和stage2可以同时成立
        if self.stage1:
            self.uncon_dss = {} 
            self.uncon_dls = {}
            self.uncon_valid_dss = {}
            self.uncon_valid_dls = {}
            self.uncon_train_valid_dss = {}  # 用于测试训练过程中训练集的指标情况变化
            self.uncon_train_valid_dls = {}
            self.uncon_dl_iters = {}
            self.uncon_valid_dl_iters = {}
            self.uncon_similarity_lookup_table_train = {}
            self.uncon_similarity_lookup_table_valid = {}
            self.uncon_similarity_lookup_table_train_valid = {}
            for i in range(len(uncon_dataset_names)):
                uncon_dataset_name = uncon_dataset_names[i]
                modality_type = uncon_modality[i]
                train_split_name = uncon_train_filter[i][(local_rank) % len(uncon_train_filter[i])]  # 只取第一个训练分割
                # self.uncon_dss[uncon_dataset_name] = CTReportDataset(os.path.join(uncon_data_train_jsonl[i],train_split_name + '.jsonl'),need_aug=True,modality=uncon_modality[i],is_train=True)
                if not train_no_aug:
                    self.uncon_dss[uncon_dataset_name] = CTReportDataset(os.path.join(uncon_data_train_jsonl[i],train_split_name + '.jsonl'),need_aug=True,modality=uncon_modality[i],is_train=True,modal_embedding=modal_embedding)
                else:
                    self.uncon_dss[uncon_dataset_name] = CTReportDataset(os.path.join(uncon_data_train_jsonl[i],train_split_name + '.jsonl'),need_aug=False,modality=uncon_modality[i],is_train=True,modal_embedding=modal_embedding)        
                    
                self.uncon_valid_dss[uncon_dataset_name] = CTReportDataset(os.path.join(uncon_data_valid_jsonl[i], uncon_valid[i] + '.jsonl'),need_aug=False,modality=uncon_modality[i],is_train=False,modal_embedding=modal_embedding)
                self.uncon_train_valid_dss[uncon_dataset_name] = CTReportDataset(os.path.join(uncon_data_train_jsonl[i],uncon_train_filter[i][0] + '.jsonl'),need_aug=False,modality=uncon_modality[i],is_train=False,modal_embedding=modal_embedding,length_=train_valid_length_)  # 用于计算soft label的相似度
                # 创建数据加载器
                uncon_dl_i = DataLoader(
                    self.uncon_dss[uncon_dataset_name],
                    num_workers = uncon_num_workers,
                    batch_size = self.uncon_batch_size[i],
                    shuffle = True,
                    drop_last = True,
                    pin_memory = pin_memory
                )
                self.uncon_dls[uncon_dataset_name] = uncon_dl_i
                
                # 验证数据加载器
                uncon_valid_dl_i = DataLoader(
                    self.uncon_valid_dss[uncon_dataset_name],
                    # num_workers = uncon_num_workers,
                    num_workers = 0,
                    
                    batch_size = self.uncon_batch_size_valid[i],
                    shuffle = False,
                    pin_memory = pin_memory
                )
                self.uncon_valid_dls[uncon_dataset_name] = uncon_valid_dl_i
                
                uncon_train_valid_dl_i = DataLoader(
                    self.uncon_train_valid_dss[uncon_dataset_name],
                    num_workers = uncon_num_workers,
                    batch_size = self.uncon_batch_size_valid[i],
                    shuffle = False,
                    pin_memory = pin_memory
                )
                self.uncon_train_valid_dls[uncon_dataset_name] = uncon_train_valid_dl_i
                
                # 准备迭代器
                self.uncon_dl_iters[uncon_dataset_name] = cycle(self.accelerator.prepare(uncon_dl_i))
                self.uncon_valid_dl_iters[uncon_dataset_name] = cycle(self.accelerator.prepare(uncon_valid_dl_i))
                
                if self.soft_label:
                    self.uncon_similarity_lookup_table_train[uncon_dataset_name] = np.load(os.path.join(uncon_similarity_lookup_table_train[i], train_split_name + '.npy'))
                    self.uncon_similarity_lookup_table_valid[uncon_dataset_name] = np.load(os.path.join(uncon_similarity_lookup_table_valid[i], uncon_valid[i] + '.npy'))
                    self.uncon_similarity_lookup_table_train[uncon_dataset_name] = np.clip(self.uncon_similarity_lookup_table_train[uncon_dataset_name],0,1)  # 已经转成了0-1之间的数值了，还要转为uint减少内存占用
                    self.uncon_similarity_lookup_table_valid[uncon_dataset_name] = np.clip(self.uncon_similarity_lookup_table_valid[uncon_dataset_name],0,1)  # 已经转成了0-1之间的数值了，还要转为uint减少内存占用
                    self.uncon_similarity_lookup_table_train[uncon_dataset_name] = np.round(self.uncon_similarity_lookup_table_train[uncon_dataset_name] * 100).astype(np.uint8)  # 转为0-100之间的uint8
                    self.uncon_similarity_lookup_table_valid[uncon_dataset_name] = np.round(self.uncon_similarity_lookup_table_valid[uncon_dataset_name] * 100).astype(np.uint8)  # 转为0-100之间的uint8
                    
                    self.uncon_similarity_lookup_table_train_valid[uncon_dataset_name] = np.load(os.path.join(uncon_similarity_lookup_table_train[i], uncon_train_filter[i][0] + '.npy'))[:train_valid_length_,:train_valid_length_]  # 只取前train_valid_length_个样本
                    self.uncon_similarity_lookup_table_train_valid[uncon_dataset_name] = np.clip(self.uncon_similarity_lookup_table_train_valid[uncon_dataset_name],0,1)  # 已经转成了0-1之间的数值了，还要转为uint减少内存占用
                    self.uncon_similarity_lookup_table_train_valid[uncon_dataset_name] = np.round(self.uncon_similarity_lookup_table_train_valid[uncon_dataset_name] * 100).astype(np.uint8)  # 转为0-100之间的uint8
                    
                    # 内容是np矩阵，使用的时候需要转换为tensor形式以及转回fp的格式
                # 打印数据集信息
                print(f"Rank {local_rank} unconditional dataset: {uncon_dataset_name}, split: {train_split_name}")
                print(f"Rank {local_rank} train samples: {len(self.uncon_dss[uncon_dataset_name])}")
                print(f"Rank {local_rank} valid samples: {len(self.uncon_valid_dss[uncon_dataset_name])}")
        if self.stage2:
            # Divide anatomies into different process
            self.anatomy2id_lss = {}
            self.dss = {}
            self.dls = {}
            self.valid_dss = {}
            self.valid_dls = {}
            self.dl_iters = {}
            self.valid_dl_iters = {}
            self.local_anatomy_filter_for_train = {}
            self.local_anatomy_filter_for_valid = {}
            # 这里要修改
            for i in range(len(dataset_names)):
                dataset_name = dataset_names[i]
                modality_type = modality[i]

                with open(f"/mnt/petrelfs/zhangtengfei/public_dataset/extend_test/all_condition/all_distribution/{dataset_name}_anatomy_distribution.json", 'r') as f:
                    anatomy_distribution = json.load(f)
                anatomy_samples = {anatomy:anatomy_distribution[anatomy]["train >0.9 links count"] for anatomy in anatomy_filter[i]}   # pos 样本越多，采样越多
                
                rank = torch.distributed.get_rank()  # 获取当前进程的 rank
                world_size = torch.distributed.get_world_size()  # 获取总进程数

                distribution_train_i, distribution_valid_i = self.distribute_equal_sample_num(anatomy_samples, world_size)
                local_anatomy_filter_for_train_i, local_anatomy_filter_for_valid_i = distribution_train_i[rank], distribution_valid_i[rank]
                self.local_anatomy_filter_for_train[dataset_name] = local_anatomy_filter_for_train_i
                self.local_anatomy_filter_for_valid[dataset_name] = local_anatomy_filter_for_valid_i
                # for training, allow repeat when processor > anatomy 
                
                # check split results
                print(f"Rank {rank} anatomy subset for training: {local_anatomy_filter_for_train_i}")
                print(f"Rank {rank} total samples: {sum(anatomy_samples[k] for k in local_anatomy_filter_for_train_i)}")
                print(f"Rank {rank} anatomy subset for validation: {local_anatomy_filter_for_valid_i}")
                print(f"Rank {rank} total samples: {sum(anatomy_samples[k] for k in local_anatomy_filter_for_valid_i)}")
                if not train_no_aug:
                    ds_i = Conditional_CTReportDataset_Train(
                        modality=modality_type,
                        local_batch_size=local_batch_size[i], 
                        jsonl_file=data_train_jsonl[i], 
                        csv_file_dir=data_train_csv_dir[i], 
                        npy_file_dir=data_train_npy_dir[i], 
                        anatomy_filter=local_anatomy_filter_for_train_i,
                        positive_threshold=positive_threshold,
                        negative_threshold=negative_threshold,
                        max_samples=train_max_samples,
                        modal_embedding=modal_embedding,
                        need_aug=True
                        )
                else:
                    ds_i = Conditional_CTReportDataset_Train(
                        modality=modality_type,
                        local_batch_size=local_batch_size[i], 
                        jsonl_file=data_train_jsonl[i], 
                        csv_file_dir=data_train_csv_dir[i], 
                        npy_file_dir=data_train_npy_dir[i], 
                        anatomy_filter=local_anatomy_filter_for_train_i,
                        positive_threshold=positive_threshold,
                        negative_threshold=negative_threshold,
                        max_samples=train_max_samples,
                        modal_embedding=modal_embedding,
                        need_aug=False
                        )
                self.dss[dataset_name] = ds_i

                valid_ds_i = Conditional_CTReportDataset_Eval(
                    modality=modality_type,
                    jsonl_file=data_valid_jsonl[i], 
                    csv_file_dir=data_valid_csv_dir[i], 
                    npy_file_dir=data_valid_npy_dir[i], 
                    anatomy_filter=local_anatomy_filter_for_valid_i,
                    max_samples=valid_max_samples,
                    modal_embedding=modal_embedding
                    )
                self.valid_dss[dataset_name] = valid_ds_i
                self.anatomy2id_lss[dataset_name] = self.valid_dss[dataset_name].get_anatomy2id_ls()
                dl_i = DataLoader(
                    ds_i,
                    num_workers = con_num_workers,
                    batch_size = self.batch_size[i],
                    shuffle = True,
                    drop_last = True,
                    pin_memory = pin_memory,
                    collate_fn = collate_fn
                )
                self.dls[dataset_name] = dl_i
                valid_dl_i = DataLoader(
                    valid_ds_i,
                    num_workers = con_num_workers,
                    batch_size = self.batch_size[i] * local_batch_size[i],
                    shuffle = False,
                    pin_memory = pin_memory,
                )

                self.valid_dls[dataset_name] = valid_dl_i
                self.dl_iters[dataset_name] = cycle(self.accelerator.prepare(dl_i))
                self.valid_dl_iters[dataset_name] = cycle(self.accelerator.prepare(valid_dl_i))

                self.pin_memory = pin_memory

        self.batch_generators = []
        self.gen_batch_generators()
        # prepare with accelerator
        self.device = self.accelerator.device
        self.CTClip.to(self.device)

        (
            self.CTClip,
            self.optim,
        ) = self.accelerator.prepare(
            self.CTClip,
            self.optim,
        )
        
        self.lr_scheduler = cosine_lr(self.optim, lr, warmup_steps, num_train_steps)
        if self.stage1:
            self.uncon_loss_m = {dataset_name: AverageMeter() for dataset_name in uncon_dataset_names}
            self.it_triplet_loss_m = {dataset_name: AverageMeter() for dataset_name in uncon_dataset_names}
            self.it_infoNCE_loss_m = {dataset_name: AverageMeter() for dataset_name in uncon_dataset_names}
            self.ii_triplet_loss_m = {dataset_name: AverageMeter() for dataset_name in uncon_dataset_names}
        if self.stage2:
            self.loss_m = {dataset_name: AverageMeter() for dataset_name in dataset_names}
            self.infoNCE_loss_m = {dataset_name: AverageMeter() for dataset_name in dataset_names}
            self.triplet_loss_m = {dataset_name: AverageMeter() for dataset_name in dataset_names}
            self.binary_loss_m = {dataset_name: AverageMeter() for dataset_name in dataset_names}
            self.rerank_loss_m = {dataset_name: AverageMeter() for dataset_name in dataset_names}
            self.anatomy_infoNCE_loss_m = {dataset_name: dict() for dataset_name in dataset_names}
            self.anatomy_triplet_loss_m = {dataset_name: dict() for dataset_name in dataset_names}
            self.anatomy_binary_loss_m = {dataset_name: dict() for dataset_name in dataset_names}
            self.anatomy_rerank_loss_m = {dataset_name: dict() for dataset_name in dataset_names}
            self.anatomy_valid_triplet_count_m = {dataset_name: dict() for dataset_name in dataset_names}

        self.save_model_every = save_model_every
        self.save_results_every = save_results_every
        self.shuffle_train_samples_every = shuffle_train_samples_every

        self.results_folder = Path(results_folder)
        tensorboard_path = os.path.join(results_folder, 'tensorboard_log')
        self.tb_writer = SummaryWriter(tensorboard_path)


        self.results_folder.mkdir(parents=True, exist_ok=True)
        
        if self.accelerator.is_local_main_process:
            # local_rank = torch.distributed.get_rank()
            # run_name = f'CTClip-Train-{local_rank}-{datetime.now().strftime("%Y%m%d-%H%M%S")}'
            visible = os.environ.get("CUDA_VISIBLE_DEVICES")
            
            physical_ids = [int(v) for v in visible.split(",")]
            
            run_name = f'CTClip-Train-phy-{physical_ids}-7.30'
            self.wandb = wandb.init(
                project="CTClip-Train_overfit",
                name=run_name,
                group ='ddp_exp',
                config={
                    "num_train_steps": self.num_train_steps,
                    "current_step": self.current_step,
                    "warmup_steps": warmup_steps,
                    "local_batch_size": self.local_batch_size,
                    "batch_size": self.batch_size,
                    "data_train_npy_dir": data_train_npy_dir,
                    "data_valid_npy_dir": data_valid_npy_dir,
                    "data_train_csv_dir": data_train_csv_dir,
                    "data_valid_csv_dir": data_valid_csv_dir,
                    "data_train_jsonl": data_train_jsonl,
                    "data_valid_jsonl": data_valid_jsonl,
                    "anatomy_filter": anatomy_filter,
                    "dataset_names": dataset_names,
                    "modality": modality,
                    "positive_threshold": positive_threshold,
                    "negative_threshold": negative_threshold,
                    "lr": lr,
                    "wd": wd,
                    "max_grad_norm": max_grad_norm,
                    "save_results_every": save_results_every,
                    "save_model_every": save_model_every,
                    "train_max_samples": train_max_samples,
                    "valid_max_samples": valid_max_samples,
                    "shuffle_train_samples_every": shuffle_train_samples_every
                }
            )
            self.wandb.watch(self.CTClip.module, log='all', log_freq=10)
            
    def gen_batch_generators(self):
        self.batch_generators = []
        if self.stage1:
            for dataset_name in self.uncon_dataset_names:
                self.batch_generators.append(("stage1",dataset_name,self.uncon_dl_iters[dataset_name]))
        if self.stage2:
            for dataset_name in self.dataset_names:
                self.batch_generators.append(("stage2",dataset_name,self.dl_iters[dataset_name]))
                
    def _reset_metrics(self):
        for dataset_name in self.dataset_names:
            self.loss_m[dataset_name].reset()
            self.infoNCE_loss_m[dataset_name].reset()
            self.triplet_loss_m[dataset_name].reset()
            self.binary_loss_m[dataset_name].reset()
            self.rerank_loss_m[dataset_name].reset()
            for m in self.anatomy_infoNCE_loss_m[dataset_name].values():
                m.reset()
            for m in self.anatomy_triplet_loss_m[dataset_name].values():
                m.reset()
            for m in self.anatomy_valid_triplet_count_m[dataset_name].values():
                m.reset()
            for m in self.anatomy_binary_loss_m[dataset_name].values():
                m.reset()
            for m in self.anatomy_rerank_loss_m[dataset_name].values():
                m.reset()
            # 重置统计的结果
    def _reset_metrics_uncon(self):
        for dataset_name in self.uncon_dataset_names:
            self.uncon_loss_m[dataset_name].reset()
            self.it_triplet_loss_m[dataset_name].reset()
            self.it_infoNCE_loss_m[dataset_name].reset()
            self.ii_triplet_loss_m[dataset_name].reset()
            # 重置统计的结果

    def distribute_equal_sample_num(self, anatomy_samples, num_processes):
        """
        改进后的分配策略：
        1. 当anatomy数量 ≤ 进程数时：
        - 每个进程分配1个anatomy（按样本量降序）
        - 多余进程循环复用高样本量anatomy
        2. 当anatomy数量 > 进程数时：
        - 使用贪心算法均衡分配，使各进程总样本量接近
        """
        # 按样本量降序、名称升序排序
        sorted_anatomy = sorted(
            anatomy_samples.items(),
            key=lambda x: (-x[1], x[0])
        )
        num_anatomy = len(sorted_anatomy)
        
        # 情况1：anatomy数量 ≤ 进程数
        if num_anatomy <= num_processes:
            distribution = []
            for i in range(num_processes):
                # 循环使用高样本量anatomy
                anatomy_idx = i % num_anatomy
                anatomy_name = sorted_anatomy[anatomy_idx][0]
                distribution.append([anatomy_name])
        
            # 针对validation分配结果（多余进程置空）
            backup_dist = []
            for i in range(num_processes):
                if i < num_anatomy:
                    anatomy_name = sorted_anatomy[i][0]
                    backup_dist.append([anatomy_name])
                else:
                    backup_dist.append([])
                    
            # return distribution, backup_dist
            return distribution,distribution
        
        # 情况2：anatomy数量 > 进程数
        else:
            # 初始化最小堆（总样本量，进程ID，anatomy列表）
            heap = []
            for i in range(num_processes):
                heapq.heappush(heap, (0, i, []))
            
            # 贪心分配每个anatomy
            for anatomy_name, samples in sorted_anatomy:
                total, proc_id, anatomy_list = heapq.heappop(heap)
                anatomy_list.append(anatomy_name)
                heapq.heappush(heap, (total + samples, proc_id, anatomy_list))
            
            # 提取分配结果并按进程ID排序
            distribution = [[] for _ in range(num_processes)]
            while heap:
                total, proc_id, anatomy_list = heapq.heappop(heap)
                distribution[proc_id] = anatomy_list
            
            return distribution, distribution

    def distribute_equal_anatomy(self, anatomy_samples, num_processes):
        """
        改进后的分配策略：
        1. 当anatomy数量 ≤ 进程数时：
        - 每个进程分配1个anatomy（按样本量降序）
        - 多余进程循环复用高样本量anatomy
        2. 当anatomy数量 > 进程数时：
        - 使用贪心算法均衡分配，使各进程总样本量接近，且每个进程内的anatomy数量尽可能平均
        """
        # 按样本量降序、名称升序排序
        sorted_anatomy = sorted(
            anatomy_samples.items(),
            key=lambda x: (-x[1], x[0])
        )
        num_anatomy = len(sorted_anatomy)
        
        # 情况1：anatomy数量 ≤ 进程数
        if num_anatomy <= num_processes:
            distribution = []
            for i in range(num_processes):
                # 循环使用高样本量anatomy
                anatomy_idx = i % num_anatomy
                anatomy_name = sorted_anatomy[anatomy_idx][0]
                distribution.append([anatomy_name])
        
            # 针对validation分配结果（多余进程置空）
            backup_dist = []
            for i in range(num_processes):
                if i < num_anatomy:
                    anatomy_name = sorted_anatomy[i][0]
                    backup_dist.append([anatomy_name])
                else:
                    backup_dist.append([])
                    
            return distribution, backup_dist
        
        # 情况2：anatomy数量 > 进程数
        else:
            # 初始化进程的anatomy数量和sample num之和
            process_anatomy_count = [0] * num_processes
            process_sample_sum = [0] * num_processes
            distribution = [[] for _ in range(num_processes)]
            
            # 分配每个anatomy
            for anatomy_name, samples in sorted_anatomy:
                # 找到当前anatomy数量最少的进程
                min_anatomy_count = min(process_anatomy_count)
                candidates = [i for i, count in enumerate(process_anatomy_count) if count == min_anatomy_count]
                
                # 在这些进程中挑选sample num之和最小的进程
                min_sample_sum = min(process_sample_sum[i] for i in candidates)
                selected_process = next(i for i in candidates if process_sample_sum[i] == min_sample_sum)
                
                # 分配anatomy到选中的进程
                distribution[selected_process].append(anatomy_name)
                process_anatomy_count[selected_process] += 1
                process_sample_sum[selected_process] += samples
            
            return distribution, distribution
        
    # def evaluate(self, current_step):
    #     with torch.no_grad():
            
    #         self.CTClip.eval()
    #         all_metrics = {}
    #         temp_dataset_names = []
    #         for dataset_name in self.dataset_names:
    #             # if 'CT_RATE' == dataset_name:
    #             #     continue
    #             temp_dataset_names.append(dataset_name)
    #         for dataset_name in temp_dataset_names:
    #             for k in [1,3, 5, 10, 20, 50, 100]:
    #                 all_metrics[f'{dataset_name}_(I2T)NDCG@{k}'] = {}
    #             for k in [1,3, 5, 10, 20, 50, 100]:
    #                 all_metrics[f'{dataset_name}_(I2T)Hard_R@{k}'] = {}
    #             for k in [1,3, 5, 10, 20, 50, 100]:
    #                 all_metrics[f'{dataset_name}_(I2T)Soft_R@{k}'] = {}
    #             for k in [1,3, 5, 10, 20, 50, 100]:
    #                 all_metrics[f'{dataset_name}_(I2T)Soft_exclude_R@{k}'] = {}
    #             for k in [1,3, 5, 10, 20, 50, 100]:
    #                 all_metrics[f'{dataset_name}_RateScore@{k}'] = {}
    #             for k in [1,3, 5, 10, 20, 50, 100]:
    #                 all_metrics[f'{dataset_name}_UpperBound_RateScore@{k}'] = {}
    #             for k in [1,3, 5, 10, 20, 50, 100]:
    #                 all_metrics[f'{dataset_name}_(I2I)NDCG@{k}'] = {}
    #             for k in [1,3, 5, 10, 20, 50, 100]:
    #                 all_metrics[f'{dataset_name}_(I2I)Soft_exclude_R@{k}'] = {}
    #             # 针对bool的结果进行计算准确率，前几个的准确率
    #             all_metrics[f'{dataset_name}_(I2I)Acc'] = {}
    #         for dataset_name in temp_dataset_names:
    #             # 修改增加到了fp16部分
    #             fused_latents_all = {anatomy:torch.zeros(len(self.anatomy2id_lss[dataset_name][anatomy]), 512, dtype=torch.float16) for anatomy in self.local_anatomy_filter_for_valid[dataset_name]}  
    #             abnormal_preds_all = {anatomy:torch.zeros(len(self.anatomy2id_lss[dataset_name][anatomy]), dtype=torch.float16) for anatomy in self.local_anatomy_filter_for_valid[dataset_name]}
    #             abnormal_gt_all = {anatomy:torch.zeros(len(self.anatomy2id_lss[dataset_name][anatomy]), dtype=torch.float16) for anatomy in self.local_anatomy_filter_for_valid[dataset_name]}
    #             local_latents_all = {anatomy:torch.zeros(len(self.anatomy2id_lss[dataset_name][anatomy]), 512, dtype=torch.float16) for anatomy in self.local_anatomy_filter_for_valid[dataset_name]} 
    #             # 每个anatomy的fused_latents顺序与dataset中的sample id顺序严格一致
    #             # 每个anatomy的fused_latents的shape是 (N, 512)，N是这个anatomy的样本数
                
    #             # First derive the fusion latent for each anatomy and each sample
    #             if not self.modal_embedding:
    #                 for video, anatomy_ls, sample_id_ls in tqdm(self.valid_dls[dataset_name], desc=f"Rank {torch.distributed.get_rank()}"):
    #                     # video: B 1 d h w
    #                     video = video.to(self.device).unsqueeze(1)  # B 1 1 d h w (凑出local_B)
    #                     anatomy_ls = list(anatomy_ls)
    #                     sample_id_ls = list(sample_id_ls)
    #                     text_tokens=self.tokenizer(anatomy_ls, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
    #                     with torch.autocast(device_type='cuda', dtype=torch.floa):
    #                         local_latents, fused_latents, _ = self.CTClip(
    #                         text_tokens, video, return_latents=True, device=self.device
    #                     )  # B 1 Dim
    #                     fused_latents = fused_latents.detach().cpu().squeeze().to(torch.float16)  # 去掉local_B，转换为float16
    #                     local_latents = local_latents.detach().cpu().squeeze().to(torch.float16)
    #                     # _, _, fused_latents, tmp = self.CTClip(text_tokens, video, return_latents=True, device=self.device) # B 1 Dim
    #                     # fused_latents = fused_latents.detach().cpu().squeeze()  # 去掉local_B
    #                     for i, (sample_id, anatomy) in enumerate(zip(sample_id_ls, anatomy_ls)):
    #                         sample_index = self.anatomy2id_lss[dataset_name][anatomy].index(sample_id)
    #                         fused_latents_all[anatomy][sample_index] = fused_latents[i]
    #                         local_latents_all[anatomy][sample_index] = local_latents[i]
    #                     del fused_latents, video, text_tokens, local_latents
    #             else:
    #                 def calculate_accuracy(pred, gt):
    #                     """
    #                     计算二分类准确率
    #                     Args:
    #                         pred (torch.Tensor): 预测值，取值 0 或 1，形状为 (B,) 或 (B, 1)
    #                         gt (torch.Tensor): 真实标签，取值 0 或 1，形状需与 pred 一致
    #                     Returns:
    #                         float: 准确率 (0.0 ~ 1.0)
    #                     """
    #                     correct = (pred == gt)
    #                     accuracy = correct.float().mean()
    #                     return accuracy.item()
    #                 for video, anatomy_ls, sample_id_ls,modal_indexs,local_text_ls,abnormal_gt in tqdm(self.valid_dls[dataset_name], desc=f"Rank {torch.distributed.get_rank()}"):
    #                     # video: B 1 d h w
    #                     video = video.to(self.device).unsqueeze(1)  # B 1 1 d h w (凑出local_B)
    #                     anatomy_ls = list(anatomy_ls)
    #                     sample_id_ls = list(sample_id_ls)
    #                     # text_tokens=self.tokenizer(anatomy_ls, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
    #                     condition_indexs = [self.condition_index[dataset_name][condition] for condition in anatomy_ls]
    #                     condition_indexs = torch.tensor(condition_indexs).to(self.device)
    #                     local_text_tokens = self.tokenizer(local_text_ls, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
    #                     with torch.autocast(device_type='cuda', dtype=torch.float16):
    #                         local_latents, fused_latents,abnormal_preds, _ = self.CTClip(
    #                             condition_indexs, video, return_latents=True, device=self.device,is_condition=True,modal_indexs=modal_indexs,modal_embedding=self.modal_embedding,local_text=local_text_tokens
    #                         )  # B 1 Dim .local_B一定是1
                            
    #                     fused_latents = fused_latents.detach().cpu().squeeze().to(torch.float16)
    #                     local_latents = local_latents.detach().cpu().squeeze().to(torch.float16)
    #                     abnormal_preds = abnormal_preds.detach().cpu().squeeze().to(torch.float16)
    #                     abnormal_gt = abnormal_gt.detach().cpu().to(torch.float16)  # 直接是B大小的结果
    #                     # _, _, fused_latents, tmp = self.CTClip(text_tokens, video, return_latents=True, device=self.device) # B 1 Dim
    #                     # fused_latents = fused_latents.detach().cpu().squeeze()  # 去掉local_B
    #                     for i, (sample_id, anatomy) in enumerate(zip(sample_id_ls, anatomy_ls)):
    #                         sample_index = self.anatomy2id_lss[dataset_name][anatomy].index(sample_id)
    #                         fused_latents_all[anatomy][sample_index] = fused_latents[i]
    #                         local_latents_all[anatomy][sample_index] = local_latents[i]
    #                         # +++ 添加验证 +++
    #                         # print(f"DEBUG - {anatomy} abnormal_preds original shape: {abnormal_preds.shape}")
    #                         # +++ 结束验证 +++
    #                         abnormal_preds_all[anatomy][sample_index] = abnormal_preds[i]
    #                         abnormal_gt_all[anatomy][sample_index] = abnormal_gt[i]
    #                     del fused_latents, video,local_latents
    #             # Now lets calculate the similarity matrix for each anatomy
    #             for anatomy, fused_latents in fused_latents_all.items():
    #                 local_latents = local_latents_all[anatomy]
    #                 abnormal_gt = abnormal_gt_all[anatomy]
    #                 abnormal_preds = abnormal_preds_all[anatomy]
    #                 local_latents = local_latents.to(self.device)
    #                 fused_latents = fused_latents.to(self.device)
    #                 # 保存fused_latents备用
    #                 # torch.save(fused_latents, os.path.join(self.results_folder, f'fused_latents_{dataset_name}_{anatomy}_step{current_step}.pt'))
    #                 # image_to_image = torch.einsum('m d, n d -> m n', fused_latents, fused_latents)
    #                 image_to_text = torch.einsum('m d, n d -> m n', fused_latents, local_latents).float()
    #                 # 修改为fp32
    #                 image_to_image = torch.einsum('m d,n d->m n', fused_latents, fused_latents).float()
                    
    #                 similarity_tab = self.valid_dss[dataset_name].get_similarity_table(anatomy)
    #                 similarity_tab = torch.tensor(similarity_tab)
    #                 similarity_tab = (similarity_tab.to(dtype=torch.float32) * 0.01).to(self.device)  # 转回0-1之间的float32
                    
    #                 assert image_to_image.shape == similarity_tab.shape, f'image_to_image {image_to_image.shape} != similarity_tab {similarity_tab.shape}'
    #                 assert image_to_text.shape == similarity_tab.shape, f'iamge_to_text {image_to_text.shape} != similarity_tab {similarity_tab.shape}'
    #                 # now calculate the ndcg
                    
    #                 print('image_to_image shape:', image_to_image.shape)
    #                 print('similarity_tab shape:', similarity_tab.shape)
    #                 # 还要计算i2t的hard_R,ndcg,soft_R,soft_exclude_R
    #                 results = hard_retrieval(image_to_text, similarity_tab, k=[1,3, 5, 10, 20, 50, 100])
    #                 for k, v in zip([1,3, 5, 10, 20, 50, 100], results):
    #                     all_metrics[f'{dataset_name}_(I2T)Hard_R@{k}'][anatomy] = v*100
    #                 ndcg_scores = compute_ndcg_uncon(image_to_text, similarity_tab, k=[1,3, 5, 10, 20, 50, 100])
    #                 for k, v in zip([1,3, 5, 10, 20, 50, 100], ndcg_scores):
    #                     all_metrics[f'{dataset_name}_(I2T)NDCG@{k}'][anatomy] = v*100
    #                 results = soft_retrieval_uncon(image_to_text, similarity_tab, k=[1,3, 5, 10, 20, 50, 100])
    #                 for k, v in zip([1,3, 5, 10, 20, 50, 100], results):
    #                     all_metrics[f'{dataset_name}_(I2T)Soft_R@{k}'][anatomy] = v*100
    #                 all_metrics[f'{dataset_name}_(I2I)Acc'][anatomy] = calculate_accuracy(abnormal_preds.to(torch.int8), abnormal_gt.to(torch.int8))*100
                    
    #                 rows_to_remove = []
    #                 n = similarity_tab.shape[0]
    #                 index_list = [i for i in range(similarity_tab.shape[0])]
                    
    #                 for i in range(n):
    #                     # 计算当前行中大于90的元素数量
    #                     # count_gt_90 = torch.sum(similarity_tab[i] > 0.9).item()
    #                     count_gt_90 = torch.sum(similarity_tab[i] > self.positive_threshold).item()
    #                     if count_gt_90 < 2:
    #                         rows_to_remove.append(i)
                    
    #                 # 保留需要保留的索引
    #                 keep_indices = [i for i in range(n) if i not in rows_to_remove]
                    
    #                 if keep_indices:
    #                     # 更新similarity_tab
    #                     similarity_tab = similarity_tab[keep_indices][:, keep_indices]
    #                     image_to_text = image_to_text[keep_indices][:, keep_indices]
    #                     image_to_image = image_to_image[keep_indices][:, keep_indices]
    #                 else:
    #                     print(f"Warning: All rows removed for {anatomy}, skipping...")
    #                     continue
                    
    #                 results = soft_retrieval_exclude_self(image_to_text, similarity_tab, index_list, k=[1,3, 5, 10, 20, 50, 100],positive_threshold=self.positive_threshold)
    #                 for k, v in zip([1,3, 5, 10, 20, 50, 100], results):
    #                     all_metrics[f'{dataset_name}_(I2T)Soft_exclude_R@{k}'][anatomy] = v*100
                    
    #                 # average similarity
    #                 pred_score, upperbound_score = soft_retrieval(image_to_image, similarity_tab, k=[1,3, 5, 10, 20, 50, 100])   # {'avg_similarity@{k}': xxx, ...}
    #                 for k, v in zip([1,3, 5, 10, 20, 50, 100], pred_score):
    #                     all_metrics[f'{dataset_name}_RateScore@{k}'][anatomy] = v*100
    #                 for k, v in zip([1,3, 5, 10, 20, 50, 100], upperbound_score):
    #                     all_metrics[f'{dataset_name}_UpperBound_RateScore@{k}'][anatomy] = v*100
                    
    #                 results = compute_ndcg(image_to_image, similarity_tab, index_list, k=[1,3, 5, 10, 20, 50, 100])
    #                 for k, v in zip([1,3, 5, 10, 20, 50, 100], results):
    #                     all_metrics[f'{dataset_name}_(I2I)NDCG@{k}'][anatomy] = v*100
    #                 # hard image-image retrieval
    #                 recall_scores = soft_retrieval_exclude_self(image_to_image, similarity_tab, index_list, k=[1,3, 5, 10, 20, 50, 100],positive_threshold=self.positive_threshold)
    #                 if recall_scores == -1:
    #                     print(f'{anatomy} has no positive samples!')
    #                     for k in [1,3, 5, 10, 20, 50, 100]:
    #                         all_metrics[f'{dataset_name}_(I2I)Soft_exclude_R@{k}'][anatomy] = -1
    #                 else:
    #                     for k, v in zip([1,3, 5, 10, 20, 50, 100], recall_scores):
    #                         all_metrics[f'{dataset_name}_(I2I)Soft_exclude_R@{k}'][anatomy] = v*100
                        
    #                     # 清理显存，释放不再需要的变量
    #                 del image_to_image, similarity_tab, fused_latents
    #             del fused_latents_all


    #         if torch.distributed.is_initialized():
    #             # Gather all_metrics dicts from all processes into a list on each process
    #             world_size = torch.distributed.get_world_size()
    #             gathered_metrics = [None for _ in range(world_size)]
                
    #             # +++ 新增：构建包含损失统计量的数据结构 +++
    #             losses = {}
    #             for dataset_name in temp_dataset_names:
    #                 losses.update({
    #                     f'{dataset_name}_main': (self.loss_m[dataset_name].sum, self.loss_m[dataset_name].count),
    #                     f'{dataset_name}_triplet': (self.triplet_loss_m[dataset_name].sum, self.triplet_loss_m[dataset_name].count),
    #                     f'{dataset_name}_binary': (self.binary_loss_m[dataset_name].sum, self.binary_loss_m[dataset_name].count),
    #                     f'{dataset_name}_rerank': (self.rerank_loss_m[dataset_name].sum, self.rerank_loss_m[dataset_name].count),
    #                     f'{dataset_name}_infoNCE': (self.infoNCE_loss_m[dataset_name].sum, self.infoNCE_loss_m[dataset_name].count),
    #                     f'{dataset_name}_anatomy_infoNCE': {k: (v.sum, v.count) for k, v in self.anatomy_infoNCE_loss_m[dataset_name].items()},
    #                     f'{dataset_name}_anatomy_triplet': {k: (v.sum, v.count) for k, v in self.anatomy_triplet_loss_m[dataset_name].items()},
    #                     f'{dataset_name}_anatomy_binary': {k: (v.sum, v.count) for k, v in self.anatomy_binary_loss_m[dataset_name].items()},
    #                     f'{dataset_name}_anatomy_rerank': {k: (v.sum, v.count) for k, v in self.anatomy_rerank_loss_m[dataset_name].items()},
    #                     f'{dataset_name}_anatomy_valid_triplet': {k: (v.sum, v.count) for k, v in self.anatomy_valid_triplet_count_m[dataset_name].items()}
    #                 })

    #             gathered_data = {
    #                 'metrics': all_metrics,
    #                 'losses': losses
    #             }
                
    #             # 执行收集（替换原 all_gather_object 调用）
    #             torch.distributed.all_gather_object(gathered_metrics, gathered_data)  # +++ 修改行 +++

    #             # On the main process, merge the dictionaries
    #             if self.accelerator.is_local_main_process:
                    
    #                 print(f'** EVAL ** Gather Done!')
                    
    #                 # +++ 新增：合并损失统计量 +++
    #                 def merge_loss(metric_name):
    #                     total_sum = sum(data['losses'][metric_name][0] for data in gathered_metrics)
    #                     total_count = sum(data['losses'][metric_name][1] for data in gathered_metrics)
    #                     return total_sum / total_count if total_count > 0 else 0

    #                 def merge_anatomy_loss(metric_name):
    #                     merged = defaultdict(lambda: [0, 0])  # [sum, count]
    #                     for data in gathered_metrics:
    #                         for anatomy, (s, c) in data['losses'][metric_name].items():
    #                             merged[anatomy][0] += s
    #                             merged[anatomy][1] += c
    #                     return {k: (v[0]/v[1] if v[1]>0 else 0) for k, v in merged.items()}
                    
    #                 # 计算全局平均损失
    #                 global_main_loss = {dataset_name: merge_loss(f'{dataset_name}_main') for dataset_name in temp_dataset_names}
    #                 global_triplet_loss = {dataset_name: merge_loss(f'{dataset_name}_triplet') for dataset_name in temp_dataset_names}
    #                 global_infoNCE_loss = {dataset_name: merge_loss(f'{dataset_name}_infoNCE') for dataset_name in temp_dataset_names}
    #                 global_binary_loss = {dataset_name: merge_loss(f'{dataset_name}_binary') for dataset_name in temp_dataset_names}
    #                 global_rerank_loss = {dataset_name: merge_loss(f'{dataset_name}_rerank') for dataset_name in temp_dataset_names}
    #                 global_anatomy_infoNCE = {dataset_name: merge_anatomy_loss(f'{dataset_name}_anatomy_infoNCE') for dataset_name in temp_dataset_names}
    #                 global_anatomy_triplet = {dataset_name: merge_anatomy_loss(f'{dataset_name}_anatomy_triplet') for dataset_name in temp_dataset_names}
    #                 global_anatomy_binary = {dataset_name: merge_anatomy_loss(f'{dataset_name}_anatomy_binary') for dataset_name in temp_dataset_names}
    #                 global_anatomy_rerank = {dataset_name: merge_anatomy_loss(f'{dataset_name}_anatomy_rerank') for dataset_name in temp_dataset_names}
    #                 global_anatomy_valid_triplet = {dataset_name: merge_anatomy_loss(f'{dataset_name}_anatomy_valid_triplet') for dataset_name in temp_dataset_names}

    #                 lr = self.optim.param_groups[0]['lr']
                    
    #                 # +++ 修改信息输出部分 +++
    #                 # info = f"Step {current_step} | LR {lr} | "\
    #                 #     f"Loss {global_main_loss:.4f} | "\
    #                 #     f"Triplet Loss {global_triplet_loss:.4f} | "\
    #                 #     f"InfoNCE Loss {global_infoNCE_loss:.4f} |"
    #                 info  = f"Step {current_step} | LR {lr} | temp {self.CTClip.module.log_temperature.exp()} | "
    #                 for dataset_name in temp_dataset_names:
    #                     info += f"{dataset_name} Loss {global_main_loss[dataset_name]:.4f} | "\
    #                             f"{dataset_name} Triplet Loss {global_triplet_loss[dataset_name]:.4f} | "\
    #                             f"{dataset_name} InfoNCE Loss {global_infoNCE_loss[dataset_name]:.4f} | "\
    #                             f"{dataset_name} Binary Loss {global_binary_loss[dataset_name]:.4f} | "\
    #                             f"{dataset_name} Rerank Loss {global_rerank_loss[dataset_name]:.4f} | "
    #                 # 创建Wandb日志字典
    #                 wandb_log_dict = {'lr': lr}
                    
    #                 if current_step > 0:
    #                     # 使用wandb替代tensorboard
    #                     # self.tb_writer.add_scalar('lr', lr, current_step)
    #                     wandb_log_dict['lr'] = lr
                        
    #                     for dataset_name in temp_dataset_names:
    #                         # 记录训练损失
    #                         wandb_log_dict[f'{dataset_name}_train_loss'] = self.loss_m[dataset_name].avg
    #                         wandb_log_dict[f'{dataset_name}_infoNCE_loss'] = self.infoNCE_loss_m[dataset_name].avg
    #                         wandb_log_dict[f'{dataset_name}_triplet_loss'] = self.triplet_loss_m[dataset_name].avg
    #                         wandb_log_dict[f'{dataset_name}_binary_loss'] = self.binary_loss_m[dataset_name].avg
    #                         wandb_log_dict[f'{dataset_name}_rerank_loss'] = self.rerank_loss_m[dataset_name].avg
    #                         # 记录解剖结构损失
    #                         for anatomy, loss in global_anatomy_infoNCE[dataset_name].items():
    #                             wandb_log_dict[f'{dataset_name}_{anatomy}_infoNCE_loss'] = loss
    #                         for anatomy, loss in global_anatomy_triplet[dataset_name].items():
    #                             wandb_log_dict[f'{dataset_name}_{anatomy}_triplet_loss'] = loss
    #                         for anatomy, loss in global_anatomy_binary[dataset_name].items():
    #                             wandb_log_dict[f'{dataset_name}_{anatomy}_binary_loss'] = loss
    #                         for anatomy, loss in global_anatomy_rerank[dataset_name].items():
    #                             wandb_log_dict[f'{dataset_name}_{anatomy}_rerank_loss'] = loss
    #                         for anatomy, count in global_anatomy_valid_triplet[dataset_name].items():
    #                             wandb_log_dict[f'{dataset_name}_{anatomy}_valid_triplet'] = count
                            
                                    
    #                 merged_metrics = {}
    #                 for metric_name in all_metrics.keys():
    #                     merged_metrics[metric_name] = {}
    #                 for gathered_data in gathered_metrics:
    #                     for metric_name, subdict in gathered_data['metrics'].items():   # metric_name: 'Recall@5' ...
    #                         for key, value in subdict.items():  # pancreas: 0.8
    #                             merged_metrics[metric_name][key] = value
    #                 all_metrics = merged_metrics
    #                 write_info = ''
    #                 for dataset_name in temp_dataset_names:
    #                     # for metrics_name in ['(I2T)NDCG', '(I2T)Hard_R', '(I2T)Soft_R', '(I2T)Soft_exclude_R', 'RateScore', 'UpperBound_RateScore', '(I2I)NDCG', '(I2I)Soft_exclude_R']:
    #                     # for metrics_name in ['(I2T)Soft_R', 'RateScore', 'UpperBound_RateScore', '(I2I)Soft_exclude_R']:
    #                     # 统计acc的值，直接可见这部分结果
    #                     avg_results = sum(all_metrics[f'{dataset_name}_(I2I)Acc'].values()) / len(all_metrics[f'{dataset_name}_(I2I)Acc'])
    #                     info += f' {dataset_name}_(I2I)Acc {avg_results} |'
    #                     wandb_log_dict[f'{dataset_name}_(I2I)Acc'] = avg_results
    #                     write_info += info
    #                     for key, value in all_metrics[f'{dataset_name}_(I2I)Acc'].items():
    #                         # 记录详细指标到wandb
    #                         wandb_log_dict[f'{dataset_name}_{key}_(I2I)Acc'] = value
    #                         write_info += f"{dataset_name}_{key}_(I2I)Acc: {value} | "
    #                     for metrics_name in ['(I2T)Soft_R', '(I2T)Soft_exclude_R', '(I2I)Soft_exclude_R']:
                        
    #                         for k in [1,3, 5, 10, 20, 50, 100]:
    #                             avg_results = sum(all_metrics[f'{dataset_name}_{metrics_name}@{k}'].values()) / len(all_metrics[f'{dataset_name}_{metrics_name}@{k}'])
    #                             info += f' {dataset_name}_{metrics_name}@{k} {avg_results} |'
                                
    #                             # 记录到wandb
    #                             wandb_log_dict[f'{dataset_name}_{metrics_name}@{k}'] = avg_results
                                
    #                             write_info += info   # the following details will be written but not displayed
                                
    #                             for key, value in all_metrics[f'{dataset_name}_{metrics_name}@{k}'].items():
    #                                 # 记录详细指标到wandb
    #                                 wandb_log_dict[f'{dataset_name}_{key}_{metrics_name}@{k}'] = value
    #                                 write_info += f"{dataset_name}_{key}_{metrics_name}@{k}: {value} | "
                            
    #                 # 使用wandb记录所有指标，step参数用于标识当前步骤
    #                 self.wandb.log(wandb_log_dict, step=current_step)

    #                 info += '\n'
    #                 write_info += '\n'
    #                 self.print(info)
    #                 with open(self.results_folder / 'log.txt', 'a') as f:
    #                     f.write(info)
    #                 def convert_to_serializable(obj):
    #                     """将非JSON可序列化类型转换为可序列化类型"""
    #                     import numpy as np
                        
    #                     if isinstance(obj, (np.integer, np.floating, np.bool_)):
    #                         # 处理numpy标量类型
    #                         return obj.item()
    #                     elif isinstance(obj, (np.ndarray, torch.Tensor)):
    #                         # 处理数组和张量
    #                         return obj.tolist()
    #                     elif isinstance(obj, dict):
    #                         # 递归处理字典
    #                         return {k: convert_to_serializable(v) for k, v in obj.items()}
    #                     elif isinstance(obj, (list, tuple)):
    #                         # 递归处理列表或元组
    #                         return [convert_to_serializable(i) for i in obj]
    #                     else:
    #                         # 返回原始值
    #                         return obj
                            
    #                 with open(self.results_folder / f'{current_step}.json', 'w') as f:
    #                     json.dump(convert_to_serializable(all_metrics), f, indent=4)
                        
    #                 print(f'** EVAL ** Log Done!')
                    
    #             self._reset_metrics()

    def evaluate(self, current_step):
        with torch.no_grad():
            self.CTClip.eval()
            temp_dataset_names = [name for name in self.dataset_names]

            if torch.distributed.is_initialized():
                world_size = torch.distributed.get_world_size()
                gathered_metrics = [None for _ in range(world_size)]

                losses = {}
                for dataset_name in temp_dataset_names:
                    losses.update({
                        f'{dataset_name}_main': (self.loss_m[dataset_name].sum, self.loss_m[dataset_name].count),
                        f'{dataset_name}_triplet': (self.triplet_loss_m[dataset_name].sum, self.triplet_loss_m[dataset_name].count),
                        f'{dataset_name}_binary': (self.binary_loss_m[dataset_name].sum, self.binary_loss_m[dataset_name].count),
                        f'{dataset_name}_rerank': (self.rerank_loss_m[dataset_name].sum, self.rerank_loss_m[dataset_name].count),
                        f'{dataset_name}_infoNCE': (self.infoNCE_loss_m[dataset_name].sum, self.infoNCE_loss_m[dataset_name].count),
                        f'{dataset_name}_anatomy_infoNCE': {k: (v.sum, v.count) for k, v in self.anatomy_infoNCE_loss_m[dataset_name].items()},
                        f'{dataset_name}_anatomy_triplet': {k: (v.sum, v.count) for k, v in self.anatomy_triplet_loss_m[dataset_name].items()},
                        f'{dataset_name}_anatomy_binary': {k: (v.sum, v.count) for k, v in self.anatomy_binary_loss_m[dataset_name].items()},
                        f'{dataset_name}_anatomy_rerank': {k: (v.sum, v.count) for k, v in self.anatomy_rerank_loss_m[dataset_name].items()},
                        f'{dataset_name}_anatomy_valid_triplet': {k: (v.sum, v.count) for k, v in self.anatomy_valid_triplet_count_m[dataset_name].items()},
                    })

                torch.distributed.all_gather_object(gathered_metrics, {'losses': losses})

                if self.accelerator.is_local_main_process:
                    def merge_loss(metric_name):
                        total_sum = sum(data['losses'][metric_name][0] for data in gathered_metrics)
                        total_count = sum(data['losses'][metric_name][1] for data in gathered_metrics)
                        return total_sum / total_count if total_count > 0 else 0

                    def merge_anatomy_loss(metric_name):
                        merged = defaultdict(lambda: [0, 0])
                        for data in gathered_metrics:
                            for anatomy, (s, c) in data['losses'][metric_name].items():
                                merged[anatomy][0] += s
                                merged[anatomy][1] += c
                        return {k: (v[0] / v[1] if v[1] > 0 else 0) for k, v in merged.items()}

                    global_main_loss = {d: merge_loss(f'{d}_main') for d in temp_dataset_names}
                    global_triplet_loss = {d: merge_loss(f'{d}_triplet') for d in temp_dataset_names}
                    global_infoNCE_loss = {d: merge_loss(f'{d}_infoNCE') for d in temp_dataset_names}
                    global_binary_loss = {d: merge_loss(f'{d}_binary') for d in temp_dataset_names}
                    global_rerank_loss = {d: merge_loss(f'{d}_rerank') for d in temp_dataset_names}

                    lr = self.optim.param_groups[0]['lr']
                    info = f"Step {current_step} | LR {lr} | temp {self.CTClip.module.log_temperature.exp()} | "
                    for dataset_name in temp_dataset_names:
                        info += (
                            f"{dataset_name} Loss {global_main_loss[dataset_name]:.4f} | "
                            f"{dataset_name} Triplet Loss {global_triplet_loss[dataset_name]:.4f} | "
                            f"{dataset_name} InfoNCE Loss {global_infoNCE_loss[dataset_name]:.4f} | "
                            f"{dataset_name} Binary Loss {global_binary_loss[dataset_name]:.4f} | "
                            f"{dataset_name} Rerank Loss {global_rerank_loss[dataset_name]:.4f} | "
                        )

                    info += '\n'
                    self.print(info)
                    with open(self.results_folder / 'log.txt', 'a') as f:
                        f.write(info)

            self._reset_metrics()

    def log_training_losses(self, current_step):
        """
        统计并记录训练过程中的各类损失
        Args:
            current_step: 当前训练步数
        """
        from collections import defaultdict
        
        temp_dataset_names = [name for name in self.dataset_names]
        
        if torch.distributed.is_initialized():
            world_size = torch.distributed.get_world_size()
            gathered_data_list = [None for _ in range(world_size)]
            
            # 构建包含损失统计量的数据结构
            losses = {}
            for dataset_name in temp_dataset_names:
                losses.update({
                    f'{dataset_name}_main': (self.loss_m[dataset_name].sum, self.loss_m[dataset_name].count),
                    f'{dataset_name}_triplet': (self.triplet_loss_m[dataset_name].sum, self.triplet_loss_m[dataset_name].count),
                    f'{dataset_name}_binary': (self.binary_loss_m[dataset_name].sum, self.binary_loss_m[dataset_name].count),
                    f'{dataset_name}_rerank': (self.rerank_loss_m[dataset_name].sum, self.rerank_loss_m[dataset_name].count),
                    f'{dataset_name}_infoNCE': (self.infoNCE_loss_m[dataset_name].sum, self.infoNCE_loss_m[dataset_name].count),
                    f'{dataset_name}_anatomy_infoNCE': {k: (v.sum, v.count) for k, v in self.anatomy_infoNCE_loss_m[dataset_name].items()},
                    f'{dataset_name}_anatomy_triplet': {k: (v.sum, v.count) for k, v in self.anatomy_triplet_loss_m[dataset_name].items()},
                    f'{dataset_name}_anatomy_binary': {k: (v.sum, v.count) for k, v in self.anatomy_binary_loss_m[dataset_name].items()},
                    f'{dataset_name}_anatomy_rerank': {k: (v.sum, v.count) for k, v in self.anatomy_rerank_loss_m[dataset_name].items()},
                    f'{dataset_name}_anatomy_valid_triplet': {k: (v.sum, v.count) for k, v in self.anatomy_valid_triplet_count_m[dataset_name].items()}
                })
            
            # 执行收集
            torch.distributed.all_gather_object(gathered_data_list, {'losses': losses})
            
            # 仅在主进程处理
            if self.accelerator.is_local_main_process:
                print(f'** Loss Logging ** Gather Done!')
                
                # 定义合并函数
                def merge_loss(metric_name):
                    total_sum = sum(data['losses'][metric_name][0] for data in gathered_data_list)
                    total_count = sum(data['losses'][metric_name][1] for data in gathered_data_list)
                    return total_sum / total_count if total_count > 0 else 0

                def merge_anatomy_loss(metric_name):
                    merged = defaultdict(lambda: [0, 0])  # [sum, count]
                    for data in gathered_data_list:
                        for anatomy, (s, c) in data['losses'][metric_name].items():
                            merged[anatomy][0] += s
                            merged[anatomy][1] += c
                    return {k: (v[0]/v[1] if v[1]>0 else 0) for k, v in merged.items()}
                
                # 计算全局平均损失
                global_main_loss = {dataset_name: merge_loss(f'{dataset_name}_main') for dataset_name in temp_dataset_names}
                global_triplet_loss = {dataset_name: merge_loss(f'{dataset_name}_triplet') for dataset_name in temp_dataset_names}
                global_infoNCE_loss = {dataset_name: merge_loss(f'{dataset_name}_infoNCE') for dataset_name in temp_dataset_names}
                global_binary_loss = {dataset_name: merge_loss(f'{dataset_name}_binary') for dataset_name in temp_dataset_names}
                global_rerank_loss = {dataset_name: merge_loss(f'{dataset_name}_rerank') for dataset_name in temp_dataset_names}
                global_anatomy_infoNCE = {dataset_name: merge_anatomy_loss(f'{dataset_name}_anatomy_infoNCE') for dataset_name in temp_dataset_names}
                global_anatomy_triplet = {dataset_name: merge_anatomy_loss(f'{dataset_name}_anatomy_triplet') for dataset_name in temp_dataset_names}
                global_anatomy_binary = {dataset_name: merge_anatomy_loss(f'{dataset_name}_anatomy_binary') for dataset_name in temp_dataset_names}
                global_anatomy_rerank = {dataset_name: merge_anatomy_loss(f'{dataset_name}_anatomy_rerank') for dataset_name in temp_dataset_names}
                global_anatomy_valid_triplet = {dataset_name: merge_anatomy_loss(f'{dataset_name}_anatomy_valid_triplet') for dataset_name in temp_dataset_names}

                lr = self.optim.param_groups[0]['lr']
                
                # 构建信息输出
                info = f"Step {current_step} | LR {lr} | "
                for dataset_name in temp_dataset_names:
                    info += f"{dataset_name} Loss {global_main_loss[dataset_name]:.4f} | "\
                            f"{dataset_name} Triplet Loss {global_triplet_loss[dataset_name]:.4f} | "\
                            f"{dataset_name} InfoNCE Loss {global_infoNCE_loss[dataset_name]:.4f} | "\
                            f"{dataset_name} Binary Loss {global_binary_loss[dataset_name]:.4f} | "\
                            f"{dataset_name} Rerank Loss {global_rerank_loss[dataset_name]:.4f} | "
                
                # 创建Wandb日志字典
                wandb_log_dict = {'lr': lr}
                
                if current_step > 0:
                    for dataset_name in temp_dataset_names:
                        # 记录训练损失
                        wandb_log_dict[f'{dataset_name}_train_loss'] = self.loss_m[dataset_name].avg
                        wandb_log_dict[f'{dataset_name}_infoNCE_loss'] = self.infoNCE_loss_m[dataset_name].avg
                        wandb_log_dict[f'{dataset_name}_triplet_loss'] = self.triplet_loss_m[dataset_name].avg
                        wandb_log_dict[f'{dataset_name}_binary_loss'] = self.binary_loss_m[dataset_name].avg
                        wandb_log_dict[f'{dataset_name}_rerank_loss'] = self.rerank_loss_m[dataset_name].avg
                        
                        # 记录解剖结构损失
                        for anatomy, loss in global_anatomy_infoNCE[dataset_name].items():
                            wandb_log_dict[f'{dataset_name}_{anatomy}_infoNCE_loss'] = loss
                        for anatomy, loss in global_anatomy_triplet[dataset_name].items():
                            wandb_log_dict[f'{dataset_name}_{anatomy}_triplet_loss'] = loss
                        for anatomy, loss in global_anatomy_binary[dataset_name].items():
                            wandb_log_dict[f'{dataset_name}_{anatomy}_binary_loss'] = loss
                        for anatomy, loss in global_anatomy_rerank[dataset_name].items():
                            wandb_log_dict[f'{dataset_name}_{anatomy}_rerank_loss'] = loss
                        for anatomy, count in global_anatomy_valid_triplet[dataset_name].items():
                            wandb_log_dict[f'{dataset_name}_{anatomy}_valid_triplet'] = count
                
                # 记录到wandb
                self.wandb.log(wandb_log_dict, step=current_step)
                
                # 打印和写入日志
                info += '\n'
                self.print(info)
                with open(self.results_folder / 'log.txt', 'a') as f:
                    f.write(info)
                
                print(f'** Loss Logging ** Done!')
                
            # 重置指标
            self._reset_metrics()
        else:
            # 非分布式环境下的简单记录
            lr = self.optim.param_groups[0]['lr']
            wandb_log_dict = {'lr': lr}
            
            if current_step > 0:
                for dataset_name in temp_dataset_names:
                    wandb_log_dict[f'{dataset_name}_train_loss'] = self.loss_m[dataset_name].avg
                    wandb_log_dict[f'{dataset_name}_infoNCE_loss'] = self.infoNCE_loss_m[dataset_name].avg
                    wandb_log_dict[f'{dataset_name}_triplet_loss'] = self.triplet_loss_m[dataset_name].avg
                    wandb_log_dict[f'{dataset_name}_binary_loss'] = self.binary_loss_m[dataset_name].avg
                    wandb_log_dict[f'{dataset_name}_rerank_loss'] = self.rerank_loss_m[dataset_name].avg
            
            self.wandb.log(wandb_log_dict, step=current_step)
            self._reset_metrics()

    def evaluate_uncon(self, current_step,is_trainset=False):
        
        accelerator = self.accelerator  # 假设你已经有accelerator
        device = accelerator.device
        self.CTClip.eval()
        if is_trainset:
            prex =  'part_trainset'
        else:
            prex = 'validset'

        for i,dataset_name in enumerate(self.uncon_dataset_names):
            text_latents_all = []
            image_latents_all = []
            all_sample_idls = []
            # valid_dl = self.uncon_valid_dls[dataset_name]
            if is_trainset == False:
                valid_sampler = DistributedSampler(self.uncon_valid_dss[dataset_name], shuffle=False)
                valid_dl  = DataLoader(
                        self.uncon_valid_dss[dataset_name], 
                        sampler=valid_sampler,
                        num_workers = 4,
                        batch_size = self.uncon_batch_size_valid[i],
                        shuffle = False,
                        drop_last = True,
                    )
            else:
                valid_sampler = DistributedSampler(self.uncon_train_valid_dss[dataset_name], shuffle=False)
                valid_dl  = DataLoader(
                        self.uncon_train_valid_dss[dataset_name], 
                        sampler=valid_sampler,
                        num_workers = 4,
                        batch_size = self.uncon_batch_size_valid[i],
                        shuffle = False,
                        drop_last = True,
                    )
            with torch.no_grad(), torch.cuda.amp.autocast():
                print('开始加载无条件数据集:', dataset_name)
                if not self.modal_embedding:
                    for video, text, sample_idls in tqdm(valid_dl, disable=not accelerator.is_local_main_process):
                        video = video.to(device)
                        text = list(text)
                        # 切换完text encoder后可能没有什么特别大的变化，FIXME: 这里可能会有新的bug
                        text_tokens = self.tokenizer(
                            text,
                            return_tensors="pt",
                            padding=True,
                            truncation=True,
                            max_length=512
                        ).to(device)
                        text_latents, image_latents, temp = self.CTClip(
                            text_tokens, video, return_latents=True, device=device,is_condition=False
                        )
                        text_latents_all.append(text_latents.detach())
                        image_latents_all.append(image_latents.detach())
                        all_sample_idls.append(sample_idls.detach())
                else:
                    for video, text, sample_idls ,modal_indexs in tqdm(valid_dl, disable=not accelerator.is_local_main_process):
                        video = video.to(device)
                        text = list(text)
                        text_tokens = self.tokenizer(
                            text,
                            return_tensors="pt",
                            padding=True,
                            truncation=True,
                            max_length=512
                        ).to(device)
                        text_latents, image_latents, temp = self.CTClip(
                            text_tokens, video, return_latents=True, device=device,is_condition=False,modal_embedding=True,modal_indexs=modal_indexs
                        )
                        text_latents_all.append(text_latents.detach())
                        image_latents_all.append(image_latents.detach())
                        all_sample_idls.append(sample_idls.detach())
            # 拼接本地的latents
            text_latents_all = torch.cat(text_latents_all, dim=0)
            image_latents_all = torch.cat(image_latents_all, dim=0)
            all_sample_idls = torch.cat(all_sample_idls)
            # print('all_sample_idls:', all_sample_idls)
            if not isinstance(all_sample_idls, torch.Tensor):
                all_sample_idls = torch.tensor(all_sample_idls, device=self.accelerator.device)
            elif all_sample_idls.device.type != 'cuda':
                all_sample_idls = all_sample_idls.to(self.accelerator.device)

            # 确保是密集张量
            if all_sample_idls.is_sparse:
                all_sample_idls = all_sample_idls.to_dense()
            # print(all_sample_idls)
            # 收集所有进程的latents
            text_latents_gathered = accelerator.gather_for_metrics(text_latents_all)
            image_latents_gathered = accelerator.gather_for_metrics(image_latents_all)
            all_sample_idls = accelerator.gather_for_metrics(all_sample_idls)
            # print('all_sample_idls', all_sample_idls)
            
            # print('text_latents_gathered', text_latents_gathered[0:5])
            # print('image_latents_gathered', image_latents_gathered[0:5])
            # 只在主进程上做日志和评测
            if accelerator.is_local_main_process:
                if is_trainset == False:
                    lr = self.optim.param_groups[0]['lr']
                    info = f"Unconditio Step {current_step} | LR {lr} | Loss {self.uncon_loss_m[dataset_name].val:.4f}({self.uncon_loss_m[dataset_name].avg:.4f}) | IT Triplet Loss {self.it_triplet_loss_m[dataset_name].val:.4f}({self.it_triplet_loss_m[dataset_name].avg:.4f}) | II Triplet Loss {self.ii_triplet_loss_m[dataset_name].val:.4f}({self.ii_triplet_loss_m[dataset_name].avg:.4f}) | IT infoNCE Loss {self.it_infoNCE_loss_m[dataset_name].val:.4f}({self.it_infoNCE_loss_m[dataset_name].avg:.4f})\n"
                    
                    # 创建wandb日志字典
                    wandb_log_dict = {
                        'lr': lr,
                        'uncon_train_loss': self.uncon_loss_m[dataset_name].avg,
                        'uncon_image_text_triplet_loss': self.it_triplet_loss_m[dataset_name].avg,
                        'uncon_image_image_triplet_loss': self.ii_triplet_loss_m[dataset_name].avg,
                        'uncon_image_text_infoNCE_loss': self.it_infoNCE_loss_m[dataset_name].avg
                    }
                    
                    if current_step > 0:
                        # 使用wandb替代tensorboard
                        pass  # 已经将要记录的指标添加到了wandb_log_dict中
                        
                    self.uncon_loss_m[dataset_name].reset()
                    self.it_triplet_loss_m[dataset_name].reset()
                    self.ii_triplet_loss_m[dataset_name].reset()
                    self.it_infoNCE_loss_m[dataset_name].reset()
                else:
                    info = ''
                    wandb_log_dict = {}

                # image-report hard retrieval
                image_to_text = einsum('m d, n d -> m n', image_latents_gathered, text_latents_gathered)
                image_to_text = image_to_text.cpu()
                gt_matrix = torch.eye(image_to_text.shape[0])
                results = hard_retrieval(image_to_text, gt_matrix, k=[1, 5, 10, 20, 50, 100])
                for k, v in zip([1, 5, 10, 20, 50, 100], results):
                    info += f'{prex}_{dataset_name}_(I2T)Hard_R@{k}: {v*100:.2f} | '
                    wandb_log_dict[f'{prex}_{dataset_name}_R@{k}'] = v
                info += '\n'
                print('image_to_text shape:', image_to_text.shape)
                sample_idls = all_sample_idls.cpu().numpy() if isinstance(all_sample_idls, torch.Tensor) else all_sample_idls
                if is_trainset == False:
                    similarity_tab = self.uncon_similarity_lookup_table_valid[dataset_name][np.ix_(sample_idls, sample_idls)]
                else:
                    similarity_tab = self.uncon_similarity_lookup_table_train_valid[dataset_name][np.ix_(sample_idls, sample_idls)]
                    
                similarity_tab = torch.from_numpy(similarity_tab) if isinstance(similarity_tab, np.ndarray) else similarity_tab
                similarity_tab = (similarity_tab.to(torch.float32) * 0.01).to(device, non_blocking=True)
                print('similarity_tab shape:', similarity_tab.shape)
                results = compute_ndcg_uncon(image_to_text, similarity_tab, k=[1, 5, 10, 20, 50, 100])
                for k, v in zip([1, 5, 10, 20, 50, 100], results):
                    info += f'{prex}_{dataset_name}_(I2T)NDCG@{k}: {v*100:.2f} | '
                    wandb_log_dict[f'{prex}_{dataset_name}_(I2T)NDCG@{k}'] = v
                info += '\n'

                results = soft_retrieval_uncon(image_to_text,similarity_tab, k=[1, 5, 10, 20, 50, 100])
                for k, v in zip([1, 5, 10, 20, 50, 100], results):
                    info += f'{prex}_{dataset_name}_(I2T)Soft_R@{k}: {v*100:.2f} | '
                    wandb_log_dict[f'{prex}_{dataset_name}_(I2T)Soft_R@{k}'] = v
                info += '\n'
                index_list = [i for i in range(similarity_tab.shape[0])]
                results = soft_retrieval_exclude_self(image_to_text,similarity_tab,index_list, k=[1, 5, 10, 20, 50, 100])
                print('dataset_name:', dataset_name,', results:', results)
                for k, v in zip([1, 5, 10, 20, 50, 100], results):
                    info += f'{prex}_{dataset_name}_(I2T)Soft_exclude_R@{k}: {v*100:.2f} | '
                    wandb_log_dict[f'{prex}_{dataset_name}_(I2T)Soft_exclude_R@{k}'] = v
                info += '\n'
                # image-image
                if self.soft_label:
                    image_to_image = einsum('m d, n d -> m n', image_latents_gathered, image_latents_gathered)
                    image_to_image = image_to_image.cpu()
                    if 'amos' in dataset_name.lower():  # 直接抛弃了这部分
                        rows_to_remove = get_rows_to_remove(similarity_tab.cpu().numpy(),threshold=0.75,min_count=1)
                    else:
                        rows_to_remove = get_rows_to_remove(similarity_tab.cpu().numpy(),threshold=0.9,min_count=1)
                    if len(rows_to_remove) > 0:
                        # 处理相似度矩阵
                        similarity_tab_processed = remove_rows_columns(similarity_tab.cpu().numpy(), rows_to_remove)
                        similarity_tab_processed = torch.from_numpy(similarity_tab_processed)
                        
                        # 处理预测矩阵
                        image_to_image_processed = remove_rows_columns(image_to_image.numpy(), rows_to_remove)
                        image_to_image_processed = torch.from_numpy(image_to_image_processed)
                        
                        # 更新index_list
                        index_list_processed = [i for i in range(len(similarity_tab_processed)) ]
                        
                        print(f"I2I处理后的矩阵形状: {image_to_image_processed.shape}")
                    else:
                        similarity_tab_processed = similarity_tab
                        image_to_image_processed = image_to_image
                        index_list_processed = index_list
                    similarity_tab = similarity_tab_processed
                    image_to_image = image_to_image_processed
                    index_list = index_list_processed
                    pred_score, upperbound_score = RateScore_retrieval(image_to_image, similarity_tab, k=[1, 5, 10, 20, 50, 100])
                    for k, v in zip([1, 5, 10, 20, 50, 100], pred_score):
                        info += f'{prex}_Uncon_{dataset_name}_RateScore@{k}: {v*100:.2f} | '
                        wandb_log_dict[f'{prex}_Uncon_{dataset_name}_RateScore@{k}'] = v
                    for k, v in zip([1, 5, 10, 20, 50, 100], upperbound_score):
                        info += f'{prex}_Uncon_{dataset_name}_Upperbound_RateScore@{k}: {v*100:.2f} | '
                        wandb_log_dict[f'{prex}_Uncon_{dataset_name}_Upperbound_RateScore@{k}'] = v
                    info += '\n'

                
                    ndcg_scores = compute_ndcg_exclude_self(image_to_image, similarity_tab, index_list, k=[1, 5, 10, 20, 50, 100])
                    for k, v in zip([1, 5, 10, 20, 50, 100], ndcg_scores):
                        info += f'{prex}_{dataset_name}_(I2I)NDCG@{k}: {v*100:.2f} | '
                        wandb_log_dict[f'{prex}_Uncon_{dataset_name}_NDCG@{k}'] = v
                    info += '\n'

                    recall_scores = soft_retrieval_exclude_self(image_to_image, similarity_tab, index_list, k=[1, 5, 10, 20, 50, 100])
                    for k, v in zip([1, 5, 10, 20, 50, 100], recall_scores):
                        info += f'{prex}_{dataset_name}_(I2I)Soft_R@{k}: {v*100:.2f} | '
                        wandb_log_dict[f'{prex}_{dataset_name}_Uncon_(I2I)Soft_R@{k}'] = v
                    info += '\n'

                # 使用wandb记录所有指标
                self.wandb.log(wandb_log_dict, step=current_step)
                    
                self.print(info)
                with open(self.results_folder / 'log.txt', 'a') as f:
                    f.write(info)
            # 清理显存
            del text_latents_all, image_latents_all, text_latents_gathered, image_latents_gathered
            torch.cuda.empty_cache()
            self._reset_metrics_uncon()

    def save(self, path):
        if not self.accelerator.is_local_main_process:
            return

        pkg = dict(
            model=self.accelerator.get_state_dict(self.CTClip),
            optim=self.optim.state_dict(),
            step=self.current_step
        )
        torch.save(pkg, path)

    def resume_training(self, path):
        self.print(f'** MODEL ** Resume Training from {path}')
            
        path = Path(path)
        assert path.exists()
        pkg = torch.load(path, map_location='cpu',weights_only=False)

        CTClip = self.accelerator.unwrap_model(self.CTClip)
        CTClip.load_state_dict(pkg['model'])
        # self.optim.load_state_dict(pkg['optim'])
        self.current_step = pkg['step'] + 1
        
    # def load_checkpoint(self, path, allow_partial_load):
    #     self.print(f'** MODEL ** Load Checkpoint from {path}')
            
    #     path = Path(path)
    #     assert path.exists()
    #     pkg = torch.load(path, map_location='cpu', weights_only=False)

    #     CTClip = self.accelerator.unwrap_model(self.CTClip)
        
    #     if allow_partial_load:
    #         model_dict = CTClip.state_dict()
    #         if 'model' in pkg:
    #             pkg_state_dict = pkg['model']  # 假设 pkg['model'] 是模型的状态字典
    #         else:
    #             pkg_state_dict = pkg
    #         # 检查差异
    #         unexpected_state_dict = [k for k in pkg_state_dict.keys() if k not in model_dict.keys()]
    #         missing_state_dict = [k for k in model_dict.keys() if k not in pkg_state_dict.keys()]
    #         unmatchd_state_dict = [k for k, v in pkg_state_dict.items() if k in model_dict.keys() and v.shape != model_dict[k].shape]

    #         # 加载部分参数
    #         state_dict = {k: v for k, v in pkg_state_dict.items() if k in model_dict.keys() and v.shape == model_dict[k].shape}
    #         model_dict.update(state_dict)
    #         CTClip.load_state_dict(model_dict)

    #         self.print('** MODEL ** The following parameters are UNEXPECTED in checkpoint:\n')
    #         self.print(unexpected_state_dict)
    #         self.print('** MODEL ** The following parameters are MISSING in checkpoint:\n')
    #         self.print(missing_state_dict)
    #         self.print('** MODEL ** The following parameters have DIFFERENT SHAPES in checkpoint:\n')
    #         self.print(unmatchd_state_dict)
    #         self.print('** MODEL ** The following parameters are LOADED in:\n')
    #         self.print(state_dict.keys())
    #     else:
    #         CTClip.load_state_dict(pkg['model'])


    def load_checkpoint(self, path, allow_partial_load):
        self.print(f'** MODEL ** Load Checkpoint from {path}')
            
        path = Path(path)
        assert path.exists()
        pkg = torch.load(path, map_location='cpu', weights_only=False)

        CTClip = self.accelerator.unwrap_model(self.CTClip)
        
        if allow_partial_load:
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
                                    self.print(f'** MODEL ** Copy expert weight: {source_key} -> {key}')
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

            self.print('** MODEL ** The following parameters are UNEXPECTED in checkpoint:\n')
            self.print(unexpected_state_dict)
            self.print('** MODEL ** The following parameters are MISSING in checkpoint:\n')
            self.print(missing_state_dict)
            self.print('** MODEL ** The following parameters have DIFFERENT SHAPES in checkpoint:\n')
            self.print(unmatchd_state_dict)
            self.print('** MODEL ** The following parameters are LOADED in:\n')
            self.print(state_dict.keys())
        else:
            # 完整加载时也需要处理MOE专家权重
            model_dict = CTClip.state_dict()
            pkg_state_dict = pkg['model'] if 'model' in pkg else pkg
            
            # 处理MOE专家权重的复制逻辑
            processed_state_dict = {}
            expert_copy_map = {3: 0, 4: 1}
            
            for key, value in pkg_state_dict.items():
                if '.experts.' in key:
                    parts = key.split('.')
                    for i, part in enumerate(parts):
                        if part == 'experts' and i + 1 < len(parts) and parts[i+1].isdigit():
                            expert_idx = int(parts[i+1])
                            if expert_idx in expert_copy_map:
                                source_expert = expert_copy_map[expert_idx]
                                source_parts = parts.copy()
                                source_parts[i+1] = str(source_expert)
                                source_key = '.'.join(source_parts)
                                
                                if source_key in pkg_state_dict:
                                    processed_state_dict[key] = pkg_state_dict[source_key]
                                    self.print(f'** MODEL ** Copy expert weight: {source_key} -> {key}')
                                    continue
                processed_state_dict[key] = value
            
            CTClip.load_state_dict(processed_state_dict)
        
    def print(self, msg):
        self.accelerator.print(msg)

    @property
    def is_main(self):
        # return self.accelerator.is_main_process
        return self.accelerator.is_local_main_process

    def train_step(self):
        # 添加时间统计
        if self.is_main:
            import time
            step_start_time = time.time()
            time_stats = {
                'data_loading': 0.0,
                'model_forward': 0.0,
                'backward': 0.0,
                'optimizer_step': 0.0,
                'total': 0.0
            }

        device = self.device
        mp = self.accelerator.mixed_precision
        
        current_step = self.current_step
        self.lr_scheduler(current_step)
        if hasattr(self.CTClip, 'module'):
            model = self.CTClip.module
        else:
            model = self.CTClip
        # 忽略这个，做一个reranker的结果展示，08也是这样训练出来吧
        progress = current_step / max(self.num_train_steps, 1)  # 0 -> 1
        tau_start, tau_end = 0.1, 0.01
        # model.smooth_ap_tau = tau_start + (tau_end - tau_start) * progress
        model.smooth_ap_tau = tau_end + 0.5 * (tau_start - tau_end) * (1 + math.cos(math.pi * progress))  # cosine decay
        print('model.smooth_ap_tau',model.smooth_ap_tau)
        self.CTClip.train()

        # update CTClip model
        # 有两种方案，一个是每个step同时做2D和3D的训练。另一个是每个step只做2D或3D的训练，但是要用到梯度累积。
        for stage, dataset_name, dl_iter in self.batch_generators:
            with self.accelerator.accumulate(self.CTClip):
                if stage == 'stage1':
                    print(f"pid={os.getpid()} ",'rank', torch.distributed.get_rank(),'dataset_name', dataset_name, 'step', current_step)
                    # ▸ 让 Accelerate 决定什么时候同步梯度 / clip / step
                    if self.debug:
                        print_gpu_memory_stats(self.accelerator,'Before video, similarity_tab, anatomy, _ = next(self.uncon_dl_iters[dataset_name])')
                    
                    # ---- 数据准备 ----
                    if self.is_main:
                        data_load_start = time.time()
                    if not self.modal_embedding:
                        video, text, sample_idls = next(dl_iter)
                    else:
                        video, text, sample_idls, modal_indexs = next(dl_iter)
                    video = video.to(device)
                    video = self.augmentor(video)
                    # 直接半精度进 GPU，减少显存与带宽
                    tgt_dtype = torch.float16 if mp == "fp16" else (
                                torch.bfloat16 if mp == "bf16" else torch.float32)

                    video = video.to(device, dtype=tgt_dtype, non_blocking=True)
                    if self.debug:
                        print_gpu_memory_stats(self.accelerator,'before tokenizer')
                    with torch.no_grad():  # tokenizer不需要梯度
                        text_tokens = self.tokenizer(list(text),return_tensors="pt", padding=True, truncation=True, max_length=512).to(device, non_blocking=True)
                    if self.soft_label:
                        sample_idls = sample_idls.cpu().numpy() if isinstance(sample_idls, torch.Tensor) else sample_idls
                        similarity_tab = self.uncon_similarity_lookup_table_train[dataset_name][np.ix_(sample_idls, sample_idls)]
                        similarity_tab = torch.from_numpy(similarity_tab) if isinstance(similarity_tab, np.ndarray) else similarity_tab
                        similarity_tab = (similarity_tab.to(torch.float32) * 0.01).to(
                                    device, dtype=tgt_dtype, non_blocking=True)
                    else:
                        similarity_tab = torch.eye(len(sample_idls), dtype=tgt_dtype, device=device)
                    
                    if self.is_main:
                        time_stats['data_loading'] += time.time() - data_load_start
                        model_forward_start = time.time()
                        print('stage', stage, 'dataset_name', dataset_name, 'data_load_time', time_stats['data_loading'])
                    
                    with self.accelerator.autocast():
                        loss, image_text_triplet_loss, image_text_infoNCE_loss, image_to_image_triplet_loss = self.CTClip(
                            text_tokens,
                            video,
                            gt_similarity_matrix=similarity_tab,
                            return_loss=True,
                            device=device,
                            is_condition=False,
                            modal_embedding=self.modal_embedding,
                            modal_indexs=modal_indexs if self.modal_embedding else None
                        )
                        print('rank: ', torch.distributed.get_rank(),'dataset_name:' , dataset_name," loss: ", loss,"device:", device)
                    
                    if self.is_main:
                        time_stats['model_forward'] += time.time() - model_forward_start
                        backward_start = time.time()
                        print('stage', stage, 'dataset_name', dataset_name, 'model_forward_time', time_stats['model_forward'])
                    
                    del video, text_tokens, similarity_tab
                    if dataset_name == 'CT_RATE':
                        weight_loss = 2
                    elif dataset_name == 'MIMIC-CXR':
                        weight_loss = 1
                    else:
                        weight_loss =1 
                    self.accelerator.backward(loss*weight_loss)                 # 自动 / grad_acc_steps 
                    
                    if self.is_main:
                        time_stats['backward'] += time.time() - backward_start
                        print('stage', stage, 'dataset_name', dataset_name, 'backward_time', time_stats['backward'])
                    
                    # ---- 立即把日志数据转到 CPU，释放 GPU 显存 ----
                    loss_val        = float(loss.detach().cpu())
                    image_text_triplet_mean = float(image_text_triplet_loss.mean().detach().cpu())
                    image_text_infoNCE_mean = float(image_text_infoNCE_loss.mean().detach().cpu())
                    image_to_image_triplet_mean = float(image_to_image_triplet_loss.mean().detach().cpu())
                    del loss, image_text_triplet_loss, image_text_infoNCE_loss, image_to_image_triplet_loss
                    self.uncon_loss_m[dataset_name].update(loss_val, 1)
                    self.it_triplet_loss_m[dataset_name].update(image_text_triplet_mean, 1)
                    self.it_infoNCE_loss_m[dataset_name].update(image_text_infoNCE_mean, 1)
                    self.ii_triplet_loss_m[dataset_name].update(image_to_image_triplet_mean, 1)
                        
                if stage == 'stage2':
                    print(f"pid={os.getpid()} ",'rank', torch.distributed.get_rank(),'dataset_name', dataset_name, 'step', current_step)
                    # ▸ 让 Accelerate 决定什么时候同步梯度 / clip / step
                    if self.debug:
                        print_gpu_memory_stats(self.accelerator,'Before video, similarity_tab, anatomy, _ = next(self.dl_iters[dataset_name])')
                    
                    # ---- 数据准备 ----
                    if self.is_main:
                        data_load_start = time.time()

                    video, similarity_tab, anatomy, _, modal_indexs, local_text, labels = next(dl_iter) # 选择一个元素即可
                    video = video.to(device)
                    video = self.augmentor(video)
                    # 直接半精度进 GPU，减少显存与带宽
                    tgt_dtype = torch.float16 if mp == "fp16" else (
                                torch.bfloat16 if mp == "bf16" else torch.float32)
                    # print('anatomy:', anatomy,'video shape:', video.shape, 'similarity_tab shape:', similarity_tab.shape)
                    video  = video.to(device, dtype=tgt_dtype, non_blocking=True)
                    similarity_tab = (similarity_tab.to(torch.float32) * 0.01).to(
                                    device, dtype=tgt_dtype, non_blocking=True)
                    if self.debug:
                        print_gpu_memory_stats(self.accelerator,'before tokenizer')

                    with torch.no_grad():  # tokenizer不需要梯度
                        # 修改text_tokens的部分
                        condition_index = [self.condition_index[dataset_name][condition] for condition in anatomy]
                        condition_index = torch.tensor(condition_index).to(device, non_blocking=True)
                        local_text_tokens = self.tokenizer(
                            list(local_text),
                            return_tensors="pt",
                            padding=True,
                            truncation=True,
                            max_length=512
                        ).to(device, non_blocking=True)
                    if self.is_main:
                        time_stats['data_loading'] += time.time() - data_load_start
                        model_forward_start = time.time()
                        print('stage', stage, 'dataset_name', dataset_name, 'data_load_time', time_stats['data_loading'])

                    # ---- 前向 & 反向 ----
                    with self.accelerator.autocast():
                        if self.debug:
                            print_gpu_memory_stats(self.accelerator,'before self.CTClip')
                        loss, infoNCE_loss, triplet_loss,binary_loss ,rerank_loss,valid_triplet_cnt = self.CTClip(
                            condition_index,
                            video,
                            gt_similarity_matrix=similarity_tab,
                            return_loss=True,
                            device=device,
                            is_condition=True,
                            local_text=local_text_tokens,
                            modal_embedding=self.modal_embedding,
                            modal_indexs=modal_indexs if self.modal_embedding else None,
                            abnormal_exist_labels=labels    # 计算accuracy的值
                        )
                    
                    if self.is_main:
                        time_stats['model_forward'] += time.time() - model_forward_start
                        print('stage', stage, 'dataset_name', dataset_name, 'model_forward_time', time_stats['model_forward'])
                    print('rank: ', torch.distributed.get_rank()," loss: ", loss)
                    if self.debug:            
                        print('rank: ', torch.distributed.get_rank(), "loss isnan:", torch.isnan(loss), "isinf:", torch.isinf(loss))
                    if self.debug:
                        print_gpu_memory_stats(self.accelerator,'after self.CTClip and before self.accelerator.backward')
                    
                    del video, similarity_tab
                    
                    if self.is_main:
                        backward_start = time.time()
                    
                    self.accelerator.backward(loss)                 # 自动 / grad_acc_steps
                    
                    if self.is_main:
                        time_stats['backward'] += time.time() - backward_start
                        print('stage', stage, 'dataset_name', dataset_name, 'backward_time', time_stats['backward'])
                    # ---- 立即把日志数据转到 CPU，释放 GPU 显存 ----
                    loss_val        = float(loss.detach().cpu())
                    triplet_mean    = float(triplet_loss.mean().detach().cpu())
                    infoNCE_mean    = float(infoNCE_loss.mean().detach().cpu())
                    binary_mean     = float(binary_loss.mean().detach().cpu())
                    rerank_mean     = float(rerank_loss.mean().detach().cpu())
                    del loss
                    self.loss_m[dataset_name].update(loss_val, 1)
                    self.triplet_loss_m[dataset_name].update(triplet_mean, 1)
                    self.infoNCE_loss_m[dataset_name].update(infoNCE_mean, 1)
                    self.binary_loss_m[dataset_name].update(binary_mean, 1)
                    self.rerank_loss_m[dataset_name].update(rerank_mean, 1)
                    for i, name in enumerate(anatomy):
                        tl = float(triplet_loss[i].detach().cpu())
                        bl = float(binary_loss[i].detach().cpu())
                        rl = float(rerank_loss[i].detach().cpu())
                        il = float(infoNCE_loss[i].detach().cpu())
                        vc = int(valid_triplet_cnt)

                        if name not in self.anatomy_infoNCE_loss_m[dataset_name]:
                            self.anatomy_infoNCE_loss_m[dataset_name][name]          = AverageMeter()
                            self.anatomy_triplet_loss_m[dataset_name][name]          = AverageMeter()
                            self.anatomy_binary_loss_m[dataset_name][name]           = AverageMeter()
                            self.anatomy_rerank_loss_m[dataset_name][name]           = AverageMeter()
                            self.anatomy_valid_triplet_count_m[dataset_name][name]   = AverageMeter()

                        if il > 0:
                            self.anatomy_infoNCE_loss_m[dataset_name][name].update(il, 1)
                        if tl > 0:
                            self.anatomy_triplet_loss_m[dataset_name][name].update(tl, 1)
                            self.anatomy_valid_triplet_count_m[dataset_name][name].update(vc, 1)
                        if bl > 0:
                            self.anatomy_binary_loss_m[dataset_name][name].update(bl, 1)
                        if rl > 0:
                            self.anatomy_rerank_loss_m[dataset_name][name].update(rl, 1)
                    # 显式删除局部大张量，帮助 Python 及时回收
                    del triplet_loss, infoNCE_loss, binary_loss, rerank_loss
            
                torch.cuda.empty_cache()
                # ---- 梯度同步步：clip / step / zero_grad ----
                if self.accelerator.sync_gradients:
                    if self.max_grad_norm is not None:
                        self.accelerator.clip_grad_norm_(self.CTClip.parameters(), self.max_grad_norm)
                    
                    if self.is_main:
                        optimizer_step_start = time.time()
                    
                    if self.debug:
                        print_gpu_memory_stats(self.accelerator,'before self.optim.step')
                    self.optim.step()
                    self.optim.zero_grad(set_to_none=True)  # 清除梯度，避免内存泄漏
                    
                    if self.is_main:
                        time_stats['optimizer_step'] += time.time() - optimizer_step_start
                    
                    torch.cuda.empty_cache()


        # save model every so often
        if not (current_step % self.save_model_every) and self.is_main:
            model_path = self.results_folder / f'CTClip.{current_step}.pt'
            self.save(model_path)
            self.print(f'Saving model to {str(self.results_folder)}')
        if self.stage2:
            if not (current_step % self.shuffle_train_samples_every):
                for i in range(len(self.dataset_names)):
                    dataset_name = self.dataset_names[i]
                    print(f"Rank {torch.distributed.get_rank()} | Step {current_step} | Shuffle Training Samples")
                    self.dss[dataset_name].prepare_anatomy_data()
                    # 重新创建 DataLoader
                    self.dls[dataset_name] = DataLoader(
                        self.dss[dataset_name],
                        num_workers=self.con_num_workers,
                        batch_size=self.batch_size[i],
                        shuffle=False,
                        drop_last=False,
                        pin_memory=self.pin_memory,
                        collate_fn=collate_fn
                    )
                    # 重新准备迭代器
                    self.dl_iters[dataset_name] = cycle(self.accelerator.prepare(self.dls[dataset_name]))
                self.gen_batch_generators()  # 重新生成 batch_generators
        
        # evaluate model every so often (ddp)
        if not (current_step % self.save_results_every):
            if self.stage1:
                self.evaluate_uncon(current_step=current_step)
                self.evaluate_uncon(current_step=current_step,is_trainset=True)
            if self.stage2:
                self.evaluate(current_step=current_step)
            if torch.distributed.is_initialized():
                torch.distributed.barrier()     
        elif not (current_step % 100):   # 每100个step统计一轮loss
        
            if torch.distributed.is_initialized():
                world_size = torch.distributed.get_world_size()
                gathered_metrics = [None for _ in range(world_size)]
                losses = {}
                if self.stage1:
                    for dataset_name in self.uncon_dataset_names:
                        losses.update({
                            f'{dataset_name}_uncon_main': (self.uncon_loss_m[dataset_name].sum, self.uncon_loss_m[dataset_name].count),
                            f'{dataset_name}_it_triplet': (self.it_triplet_loss_m[dataset_name].sum, self.it_triplet_loss_m[dataset_name].count),
                            f'{dataset_name}_it_infoNCE': (self.it_infoNCE_loss_m[dataset_name].sum, self.it_infoNCE_loss_m[dataset_name].count),
                            f'{dataset_name}_ii_triplet': (self.ii_triplet_loss_m[dataset_name].sum, self.ii_triplet_loss_m[dataset_name].count)
                        })
                if self.stage2:
                    for dataset_name in self.dataset_names:
                        losses.update({
                            f'{dataset_name}_main': (self.loss_m[dataset_name].sum, self.loss_m[dataset_name].count),
                            f'{dataset_name}_con_triplet': (self.triplet_loss_m[dataset_name].sum, self.triplet_loss_m[dataset_name].count),
                            f'{dataset_name}_con_infoNCE': (self.infoNCE_loss_m[dataset_name].sum, self.infoNCE_loss_m[dataset_name].count),
                            f'{dataset_name}_con_binary': (self.binary_loss_m[dataset_name].sum, self.binary_loss_m[dataset_name].count),
                            f'{dataset_name}_con_rerank': (self.rerank_loss_m[dataset_name].sum, self.rerank_loss_m[dataset_name].count),
                            f'{dataset_name}_anatomy_infoNCE': {k: (v.sum, v.count) for k, v in self.anatomy_infoNCE_loss_m[dataset_name].items()},
                            f'{dataset_name}_anatomy_triplet': {k: (v.sum, v.count) for k, v in self.anatomy_triplet_loss_m[dataset_name].items()},
                            f'{dataset_name}_anatomy_binary': {k: (v.sum, v.count) for k, v in self.anatomy_binary_loss_m[dataset_name].items()},
                            f'{dataset_name}_anatomy_rerank': {k: (v.sum, v.count) for k, v in self.anatomy_rerank_loss_m[dataset_name].items()},
                            f'{dataset_name}_anatomy_valid_triplet': {k: (v.sum, v.count) for k, v in self.anatomy_valid_triplet_count_m[dataset_name].items()}
                        })

                gathered_data = {
                    'losses': losses
                }
                torch.distributed.all_gather_object(gathered_metrics, gathered_data)
                torch.distributed.barrier()
                
                if self.is_main:
                    # ==== 合并损失统计量 ====
                    def merge_loss(metric_name):
                        total_sum = sum(data['losses'][metric_name][0] for data in gathered_metrics)
                        total_count = sum(data['losses'][metric_name][1] for data in gathered_metrics)
                        return total_sum / total_count if total_count > 0 else 0

                    def merge_anatomy_loss(metric_name):
                        merged = defaultdict(lambda: [0, 0])  # [sum, count]
                        for data in gathered_metrics:
                            for anatomy, (s, c) in data['losses'][metric_name].items():
                                merged[anatomy][0] += s
                                merged[anatomy][1] += c
                        return {k: (v[0]/v[1] if v[1]>0 else 0) for k, v in merged.items()}
                    
                    # 计算全局平均损失
                    if self.stage1:
                        global_uncon_loss = {dataset_name: merge_loss(f'{dataset_name}_uncon_main') for dataset_name in self.uncon_dataset_names}
                        global_it_triplet_loss = {dataset_name: merge_loss(f'{dataset_name}_it_triplet') for dataset_name in self.uncon_dataset_names}
                        global_it_infoNCE_loss = {dataset_name: merge_loss(f'{dataset_name}_it_infoNCE') for dataset_name in self.uncon_dataset_names}
                        global_ii_triplet_loss = {dataset_name: merge_loss(f'{dataset_name}_ii_triplet') for dataset_name in self.uncon_dataset_names} 
                        
                    if self.stage2:
                        global_main_loss = {dataset_name: merge_loss(f'{dataset_name}_main') for dataset_name in self.dataset_names}
                        global_triplet_loss = {dataset_name: merge_loss(f'{dataset_name}_con_triplet') for dataset_name in self.dataset_names}
                        global_infoNCE_loss = {dataset_name: merge_loss(f'{dataset_name}_con_infoNCE') for dataset_name in self.dataset_names}
                        global_binary_loss = {dataset_name: merge_loss(f'{dataset_name}_con_binary') for dataset_name in self.dataset_names}
                        global_rerank_loss = {dataset_name: merge_loss(f'{dataset_name}_con_rerank') for dataset_name in self.dataset_names}
                        global_anatomy_infoNCE = {dataset_name: merge_anatomy_loss(f'{dataset_name}_anatomy_infoNCE') for dataset_name in self.dataset_names}
                        global_anatomy_triplet = {dataset_name: merge_anatomy_loss(f'{dataset_name}_anatomy_triplet') for dataset_name in self.dataset_names}
                        global_anatomy_binary = {dataset_name: merge_anatomy_loss(f'{dataset_name}_anatomy_binary') for dataset_name in self.dataset_names}
                        global_anatomy_rerank = {dataset_name: merge_anatomy_loss(f'{dataset_name}_anatomy_rerank') for dataset_name in self.dataset_names}
                        global_anatomy_valid_triplet = {dataset_name: merge_anatomy_loss(f'{dataset_name}_anatomy_valid_triplet') for dataset_name in self.dataset_names}

                    # ==== 日志记录 ====
                    lr = self.optim.param_groups[0]['lr']
                    current_time = datetime.now().strftime("%m-%d %H:%M")
        
                    if self.stage1:
                        for dataset_name in self.uncon_dataset_names:
                            self.print(f"[{dataset_name}] Step {self.current_step} at {current_time} | LR {lr} | "
                                f"Uncon Loss {global_uncon_loss[dataset_name]:.4f} | "
                                f"Image-Text Triplet {global_it_triplet_loss[dataset_name]:.4f} | "
                                f"Image-Text InfoNCE {global_it_infoNCE_loss[dataset_name]:.4f} | "
                                f"Image-Image Triplet {global_ii_triplet_loss[dataset_name]:.4f}")
                    if self.stage2:
                        for dataset_name in self.dataset_names:
                            self.print(f"[{dataset_name}] Step {self.current_step} at {current_time} | LR {lr} | "
                                f"Con Loss {global_main_loss[dataset_name]:.4f} | Con InfoNCE {global_infoNCE_loss[dataset_name]:.4f} | "
                                f"Con Triplet {global_triplet_loss[dataset_name]:.4f} | Con Binary {global_binary_loss[dataset_name]:.4f} | Con Rerank {global_rerank_loss[dataset_name]:.4f}")
                    log_dict = {'lr': lr}
                    

                    if self.stage1:
                        for dataset_name in self.uncon_dataset_names:
                            log_dict.update({
                                f'{dataset_name}_uncon_train_loss': self.uncon_loss_m[dataset_name].avg,
                                f'{dataset_name}_it_triplet_loss': self.it_triplet_loss_m[dataset_name].avg,
                                f'{dataset_name}_it_infoNCE_loss': self.it_infoNCE_loss_m[dataset_name].avg,
                                f'{dataset_name}_ii_triplet_loss': self.ii_triplet_loss_m[dataset_name].avg
                            })
                    
                    if self.stage2:
                        for dataset_name in self.dataset_names:
                            log_dict.update({
                                f'{dataset_name}_train_loss': self.loss_m[dataset_name].avg,
                                f'{dataset_name}_infoNCE_loss': self.infoNCE_loss_m[dataset_name].avg,
                                f'{dataset_name}_triplet_loss': self.triplet_loss_m[dataset_name].avg,
                                f'{dataset_name}_binary_loss': self.binary_loss_m[dataset_name].avg,
                                f'{dataset_name}_rerank_loss': self.rerank_loss_m[dataset_name].avg
                            })
                            
                            # 记录解剖结构损失
                            for anatomy, loss in global_anatomy_infoNCE[dataset_name].items():
                                log_dict[f'{dataset_name}_{anatomy}_infoNCE_loss'] = loss
                            
                            for anatomy, loss in global_anatomy_triplet[dataset_name].items():
                                log_dict[f'{dataset_name}_{anatomy}_triplet_loss'] = loss
                            for anatomy, loss in global_anatomy_binary[dataset_name].items():
                                log_dict[f'{dataset_name}_{anatomy}_binary_loss'] = loss
                            for anatomy, loss in global_anatomy_rerank[dataset_name].items():
                                log_dict[f'{dataset_name}_{anatomy}_rerank_loss'] = loss
                            for anatomy, count in global_anatomy_valid_triplet[dataset_name].items():
                                log_dict[f'{dataset_name}_{anatomy}_valid_triplet'] = count
                    # 使用wandb记录所有指标
                    self.wandb.log(log_dict, step=current_step)
                    
                if self.stage2:
                    self._reset_metrics()
                if self.stage1:
                    self._reset_metrics_uncon()
        self.current_step += 1

    def train(self, evaluate_before_train):
        
        if evaluate_before_train:
            if self.stage1:
                self.evaluate_uncon(current_step=self.current_step)  # Evaluate before training
                self.evaluate_uncon(current_step=self.current_step,is_trainset=True)  # Evaluate before training
            if self.stage2:
                self.evaluate(current_step=self.current_step)   # Evaluate before training

        while self.current_step <= self.num_train_steps:

            # print(f'Rank {torch.distributed.get_rank()} | Step {self.current_step} | Training...','allocated', torch.cuda.memory_allocated()/1e6,
            #     'reserved', torch.cuda.memory_reserved()/1e6,
            #     'peak', torch.cuda.max_memory_allocated()/1e6)
            self.train_step()



if __name__ == '__main__':
    anatomy_filter = ['trachea', 'bronchie', 'bone', 'heart', 'liver', 'vertebrae', 'aorta', 'pleura', 'spinal canal', 'heart ascending aorta', 'clavicle', 'breast', 'heart ventricle', 'heart atrium', 'gallbladder', 'pancreas', 'pulmonary artery', 'stomach']
    # ['esophagus', 'trachea', 'bronchie', 'bone', 'heart', 'liver', 'vertebrae', 'aorta', 'pleura'] # - 'adrenal gland'
    # ['spinal canal', 'heart ascending aorta', 'clavicle', 'breast', 'heart ventricle', 'heart atrium', 'gallbladder', 'pancreas', 'pulmonary artery', 'stomach'] # - 'thoracic vertebrae'
    # ['adrenal gland', 'aorta', 'bone', 'breast', 'bronchie', 'clavicle', 'esophagus', 'gallbladder', 'heart ascending aorta', 'heart atrium', 'heart ventricle', 'heart', 'liver', 'pancreas', 'pleura', 'pulmonary artery', 'spinal canal', 'stomach', 'thoracic vertebrae', 'vertebrae', 'trachea']
    # 最终去除了esophagus, adrenal gland和thoracic vertebrae
    
    with open("/DB/data/haoningwu-1/zihengzhao/data/tengfei_proj/(all)anatomy_positive_elements_count[max_train_12k].json", 'r') as f:
        anatomy_distribution = json.load(f)
    anatomy_samples = {anatomy:anatomy_distribution[anatomy]["train >0.9 links count"] for anatomy in anatomy_filter}   # pos 样本越多，采样越多 # {'anatomy':num, ...}
    # 将 anatomies 分配到两个 8 卡机器（共 16 个进程），保证同一机器内 8 个进程的样本总数尽量接近
    
    def distribute_equal_anatomy(anatomy_samples, num_processes):
        """
        改进后的分配策略：
        1. 当anatomy数量 ≤ 进程数时：
        - 每个进程分配1个anatomy（按样本量降序）
        - 多余进程循环复用高样本量anatomy
        2. 当anatomy数量 > 进程数时：
        - 使用贪心算法均衡分配，使各进程总样本量接近，且每个进程内的anatomy数量尽可能平均
        """
        # 按样本量降序、名称升序排序
        sorted_anatomy = sorted(
            anatomy_samples.items(),
            key=lambda x: (-x[1], x[0])
        )
        num_anatomy = len(sorted_anatomy)
        
        # 情况1：anatomy数量 ≤ 进程数
        if num_anatomy <= num_processes:
            distribution = []
            for i in range(num_processes):
                # 循环使用高样本量anatomy
                anatomy_idx = i % num_anatomy
                anatomy_name = sorted_anatomy[anatomy_idx][0]
                distribution.append([anatomy_name])
        
            # 针对validation分配结果（多余进程置空）
            backup_dist = []
            for i in range(num_processes):
                if i < num_anatomy:
                    anatomy_name = sorted_anatomy[i][0]
                    backup_dist.append([anatomy_name])
                else:
                    backup_dist.append([])
                    
            return distribution, backup_dist
        
        # 情况2：anatomy数量 > 进程数
        else:
            # 初始化进程的anatomy数量和sample num之和
            process_anatomy_count = [0] * num_processes
            process_sample_sum = [0] * num_processes
            distribution = [[] for _ in range(num_processes)]
            
            # 分配每个anatomy
            for anatomy_name, samples in sorted_anatomy:
                # 找到当前anatomy数量最少的进程
                min_anatomy_count = min(process_anatomy_count)
                candidates = [i for i, count in enumerate(process_anatomy_count) if count == min_anatomy_count]
                
                # 在这些进程中挑选sample num之和最小的进程
                min_sample_sum = min(process_sample_sum[i] for i in candidates)
                selected_process = next(i for i in candidates if process_sample_sum[i] == min_sample_sum)
                
                # 分配anatomy到选中的进程
                distribution[selected_process].append(anatomy_name)
                process_anatomy_count[selected_process] += 1
                process_sample_sum[selected_process] += samples
            
            return distribution, distribution
        
    distribution, _ = distribute_equal_anatomy(anatomy_samples, 8)
    
    print('Process == 8')
    
    for rank_split in distribution:
        anatomy_sample_num = [anatomy_samples[k] for k in rank_split]
        anatomy_weight = [round(math.log(num)) for num in anatomy_sample_num]   # weight 与 sqrt(size) 相关
        anatomy_weight = [w/sum(anatomy_weight) for w in anatomy_weight]  # NOTE: Balancing between anatomys
        print(rank_split, sum(anatomy_samples[k] for k in rank_split)/1000000)
        for k, w, n in zip(rank_split, anatomy_weight, anatomy_sample_num):
            print(k, w, n/1000000)
    
    # distribution, _ = distribute_anatomy(anatomy_samples, 16)
    
    # print('Process == 16')
    
    # for rank_split in distribution:
    #     print(rank_split, sum(anatomy_samples[k] for k in rank_split)/1000000)
    
    
    
    # def evaluate(self, current_step):
    #     with torch.no_grad():
            
    #         self.CTClip.eval()
    #         all_metrics = {}
    #         temp_dataset_names = []
    #         for dataset_name in self.dataset_names:
    #             # if 'CT_RATE' == dataset_name:
    #             #     continue
    #             temp_dataset_names.append(dataset_name)
    #         for dataset_name in temp_dataset_names:
    #             for data_type in ['train','test']:
    #                 for k in [1,3, 5, 10, 20, 50, 100]:
    #                     all_metrics[f'{dataset_name}_{data_type}_(I2T)NDCG@{k}'] = {}
    #                 for k in [1,3, 5, 10, 20, 50, 100]:
    #                     all_metrics[f'{dataset_name}_{data_type}_(I2T)Hard_R@{k}'] = {}
    #                 for k in [1,3, 5, 10, 20, 50, 100]:
    #                     all_metrics[f'{dataset_name}_{data_type}_(I2T)Soft_R@{k}'] = {}
    #                 for k in [1,3, 5, 10, 20, 50, 100]:
    #                     all_metrics[f'{dataset_name}_{data_type}_(I2T)Soft_exclude_R@{k}'] = {}
    #                 for k in [1,3, 5, 10, 20, 50, 100]:
    #                     all_metrics[f'{dataset_name}_{data_type}_RateScore@{k}'] = {}
    #                 for k in [1,3, 5, 10, 20, 50, 100]:
    #                     all_metrics[f'{dataset_name}_{data_type}_UpperBound_RateScore@{k}'] = {}
    #                 for k in [1,3, 5, 10, 20, 50, 100]:
    #                     all_metrics[f'{dataset_name}_{data_type}_(I2I)NDCG@{k}'] = {}
    #                 for k in [1,3, 5, 10, 20, 50, 100]:
    #                     all_metrics[f'{dataset_name}_{data_type}_(I2I)Soft_exclude_R@{k}'] = {}
    #         for dataset_name in temp_dataset_names:
    #             # 修改增加到了fp16部分
    #             fused_latents_all = {anatomy:torch.zeros(len(self.anatomy2id_lss[dataset_name][anatomy]), 512, dtype=torch.float16) for anatomy in self.local_anatomy_filter_for_valid[dataset_name]}  
    #             local_latents_all = {anatomy:torch.zeros(len(self.anatomy2id_lss[dataset_name][anatomy]), 512, dtype=torch.float16) for anatomy in self.local_anatomy_filter_for_valid[dataset_name]} 
    #             # 每个anatomy的fused_latents顺序与dataset中的sample id顺序严格一致
    #             # 每个anatomy的fused_latents的shape是 (N, 512)，N是这个anatomy的样本数
                
    #             # First derive the fusion latent for each anatomy and each sample
    #             if not self.modal_embedding:
    #                 for video, anatomy_ls, sample_id_ls in tqdm(self.valid_dls[dataset_name], desc=f"Rank {torch.distributed.get_rank()}"):
    #                     # video: B 1 d h w
    #                     video = video.to(self.device).unsqueeze(1)  # B 1 1 d h w (凑出local_B)
    #                     anatomy_ls = list(anatomy_ls)
    #                     sample_id_ls = list(sample_id_ls)
    #                     text_tokens=self.tokenizer(anatomy_ls, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
    #                     with torch.autocast(device_type='cuda', dtype=torch.float16):
    #                         local_latents, fused_latents, _ = self.CTClip(
    #                         text_tokens, video, return_latents=True, device=self.device
    #                     )  # B 1 Dim
    #                     fused_latents = fused_latents.detach().cpu().squeeze().to(torch.float16)  # 去掉local_B，转换为float16
    #                     local_latents = local_latents.detach().cpu().squeeze().to(torch.float16)
    #                     # _, _, fused_latents, tmp = self.CTClip(text_tokens, video, return_latents=True, device=self.device) # B 1 Dim
    #                     # fused_latents = fused_latents.detach().cpu().squeeze()  # 去掉local_B
    #                     for i, (sample_id, anatomy) in enumerate(zip(sample_id_ls, anatomy_ls)):
    #                         sample_index = self.anatomy2id_lss[dataset_name][anatomy].index(sample_id)
    #                         fused_latents_all[anatomy][sample_index] = fused_latents[i]
    #                         local_latents_all[anatomy][sample_index] = local_latents[i]
    #                     del fused_latents, video, text_tokens, local_latents
    #             else:
    #                 for video, anatomy_ls, sample_id_ls,modal_indexs,local_text_ls in tqdm(self.valid_dls[dataset_name], desc=f"Rank {torch.distributed.get_rank()}"):
    #                     # video: B 1 d h w
    #                     video = video.to(self.device).unsqueeze(1)  # B 1 1 d h w (凑出local_B)
    #                     anatomy_ls = list(anatomy_ls)
    #                     sample_id_ls = list(sample_id_ls)
    #                     text_tokens=self.tokenizer(anatomy_ls, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
    #                     local_text_tokens = self.tokenizer(local_text_ls, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
    #                     with torch.autocast(device_type='cuda', dtype=torch.float16):
    #                         local_latents, fused_latents, _ = self.CTClip(
    #                             text_tokens, video, return_latents=True, device=self.device,is_condition=True,modal_indexs=modal_indexs,modal_embedding=self.modal_embedding,local_text=local_text_tokens
    #                         )  # B 1 Dim
                            
    #                     fused_latents = fused_latents.detach().cpu().squeeze().to(torch.float16)  # 去掉local_B，转换为float16
    #                     local_latents = local_latents.detach().cpu().squeeze().to(torch.float16)
    #                     # _, _, fused_latents, tmp = self.CTClip(text_tokens, video, return_latents=True, device=self.device) # B 1 Dim
    #                     # fused_latents = fused_latents.detach().cpu().squeeze()  # 去掉local_B
    #                     for i, (sample_id, anatomy) in enumerate(zip(sample_id_ls, anatomy_ls)):
    #                         sample_index = self.anatomy2id_lss[dataset_name][anatomy].index(sample_id)
    #                         fused_latents_all[anatomy][sample_index] = fused_latents[i]
    #                         local_latents_all[anatomy][sample_index] = local_latents[i]
    #                     del fused_latents, video, text_tokens,local_latents
    #             # Now lets calculate the similarity matrix for each anatomy
    #             for anatomy, fused_latents in fused_latents_all.items():
    #                 local_latents = local_latents_all[anatomy]
    #                 local_latents = local_latents.to(self.device)
    #                 fused_latents = fused_latents.to(self.device)
    #                 # image_to_image = torch.einsum('m d, n d -> m n', fused_latents, fused_latents)
    #                 image_to_text = torch.einsum('m d, n d -> m n', fused_latents, local_latents).float()
    #                 # 修改为fp32
    #                 image_to_image = torch.einsum('m d,n d->m n', fused_latents, fused_latents).float()
                    
    #                 similarity_tab = self.valid_dss[dataset_name].get_similarity_table(anatomy)
    #                 similarity_tab = torch.tensor(similarity_tab)
    #                 similarity_tab = (similarity_tab.to(dtype=torch.float32) * 0.01).to(self.device)  # 转回0-1之间的float32
                    
    #                 assert image_to_image.shape == similarity_tab.shape, f'image_to_image {image_to_image.shape} != similarity_tab {similarity_tab.shape}'
    #                 assert image_to_text.shape == similarity_tab.shape, f'iamge_to_text {image_to_text.shape} != similarity_tab {similarity_tab.shape}'
    #                 # now calculate the ndcg
    #                 def get_test_tab(simi_tab):
    #                     test_test = simi_tab[2000: , 2000: ]
    #                     test_train = simi_tab[2000: , :2000 ]
    #                     reordered_test_tab = torch.cat([test_test,test_train], dim=1)
    #                     return reordered_test_tab
    #                 train_image_to_text = image_to_text[:2000,:2000]
    #                 train_image_to_image = image_to_image[:2000,:2000]
    #                 train_similarity_tab = similarity_tab[:2000,:2000]
    #                 test_image_to_text = get_test_tab(image_to_text)
    #                 test_image_to_image = get_test_tab(image_to_image)
    #                 test_similarity_tab = get_test_tab(similarity_tab)
                    
    #                 print('image_to_image shape:', image_to_image.shape)
    #                 print('similarity_tab shape:', similarity_tab.shape)
    #                 # 还要计算i2t的hard_R,ndcg,soft_R,soft_exclude_R
                    
    #                 results = hard_retrieval(train_image_to_text, train_similarity_tab, k=[1,3, 5, 10, 20, 50, 100])
    #                 for k, v in zip([1,3, 5, 10, 20, 50, 100], results):
    #                     all_metrics[f'{dataset_name}_train_(I2T)Hard_R@{k}'][anatomy] = v*100
    #                 ndcg_scores = compute_ndcg_uncon(train_image_to_text, train_similarity_tab, k=[1,3, 5, 10, 20, 50, 100])
    #                 for k, v in zip([1,3, 5, 10, 20, 50, 100], ndcg_scores):
    #                     all_metrics[f'{dataset_name}_train_(I2T)NDCG@{k}'][anatomy] = v*100
    #                 results = soft_retrieval_uncon(train_image_to_text, train_similarity_tab, k=[1,3, 5, 10, 20, 50, 100])
    #                 for k, v in zip([1,3, 5, 10, 20, 50, 100], results):
    #                     all_metrics[f'{dataset_name}_train_(I2T)Soft_R@{k}'][anatomy] = v*100
                        
    #                 results = hard_retrieval(test_image_to_text, test_similarity_tab, k=[1,3, 5, 10, 20, 50, 100])
    #                 for k, v in zip([1,3, 5, 10, 20, 50, 100], results):
    #                     all_metrics[f'{dataset_name}_test_(I2T)Hard_R@{k}'][anatomy] = v*100
    #                 ndcg_scores = compute_ndcg_uncon(test_image_to_text, test_similarity_tab, k=[1,3, 5, 10, 20, 50, 100])
    #                 for k, v in zip([1,3, 5, 10, 20, 50, 100], ndcg_scores):
    #                     all_metrics[f'{dataset_name}_test_(I2T)NDCG@{k}'][anatomy] = v*100
    #                 results = soft_retrieval_uncon(test_image_to_text, test_similarity_tab, k=[1,3, 5, 10, 20, 50, 100])
    #                 for k, v in zip([1,3, 5, 10, 20, 50, 100], results):
    #                     all_metrics[f'{dataset_name}_test_(I2T)Soft_R@{k}'][anatomy] = v*100
                        
    #                 rows_to_remove = []
    #                 n = train_similarity_tab.shape[0]
    #                 index_list = [i for i in range(train_similarity_tab.shape[0])]
                    
    #                 for i in range(n):
    #                     # 计算当前行中大于90的元素数量
    #                     count_gt_90 = torch.sum(train_similarity_tab[i] > 0.9).item()
    #                     if count_gt_90 < 2:
    #                         rows_to_remove.append(i)
                    
    #                 # 保留需要保留的索引
    #                 keep_indices = [i for i in range(n) if i not in rows_to_remove]
                    
    #                 if keep_indices:
    #                     # 更新similarity_tab
    #                     train_similarity_tab = train_similarity_tab[keep_indices][:, keep_indices]
    #                     train_image_to_text = train_image_to_text[keep_indices][:, keep_indices]
    #                     train_image_to_image = train_image_to_image[keep_indices][:, keep_indices]
    #                 else:
    #                     print(f"Warning: All rows removed for {anatomy}, skipping...")
    #                     continue
                    
    #                 results = soft_retrieval_exclude_self(train_image_to_text, train_similarity_tab, index_list, k=[1,3, 5, 10, 20, 50, 100])
    #                 for k, v in zip([1,3, 5, 10, 20, 50, 100], results):
    #                     all_metrics[f'{dataset_name}_train_(I2T)Soft_exclude_R@{k}'][anatomy] = v*100
                    
    #                 # average similarity
    #                 pred_score, upperbound_score = soft_retrieval(train_image_to_image, train_similarity_tab, k=[1,3, 5, 10, 20, 50, 100])   # {'avg_similarity@{k}': xxx, ...}
    #                 for k, v in zip([1,3, 5, 10, 20, 50, 100], pred_score):
    #                     all_metrics[f'{dataset_name}_train_RateScore@{k}'][anatomy] = v*100
    #                 for k, v in zip([1,3, 5, 10, 20, 50, 100], upperbound_score):
    #                     all_metrics[f'{dataset_name}_train_UpperBound_RateScore@{k}'][anatomy] = v*100
                    
    #                 results = compute_ndcg(train_image_to_image, train_similarity_tab, index_list, k=[1,3, 5, 10, 20, 50, 100])
    #                 for k, v in zip([1,3, 5, 10, 20, 50, 100], results):
    #                     all_metrics[f'{dataset_name}_train_(I2I)NDCG@{k}'][anatomy] = v*100
    #                 # hard image-image retrieval
    #                 recall_scores = soft_retrieval_exclude_self(train_image_to_image, train_similarity_tab, index_list, k=[1,3, 5, 10, 20, 50, 100])
    #                 if recall_scores == -1:
    #                     print(f'{anatomy} has no positive samples!')
    #                     for k in [1,3, 5, 10, 20, 50, 100]:
    #                         all_metrics[f'{dataset_name}_train_(I2I)Soft_exclude_R@{k}'][anatomy] = -1
    #                 else:
    #                     for k, v in zip([1,3, 5, 10, 20, 50, 100], recall_scores):
    #                         all_metrics[f'{dataset_name}_train_(I2I)Soft_exclude_R@{k}'][anatomy] = v*100
                    
    #                 results = soft_retrieval_exclude_self(train_image_to_text, train_similarity_tab, index_list, k=[1,3, 5, 10, 20, 50, 100])
    #                 for k, v in zip([1,3, 5, 10, 20, 50, 100], results):
    #                     all_metrics[f'{dataset_name}_train_(I2T)Soft_exclude_R@{k}'][anatomy] = v*100
                    
                    
    #                 # average similarity
    #                 results = soft_retrieval_exclude_self(test_image_to_text, test_similarity_tab, index_list, k=[1,3, 5, 10, 20, 50, 100])
    #                 for k, v in zip([1,3, 5, 10, 20, 50, 100], results):
    #                     all_metrics[f'{dataset_name}_test_(I2T)Soft_exclude_R@{k}'][anatomy] = v*100
                    
    #                 index_list = [i for i in range(test_similarity_tab.shape[0])]
    #                 pred_score, upperbound_score = soft_retrieval(test_image_to_image, test_similarity_tab, k=[1,3, 5, 10, 20, 50, 100])   # {'avg_similarity@{k}': xxx, ...}
    #                 for k, v in zip([1,3, 5, 10, 20, 50, 100], pred_score):
    #                     all_metrics[f'{dataset_name}_test_RateScore@{k}'][anatomy] = v*100
    #                 for k, v in zip([1,3, 5, 10, 20, 50, 100], upperbound_score):
    #                     all_metrics[f'{dataset_name}_test_UpperBound_RateScore@{k}'][anatomy] = v*100
                    
    #                 results = compute_ndcg(test_image_to_image, test_similarity_tab, index_list, k=[1,3, 5, 10, 20, 50, 100])
    #                 for k, v in zip([1,3, 5, 10, 20, 50, 100], results):
    #                     all_metrics[f'{dataset_name}_test_(I2I)NDCG@{k}'][anatomy] = v*100
    #                 # hard image-image retrieval
    #                 recall_scores = soft_retrieval_exclude_self(test_image_to_image, test_similarity_tab, index_list, k=[1,3, 5, 10, 20, 50, 100])
    #                 if recall_scores == -1:
    #                     print(f'{anatomy} has no positive samples!')
    #                     for k in [1,3, 5, 10, 20, 50, 100]:
    #                         all_metrics[f'{dataset_name}_test_(I2I)Soft_exclude_R@{k}'][anatomy] = -1
    #                 else:
    #                     for k, v in zip([1,3, 5, 10, 20, 50, 100], recall_scores):
    #                         all_metrics[f'{dataset_name}_test_(I2I)Soft_exclude_R@{k}'][anatomy] = v*100

    #                     # 清理显存，释放不再需要的变量
    #                 del image_to_image, similarity_tab, fused_latents
    #                 del train_image_to_image, train_similarity_tab, train_image_to_text
    #                 del test_image_to_image, test_similarity_tab, test_image_to_text
    #             del fused_latents_all


    #         if torch.distributed.is_initialized():
    #             # Gather all_metrics dicts from all processes into a list on each process
    #             world_size = torch.distributed.get_world_size()
    #             gathered_metrics = [None for _ in range(world_size)]
                
    #             # +++ 新增：构建包含损失统计量的数据结构 +++
    #             losses = {}
    #             for dataset_name in temp_dataset_names:
    #                 losses.update({
    #                     f'{dataset_name}_main': (self.loss_m[dataset_name].sum, self.loss_m[dataset_name].count),
    #                     f'{dataset_name}_triplet': (self.triplet_loss_m[dataset_name].sum, self.triplet_loss_m[dataset_name].count),
    #                     f'{dataset_name}_infoNCE': (self.infoNCE_loss_m[dataset_name].sum, self.infoNCE_loss_m[dataset_name].count),
    #                     f'{dataset_name}_anatomy_infoNCE': {k: (v.sum, v.count) for k, v in self.anatomy_infoNCE_loss_m[dataset_name].items()},
    #                     f'{dataset_name}_anatomy_triplet': {k: (v.sum, v.count) for k, v in self.anatomy_triplet_loss_m[dataset_name].items()},
    #                     f'{dataset_name}_anatomy_valid_triplet': {k: (v.sum, v.count) for k, v in self.anatomy_valid_triplet_count_m[dataset_name].items()}
    #                 })

    #             gathered_data = {
    #                 'metrics': all_metrics,
    #                 'losses': losses
    #             }
                
    #             # 执行收集（替换原 all_gather_object 调用）
    #             torch.distributed.all_gather_object(gathered_metrics, gathered_data)  # +++ 修改行 +++

    #             # On the main process, merge the dictionaries
    #             if self.accelerator.is_local_main_process:
                    
    #                 print(f'** EVAL ** Gather Done!')
                    
    #                 # +++ 新增：合并损失统计量 +++
    #                 def merge_loss(metric_name):
    #                     total_sum = sum(data['losses'][metric_name][0] for data in gathered_metrics)
    #                     total_count = sum(data['losses'][metric_name][1] for data in gathered_metrics)
    #                     return total_sum / total_count if total_count > 0 else 0

    #                 def merge_anatomy_loss(metric_name):
    #                     merged = defaultdict(lambda: [0, 0])  # [sum, count]
    #                     for data in gathered_metrics:
    #                         for anatomy, (s, c) in data['losses'][metric_name].items():
    #                             merged[anatomy][0] += s
    #                             merged[anatomy][1] += c
    #                     return {k: (v[0]/v[1] if v[1]>0 else 0) for k, v in merged.items()}
                    
    #                 # 计算全局平均损失
    #                 global_main_loss = {dataset_name: merge_loss(f'{dataset_name}_main') for dataset_name in temp_dataset_names}
    #                 global_triplet_loss = {dataset_name: merge_loss(f'{dataset_name}_triplet') for dataset_name in temp_dataset_names}
    #                 global_infoNCE_loss = {dataset_name: merge_loss(f'{dataset_name}_infoNCE') for dataset_name in temp_dataset_names}
    #                 global_anatomy_infoNCE = {dataset_name: merge_anatomy_loss(f'{dataset_name}_anatomy_infoNCE') for dataset_name in temp_dataset_names}
    #                 global_anatomy_triplet = {dataset_name: merge_anatomy_loss(f'{dataset_name}_anatomy_triplet') for dataset_name in temp_dataset_names}
    #                 global_anatomy_valid_triplet = {dataset_name: merge_anatomy_loss(f'{dataset_name}_anatomy_valid_triplet') for dataset_name in temp_dataset_names}

    #                 lr = self.optim.param_groups[0]['lr']
                    
    #                 # +++ 修改信息输出部分 +++
    #                 # info = f"Step {current_step} | LR {lr} | "\
    #                 #     f"Loss {global_main_loss:.4f} | "\
    #                 #     f"Triplet Loss {global_triplet_loss:.4f} | "\
    #                 #     f"InfoNCE Loss {global_infoNCE_loss:.4f} |"
    #                 info  = f"Step {current_step} | LR {lr} | "
    #                 for dataset_name in temp_dataset_names:
    #                     info += f"{dataset_name} Loss {global_main_loss[dataset_name]:.4f} | "\
    #                             f"{dataset_name} Triplet Loss {global_triplet_loss[dataset_name]:.4f} | "\
    #                             f"{dataset_name} InfoNCE Loss {global_infoNCE_loss[dataset_name]:.4f} | "
    #                 # 创建Wandb日志字典
    #                 wandb_log_dict = {'lr': lr}
                    
    #                 if current_step > 0:
    #                     # 使用wandb替代tensorboard
    #                     # self.tb_writer.add_scalar('lr', lr, current_step)
    #                     wandb_log_dict['lr'] = lr
                        
    #                     for dataset_name in temp_dataset_names:
    #                         # 记录训练损失
    #                         wandb_log_dict[f'{dataset_name}_train_loss'] = self.loss_m[dataset_name].avg
    #                         wandb_log_dict[f'{dataset_name}_infoNCE_loss'] = self.infoNCE_loss_m[dataset_name].avg
    #                         wandb_log_dict[f'{dataset_name}_triplet_loss'] = self.triplet_loss_m[dataset_name].avg
                            
    #                         # 记录解剖结构损失
    #                         for anatomy, loss in global_anatomy_infoNCE[dataset_name].items():
    #                             wandb_log_dict[f'{dataset_name}_{anatomy}_infoNCE_loss'] = loss
    #                         for anatomy, loss in global_anatomy_triplet[dataset_name].items():
    #                             wandb_log_dict[f'{dataset_name}_{anatomy}_triplet_loss'] = loss
    #                         for anatomy, count in global_anatomy_valid_triplet[dataset_name].items():
    #                             wandb_log_dict[f'{dataset_name}_{anatomy}_valid_triplet'] = count
                                    
    #                 merged_metrics = {}
    #                 for metric_name in all_metrics.keys():
    #                     merged_metrics[metric_name] = {}
    #                 for gathered_data in gathered_metrics:
    #                     # print(gathered_data['metrics'])
    #                     for metric_name, subdict in gathered_data['metrics'].items():   # metric_name: 'Recall@5' ...
    #                         for key, value in subdict.items():  # pancreas: 0.8
    #                             merged_metrics[metric_name][key] = value
    #                 all_metrics = merged_metrics
    #                 write_info = ''
    #                 for dataset_name in temp_dataset_names:
    #                     for data_type in ['train','test']:
    #                         for metrics_name in ['(I2T)NDCG', '(I2T)Hard_R', '(I2T)Soft_R', '(I2T)Soft_exclude_R', 'RateScore', 'UpperBound_RateScore', '(I2I)NDCG', '(I2I)Soft_exclude_R']:
    #                             for k in [1,3, 5, 10, 20, 50, 100]:
    #                                 # print(f'Processing {dataset_name} {data_type} {metrics_name}@{k}')
    #                                 # print(all_metrics[f'{dataset_name}_{data_type}_{metrics_name}@{k}'])
    #                                 avg_results = sum(all_metrics[f'{dataset_name}_{data_type}_{metrics_name}@{k}'].values()) / len(all_metrics[f'{dataset_name}_{data_type}_{metrics_name}@{k}'])
    #                                 info += f' {dataset_name}_{data_type}_{metrics_name}@{k} {avg_results} |'
                                    
    #                                 # 记录到wandb
    #                                 wandb_log_dict[f'{dataset_name}_{data_type}_{metrics_name}@{k}'] = avg_results
                                    
    #                                 write_info += info   # the following details will be written but not displayed
                                    
    #                                 for key, value in all_metrics[f'{dataset_name}_{data_type}_{metrics_name}@{k}'].items():
    #                                     # 记录详细指标到wandb
    #                                     wandb_log_dict[f'{dataset_name}_{data_type}_{key}_{metrics_name}@{k}'] = value
    #                                     write_info += f"{dataset_name}_{data_type}_{key}_{metrics_name}@{k}: {value} | "
    #                 # 使用wandb记录所有指标，step参数用于标识当前步骤
    #                 self.wandb.log(wandb_log_dict, step=current_step)

    #                 info += '\n'
    #                 write_info += '\n'
    #                 self.print(info)
    #                 with open(self.results_folder / 'log.txt', 'a') as f:
    #                     f.write(info)
    #                 def convert_to_serializable(obj):
    #                     """将非JSON可序列化类型转换为可序列化类型"""
    #                     import numpy as np
                        
    #                     if isinstance(obj, (np.integer, np.floating, np.bool_)):
    #                         # 处理numpy标量类型
    #                         return obj.item()
    #                     elif isinstance(obj, (np.ndarray, torch.Tensor)):
    #                         # 处理数组和张量
    #                         return obj.tolist()
    #                     elif isinstance(obj, dict):
    #                         # 递归处理字典
    #                         return {k: convert_to_serializable(v) for k, v in obj.items()}
    #                     elif isinstance(obj, (list, tuple)):
    #                         # 递归处理列表或元组
    #                         return [convert_to_serializable(i) for i in obj]
    #                     else:
    #                         # 返回原始值
    #                         return obj
                            
    #                 with open(self.results_folder / f'{current_step}.json', 'w') as f:
    #                     json.dump(convert_to_serializable(all_metrics), f, indent=4)
                        
    #                 print(f'** EVAL ** Log Done!')
                    
    #             self._reset_metrics()
