# # 合并两个权重
# import torch
# import os
# from transformers import AutoConfig, AutoModel
# from modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
# # 定义路径
# original_model_path = "/mnt/petrelfs/zhangtengfei/RadIR/Qwen2.5-VL-7B-Instruct"  # 原始模型文件夹路径
# new_config_path = "/mnt/petrelfs/zhangtengfei/RadIR/QWEN_test/combine_vision_v2/new_temp_config"  # 新配置文件路径
# ctvit_weights_path = "/mnt/petrelfs/zhangtengfei/RadIR/QWEN_test/combine_vision_v2/vision_encoder_pretrain_strict.pt"  # CTViT权重路径
# # output_path = "/mnt/petrelfs/zhangtengfei/RadIR/QWEN_test/combine_vision/result"  # 输出路径
# output_path = "/mnt/petrelfs/zhangtengfei/RadIR/QWEN_test/combine_vision_v2/Qwen2.5_result"  # 输出路径

# # 3. 加载原始模型权重
# print('0000')
# # config = AutoConfig.from_pretrained(original_model_path)
# original_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(original_model_path)
# original_state_dict = original_model.state_dict()
# # original_model = Qwen2_5_VLForConditionalGeneration(config)
# # original_state_dict = original_model.state_dict()

# from modeling_qwen2_5_vl_ctvit import Qwen2_5_VLForConditionalGeneration
# from ctvit import CTViT
# print('1111')
# # 1. 加载新的配置文件
# config = AutoConfig.from_pretrained(new_config_path)
# print('2222')
# # 2. 创建新模型实例
# model = Qwen2_5_VLForConditionalGeneration(config)
# print('3333')
# # 4. 加载CTViT权重
# ctvit_weights = torch.load(ctvit_weights_path)
# # ctvit = CTViT(
# #         dim = 512,
# #         final_dim=3584,
# #         image_size = 480,
# #         patch_size = 20,
# #         temporal_patch_size = 10,
# #         spatial_depth = 8,
# #         temporal_depth = 6,
# #         cls_depth = 4,
# #         dim_head = 32,
# #         heads = 8,
# #         spatial_merge_size = 1,
# # )
# # ctvit_weights = ctvit.state_dict()
# print('4444')
# # 5. 合并权重
# # 首先复制除visual部分外的所有权重
# new_state_dict = {}
# for name, param in original_state_dict.items():
#     if not name.startswith("model.visual"):
#         new_state_dict[name] = param
#     # else:
#         # print(f'old model visual parameters key {name}')
# for name,param in model.state_dict().items():
#     if name.startswith("model.visual"):
#         print(f'new model visual parameters key {name}')
# print('5555')
# # 加载CTViT权重到模型中
# # 假设CTViT权重键的格式与模型中期望的格式匹配
# # for name, param in ctvit_weights.items():
# #     if name in model.visual.ctvit.state_dict():
# #         new_state_dict[f"model.visual.{name}"] = param
# #         if param.numel() == 0:
# #             print(f"Empty parameter found: {name}")
# #     else:
# #         print(f"警告: CTViT权重中的{name}在模型中不存在")
# import torch

# # 假设 ctvit_weights 和 model 已经定义好了
# # new_state_dict = {} # 确保 new_state_dict 已经初始化

# for name, param in ctvit_weights.items():
#     # 检查模型中是否存在对应的参数
#     if name in model.visual.ctvit.state_dict():
#         # 检查加载的参数张量是否为空 (元素数量为0)
#         if param.numel() == 0:
#             print(f"Empty parameter found: '{name}'. Initializing with random normal values.")
            
#             # 1. 从你的模型定义中获取目标参数的正确形状和设备信息
#             target_param = model.visual.ctvit.state_dict()[name]
#             target_shape = target_param.shape
#             target_device = target_param.device
#             target_dtype = target_param.dtype
            
#             # 2. 使用 torch.randn 创建一个符合目标形状、设备和数据类型的随机张量
#             #    torch.randn 默认生成标准正态分布的值
#             initialized_param = torch.randn(target_shape, device=target_device, dtype=target_dtype)
            
#             # 3. 将新创建的张量放入 new_state_dict
#             new_state_dict[f"model.visual.ctvit.{name}"] = initialized_param
#         else:
#             # 如果张量不为空，则正常赋值
#             new_state_dict[f"model.visual.ctvit.{name}"] = param
#     else:
#         print(f"警告: CTViT权重中的{name}在模型中不存在")
# print('1111')
# # 加载合并后的权重
# missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
# print(f"缺失的键: {missing_keys}")
# print(f"意外的键: {unexpected_keys}")

# # 6. 保存完整模型
# model.config.save_pretrained(output_path)
# model.save_pretrained(output_path)
# print(f"模型已保存到: {output_path}")


# 合并两个权重
import torch
import os
from transformers import AutoConfig, AutoModel
from modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
# 定义路径
# original_model_path = "/mnt/petrelfs/zhangtengfei/RadIR/Qwen2.5-VL-7B-Instruct"  # 原始模型文件夹路径
# new_config_path = "/mnt/petrelfs/zhangtengfei/RadIR/QWEN_test/combine_vision_our/new_temp_config"  # 新配置文件路径
# ctvit_weights_path = "/mnt/petrelfs/zhangtengfei/RadIR/QWEN_test/combine_vision_our/vision_encoder_pretrain_strict.pt"  # CTViT权重路径
# # output_path = "/mnt/petrelfs/zhangtengfei/RadIR/QWEN_test/combine_vision/result"  # 输出路径
# output_path = "/mnt/petrelfs/zhangtengfei/RadIR/QWEN_test/combine_vision_our/Qwen2.5_result"  # 输出路径

original_model_path = "/mnt/petrelfs/zhangtengfei/RadIR/Qwen2.5-VL-7B-Instruct"  # 原始模型文件夹路径
new_config_path = "/mnt/petrelfs/zhangtengfei/RadIR/QWEN_test/combine_vision_527_our/new_temp_config"  # 新配置文件路径
ctvit_weights_path = "/mnt/petrelfs/zhangtengfei/RadIR/QWEN_test/combine_vision_527_our/vision_encoder_pretrain_strict.pt"  # CTViT权重路径
# output_path = "/mnt/petrelfs/zhangtengfei/RadIR/QWEN_test/combine_vision/result"  # 输出路径
output_path = "/mnt/petrelfs/zhangtengfei/RadIR/QWEN_test/combine_vision_527_our/Qwen2.5_result"  # 输出路径

# 3. 加载原始模型权重
print('0000')
# config = AutoConfig.from_pretrained(original_model_path)
original_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(original_model_path)
original_state_dict = original_model.state_dict()
# original_model = Qwen2_5_VLForConditionalGeneration(config)
# original_state_dict = original_model.state_dict()

from modeling_qwen2_5_vl_ctvit import Qwen2_5_VLForConditionalGeneration
from ctvit import CTViT
print('1111')
# 1. 加载新的配置文件
config = AutoConfig.from_pretrained(new_config_path)
print('2222')
# 2. 创建新模型实例
model = Qwen2_5_VLForConditionalGeneration(config)
print('3333')
# 4. 加载CTViT权重
ctvit_weights = torch.load(ctvit_weights_path)
# ctvit = CTViT(
#         dim = 512,
#         final_dim=3584,
#         image_size = 480,
#         patch_size = 20,
#         temporal_patch_size = 10,
#         spatial_depth = 8,
#         temporal_depth = 6,
#         cls_depth = 4,
#         dim_head = 32,
#         heads = 8,
#         spatial_merge_size = 1,
# )
# ctvit_weights = ctvit.state_dict()
print('4444')
# 5. 合并权重
# 首先复制除visual部分外的所有权重
new_state_dict = {}
for name, param in original_state_dict.items():
    if not name.startswith("model.visual"):
        new_state_dict[name] = param
    # else:
        # print(f'old model visual parameters key {name}')
for name,param in model.state_dict().items():
    if name.startswith("model.visual"):
        print(f'new model visual parameters key {name}')
print('5555')
# 加载CTViT权重到模型中
# 假设CTViT权重键的格式与模型中期望的格式匹配
# for name, param in ctvit_weights.items():
#     if name in model.visual.ctvit.state_dict():
#         new_state_dict[f"model.visual.{name}"] = param
#         if param.numel() == 0:
#             print(f"Empty parameter found: {name}")
#     else:
#         print(f"警告: CTViT权重中的{name}在模型中不存在")
import torch

# 假设 ctvit_weights 和 model 已经定义好了
# new_state_dict = {} # 确保 new_state_dict 已经初始化

for name, param in ctvit_weights.items():
    # 检查模型中是否存在对应的参数
    if name in model.visual.ctvit.state_dict():
        # 检查加载的参数张量是否为空 (元素数量为0)
        if param.numel() == 0:
            print(f"Empty parameter found: '{name}'. Initializing with random normal values.")
            
            # 1. 从你的模型定义中获取目标参数的正确形状和设备信息
            target_param = model.visual.ctvit.state_dict()[name]
            target_shape = target_param.shape
            target_device = target_param.device
            target_dtype = target_param.dtype
            
            # 2. 使用 torch.randn 创建一个符合目标形状、设备和数据类型的随机张量
            #    torch.randn 默认生成标准正态分布的值
            initialized_param = torch.randn(target_shape, device=target_device, dtype=target_dtype)
            
            # 3. 将新创建的张量放入 new_state_dict
            new_state_dict[f"model.visual.ctvit.{name}"] = initialized_param
        else:
            # 如果张量不为空，则正常赋值
            new_state_dict[f"model.visual.ctvit.{name}"] = param
    else:
        print(f"警告: CTViT权重中的{name}在模型中不存在")
print('1111')
# 加载合并后的权重
missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
print(f"缺失的键: {missing_keys}")
print(f"意外的键: {unexpected_keys}")

# 6. 保存完整模型
model.config.save_pretrained(output_path)
model.save_pretrained(output_path)
print(f"模型已保存到: {output_path}")