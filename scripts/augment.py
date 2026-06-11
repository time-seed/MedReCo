
import torch.nn.functional as F
import math

# class MedicalOnGPUAugmenter(nn.Module):
#     """
#     针对 (B, C, D, H, W) 格式的医学图像进行 GPU 在线增强。
#     支持 CXR (D=1) 和 CT/MRI (D>1)。
#     严格遵守：不包含任何翻转操作。
#     """
#     def __init__(self, 
#                  aug_prob=0.65, 
#                  noise_sigma=0.02, 
#                  contrast_range=(0.85, 1.15),
#                  brightness_delta=0.1,
#                  gamma_range=(0.85, 1.15),
#                  enable_spatial=True):
#         super().__init__()
#         self.aug_prob = aug_prob
#         self.noise_sigma = noise_sigma
#         self.contrast_range = contrast_range
#         self.brightness_delta = brightness_delta
#         self.gamma_range = gamma_range
#         self.enable_spatial = enable_spatial

#     def forward(self, x):
#         """
#         输入 x: Tensor, shape (B, C, D, H, W), value range [0, 1]
#         """
#         is_condition = False
#         if len(x.shape) == 6:
#             is_condition=True
#             global_B = x.shape[0]
#             local_B = x.shape[1]
#             x = rearrange(x,'B N O D H W -> (B N) O D H W')
            
#         if (len(x.shape) != 5):
#             raise ValueError("输入张量必须是5维的 (B, C, D, H, W) 格式")
#         # 1. 确保在 GPU 上且不需要梯度
#         with torch.no_grad():
#             # 判断是否应用增强 (batch 级别或 sample 级别，这里为了效率做 batch 级别判断，
#             # 或者对每个样本独立判断。为了并行效率，以下实现为对 Batch 内部分样本应用)
            
#             B = x.shape[0]
            
#             # --------------- 强度变换 (Intensity Transforms) ---------------
            
#             # 1. 高斯噪声 (Gaussian Noise)
#             if torch.rand(1) < self.aug_prob:
#                 noise = torch.randn_like(x) * self.noise_sigma
#                 x = x + noise

#             # 2. 随机对比度 (Random Contrast)
#             if torch.rand(1) < self.aug_prob:
#                 # 生成 B 个随机因子，并调整形状以进行广播
#                 factor = torch.empty(B, 1, 1, 1, 1, device=x.device).uniform_(*self.contrast_range)
#                 mean = x.mean(dim=(2, 3, 4), keepdim=True)
#                 x = (x - mean) * factor + mean

#             # 3. 随机亮度 (Random Brightness)
#             if torch.rand(1) < self.aug_prob:
#                 delta = torch.empty(B, 1, 1, 1, 1, device=x.device).uniform_(-self.brightness_delta, self.brightness_delta)
#                 x = x + delta

#             # 4. 随机 Gamma 变换 (模拟不同设备的成像特性)
#             if torch.rand(1) < self.aug_prob:
#                 gamma = torch.empty(B, 1, 1, 1, 1, device=x.device).uniform_(*self.gamma_range)
#                 # 添加 epsilon 防止 log(0)
#                 x = x.clamp(1e-7, 1) ** gamma

#             # --------------- 空间变换 (Spatial Transforms) ---------------
#             # 注意：不使用翻转。仅使用平移和随机遮挡(Cutout)
            
#             if self.enable_spatial and torch.rand(1) < self.aug_prob:
#                 x = self.random_shift(x, max_shift_percent=0.08)
#             # 针对condition还是删除了这部分
#             # if self.enable_spatial and torch.rand(1) < self.aug_prob:
#             #     # x = self.random_cutout(x, hole_size_ratio=0.15)
#             #     x = self.random_cutout(x, hole_size_ratio=0.05)     # 或者0.05.问题日,训练集过拟合不上?                

#             # --------------- 最终截断 ---------------
#             x = torch.clamp(x, 0.0, 1.0)
#         if is_condition:
#             x = rearrange(x,'(B N) O D H W -> B N O D H W', B=global_B, N=local_B)
#         return x

#     def random_shift(self, x, max_shift_percent=0.1):
#         """
#         随机平移 (Translation)。
#         对于 CXR (D=1)，不移动深度方向。
#         对于 CT/MRI (D>1)，三维都可能微调。
#         """
#         B, C, D, H, W = x.shape
        
#         # 计算最大移动像素数
#         d_shift_max = int(D * max_shift_percent) if D > 1 else 0
#         h_shift_max = int(H * max_shift_percent)
#         w_shift_max = int(W * max_shift_percent)

#         # 生成平移量
#         d_shift = torch.randint(-d_shift_max, d_shift_max + 1, (1,)).item() if d_shift_max > 0 else 0
#         h_shift = torch.randint(-h_shift_max, h_shift_max + 1, (1,)).item()
#         w_shift = torch.randint(-w_shift_max, w_shift_max + 1, (1,)).item()

#         # 使用 padding 和 cropping 实现平移 (比 grid_sample 更快且无插值损失)
#         # Padding顺序: (w_left, w_right, h_top, h_bottom, d_front, d_back)
#         pad_w = (max(0, w_shift), max(0, -w_shift))
#         pad_h = (max(0, h_shift), max(0, -h_shift))
#         pad_d = (max(0, d_shift), max(0, -d_shift))
        
#         # 先 Pad 扩展
#         x_padded = F.pad(x, pad_w + pad_h + pad_d, value=0)  # 背景补0
        
#         # 再 Crop 回原大小
#         # 计算裁剪起始点
#         start_d = pad_d[0] if d_shift >=0 else 0
#         start_h = pad_h[0] if h_shift >=0 else 0
#         start_w = pad_w[0] if w_shift >=0 else 0
        
#         # 修正切片逻辑：如果是负向移动，padding 在右/下/后，起始点要变
#         if d_shift < 0: start_d = -d_shift
#         if h_shift < 0: start_h = -h_shift
#         if w_shift < 0: start_w = -w_shift

#         return x_padded[:, :, start_d:start_d+D, start_h:start_h+H, start_w:start_w+W]

#     def random_cutout(self, x, hole_size_ratio=0.2):
#         """
#         随机遮挡 (Cutout / Coarse Dropout)
#         这能迫使模型关注上下文，而不是只依赖某个特定特征点。
#         """
#         B, C, D, H, W = x.shape
        
#         mask_d = max(1, int(D * hole_size_ratio))
#         mask_h = int(H * hole_size_ratio)
#         mask_w = int(W * hole_size_ratio)

#         # 随机选择遮挡块的左上角
#         d_start = torch.randint(0, D - mask_d + 1, (B,))
#         h_start = torch.randint(0, H - mask_h + 1, (B,))
#         w_start = torch.randint(0, W - mask_w + 1, (B,))

#         # 由于 Tensor 索引在 GPU 上比较 tricky，我们使用 mask 相乘的方式
#         mask = torch.ones_like(x)
        
#         for i in range(B):
#             mask[i, :, 
#                  d_start[i]:d_start[i]+mask_d, 
#                  h_start[i]:h_start[i]+mask_h, 
#                  w_start[i]:w_start[i]+mask_w] = 0.0
        
#         return x * mask


import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math

class MedicalOnGPUAugmenter(nn.Module):
    def __init__(self, # 复现版本的aug是否是有问题的，保持和原来的完全一致哈
                 aug_prob=0.3, 
                #  aug_prob=0.5, 
                 noise_sigma=0.05,
                 contrast_range=(0.85, 1.15),
                 brightness_delta=0.1,
                 gamma_range=(0.85, 1.15),
                 # 新增参数
                 rotate_range_deg=(-15, 15), # 限制在正负15度，模拟患者体位偏差
                 scale_range=(0.9, 1.1),     # 限制缩放比例，过大或过小会丢失上下文
                 enable_spatial=True):
        super().__init__()
        self.aug_prob = aug_prob
        self.noise_sigma = noise_sigma
        self.contrast_range = contrast_range
        self.brightness_delta = brightness_delta
        self.gamma_range = gamma_range
        
        self.rotate_range_deg = rotate_range_deg
        self.scale_range = scale_range
        self.enable_spatial = enable_spatial

    def forward(self, x):
        is_condition = False
        if len(x.shape) == 6:
            is_condition = True
            global_B = x.shape[0]
            local_B = x.shape[1]
            x = rearrange(x, 'B N O D H W -> (B N) O D H W')
            
        if len(x.shape) != 5:
            raise ValueError("Input tensor must be (B, C, D, H, W)")

        with torch.no_grad():
            B, C, D, H, W = x.shape
            device = x.device

            # --- 像素级增强 (Intensity Augmentations) ---
            
            # 1. 高斯噪声
            mask = (torch.rand(B, 1, 1, 1, 1, device=device) < self.aug_prob).float()
            if mask.any():
                noise = torch.randn_like(x) * self.noise_sigma
                x = x + noise * mask

            # 2. 随机对比度
            mask = (torch.rand(B, 1, 1, 1, 1, device=device) < self.aug_prob).float()
            if mask.any():
                factor = torch.empty(B, 1, 1, 1, 1, device=device).uniform_(*self.contrast_range)
                x = (x - 0.5) * (factor * mask + (1 - mask)) + 0.5

            # 3. 随机亮度
            mask = (torch.rand(B, 1, 1, 1, 1, device=device) < self.aug_prob).float()
            if mask.any():
                delta = torch.empty(B, 1, 1, 1, 1, device=device).uniform_(-self.brightness_delta, self.brightness_delta)
                x = x + delta * mask

            # 4. 随机 Gamma
            mask = (torch.rand(B, 1, 1, 1, 1, device=device) < self.aug_prob).float()
            if mask.any():
                gamma = torch.empty(B, 1, 1, 1, 1, device=device).uniform_(*self.gamma_range)
                final_gamma = gamma * mask + 1.0 * (1 - mask)
                x = x.clamp(1e-7, 1) ** final_gamma
            
            # 最终截断 (像素增强后先截断，防止溢出影响后续插值)
            x = torch.clamp(x, 0.0, 1.0)

            # --- 空间变换 (Spatial Augmentations) ---
            
            if self.enable_spatial:
                # 5. 仿射变换 (旋转 + 缩放)
                # 使用概率触发，且避免和Shift同时进行导致过度变形，或者串行执行
                # 建议：如果进行了仿射变换，可以稍微降低Shift的概率或幅度，这里为了代码简洁依然独立判断
                if torch.rand(1) < self.aug_prob:
                    x = self.apply_affine_transform(x)

                # 6. 随机 Shift (保持你原有的高效实现)
                if torch.rand(1) < self.aug_prob:
                    x = self.random_shift(x, max_shift_percent=0.1)

        if is_condition:
            x = rearrange(x, '(B N) O D H W -> B N O D H W', B=global_B, N=local_B)
        return x

    # def apply_affine_transform(self, x):
    #     """
    #     应用批量的旋转和缩放。
    #     旋转：仅在 H-W 平面 (Axial) 旋转，模拟患者摆位不正。
    #     缩放：整体缩放。
    #     """
    #     B, C, D, H, W = x.shape
    #     device = x.device

    #     # 1. 采样旋转角度 (Radians)
    #     # 既然是 affine_grid，我们需要的是 target -> source 的映射
    #     # 旋转 theta，矩阵需要用 -theta
    #     min_deg, max_deg = self.rotate_range_deg
    #     angle_deg = torch.empty(B, device=device).uniform_(min_deg, max_deg)
    #     angle_rad = angle_deg * (math.pi / 180.0)
        
    #     # 2. 采样缩放因子
    #     # 放大图片 = 采样网格缩小 = 坐标 * (1/scale)
    #     min_s, max_s = self.scale_range
    #     scale = torch.empty(B, device=device).uniform_(min_s, max_s)
    #     inv_scale = 1.0 / scale 

    #     # 3. 构建 3x4 仿射矩阵 (针对 3D volume: D, H, W)
    #     # PyTorch grid_sample 坐标顺序是 (x, y, z) 即 (W, H, D)
    #     # 我们希望绕 Z 轴 (Depth) 旋转，即在 X-Y (W-H) 平面上旋转
        
    #     cos_a = torch.cos(-angle_rad) # 反向旋转
    #     sin_a = torch.sin(-angle_rad)
        
    #     # 初始化为单位矩阵的扩展 [B, 3, 4]
    #     theta = torch.zeros(B, 3, 4, device=device)
        
    #     # 填充矩阵
    #     # x_src = s * (x*cos - y*sin)
    #     # y_src = s * (x*sin + y*cos)
    #     # z_src = s * z  (或者 z 轴不缩放，看需求，这里假设各向同性缩放)
        
    #     # Row 0 (X / Width)
    #     theta[:, 0, 0] = inv_scale * cos_a
    #     theta[:, 0, 1] = inv_scale * (-sin_a)
        
    #     # Row 1 (Y / Height)
    #     theta[:, 1, 0] = inv_scale * sin_a
    #     theta[:, 1, 1] = inv_scale * cos_a
        
    #     # Row 2 (Z / Depth)
    #     # 保持 Z 轴不旋转，但应用缩放 (如果希望Z轴不缩放，把 inv_scale 改为 1.0)
    #     theta[:, 2, 2] = inv_scale 
        
    #     # 生成 Grid
    #     # size 需要是 (B, C, D, H, W)，但 affine_grid 只需要 (B, C, D, H, W) 的 size 元组
    #     grid = F.affine_grid(theta, x.size(), align_corners=False)
    #     if x.dtype != grid.dtype:
    #         grid = grid.to(x.dtype)
    #     # 采样
    #     # padding_mode='zeros' 非常重要，医学图像背景是黑的，不能用 border/reflection
    #     # align_corners=False 是现在的默认推荐
    #     x_transformed = F.grid_sample(x, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
        
    #     return x_transformed

    def apply_affine_transform(self, x, chunk_size=6):
        """
        应用批量的旋转和缩放。
        分块执行 grid_sample 以节省显存。
        """
        B, C, D, H, W = x.shape
        if D==1:
            chunk_size = 60
        device = x.device

        # --- 1. 参数准备 (这部分显存占用很小，可以一次性算完) ---
        
        # 1.1 采样旋转角度
        min_deg, max_deg = self.rotate_range_deg
        angle_deg = torch.empty(B, device=device).uniform_(min_deg, max_deg)
        angle_rad = angle_deg * (math.pi / 180.0)
        
        # 1.2 采样缩放因子
        min_s, max_s = self.scale_range
        scale = torch.empty(B, device=device).uniform_(min_s, max_s)
        inv_scale = 1.0 / scale 

        # 1.3 构建仿射矩阵
        cos_a = torch.cos(-angle_rad)
        sin_a = torch.sin(-angle_rad)
        
        # 初始化 [B, 3, 4]
        theta = torch.zeros(B, 3, 4, device=device)
        
        # Row 0 (X / Width)
        theta[:, 0, 0] = inv_scale * cos_a
        theta[:, 0, 1] = inv_scale * (-sin_a)
        
        # Row 1 (Y / Height)
        theta[:, 1, 0] = inv_scale * sin_a
        theta[:, 1, 1] = inv_scale * cos_a
        
        # Row 2 (Z / Depth)
        theta[:, 2, 2] = inv_scale 

        # --- 2. 分块执行仿射变换 (显存密集区) ---
        
        # 结果列表
        transformed_chunks = []
        
        # 按 chunk_size 步长循环
        for i in range(0, B, chunk_size):
            # 确定当前 chunk 的结束位置
            end = min(i + chunk_size, B)
            
            # 切片：取出一部分数据和对应的 theta
            x_chunk = x[i:end]        # shape: [chunk_size, C, D, H, W]
            theta_chunk = theta[i:end] # shape: [chunk_size, 3, 4]
            
            # 生成 Grid (只生成当前小批量的 grid，大幅节省显存)
            # F.affine_grid 需要传入当前 chunk 的 shape
            grid_chunk = F.affine_grid(theta_chunk, x_chunk.size(), align_corners=False)
            
            # 类型转换 (解决你之前的 invalid argument 报错点)
            if x_chunk.dtype != grid_chunk.dtype:
                grid_chunk = grid_chunk.to(x_chunk.dtype)
            
            # 采样
            x_transformed_chunk = F.grid_sample(
                x_chunk, 
                grid_chunk, 
                mode='bilinear', 
                padding_mode='zeros', 
                align_corners=False
            )
            
            transformed_chunks.append(x_transformed_chunk)
            
            # 显式删除临时变量，虽非必须但有助于立即释放显存
            del grid_chunk
            del x_chunk
            del theta_chunk

        # --- 3. 拼接结果 ---
        x_transformed = torch.cat(transformed_chunks, dim=0)
        
        return x_transformed
    
    def random_shift(self, x, max_shift_percent=0.1):
        # 你的原始代码，保持不变
        B, C, D, H, W = x.shape
        d_shift_max = int(D * max_shift_percent) if D > 1 else 0
        h_shift_max = int(H * max_shift_percent)
        w_shift_max = int(W * max_shift_percent)

        d_shift = torch.randint(-d_shift_max, d_shift_max + 1, (1,)).item() if d_shift_max > 0 else 0
        h_shift = torch.randint(-h_shift_max, h_shift_max + 1, (1,)).item()
        w_shift = torch.randint(-w_shift_max, w_shift_max + 1, (1,)).item()

        pad_w = (max(0, w_shift), max(0, -w_shift))
        pad_h = (max(0, h_shift), max(0, -h_shift))
        pad_d = (max(0, d_shift), max(0, -d_shift))
        
        x_padded = F.pad(x, pad_w + pad_h + pad_d, value=0)
        
        start_d = pad_d[0] if d_shift >=0 else 0
        start_h = pad_h[0] if h_shift >=0 else 0
        start_w = pad_w[0] if w_shift >=0 else 0
        
        if d_shift < 0: start_d = -d_shift
        if h_shift < 0: start_h = -h_shift
        if w_shift < 0: start_w = -w_shift

        return x_padded[:, :, start_d:start_d+D, start_h:start_h+H, start_w:start_w+W]

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import math
# from einops import rearrange

# class MedicalOnGPUAugmenter(nn.Module):
#     def __init__(self, 
#                  aug_prob=0.45, 
#                  noise_sigma=0.04,
#                  contrast_range=(0.85, 1.15),
#                  brightness_delta=0.15,
#                  gamma_range=(0.85, 1.15),
#                  rotate_range_deg=(-25, 25),
#                  scale_range=(0.8, 1.2),
#                  enable_spatial=True,
#                  # --- 新增参数 ---
#                  enable_flip=True,       # 开关翻转
#                  flip_axis=[4],          # 默认沿W轴(左右)翻转。[2,3,4]对应[D,H,W]
#                  enable_sharpen=True,    # 开关锐化
#                  sharpen_alpha=(0.5, 1.5), # 锐化强度范围
#                  channels=1              # 输入通道数，用于初始化卷积核
#                  ):
#         super().__init__()
#         self.aug_prob = aug_prob
#         self.noise_sigma = noise_sigma
#         self.contrast_range = contrast_range
#         self.brightness_delta = brightness_delta
#         self.gamma_range = gamma_range
        
#         self.rotate_range_deg = rotate_range_deg
#         self.scale_range = scale_range
#         self.enable_spatial = enable_spatial
        
#         # --- 翻转设置 ---
#         self.enable_flip = enable_flip
#         self.flip_axis = flip_axis # 4 代表 W (Batch, Channel, Depth, Height, Width)

#         # --- 锐化设置 (预生成高斯核) ---
#         self.enable_sharpen = enable_sharpen
#         self.sharpen_alpha = sharpen_alpha
#         if self.enable_sharpen:
#             # 创建一个 3x3x3 的高斯核用于模糊
#             kernel_size = 3
#             sigma = 1.0
#             x_coord = torch.arange(kernel_size) - (kernel_size - 1) / 2
#             grid = torch.stack(torch.meshgrid([x_coord, x_coord, x_coord], indexing='ij'))
#             kernel = torch.exp(-(grid ** 2).sum(0) / (2 * sigma ** 2))
#             kernel = kernel / kernel.sum()
#             # Reshape 为 (Out_C, In_C/groups, k, k, k) -> (C, 1, 3, 3, 3)
#             # 我们将使用 depth-wise convolution (groups=channels)
#             kernel = kernel.view(1, 1, kernel_size, kernel_size, kernel_size)
#             kernel = kernel.repeat(channels, 1, 1, 1, 1)
#             # 注册为 buffer，这样它会自动随模型移动到 GPU，且不会被视为可训练参数
#             self.register_buffer('gaussian_kernel', kernel)

#     def forward(self, x):
#         is_condition = False
#         if len(x.shape) == 6:
#             is_condition = True
#             global_B = x.shape[0]
#             local_B = x.shape[1]
#             x = rearrange(x, 'B N O D H W -> (B N) O D H W')
            
#         if len(x.shape) != 5:
#             raise ValueError("Input tensor must be (B, C, D, H, W)")

#         with torch.no_grad():
#             B, C, D, H, W = x.shape
#             device = x.device

#             # --- 1. 随机翻转 (Spatial - Flip) ---
#             # # 放在最前面或最后面都可以，这里放在最前面
#             # if self.enable_flip and len(self.flip_axis) > 0:
#             #     x = self.apply_random_flip(x)

#             # --- 2. 像素级增强 (Intensity) ---
            
#             # 高斯噪声
#             mask = (torch.rand(B, 1, 1, 1, 1, device=device) < self.aug_prob).float()
#             if mask.any():
#                 noise = torch.randn_like(x) * self.noise_sigma
#                 x = x + noise * mask

#             # 随机锐化 (新增)
#             # 放在噪声之后，可能会强化噪声，或者放在噪声之前。
#             # 这里建议放在噪声之后，模拟重建算法对含噪原始数据的处理
#             if self.enable_sharpen:
#                 x = self.apply_random_sharpen(x)

#             # 随机对比度
#             mask = (torch.rand(B, 1, 1, 1, 1, device=device) < self.aug_prob).float()
#             if mask.any():
#                 factor = torch.empty(B, 1, 1, 1, 1, device=device).uniform_(*self.contrast_range)
#                 x = (x - 0.5) * (factor * mask + (1 - mask)) + 0.5

#             # 随机亮度
#             mask = (torch.rand(B, 1, 1, 1, 1, device=device) < self.aug_prob).float()
#             if mask.any():
#                 delta = torch.empty(B, 1, 1, 1, 1, device=device).uniform_(-self.brightness_delta, self.brightness_delta)
#                 x = x + delta * mask

#             # 随机 Gamma
#             mask = (torch.rand(B, 1, 1, 1, 1, device=device) < self.aug_prob).float()
#             if mask.any():
#                 gamma = torch.empty(B, 1, 1, 1, 1, device=device).uniform_(*self.gamma_range)
#                 final_gamma = gamma * mask + 1.0 * (1 - mask)
#                 x = x.clamp(1e-7, 1) ** final_gamma
            
#             # 截断
#             x = torch.clamp(x, 0.0, 1.0)

#             # --- 3. 空间变换 (Spatial - Affine/Shift) ---
#             if self.enable_spatial:
#                 if torch.rand(1) < self.aug_prob:
#                     x = self.apply_affine_transform(x)

#                 if torch.rand(1) < self.aug_prob:
#                     x = self.random_shift(x, max_shift_percent=0.08)

#         if is_condition:
#             x = rearrange(x, '(B N) O D H W -> B N O D H W', B=global_B, N=local_B)
#         return x

#     def apply_random_flip(self, x):
#         """
#         批量随机翻转。
#         策略：先生成全batch翻转的结果，然后通过mask混合。
#         虽然看起来计算了多余的翻转，但在GPU上这通常比循环处理快，且保持了向量化。
#         """
#         B, C, D, H, W = x.shape
#         device = x.device
        
#         # 决定哪些样本需要翻转
#         flip_mask = (torch.rand(B, 1, 1, 1, 1, device=device) < self.aug_prob).float()
        
#         if flip_mask.any():
#             # 随机选择一个轴进行翻转 (如果 flip_axis 定义了多个)
#             # 这里简单起见，我们假设只针对 flip_axis 列表中的所有轴同时翻转，
#             # 或者你可以修改逻辑为随机选一个轴。
#             # 下面逻辑是：严格按照 flip_axis 指定的轴进行翻转
            
#             flipped_x = torch.flip(x, dims=self.flip_axis)
            
#             # 混合：需要翻转的用flipped_x，不需要的用原x
#             x = x * (1 - flip_mask) + flipped_x * flip_mask
            
#         return x

#     def apply_random_sharpen(self, x):
#         """
#         批量随机锐化 (Unsharp Masking)。
#         Formula: Sharpened = Original + alpha * (Original - Blurred)
#         """
#         B, C, D, H, W = x.shape
#         device = x.device
        
#         # 决定哪些样本需要锐化
#         sharpen_mask = (torch.rand(B, 1, 1, 1, 1, device=device) < self.aug_prob).float()
        
#         if sharpen_mask.any():
#             # 1. 计算模糊图像 (Depth-wise Conv3d)
#             # Padding=1 保持尺寸不变 (kernel=3)
#             kernel = self.gaussian_kernel.to(device)
#             # groups=C 确保每个通道独立模糊，不混合通道信息
#             blurred = F.conv3d(x, kernel, padding=1, groups=C)
            
#             # 2. 计算高频分量 (细节)
#             high_freq = x - blurred
            
#             # 3. 随机采样锐化强度 alpha
#             min_a, max_a = self.sharpen_alpha
#             alpha = torch.empty(B, 1, 1, 1, 1, device=device).uniform_(min_a, max_a)
            
#             # 4. 应用锐化
#             # 只对 mask 选中的样本应用 alpha，未选中的 alpha 视为 0 (即保持原样)
#             effective_alpha = alpha * sharpen_mask
            
#             x = x + effective_alpha * high_freq
            
#         return x

#     def apply_affine_transform(self, x):
#         # ... (保持你原有的代码不变) ...
#         # (为了节省篇幅，这里省略，请保留原本的 apply_affine_transform 实现)
#         B, C, D, H, W = x.shape
#         device = x.device
#         min_deg, max_deg = self.rotate_range_deg
#         angle_deg = torch.empty(B, device=device).uniform_(min_deg, max_deg)
#         angle_rad = angle_deg * (math.pi / 180.0)
#         min_s, max_s = self.scale_range
#         scale = torch.empty(B, device=device).uniform_(min_s, max_s)
#         inv_scale = 1.0 / scale 
#         cos_a = torch.cos(-angle_rad)
#         sin_a = torch.sin(-angle_rad)
#         theta = torch.zeros(B, 3, 4, device=device)
#         theta[:, 0, 0] = inv_scale * cos_a
#         theta[:, 0, 1] = inv_scale * (-sin_a)
#         theta[:, 1, 0] = inv_scale * sin_a
#         theta[:, 1, 1] = inv_scale * cos_a
#         theta[:, 2, 2] = inv_scale 
#         grid = F.affine_grid(theta, x.size(), align_corners=False)
#         x_transformed = F.grid_sample(x, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
#         return x_transformed

#     def random_shift(self, x, max_shift_percent=0.1):
#         # ... (保持你原有的代码不变) ...
#         # (为了节省篇幅，这里省略，请保留原本的 random_shift 实现)
#         B, C, D, H, W = x.shape
#         d_shift_max = int(D * max_shift_percent) if D > 1 else 0
#         h_shift_max = int(H * max_shift_percent)
#         w_shift_max = int(W * max_shift_percent)
#         d_shift = torch.randint(-d_shift_max, d_shift_max + 1, (1,)).item() if d_shift_max > 0 else 0
#         h_shift = torch.randint(-h_shift_max, h_shift_max + 1, (1,)).item()
#         w_shift = torch.randint(-w_shift_max, w_shift_max + 1, (1,)).item()
#         pad_w = (max(0, w_shift), max(0, -w_shift))
#         pad_h = (max(0, h_shift), max(0, -h_shift))
#         pad_d = (max(0, d_shift), max(0, -d_shift))
#         x_padded = F.pad(x, pad_w + pad_h + pad_d, value=0)
#         start_d = pad_d[0] if d_shift >=0 else 0
#         start_h = pad_h[0] if h_shift >=0 else 0
#         start_w = pad_w[0] if w_shift >=0 else 0
#         if d_shift < 0: start_d = -d_shift
#         if h_shift < 0: start_h = -h_shift
#         if w_shift < 0: start_w = -w_shift
#         return x_padded[:, :, start_d:start_d+D, start_h:start_h+H, start_w:start_w+W]
