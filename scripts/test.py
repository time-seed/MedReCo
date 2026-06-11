import torch
import torch.nn as nn
from torchvision.transforms import v2

class CXRAugmentation(nn.Module):
    def __init__(self):
        """
        针对CXR设计的增强模块
        Args:
            img_size: 目标图像大小 (H, W)，如果不需要Resize可忽略
        """
        super().__init__()
        
        # 定义增强管道
        self.transforms = v2.Compose([
            # 1. 几何变换 (注意：严禁 Flip)
            # 旋转：CXR拍摄可能会有轻微倾斜，+/- 10度是安全的
            # 平移/缩放：模拟拍摄时的位置偏差
            v2.RandomAffine(
                degrees=10,              # 旋转范围 (-10, 10)
                translate=(0.05, 0.05),  # 水平/垂直平移范围 5%
                scale=(0.95, 1.05),      # 缩放范围 95%-105%
                fill=0                   # 填充背景色，X光背景通常为黑色(0)
            ),
            
            # 2. 像素强度变换 (模拟不同设备的曝光差异)
            # 亮度与对比度对X光诊断很重要
            v2.ColorJitter(
                brightness=0.15, 
                contrast=0.15
            ),
            
            # 3. (可选) 高斯噪声，模拟传感器噪声
            # v2.GaussianNoise(sigma=0.01), 
            
            # 确保输出数据类型一致（通常医学图像归一化前是float）
            v2.ToDtype(torch.float32, scale=True),
        ])

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (B, 1, 1, H, W)
        Returns:
            out: Output tensor of shape (B, 1, 1, H, W)
        """
        # 1. 记录原始形状
        B, C, D, H, W = x.shape
        
        # 2. 维度压缩 (Squeeze)
        # 将 (B, 1, 1, H, W) -> (B, 1, H, W)
        # 大多数 2D transform 期望输入是 (B, C, H, W)
        x_2d = x.view(B, -1, H, W) 
        
        # 3. 执行增强
        # v2 transform 可以直接处理 Batch，且支持 GPU
        out_2d = self.transforms(x_2d)
        
        # 4. 维度恢复 (Unsqueeze)
        # 将 (B, 1, H, W) -> (B, 1, 1, H, W)
        out = out_2d.view(B, C, D, H, W)
        
        return out

# --- 测试代码 ---
if __name__ == "__main__":
    # 模拟输入数据: Batch=2, Channel=1, Depth=1, H=512, W=512
    # 假设数据在 GPU 上
    device = "cuda" if torch.cuda.is_available() else "cpu"
    input_tensor = torch.randn(2, 1, 1, 480, 480).to(device)
    
    augmentor = CXRAugmentation().to(device)
    
    # 前向传播
    output_tensor = augmentor(input_tensor)
    
    print(f"输入形状: {input_tensor.shape}")
    print(f"输出形状: {output_tensor.shape}")
    
    # 验证是否引入了非法值 (可选)
    print(f"Device: {output_tensor.device}")