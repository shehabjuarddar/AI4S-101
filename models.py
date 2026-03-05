"""
流体场自编码器模型 (Flow Field Autoencoder)

用于学习流体涡量场的低维表征
- Encoder: 卷积神经网络，将 64x128 的流场压缩到 16 维潜在空间
- Decoder: 转置卷积网络，将 16 维向量重构回 64x128 的流场

浙大 AI4S 公开课 - Shikai 老师
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    """
    编码器网络
    
    将输入的流场图像 (1, 64, 128) 编码为 latent_dim 维的潜在向量
    
    网络结构:
    - Conv2d(1, 16, 4, stride=2) -> (16, 32, 64)
    - Conv2d(16, 32, 4, stride=2) -> (32, 16, 32)
    - Conv2d(32, 64, 4, stride=2) -> (64, 8, 16)
    - Flatten -> 64 * 8 * 16 = 8192
    - Linear(8192, latent_dim) -> latent_dim
    """
    
    def __init__(self, latent_dim=16):
        super().__init__()
        
        # 卷积层: 通道数 1 -> 16 -> 32 -> 64
        # stride=2 实现下采样
        self.conv1 = nn.Conv2d(1, 16, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        
        # 批归一化层
        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(64)
        
        # 全连接层: 将特征图展平后映射到潜在空间
        # 输入尺寸 64x128 经过 3 次 stride=2 下采样后变为 8x16
        self.fc = nn.Linear(64 * 8 * 16, latent_dim)
        
    def forward(self, x):
        # x: (batch, 1, 64, 128)
        
        # 卷积块 1
        x = self.conv1(x)  # -> (batch, 16, 32, 64)
        x = self.bn1(x)
        x = F.leaky_relu(x, 0.2)
        
        # 卷积块 2
        x = self.conv2(x)  # -> (batch, 32, 16, 32)
        x = self.bn2(x)
        x = F.leaky_relu(x, 0.2)
        
        # 卷积块 3
        x = self.conv3(x)  # -> (batch, 64, 8, 16)
        x = self.bn3(x)
        x = F.leaky_relu(x, 0.2)
        
        # 展平并映射到潜在空间
        x = x.view(x.size(0), -1)  # -> (batch, 8192)
        z = self.fc(x)  # -> (batch, latent_dim)
        
        return z


class Decoder(nn.Module):
    """
    解码器网络
    
    将 latent_dim 维的潜在向量解码回 (1, 64, 128) 的流场图像
    
    网络结构:
    - Linear(latent_dim, 8192) -> 8192
    - Reshape -> (64, 8, 16)
    - ConvTranspose2d(64, 32, 4, stride=2) -> (32, 16, 32)
    - ConvTranspose2d(32, 16, 4, stride=2) -> (16, 32, 64)
    - ConvTranspose2d(16, 1, 4, stride=2) -> (1, 64, 128)
    - Tanh -> 输出范围 [-1, 1]
    """
    
    def __init__(self, latent_dim=16):
        super().__init__()
        
        # 全连接层: 将潜在向量映射回特征图尺寸
        self.fc = nn.Linear(latent_dim, 64 * 8 * 16)
        
        # 转置卷积层: 通道数 64 -> 32 -> 16 -> 1
        # stride=2 实现上采样
        self.deconv1 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=1)
        
        # 批归一化层
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(16)
        
    def forward(self, z):
        # z: (batch, latent_dim)
        
        # 全连接层并重塑为特征图
        x = self.fc(z)  # -> (batch, 8192)
        x = F.leaky_relu(x, 0.2)
        x = x.view(x.size(0), 64, 8, 16)  # -> (batch, 64, 8, 16)
        
        # 转置卷积块 1
        x = self.deconv1(x)  # -> (batch, 32, 16, 32)
        x = self.bn1(x)
        x = F.leaky_relu(x, 0.2)
        
        # 转置卷积块 2
        x = self.deconv2(x)  # -> (batch, 16, 32, 64)
        x = self.bn2(x)
        x = F.leaky_relu(x, 0.2)
        
        # 转置卷积块 3 + Tanh 激活
        x = self.deconv3(x)  # -> (batch, 1, 64, 128)
        x = torch.tanh(x)  # 输出范围 [-1, 1]，匹配数据分布
        
        return x


class FlowAE(nn.Module):
    """
    流体场自编码器 (Flow Field Autoencoder)
    
    将高维流场数据 (64x128 = 8192 维) 压缩到低维潜在空间 (默认 16 维)
    这就像用 16 个数字来表示整个流场的"数字全息图"
    
    参数:
        latent_dim: 潜在空间维度 (默认 16)
                   可以尝试改变这个值来观察重构质量的变化
    """
    
    def __init__(self, latent_dim=16):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)
        
    def encode(self, x):
        """编码: 流场 -> 潜在向量"""
        return self.encoder(x)
    
    def decode(self, z):
        """解码: 潜在向量 -> 流场"""
        return self.decoder(z)
    
    def forward(self, x):
        """前向传播: 流场 -> 潜在向量 -> 重构流场"""
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, z


def mse_loss(x_recon, x):
    """
    均方误差损失 (Mean Squared Error Loss)
    
    衡量重构流场与原始流场之间的差异
    """
    return F.mse_loss(x_recon, x)


def physics_informed_loss(x_recon, x, lambda_physics=0.1):
    """
    物理一致性损失 (Physics-Informed Loss)
    
    除了重构误差外，还考虑物理约束:
    - 连续性方程: ∂u/∂x + ∂v/∂y = 0 (不可压缩流体)
    - 这里简化为梯度平滑性约束
    
    参数:
        x_recon: 重构流场
        x: 原始流场
        lambda_physics: 物理约束权重
    """
    # 重构误差
    recon_loss = F.mse_loss(x_recon, x)
    
    # 梯度平滑性约束 (鼓励重构流场具有平滑的空间梯度)
    # 计算 x 方向梯度
    grad_x = x_recon[:, :, :, 1:] - x_recon[:, :, :, :-1]
    # 计算 y 方向梯度
    grad_y = x_recon[:, :, 1:, :] - x_recon[:, :, :-1, :]
    
    # 梯度的二阶导数 (拉普拉斯算子的近似)
    grad_xx = grad_x[:, :, :, 1:] - grad_x[:, :, :, :-1]
    grad_yy = grad_y[:, :, 1:, :] - grad_y[:, :, :-1, :]
    
    # 平滑性损失
    smoothness_loss = torch.mean(grad_xx ** 2) + torch.mean(grad_yy ** 2)
    
    # 总损失
    total_loss = recon_loss + lambda_physics * smoothness_loss
    
    return total_loss, recon_loss, smoothness_loss


if __name__ == "__main__":
    # 测试模型
    print("测试 FlowAE 模型...")
    
    # 创建模型
    model = FlowAE(latent_dim=16)
    
    # 打印模型结构
    print("\n模型结构:")
    print(model)
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    
    # 测试前向传播
    batch_size = 4
    x = torch.randn(batch_size, 1, 64, 128)
    
    print(f"\n输入形状: {x.shape}")
    
    x_recon, z = model(x)
    
    print(f"潜在向量形状: {z.shape}")
    print(f"重构输出形状: {x_recon.shape}")
    
    # 测试损失函数
    loss = mse_loss(x_recon, x)
    print(f"\nMSE Loss: {loss.item():.4f}")
    
    total_loss, recon_loss, smooth_loss = physics_informed_loss(x_recon, x)
    print(f"Physics-Informed Loss: {total_loss.item():.4f}")
    print(f"  - Reconstruction: {recon_loss.item():.4f}")
    print(f"  - Smoothness: {smooth_loss.item():.4f}")
    
    print("\n模型测试完成!")
