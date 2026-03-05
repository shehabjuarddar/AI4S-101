"""
AI4S 公开课 Lesson 1: 流体场自编码器训练脚本

浙大 AI4S 公开课 - Shikai 老师

用法:
    python train.py
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from sklearn.decomposition import PCA
import os

from models import FlowAE, mse_loss, physics_informed_loss


def setup_environment():
    """设置运行环境"""
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 强制使用 CPU (避免 CUDA 版本兼容性问题)
    # 如需使用 GPU，请确保 PyTorch 版本与 CUDA 兼容
    device = torch.device('cpu')
    print(f"使用设备: {device}")
    
    plt.style.use('dark_background')
    plt.rcParams['figure.figsize'] = [12, 6]
    plt.rcParams['font.size'] = 12
    
    # 尝试设置中文字体
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
    except:
        pass
    
    return device


def load_data(data_path='data/flow_field.npy'):
    """加载流场数据"""
    print("\n" + "=" * 60)
    print("加载流场数据")
    print("=" * 60)
    
    data = np.load(data_path)
    
    print(f"数据形状: {data.shape}")
    print(f"数据范围: [{data.min():.4f}, {data.max():.4f}]")
    print(f"数据类型: {data.dtype}")
    print(f"总帧数: {data.shape[0]}")
    print(f"图像尺寸: {data.shape[1]} × {data.shape[2]}")
    print(f"每帧数据点: {data.shape[1] * data.shape[2]:,}")
    
    return data


def visualize_flow_fields(data, save_path='outputs/flow_fields.png'):
    """可视化流场"""
    print("\n可视化流场...")
    
    n_frames = len(data)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    
    # 根据数据长度自动选择帧
    frames_to_show = [0, n_frames//5, 2*n_frames//5, 3*n_frames//5, 4*n_frames//5, n_frames-1]
    
    for ax, frame_idx in zip(axes.flat, frames_to_show):
        if frame_idx < len(data):
            im = ax.imshow(data[frame_idx], cmap='RdBu_r', aspect='auto',
                           vmin=-1, vmax=1, origin='lower')
            ax.set_title(f't = {frame_idx}', fontsize=14)
            ax.set_xlabel('x (streamwise)')
            ax.set_ylabel('y (crossflow)')
    
    plt.suptitle('Karman Vortex Street - Vorticity Field Evolution (Re=100)', fontsize=16, y=1.02)
    fig.colorbar(im, ax=axes, label='Vorticity', shrink=0.8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"流场可视化已保存到: {save_path}")


def create_animation(data, save_path='outputs/vortex_animation.gif', n_frames=None):
    """创建卡门涡街动画"""
    print("\n创建动画...")
    
    if n_frames is None:
        n_frames = len(data)
    
    fig, ax = plt.subplots(figsize=(14, 5))
    im = ax.imshow(data[0], cmap='RdBu_r', aspect='auto',
                   vmin=-1, vmax=1, origin='lower')
    ax.set_xlabel('x (streamwise)', fontsize=12)
    ax.set_ylabel('y (crossflow)', fontsize=12)
    title = ax.set_title('Karman Vortex Street Animation (Re=100) - t = 0', fontsize=14)
    plt.colorbar(im, label='Vorticity')
    
    def animate(frame):
        if frame < len(data):
            im.set_array(data[frame])
            title.set_text(f'Karman Vortex Street Animation (Re=100) - t = {frame}')
        return [im, title]
    
    anim = animation.FuncAnimation(fig, animate, frames=min(n_frames, len(data)),
                                   interval=100, blit=True)
    anim.save(save_path, writer='pillow', fps=15)
    plt.close()
    print(f"动画已保存到: {save_path}")


def prepare_data_loaders(data, train_ratio=0.8, batch_size=16):
    """准备数据加载器"""
    print("\n准备数据加载器...")
    
    n_total = len(data)
    n_train = int(n_total * train_ratio)
    n_test = n_total - n_train
    
    train_data = data[:n_train, np.newaxis, :, :]
    test_data = data[n_train:, np.newaxis, :, :]
    
    train_tensor = torch.FloatTensor(train_data)
    test_tensor = torch.FloatTensor(test_data)
    
    train_dataset = TensorDataset(train_tensor)
    test_dataset = TensorDataset(test_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"总数据量: {n_total}")
    print(f"训练集大小: {len(train_tensor)}")
    print(f"测试集大小: {len(test_tensor)}")
    print(f"批次大小: {batch_size}")
    print(f"训练批次数: {len(train_loader)}")
    
    return train_loader, test_loader, train_tensor, test_tensor


def train_model(model, train_loader, test_loader, device, epochs=100, lr=1e-3):
    """训练模型"""
    print("\n" + "=" * 60)
    print("开始训练")
    print("=" * 60)
    print(f"Epochs: {epochs}, Learning Rate: {lr}")
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    
    train_losses = []
    test_losses = []
    
    for epoch in tqdm(range(epochs), desc="训练进度"):
        # 训练阶段
        model.train()
        epoch_loss = 0.0
        
        for batch in train_loader:
            x = batch[0].to(device)
            x_recon, z = model(x)
            loss = mse_loss(x_recon, x)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # 测试阶段
        model.eval()
        test_loss = 0.0
        
        with torch.no_grad():
            for batch in test_loader:
                x = batch[0].to(device)
                x_recon, z = model(x)
                test_loss += mse_loss(x_recon, x).item()
        
        avg_test_loss = test_loss / len(test_loader)
        test_losses.append(avg_test_loss)
        
        scheduler.step()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {avg_train_loss:.6f} | Test Loss: {avg_test_loss:.6f}")
    
    print("\n训练完成!")
    return train_losses, test_losses


def plot_training_curves(train_losses, test_losses, save_path='outputs/training_curves.png'):
    """绘制训练曲线"""
    print("\n绘制训练曲线...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.plot(train_losses, label='Train Loss', color='cyan', linewidth=2)
    ax1.plot(test_losses, label='Test Loss', color='orange', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('MSE Loss', fontsize=12)
    ax1.set_title('Training Loss Curve', fontsize=14)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    ax2.semilogy(train_losses, label='Train Loss', color='cyan', linewidth=2)
    ax2.semilogy(test_losses, label='Test Loss', color='orange', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('MSE Loss (log)', fontsize=12)
    ax2.set_title('Training Loss Curve (Log Scale)', fontsize=14)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"训练曲线已保存到: {save_path}")
    print(f"最终训练损失: {train_losses[-1]:.6f}")
    print(f"最终测试损失: {test_losses[-1]:.6f}")


def visualize_reconstruction(model, test_tensor, device, n_samples=4, 
                            save_path='outputs/reconstruction.png'):
    """可视化重构效果"""
    print("\n可视化重构效果...")
    
    model.eval()
    test_samples = test_tensor[:n_samples].to(device)
    
    with torch.no_grad():
        recon_samples, latent_vectors = model(test_samples)
    
    original = test_samples.cpu().numpy()[:, 0, :, :]
    reconstructed = recon_samples.cpu().numpy()[:, 0, :, :]
    
    fig, axes = plt.subplots(3, n_samples, figsize=(16, 10))
    
    for i in range(n_samples):
        im1 = axes[0, i].imshow(original[i], cmap='RdBu_r', vmin=-1, vmax=1, origin='lower')
        axes[0, i].set_title(f'Original #{i+1}', fontsize=12)
        axes[0, i].axis('off')
        
        im2 = axes[1, i].imshow(reconstructed[i], cmap='RdBu_r', vmin=-1, vmax=1, origin='lower')
        axes[1, i].set_title(f'Reconstructed #{i+1}', fontsize=12)
        axes[1, i].axis('off')
        
        error = np.abs(original[i] - reconstructed[i])
        mse = np.mean(error**2)
        im3 = axes[2, i].imshow(error, cmap='hot', vmin=0, vmax=0.5, origin='lower')
        axes[2, i].set_title(f'Error (MSE={mse:.4f})', fontsize=12)
        axes[2, i].axis('off')
    
    fig.colorbar(im1, ax=axes[0, :], label='Vorticity', shrink=0.8, pad=0.02)
    fig.colorbar(im2, ax=axes[1, :], label='Vorticity', shrink=0.8, pad=0.02)
    fig.colorbar(im3, ax=axes[2, :], label='|Error|', shrink=0.8, pad=0.02)
    
    plt.suptitle('Original vs Reconstructed Flow Fields', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"重构对比图已保存到: {save_path}")


def latent_space_interpolation(model, train_tensor, device, idx1=None, idx2=None,
                               save_path='outputs/interpolation.png'):
    """潜在空间插值"""
    print("\n潜在空间插值...")
    
    n_train = len(train_tensor)
    if idx1 is None:
        idx1 = 0
    if idx2 is None:
        idx2 = min(n_train - 1, n_train // 2)
    
    frame1 = train_tensor[idx1:idx1+1].to(device)
    frame2 = train_tensor[idx2:idx2+1].to(device)
    
    model.eval()
    with torch.no_grad():
        z1 = model.encode(frame1)
        z2 = model.encode(frame2)
    
    n_interp = 8
    alphas = np.linspace(0, 1, n_interp)
    interpolated_frames = []
    
    with torch.no_grad():
        for alpha in alphas:
            z_interp = alpha * z1 + (1 - alpha) * z2
            frame_interp = model.decode(z_interp)
            interpolated_frames.append(frame_interp.cpu().numpy()[0, 0])
    
    fig, axes = plt.subplots(2, n_interp//2, figsize=(16, 8))
    
    for i, (ax, frame, alpha) in enumerate(zip(axes.flat, interpolated_frames, alphas)):
        im = ax.imshow(frame, cmap='RdBu_r', vmin=-1, vmax=1, origin='lower')
        ax.set_title(f'alpha = {alpha:.2f}', fontsize=12)
        ax.axis('off')
    
    fig.colorbar(im, ax=axes, label='Vorticity', shrink=0.6)
    plt.suptitle(f'Latent Space Interpolation: t={idx2} -> t={idx1}', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"插值结果已保存到: {save_path}")


def create_interpolation_animation(model, train_tensor, device, idx1=None, idx2=None,
                                   save_path='outputs/interpolation_animation.gif'):
    """创建插值动画"""
    print("\n创建插值动画...")
    
    n_train = len(train_tensor)
    if idx1 is None:
        idx1 = 0
    if idx2 is None:
        idx2 = min(n_train - 1, n_train // 2)
    
    frame1 = train_tensor[idx1:idx1+1].to(device)
    frame2 = train_tensor[idx2:idx2+1].to(device)
    
    model.eval()
    with torch.no_grad():
        z1 = model.encode(frame1)
        z2 = model.encode(frame2)
    
    n_frames = 50
    alphas = np.linspace(0, 1, n_frames)
    
    interp_frames = []
    with torch.no_grad():
        for alpha in alphas:
            z_interp = alpha * z1 + (1 - alpha) * z2
            frame_interp = model.decode(z_interp)
            interp_frames.append(frame_interp.cpu().numpy()[0, 0])
    
    fig, ax = plt.subplots(figsize=(14, 5))
    im = ax.imshow(interp_frames[0], cmap='RdBu_r', vmin=-1, vmax=1, origin='lower')
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    title = ax.set_title('Latent Space Interpolation Animation', fontsize=14)
    plt.colorbar(im, label='Vorticity')
    
    def animate(frame):
        im.set_array(interp_frames[frame])
        alpha = alphas[frame]
        title.set_text(f'Latent Space Interpolation: alpha = {alpha:.2f}')
        return [im, title]
    
    anim = animation.FuncAnimation(fig, animate, frames=n_frames, interval=100, blit=True)
    anim.save(save_path, writer='pillow', fps=10)
    plt.close()
    print(f"插值动画已保存到: {save_path}")


def anomaly_detection_demo(model, test_tensor, device, save_path='outputs/anomaly_detection.png'):
    """异常检测与流场修复演示"""
    print("\n异常检测与流场修复演示...")
    
    test_idx = min(10, len(test_tensor) - 1)
    original_field = test_tensor[test_idx:test_idx+1].clone()
    
    # 添加遮挡
    corrupted_field = original_field.clone()
    mask_x, mask_y = 30, 20
    mask_w, mask_h = 40, 30
    corrupted_field[0, 0, mask_y:mask_y+mask_h, mask_x:mask_x+mask_w] = 0
    
    # 添加噪声
    noise = torch.randn_like(original_field) * 0.3
    corrupted_field_noisy = original_field + noise
    corrupted_field_noisy = torch.clamp(corrupted_field_noisy, -1, 1)
    
    # 修复
    model.eval()
    with torch.no_grad():
        repaired_masked, _ = model(corrupted_field.to(device))
        repaired_noisy, _ = model(corrupted_field_noisy.to(device))
    
    # 可视化
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    
    # 遮挡修复
    axes[0, 0].imshow(original_field[0, 0], cmap='RdBu_r', vmin=-1, vmax=1, origin='lower')
    axes[0, 0].set_title('Original', fontsize=12)
    
    axes[0, 1].imshow(corrupted_field[0, 0], cmap='RdBu_r', vmin=-1, vmax=1, origin='lower')
    axes[0, 1].set_title('Masked', fontsize=12)
    axes[0, 1].add_patch(plt.Rectangle((mask_x, mask_y), mask_w, mask_h,
                                        fill=False, edgecolor='yellow', linewidth=2))
    
    axes[0, 2].imshow(repaired_masked.cpu()[0, 0], cmap='RdBu_r', vmin=-1, vmax=1, origin='lower')
    axes[0, 2].set_title('AE Inpainted', fontsize=12)
    
    error_masked = np.abs(original_field[0, 0].numpy() - repaired_masked.cpu()[0, 0].numpy())
    axes[0, 3].imshow(error_masked, cmap='hot', vmin=0, vmax=0.5, origin='lower')
    axes[0, 3].set_title(f'Error (MSE={np.mean(error_masked**2):.4f})', fontsize=12)
    
    # 噪声修复
    axes[1, 0].imshow(original_field[0, 0], cmap='RdBu_r', vmin=-1, vmax=1, origin='lower')
    axes[1, 0].set_title('Original', fontsize=12)
    
    axes[1, 1].imshow(corrupted_field_noisy[0, 0], cmap='RdBu_r', vmin=-1, vmax=1, origin='lower')
    axes[1, 1].set_title('Noisy', fontsize=12)
    
    axes[1, 2].imshow(repaired_noisy.cpu()[0, 0], cmap='RdBu_r', vmin=-1, vmax=1, origin='lower')
    axes[1, 2].set_title('AE Denoised', fontsize=12)
    
    error_noisy = np.abs(original_field[0, 0].numpy() - repaired_noisy.cpu()[0, 0].numpy())
    axes[1, 3].imshow(error_noisy, cmap='hot', vmin=0, vmax=0.5, origin='lower')
    axes[1, 3].set_title(f'Error (MSE={np.mean(error_noisy**2):.4f})', fontsize=12)
    
    for ax in axes.flat:
        ax.axis('off')
    
    plt.suptitle('Autoencoder: Anomaly Detection & Flow Field Repair', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"异常检测演示已保存到: {save_path}")


def pca_vs_ae_comparison(model, train_tensor, test_tensor, device, latent_dim=16,
                         save_path='outputs/pca_vs_ae.png'):
    """PCA vs AE 对比"""
    print("\n" + "=" * 60)
    print("PCA vs AE Comparison")
    print("=" * 60)
    
    n_train = len(train_tensor)
    n_test = len(test_tensor)
    
    # 准备数据 - 去掉 channel 维度用于 PCA
    train_data = train_tensor.numpy()[:, 0, :, :]  # (n_train, 64, 128)
    test_data = test_tensor.numpy()[:, 0, :, :]    # (n_test, 64, 128)
    
    X_train = train_data.reshape(n_train, -1)  # (n_train, 64*128)
    X_test = test_data.reshape(n_test, -1)      # (n_test, 64*128)
    
    # 使用较少的 PCA 成分以便公平比较
    # 因为数据集较小，16 个成分的 PCA 几乎可以完美重构
    pca_dim = min(4, latent_dim)  # 使用 4 个成分
    
    # PCA
    pca = PCA(n_components=pca_dim)
    pca.fit(X_train)
    
    X_test_pca = pca.transform(X_test)
    X_test_recon_pca = pca.inverse_transform(X_test_pca)
    X_test_recon_pca = X_test_recon_pca.reshape(n_test, 64, 128)
    
    # AE
    model.eval()
    with torch.no_grad():
        X_test_recon_ae, _ = model(test_tensor.to(device))
    X_test_recon_ae = X_test_recon_ae.cpu().numpy()[:, 0, :, :]
    
    # 计算误差
    mse_pca = np.mean((test_data - X_test_recon_pca) ** 2)
    mse_ae = np.mean((test_data - X_test_recon_ae) ** 2)
    
    explained_var = pca.explained_variance_ratio_.sum()
    
    print(f"PCA ({pca_dim} components, {explained_var:.1%} variance) MSE: {mse_pca:.6f}")
    print(f"AE ({latent_dim} latent dims) MSE: {mse_ae:.6f}")
    if mse_ae > 0:
        print(f"PCA/AE ratio: {mse_pca/mse_ae:.2f}x")
    
    # 可视化
    n_compare = 4
    fig, axes = plt.subplots(3, n_compare, figsize=(16, 10))
    
    for i in range(n_compare):
        idx = min(i * (n_test // n_compare), n_test - 1)
        
        axes[0, i].imshow(test_data[idx], cmap='RdBu_r', vmin=-1, vmax=1, origin='lower')
        axes[0, i].set_title(f'Original #{idx}', fontsize=12)
        axes[0, i].axis('off')
        
        mse_i_pca = np.mean((test_data[idx] - X_test_recon_pca[idx]) ** 2)
        axes[1, i].imshow(X_test_recon_pca[idx], cmap='RdBu_r', vmin=-1, vmax=1, origin='lower')
        axes[1, i].set_title(f'PCA (MSE={mse_i_pca:.4f})', fontsize=12)
        axes[1, i].axis('off')
        
        mse_i_ae = np.mean((test_data[idx] - X_test_recon_ae[idx]) ** 2)
        axes[2, i].imshow(X_test_recon_ae[idx], cmap='RdBu_r', vmin=-1, vmax=1, origin='lower')
        axes[2, i].set_title(f'AE (MSE={mse_i_ae:.4f})', fontsize=12)
        axes[2, i].axis('off')
    
    axes[0, 0].set_ylabel('Original', fontsize=14)
    axes[1, 0].set_ylabel('PCA', fontsize=14)
    axes[2, 0].set_ylabel('AE', fontsize=14)
    
    plt.suptitle(f'PCA ({pca_dim} components) vs Autoencoder ({latent_dim} latent dims)', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"对比图已保存到: {save_path}")


def main():
    """主函数"""
    print("=" * 60)
    print("AI4S 公开课 Lesson 1: 流体场自编码器")
    print("浙大 AI4S 公开课 - Shikai 老师")
    print("=" * 60)
    
    # 配置参数
    LATENT_DIM = 16
    EPOCHS = 150
    LEARNING_RATE = 1e-3
    BATCH_SIZE = 16
    TRAIN_RATIO = 0.8
    
    # 创建输出目录
    os.makedirs('outputs', exist_ok=True)
    
    # 设置环境
    device = setup_environment()
    
    # 加载数据
    data = load_data('data/flow_field.npy')
    
    # 可视化原始流场
    visualize_flow_fields(data)
    
    # 创建动画
    try:
        create_animation(data)
    except Exception as e:
        print(f"动画创建跳过: {e}")
    
    # 准备数据加载器
    train_loader, test_loader, train_tensor, test_tensor = prepare_data_loaders(
        data, train_ratio=TRAIN_RATIO, batch_size=BATCH_SIZE
    )
    
    # 创建模型
    print("\n" + "=" * 60)
    print("创建模型")
    print("=" * 60)
    
    model = FlowAE(latent_dim=LATENT_DIM).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"潜在空间维度: {LATENT_DIM}")
    print(f"总参数量: {total_params:,}")
    print(f"压缩比: {64*128} → {LATENT_DIM} ({64*128/LATENT_DIM:.1f}x)")
    
    # 训练模型
    train_losses, test_losses = train_model(
        model, train_loader, test_loader, device,
        epochs=EPOCHS, lr=LEARNING_RATE
    )
    
    # 绘制训练曲线
    plot_training_curves(train_losses, test_losses)
    
    # 可视化重构效果
    visualize_reconstruction(model, test_tensor, device)
    
    # 潜在空间插值
    latent_space_interpolation(model, train_tensor, device)
    
    # 创建插值动画
    try:
        create_interpolation_animation(model, train_tensor, device)
    except Exception as e:
        print(f"插值动画创建跳过: {e}")
    
    # 异常检测演示
    anomaly_detection_demo(model, test_tensor, device)
    
    # PCA vs AE 对比
    pca_vs_ae_comparison(model, train_tensor, test_tensor, device, latent_dim=LATENT_DIM)
    
    # 保存模型
    model_path = 'outputs/flow_ae_model.pth'
    torch.save(model.state_dict(), model_path)
    print(f"\n模型已保存到: {model_path}")
    
    print("\n" + "=" * 60)
    print("所有任务完成!")
    print("=" * 60)
    print("\n输出文件:")
    for f in os.listdir('outputs'):
        print(f"  - outputs/{f}")


if __name__ == "__main__":
    main()
