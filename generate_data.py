"""
流体流动数据生成和处理模块
用于加载和预处理圆柱绕流数据（卡门涡街）

数据来源: http://dmdbook.com/DATA.zip
Re = 100 圆柱绕流直接数值模拟数据
"""

import numpy as np
import scipy.io as sio
from scipy.interpolate import RegularGridInterpolator
import os


def load_cylinder_data(mat_file='data/DATA/FLUIDS/CYLINDER_ALL.mat'):
    """
    加载圆柱绕流 MAT 数据文件
    
    Parameters:
    -----------
    mat_file : str
        MAT 文件路径
        
    Returns:
    --------
    vorticity : ndarray
        涡量场数据，形状 (n_snapshots, ny, nx)
    """
    print(f"加载数据文件: {mat_file}")
    
    data = sio.loadmat(mat_file)
    
    # 获取涡量数据
    vort_all = data['VORTALL']  # shape: (89351, 151)
    
    # 获取网格尺寸 (449 x 199)
    # 89351 = 449 * 199
    nx = 449
    ny = 199
    n_snapshots = vort_all.shape[1]
    
    print(f"原始数据形状: {vort_all.shape}")
    print(f"网格尺寸: {nx} x {ny}")
    print(f"快照数量: {n_snapshots}")
    
    # 重塑为 (n_snapshots, ny, nx)
    vorticity = np.zeros((n_snapshots, ny, nx), dtype=np.float32)
    for i in range(n_snapshots):
        # MATLAB 按列存储，需要转置
        vorticity[i] = vort_all[:, i].reshape((nx, ny)).T
    
    print(f"重塑后数据形状: {vorticity.shape}")
    print(f"涡量范围: [{vorticity.min():.4f}, {vorticity.max():.4f}]")
    
    return vorticity


def resample_to_target_size(data, target_shape=(64, 128)):
    """
    将数据重采样到目标尺寸
    
    Parameters:
    -----------
    data : ndarray
        输入数据，形状 (n_snapshots, ny, nx)
    target_shape : tuple
        目标形状 (target_ny, target_nx)
        
    Returns:
    --------
    resampled : ndarray
        重采样后的数据
    """
    n_snapshots, ny, nx = data.shape
    target_ny, target_nx = target_shape
    
    print(f"重采样: ({ny}, {nx}) -> ({target_ny}, {target_nx})")
    
    # 创建原始网格坐标
    y_orig = np.linspace(0, 1, ny)
    x_orig = np.linspace(0, 1, nx)
    
    # 创建目标网格坐标
    y_target = np.linspace(0, 1, target_ny)
    x_target = np.linspace(0, 1, target_nx)
    
    # 创建目标网格
    yy, xx = np.meshgrid(y_target, x_target, indexing='ij')
    target_points = np.stack([yy.ravel(), xx.ravel()], axis=-1)
    
    # 重采样每个快照
    resampled = np.zeros((n_snapshots, target_ny, target_nx), dtype=np.float32)
    
    for i in range(n_snapshots):
        interpolator = RegularGridInterpolator(
            (y_orig, x_orig), 
            data[i], 
            method='linear',
            bounds_error=False,
            fill_value=0
        )
        resampled[i] = interpolator(target_points).reshape(target_ny, target_nx)
    
    return resampled


def normalize_data(data, method='minmax', target_range=(-1, 1)):
    """
    归一化数据
    
    Parameters:
    -----------
    data : ndarray
        输入数据
    method : str
        归一化方法 ('minmax' 或 'zscore')
    target_range : tuple
        目标范围（仅用于 minmax）
        
    Returns:
    --------
    normalized : ndarray
        归一化后的数据
    params : dict
        归一化参数（用于反归一化）
    """
    if method == 'minmax':
        data_min = data.min()
        data_max = data.max()
        
        # 归一化到 [0, 1]
        normalized = (data - data_min) / (data_max - data_min + 1e-8)
        
        # 缩放到目标范围
        low, high = target_range
        normalized = normalized * (high - low) + low
        
        params = {
            'method': 'minmax',
            'data_min': data_min,
            'data_max': data_max,
            'target_range': target_range
        }
        
    elif method == 'zscore':
        data_mean = data.mean()
        data_std = data.std()
        
        normalized = (data - data_mean) / (data_std + 1e-8)
        
        params = {
            'method': 'zscore',
            'data_mean': data_mean,
            'data_std': data_std
        }
    else:
        raise ValueError(f"未知的归一化方法: {method}")
    
    print(f"归一化后数据范围: [{normalized.min():.4f}, {normalized.max():.4f}]")
    
    return normalized.astype(np.float32), params


def generate_synthetic_vortex_data(n_snapshots=150, ny=64, nx=128, seed=42):
    """
    生成合成的卡门涡街数据（备用方案）
    
    Parameters:
    -----------
    n_snapshots : int
        快照数量
    ny, nx : int
        网格尺寸
    seed : int
        随机种子
        
    Returns:
    --------
    vorticity : ndarray
        合成涡量场数据
    """
    np.random.seed(seed)
    
    print(f"生成合成卡门涡街数据: {n_snapshots} 快照, {ny}x{nx} 网格")
    
    # 创建网格
    x = np.linspace(-2, 10, nx)
    y = np.linspace(-3, 3, ny)
    X, Y = np.meshgrid(x, y)
    
    # 涡街参数
    vortex_spacing = 2.0  # 涡旋间距
    vortex_strength = 2.0  # 涡旋强度
    convection_speed = 0.5  # 对流速度
    
    vorticity = np.zeros((n_snapshots, ny, nx), dtype=np.float32)
    
    for t in range(n_snapshots):
        time = t * 0.1
        field = np.zeros((ny, nx))
        
        # 添加交替的正负涡旋
        for i in range(-2, 15):
            # 上排涡旋（正）
            x_center = i * vortex_spacing - convection_speed * time
            y_center = 0.8
            r2 = (X - x_center)**2 + (Y - y_center)**2
            field += vortex_strength * np.exp(-r2 / 0.3)
            
            # 下排涡旋（负）
            x_center = (i + 0.5) * vortex_spacing - convection_speed * time
            y_center = -0.8
            r2 = (X - x_center)**2 + (Y - y_center)**2
            field -= vortex_strength * np.exp(-r2 / 0.3)
        
        # 添加圆柱遮挡效果
        cylinder_mask = (X**2 + Y**2) < 0.5**2
        field[cylinder_mask] = 0
        
        vorticity[t] = field
    
    return vorticity


def prepare_flow_data(
    mat_file='data/DATA/FLUIDS/CYLINDER_ALL.mat',
    target_shape=(64, 128),
    output_file='data/flow_field.npy',
    use_synthetic=False
):
    """
    准备流体流动数据的主函数
    
    Parameters:
    -----------
    mat_file : str
        MAT 数据文件路径
    target_shape : tuple
        目标网格尺寸
    output_file : str
        输出文件路径
    use_synthetic : bool
        是否使用合成数据
        
    Returns:
    --------
    data : ndarray
        处理后的数据
    """
    if use_synthetic or not os.path.exists(mat_file):
        print("使用合成数据...")
        vorticity = generate_synthetic_vortex_data(
            n_snapshots=150, 
            ny=target_shape[0], 
            nx=target_shape[1]
        )
    else:
        # 加载真实数据
        vorticity = load_cylinder_data(mat_file)
        
        # 重采样到目标尺寸
        vorticity = resample_to_target_size(vorticity, target_shape)
    
    # 归一化到 [-1, 1]
    vorticity_normalized, norm_params = normalize_data(
        vorticity, 
        method='minmax', 
        target_range=(-1, 1)
    )
    
    # 保存数据
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    np.save(output_file, vorticity_normalized)
    
    # 保存归一化参数
    params_file = output_file.replace('.npy', '_params.npy')
    np.save(params_file, norm_params)
    
    print(f"数据已保存到: {output_file}")
    print(f"最终数据形状: {vorticity_normalized.shape}")
    
    return vorticity_normalized


if __name__ == '__main__':
    # 处理数据
    data = prepare_flow_data(
        mat_file='data/DATA/FLUIDS/CYLINDER_ALL.mat',
        target_shape=(64, 128),
        output_file='data/flow_field.npy',
        use_synthetic=False
    )
    
    print("\n数据处理完成!")
    print(f"数据形状: {data.shape}")
    print(f"数据范围: [{data.min():.4f}, {data.max():.4f}]")
