import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib
# 添加重采样相关导入
from scipy import signal as scipy_signal
from scipy.signal import butter, filtfilt, resample

matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 导入配置
try:
    from config import Config
    # 计算step_size
    STEP_SIZE = int(Config.WINDOW_SIZE * (1 - Config.OVERLAP_RATIO))
except ImportError:
    # 如果导入失败，使用默认值
    class Config:
        WINDOW_SIZE = 4096
        OVERLAP_RATIO = 0.5
        BATCH_SIZE = 32
        TRAIN_RATIO = 0.8
        TRANSFORM_METHOD = 'stft'
    STEP_SIZE = 2048

class BearingDataset(Dataset):
    def __init__(self, data_path, window_size=8192, step_size=STEP_SIZE, 
                 transform_to_2d=True, transform_method='stft', enable_resampling=True, target_fs=12000):
        self.data_path = data_path
        self.window_size = window_size
        self.step_size = step_size 
        self.transform_to_2d = transform_to_2d
        self.transform_method = transform_method
        
        # 重采样相关参数
        self.enable_resampling = enable_resampling
        self.target_fs = target_fs  # 目标采样频率，默认12kHz
        
        # 简化为4类分类故障映射
        self.fault_mapping = {
            'Normal': 0,    # N - 正常
            'IR': 1,        # IR - 内圈故障（所有尺寸合并）
            'B': 2,         # B - 滚动体故障（所有尺寸合并）
            'OR': 3         # OR - 外圈故障（所有尺寸合并）
        }
        
        # 定义要处理的故障类型（只处理B、IR、OR）
        self.target_fault_types = ['B', 'IR', 'OR']
        
        self.samples = []
        self.labels = []
        self.file_info = []
        self.raw_signals = {}  # 存储原始信号用于可视化
        
        # 加载数据
        self._load_data()
        
        # 标准化数据
        self._normalize_data()
        
        # 打印类别分布
        self._print_class_distribution()
    
    def _resample_signal(self, signal, original_fs, target_fs):
        """
        重采样信号到目标采样频率
        
        Args:
            signal: 原始信号
            original_fs: 原始采样频率
            target_fs: 目标采样频率
            
        Returns:
            resampled_signal: 重采样后的信号
        """
        if not self.enable_resampling or original_fs == target_fs:
            return signal
        
        # 设计抗混叠滤波器
        nyquist_freq = target_fs / 2
        cutoff_freq = nyquist_freq * 0.9  # 设置截止频率为奈奎斯特频率的90%
        
        # 8阶Butterworth低通滤波器
        b, a = butter(8, cutoff_freq / (original_fs / 2), btype='low')
        
        # 应用抗混叠滤波器
        filtered_signal = filtfilt(b, a, signal)
        
        # 计算重采样比例
        resample_ratio = target_fs / original_fs
        
        # 重采样
        num_samples = int(len(filtered_signal) * resample_ratio)
        resampled_signal = resample(filtered_signal, num_samples)
        
        return resampled_signal
    
    def _load_data(self):
        """加载数据 - 只加载指定的三个文件夹和特定故障类型"""
        print("开始加载轴承数据...")
        if self.enable_resampling:
            print(f"启用重采样功能，目标采样频率: {self.target_fs}Hz")
        
        # 定义要使用的数据源及其采样频率 - 排除12kHz_FE_data
        data_sources = [
            ('48kHz_Normal_data', 48000, 'normal'),
            ('48kHz_DE_data', 48000, 'fault'),
            ('12kHz_DE_data', 12000, 'fault')
            # 排除 12kHz_FE_data，包含所有其他数据
        ]
        
        for folder_name, fs, data_type in data_sources:
            folder_path = os.path.join(self.data_path, folder_name)
            if os.path.exists(folder_path):
                print(f"处理 {folder_name} 数据 (采样频率: {fs}Hz)...")
                if data_type == 'normal':
                    self._process_normal_data_with_fs(folder_path, fs)
                else:
                    self._process_fault_data_with_fs(folder_path, fs)
            else:
                print(f"警告: 未找到 {folder_name} 文件夹")
        
        print(f"数据加载完成: 总共 {len(self.samples)} 个样本")
        print("目标分类: B(滚动体故障), IR(内圈故障), OR(外圈故障), N(正常)")
    
    def _process_normal_data_with_fs(self, normal_folder, fs):
        """处理正常数据 - 使用指定采样频率"""
        for file_name in os.listdir(normal_folder):
            if file_name.endswith('.mat'):
                file_path = os.path.join(normal_folder, file_name)
                
                try:
                    # 加载.mat文件
                    mat_data = sio.loadmat(file_path)
                    
                    # 查找振动数据
                    vibration_data = None
                    for key in mat_data.keys():
                        if not key.startswith('_') and isinstance(mat_data[key], np.ndarray):
                            if mat_data[key].ndim == 2 and mat_data[key].shape[0] > mat_data[key].shape[1]:
                                vibration_data = mat_data[key].flatten()
                                break
                            elif mat_data[key].ndim == 1 and len(mat_data[key]) > 1000:
                                vibration_data = mat_data[key]
                                break
                    
                    if vibration_data is not None:
                        # 重采样处理
                        if self.enable_resampling and fs != self.target_fs:
                            print(f"  重采样 {file_name}: {fs}Hz -> {self.target_fs}Hz")
                            vibration_data = self._resample_signal(vibration_data, fs, self.target_fs)
                            effective_fs = self.target_fs
                        else:
                            effective_fs = fs
                        
                        # 滑窗采样 - 使用统一的窗口大小
                        windows = self._sliding_window_sampling_unified(vibration_data, effective_fs)
                        
                        # 添加到数据集
                        for window in windows:
                            if len(window) == self.window_size:
                                if self.transform_to_2d:
                                    image_2d = self._signal_to_2d_unified(window, effective_fs)
                                    self.samples.append(image_2d)
                                else:
                                    self.samples.append(window)
                                
                                self.labels.append(self.fault_mapping['Normal'])
                                self.file_info.append({
                                    'file_name': file_name,
                                    'fault_type': 'Normal',
                                    'fault_size': 'N/A',
                                    'fs': effective_fs
                                })
                        
                        # 存储原始信号用于可视化
                        signal_key = f"Normal_{file_name}_{effective_fs}Hz"
                        self.raw_signals[signal_key] = {
                            'signal': vibration_data,
                            'fs': effective_fs,
                            'fault_type': 'Normal',
                            'fault_size': 'N/A',
                            'file_name': file_name
                        }
                        
                        print(f"  已处理: {file_name} - {len(windows)} 个窗口")
                        
                except Exception as e:
                    print(f"处理文件 {file_name} 时出错: {e}")

    def _process_fault_data_with_fs(self, fault_folder_path, fs):
        """处理故障数据 - 使用指定采样频率，只处理指定的故障类型"""
        for fault_type in self.target_fault_types:  # 只处理B、IR、OR
            fault_path = os.path.join(fault_folder_path, fault_type)
            if not os.path.exists(fault_path):
                continue
                
            print(f"处理 {fault_type} 故障数据...")
            
            if fault_type == 'OR':
                # 外圈故障处理所有方向：Centered、Opposite、Orthogonal
                or_directions = ['Centered', 'Opposite', 'Orthogonal']
                for direction in or_directions:
                    direction_path = os.path.join(fault_path, direction)
                    if os.path.exists(direction_path):
                        print(f"  处理 OR-{direction} 数据...")
                        self._process_fault_sizes_with_fs(direction_path, fault_type, fs, direction)
                    else:
                        print(f"  警告: 未找到 OR-{direction} 文件夹")
            else:
                # 内圈和滚动体故障处理所有尺寸
                self._process_fault_sizes_with_fs(fault_path, fault_type, fs)

    def _process_fault_sizes_with_fs(self, fault_path, fault_type, fs, or_direction=None):
        """处理特定故障类型的不同尺寸 - 使用指定采样频率，所有尺寸合并为同一类"""
        # 定义尺寸映射
        if fault_type == 'OR':
            size_folders = ['0007', '0021']  # 外圈故障只有这两种尺寸
        else:
            size_folders = ['0007', '0014', '0021', '0028']  # 内圈和滚动体故障
        
        for size_folder in size_folders:
            size_path = os.path.join(fault_path, size_folder)
            if os.path.exists(size_path):
                # 使用故障类型作为标签（不区分尺寸和方向）
                self._process_size_folder_with_fs(size_path, fault_type, fs, size_folder, or_direction)

    def _process_size_folder_with_fs(self, size_path, fault_type, fs, size_folder, or_direction=None):
        """处理特定尺寸文件夹中的所有.mat文件 - 使用指定采样频率"""
        sample_count = 0
        
        for file_name in os.listdir(size_path):
            if file_name.endswith('.mat'):
                file_path = os.path.join(size_path, file_name)
                
                try:
                    # 加载.mat文件
                    mat_data = sio.loadmat(file_path)
                    
                    # 查找振动数据
                    vibration_data = None
                    for key in mat_data.keys():
                        if not key.startswith('_') and isinstance(mat_data[key], np.ndarray):
                            if mat_data[key].ndim == 2 and mat_data[key].shape[0] > mat_data[key].shape[1]:
                                vibration_data = mat_data[key].flatten()
                                break
                    
                    if vibration_data is not None:
                        # 重采样处理
                        if self.enable_resampling and fs != self.target_fs:
                            print(f"  重采样 {file_name}: {fs}Hz -> {self.target_fs}Hz")
                            vibration_data = self._resample_signal(vibration_data, fs, self.target_fs)
                            effective_fs = self.target_fs
                        else:
                            effective_fs = fs
                        
                        # 使用统一的滑窗采样
                        windows = self._sliding_window_sampling_unified(vibration_data, effective_fs)
                        
                        # 添加到数据集
                        for window in windows:
                            if len(window) == self.window_size:
                                if self.transform_to_2d:
                                    image_2d = self._signal_to_2d_unified(window, effective_fs)
                                    self.samples.append(image_2d)
                                else:
                                    self.samples.append(window)
                                
                                self.labels.append(self.fault_mapping[fault_type])  # 使用故障类型标签
                                
                                # 构建详细标签信息
                                if or_direction:
                                    detailed_label = f"{fault_type}_{or_direction}_{size_folder}"
                                    display_type = f"{fault_type}-{or_direction}"
                                else:
                                    detailed_label = f"{fault_type}_{size_folder}"
                                    display_type = fault_type
                                
                                self.file_info.append({
                                    'file': file_name,
                                    'fault_type': fault_type,
                                    'fault_size': size_folder,
                                    'or_direction': or_direction,
                                    'detailed_label': detailed_label,
                                    'display_type': display_type,
                                    'fs': effective_fs
                                })
                        
                        sample_count += len(windows)
                        
                        # 存储原始信号用于可视化
                        if or_direction:
                            signal_key = f"{fault_type}_{or_direction}_{size_folder}_{file_name}_{effective_fs}Hz"
                        else:
                            signal_key = f"{fault_type}_{size_folder}_{file_name}_{effective_fs}Hz"
                        
                        self.raw_signals[signal_key] = {
                            'signal': vibration_data,
                            'fs': effective_fs,
                            'fault_type': fault_type,
                            'fault_size': size_folder,
                            'or_direction': or_direction,
                            'file_name': file_name
                        }
                        
                except Exception as e:
                    print(f"处理文件 {file_name} 时出错: {e}")
        
        if or_direction:
            print(f"  {fault_type}-{or_direction}_{size_folder}: {sample_count} 个样本")
        else:
            print(f"  {fault_type}_{size_folder}: {sample_count} 个样本")

    def _sliding_window_sampling_unified(self, signal_data, fs):
        """统一的滑动窗口采样 - 使用固定窗口大小"""
        windows = []
        signal_data = signal_data.flatten()
        
        # 使用固定的窗口大小和步长
        window_size = self.window_size
        step_size = self.step_size
        
        for i in range(0, len(signal_data) - window_size + 1, step_size):
            window = signal_data[i:i + window_size]
            windows.append(window)
        
        return windows
    
    def _signal_to_2d_unified(self, signal, fs):
        """
        统一的信号到2D转换 - 使用固定参数
        
        Args:
            signal: 输入信号
            fs: 采样频率（重采样后应该是统一的）
        """
        method = self.transform_method
        
        if method == 'stft':
            # 统一的STFT参数（适用于12kHz采样频率）
            nperseg = 256
            noverlap = nperseg // 2
            nfft = nperseg
            
            # 计算STFT
            f, t, Zxx = scipy_signal.stft(
                signal, 
                fs=fs,
                window='hann',
                nperseg=nperseg,
                noverlap=noverlap,
                nfft=nfft,
                return_onesided=True,
                boundary=None,
                padded=False
            )
            
            # 计算幅度谱
            magnitude = np.abs(Zxx)
            
            # 对数变换以增强动态范围
            log_magnitude = 20 * np.log10(magnitude + 1e-10)
            
            # 调整到目标尺寸 (64x64)
            target_size = 64
            from scipy.ndimage import zoom
            zoom_factor_0 = target_size / log_magnitude.shape[0]
            zoom_factor_1 = target_size / log_magnitude.shape[1]
            image_2d = zoom(log_magnitude, (zoom_factor_0, zoom_factor_1))
            
            return image_2d
        
        elif method == 'cwt':
            # 统一的CWT参数
            try:
                import pywt
                
                # 固定尺度范围（适用于12kHz）
                scales = np.arange(1, 65)  # 64个尺度
                wavelet = 'morl'
                
                # 计算CWT
                coefficients, frequencies = pywt.cwt(signal, scales, wavelet)
                
                # 取绝对值
                magnitude = np.abs(coefficients)
                
                # 对数变换
                log_magnitude = np.log10(magnitude + 1e-10)
                
                # 调整到64x64
                target_size = 64
                from scipy.ndimage import zoom
                zoom_factor_0 = target_size / log_magnitude.shape[0]
                zoom_factor_1 = target_size / log_magnitude.shape[1]
                image_2d = zoom(log_magnitude, (zoom_factor_0, zoom_factor_1))
                
                return image_2d
                
            except ImportError:
                print("警告: 未安装pywt库，使用reshape方法")
                method = 'reshape'
        
        elif method == 'spectrogram':
            # 统一的频谱图参数
            nperseg = 256
            
            f, t, Sxx = scipy_signal.spectrogram(
                signal, 
                fs=fs,
                nperseg=nperseg,
                noverlap=nperseg//2,
                window='hann'
            )
            
            # 对数变换
            log_Sxx = 10 * np.log10(Sxx + 1e-10)
            
            # 调整到64x64
            target_size = 64
            from scipy.ndimage import zoom
            zoom_factor_0 = target_size / log_Sxx.shape[0]
            zoom_factor_1 = target_size / log_Sxx.shape[1]
            image_2d = zoom(log_Sxx, (zoom_factor_0, zoom_factor_1))
            
            return image_2d
        
        # 默认reshape方法
        if method == 'reshape' or method not in ['stft', 'cwt', 'spectrogram']:
            # 简单的reshape方法
            target_size = 64
            signal_length = len(signal)
            
            if signal_length == target_size * target_size:
                # 如果长度正好是64*64，直接reshape
                image_2d = signal.reshape(target_size, target_size)
            else:
                # 否则需要调整长度
                if signal_length > target_size * target_size:
                    # 截断
                    signal = signal[:target_size * target_size]
                else:
                    # 填充
                    padding = target_size * target_size - signal_length
                    signal = np.pad(signal, (0, padding), mode='constant', constant_values=0)
                
                image_2d = signal.reshape(target_size, target_size)
            
            return image_2d

    def _normalize_data(self):
        """标准化数据 - 处理不同长度的窗口"""
        if len(self.samples) == 0:
            print("警告: 没有样本数据可以标准化")
            return
        
        print("开始标准化数据...")
        
        if self.transform_to_2d:
            # 2D数据标准化
            samples_array = np.array(self.samples)
            
            # 计算全局均值和标准差
            global_mean = np.mean(samples_array)
            global_std = np.std(samples_array)
            
            # 标准化
            self.samples = [(sample - global_mean) / (global_std + 1e-8) for sample in self.samples]
            
            print(f"2D数据标准化完成: 均值={global_mean:.4f}, 标准差={global_std:.4f}")
        else:
            # 1D数据标准化
            # 将所有样本转换为相同长度
            target_length = self.window_size
            normalized_samples = []
            
            for sample in self.samples:
                if len(sample) == target_length:
                    normalized_samples.append(sample)
                elif len(sample) > target_length:
                    # 截断
                    normalized_samples.append(sample[:target_length])
                else:
                    # 填充
                    padded_sample = np.pad(sample, (0, target_length - len(sample)), mode='constant')
                    normalized_samples.append(padded_sample)
            
            # 转换为numpy数组并标准化
            samples_array = np.array(normalized_samples)
            global_mean = np.mean(samples_array)
            global_std = np.std(samples_array)
            
            # 标准化
            self.samples = [(sample - global_mean) / (global_std + 1e-8) for sample in normalized_samples]
            
            print(f"1D数据标准化完成: 均值={global_mean:.4f}, 标准差={global_std:.4f}")
    







        


    # def visualize_2d_window_comparison(self):
    #     """
    #     专门可视化不同类别的2D窗口对比
    #     """
    #     print("生成2D窗口对比图...")
        
    #     class_names = self.get_class_names()
    #     num_classes = len(class_names)
        
    #     # 创建网格布局：11行5列
    #     fig, axes = plt.subplots(num_classes, 5, figsize=(20, 3 * num_classes))
        
    #     # 收集每个类别的数据
    #     class_data = self._collect_class_data()
        
    #     for class_idx in range(num_classes):
    #         for window_idx in range(5):
    #             ax = axes[class_idx, window_idx] if num_classes > 1 else axes[window_idx]
                
    #             if class_idx in class_data and window_idx < len(class_data[class_idx]['windows']):
    #                 # 获取窗口数据并转换为2D
    #                 window_data = class_data[class_idx]['windows'][window_idx]
    #                 window_2d = self._signal_to_2d(window_data)
                    
    #                 # 显示2D图像
    #                 im = ax.imshow(window_2d, cmap='viridis', aspect='auto')
                    
    #                 if window_idx == 0:
    #                     ax.set_ylabel(f'{class_names[class_idx]}', fontsize=10)
    #                 if class_idx == 0:
    #                     ax.set_title(f'窗口 {window_idx + 1}', fontsize=10)
                    
    #                 ax.set_xticks([])
    #                 ax.set_yticks([])
                    
    #                 # 添加小的颜色条
    #                 plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    #             else:
    #                 # 无数据时显示空白
    #                 ax.text(0.5, 0.5, '无数据', ha='center', va='center', 
    #                        transform=ax.transAxes, fontsize=10)
    #                 ax.set_xticks([])
    #                 ax.set_yticks([])
        
    #     plt.tight_layout()
    #     plt.savefig('imgs/2d_windows_comparison.png', dpi=300, bbox_inches='tight')
    #     plt.show()
        
    #     print("2D窗口对比图已保存到 imgs/2d_windows_comparison.png")

    # def visualize_samples(self, num_samples=11, transform_to_2d=False):
    #     """可视化样本 - 修改为显示11个样本"""
    #     # 尝试每个类别显示一个样本
    #     fig, axes = plt.subplots(3, 4, figsize=(20, 12))
    #     axes = axes.flatten()
        
    #     class_names = self.get_class_names()
        
    #     # 尝试为每个类别找一个样本
    #     samples_to_show = []
    #     labels_to_show = []
        
    #     for class_idx in range(len(class_names)):
    #         # 找到该类别的第一个样本
    #         found = False
    #         for i, label in enumerate(self.labels):
    #             if label == class_idx:
    #                 samples_to_show.append(self.samples[i])
    #                 labels_to_show.append(label)
    #                 found = True
    #                 break
            
    #         if not found:
    #             # 如果某个类别没有样本，用其他样本填充
    #             if len(self.samples) > 0:
    #                 samples_to_show.append(self.samples[0])
    #                 labels_to_show.append(0)
        
    #     # 确保有12个样本
    #     while len(samples_to_show) < 12:
    #         if len(self.samples) > 0:
    #             samples_to_show.append(self.samples[0])
    #             labels_to_show.append(0)
    #         else:
    #             break
        
    #     # 绘制样本
    #     for i in range(min(12, len(samples_to_show))):
    #         sample = samples_to_show[i]
    #         label = labels_to_show[i]
            
    #         if transform_to_2d:
    #             # 2D热力图
    #             sample_2d = self._signal_to_2d(sample)
    #             im = axes[i].imshow(sample_2d, cmap='viridis', aspect='auto')
    #             axes[i].set_title(f'{class_names[label]}', fontsize=10)
    #             plt.colorbar(im, ax=axes[i])
    #         else:
    #             # 1D折线图
    #             axes[i].plot(sample)
    #             axes[i].set_title(f'{class_names[label]}', fontsize=10)
    #             axes[i].set_xlabel('时间点')
    #             axes[i].set_ylabel('振幅')
        
    #     # 隐藏多余的子图
    #     for i in range(len(samples_to_show), 12):
    #         axes[i].set_visible(False)
        
    #     plt.tight_layout()
    #     plt.savefig('imgs/bearing_samples_visualization_12class.png', dpi=300, bbox_inches='tight')
    #     plt.show()

    def __len__(self):
        """返回数据集大小"""
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        label = self.labels[idx]
        
        if self.transform_to_2d:
            # 2D数据已经在加载时转换好了
            sample = torch.FloatTensor(sample).unsqueeze(0)  # 添加通道维度
        else:
            sample = torch.FloatTensor(sample)
        
        return sample, torch.LongTensor([label])[0]
    
    def get_class_names(self):
        """获取类别名称"""
        class_names = [''] * len(self.fault_mapping)
        for fault_name, label in self.fault_mapping.items():
            if fault_name == 'Normal':
                class_names[label] = '正常'
            elif fault_name == 'IR':
                class_names[label] = '内圈故障'
            elif fault_name == 'B':
                class_names[label] = '滚动体故障'
            elif fault_name == 'OR':
                class_names[label] = '外圈故障'
        return class_names

    def _print_class_distribution(self):
        """打印类别分布"""
        if len(self.labels) == 0:
            print("没有标签数据")
            return
        
        # 统计各类别数量
        unique_labels, counts = np.unique(self.labels, return_counts=True)
        
        print("\n类别分布:")
        print("-" * 40)
        
        # 反向映射
        label_to_name = {v: k for k, v in self.fault_mapping.items()}
        
        total_samples = len(self.labels)
        for label, count in zip(unique_labels, counts):
            class_name = label_to_name.get(label, f"Unknown_{label}")
            percentage = (count / total_samples) * 100
            print(f"{class_name:>10}: {count:>6} 样本 ({percentage:>5.1f}%)")
        
        print("-" * 40)
        print(f"{'总计':>10}: {total_samples:>6} 样本")

def create_bearing_dataloaders(data_path, batch_size=32, train_ratio=0.8, window_size=4096, step_size=None, overlap_ratio=0.5, transform_to_2d=False, transform_method='stft', enable_resampling=True, target_fs=12000):
    """创建训练和测试数据加载器"""
    # 如果没有指定step_size，根据overlap_ratio计算
    if step_size is None:
        step_size = int(window_size * (1 - overlap_ratio))
    
    # 创建数据集
    dataset = BearingDataset(data_path, window_size, step_size, transform_to_2d, transform_method, enable_resampling, target_fs)
    
    # 计算训练集大小
    dataset_size = len(dataset)
    train_size = int(train_ratio * dataset_size)
    test_size = dataset_size - train_size
    
    # 随机分割数据集
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"训练集大小: {train_size}")
    print(f"测试集大小: {test_size}")
    
    return train_loader, test_loader, dataset

if __name__ == '__main__':
    # 测试重采样功能
    data_path = r"数据集\数据集\源域数据集"
    
    print("测试重采样功能:")
    print("="*60)
    
    # 测试启用重采样
    train_loader, test_loader, dataset = create_bearing_dataloaders(
        data_path, 
        batch_size=Config.BATCH_SIZE, 
        train_ratio=Config.TRAIN_RATIO,
        window_size=Config.WINDOW_SIZE,
        step_size=None,
        overlap_ratio=Config.OVERLAP_RATIO,
        transform_to_2d=True,
        transform_method=Config.TRANSFORM_METHOD,
        enable_resampling=True,  # 启用重采样
        target_fs=12000  # 目标采样频率12kHz
    )
    
    print("\n测试数据加载:")
    for i, (batch_data, batch_labels) in enumerate(train_loader):
        print(f"批次 {i+1}: 数据形状 {batch_data.shape}, 标签形状 {batch_labels.shape}")
        if i >= 2:  # 只显示前3个批次
            break
    
    print("\n重采样功能测试完成！")
    print("="*60)