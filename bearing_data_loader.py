import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib
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
                 transform_to_2d=True, transform_method='stft'):
        self.data_path = data_path
        self.window_size = window_size
        self.step_size = step_size 
        self.transform_to_2d = transform_to_2d
        self.transform_method = transform_method
        
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
    
    def _load_data(self):
        """加载数据 - 只加载指定的三个文件夹和特定故障类型"""
        print("开始加载轴承数据...")
        
        # 定义要使用的数据源及其采样频率
        data_sources = [
            ('48kHz_Normal_data', 48000, 'normal'),
            ('48kHz_DE_data', 48000, 'fault'),
            ('12kHz_DE_data', 12000, 'fault')
            # 移除 12kHz_FE_data
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
                    
                    if vibration_data is not None:
                        # 使用指定采样频率进行滑窗采样
                        windows = self._sliding_window_sampling(vibration_data, fs=fs)
                        
                        # 添加到数据集
                        for window in windows:
                            self.samples.append(window)
                            self.labels.append(self.fault_mapping['Normal'])
                            self.file_info.append({
                                'file': file_name,
                                'fault_type': 'Normal',
                                'fault_size': 'N/A',
                                'detailed_label': 'Normal',
                                'fs': fs
                            })
                        
                        # 存储原始信号用于可视化
                        signal_key = f"Normal_{file_name}_{fs}Hz"
                        self.raw_signals[signal_key] = {
                            'signal': vibration_data,
                            'fs': fs,
                            'fault_type': 'Normal',
                            'fault_size': 'N/A',
                            'file_name': file_name
                        }
                        
                        print(f"  正常数据 {file_name}: {len(windows)} 个样本")
                        
                except Exception as e:
                    print(f"处理正常数据文件 {file_name} 时出错: {e}")

    def _process_fault_data_with_fs(self, fault_folder_path, fs):
        """处理故障数据 - 使用指定采样频率，只处理指定的故障类型"""
        for fault_type in self.target_fault_types:  # 只处理B、IR、OR
            fault_path = os.path.join(fault_folder_path, fault_type)
            if not os.path.exists(fault_path):
                continue
                
            print(f"处理 {fault_type} 故障数据...")
            
            if fault_type == 'OR':
                # 外圈故障只处理Opposite方向
                opposite_path = os.path.join(fault_path, 'Opposite')
                if os.path.exists(opposite_path):
                    self._process_fault_sizes_with_fs(opposite_path, fault_type, fs)
            else:
                # 内圈和滚动体故障处理所有尺寸
                self._process_fault_sizes_with_fs(fault_path, fault_type, fs)
    
    def _process_fault_sizes_with_fs(self, fault_path, fault_type, fs):
        """处理特定故障类型的不同尺寸 - 使用指定采样频率，所有尺寸合并为同一类"""
        # 定义尺寸映射
        if fault_type == 'OR':
            size_folders = ['0007', '0021']  # 外圈故障只有这两种尺寸
        else:
            size_folders = ['0007', '0014', '0021', '0028']  # 内圈和滚动体故障
        
        for size_folder in size_folders:
            size_path = os.path.join(fault_path, size_folder)
            if os.path.exists(size_path):
                # 使用故障类型作为标签（不区分尺寸）
                self._process_size_folder_with_fs(size_path, fault_type, fs, size_folder)
    
    def _process_size_folder_with_fs(self, size_path, fault_type, fs, size_folder):
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
                        # 使用指定采样频率进行滑窗采样
                        windows = self._sliding_window_sampling(vibration_data, fs=fs)
                        
                        # 添加到数据集
                        for window in windows:
                            self.samples.append(window)
                            self.labels.append(self.fault_mapping[fault_type])  # 使用故障类型标签
                            self.file_info.append({
                                'file': file_name,
                                'fault_type': fault_type,
                                'fault_size': size_folder,
                                'detailed_label': f"{fault_type}_{size_folder}",
                                'fs': fs
                            })
                        
                        sample_count += len(windows)
                        
                        # 存储原始信号用于可视化
                        signal_key = f"{fault_type}_{size_folder}_{file_name}_{fs}Hz"
                        self.raw_signals[signal_key] = {
                            'signal': vibration_data,
                            'fs': fs,
                            'fault_type': fault_type,
                            'fault_size': size_folder,
                            'file_name': file_name
                        }
                        
                except Exception as e:
                    print(f"处理文件 {file_name} 时出错: {e}")
        
        print(f"  {fault_type}_{size_folder}: {sample_count} 个样本")

    def _get_window_size_for_fs(self, fs):
        """根据采样频率计算窗口大小"""
        # 基准: 12kHz采样频率下窗口大小为self.window_size
        base_fs = 12000
        if fs == base_fs:
            return self.window_size
        else:
            # 按采样频率比例调整窗口大小
            adjusted_window_size = int(self.window_size * fs / base_fs)
            # 确保窗口大小是64的倍数（便于2D转换）
            adjusted_window_size = ((adjusted_window_size + 63) // 64) * 64
            return adjusted_window_size
    
    def _get_step_size_for_fs(self, fs):
        """根据采样频率和重叠比例计算步长"""
        # 先获取对应采样频率的窗口大小
        window_size = self._get_window_size_for_fs(fs)
        
        # 根据重叠比例计算步长
        step_size = int(window_size * (1 - Config.OVERLAP_RATIO))
        
        return step_size

    def _sliding_window_sampling(self, signal_data, fs=12000):
        """滑动窗口采样 - 根据采样频率调整窗口大小"""
        windows = []
        
        # 根据采样频率获取相应的窗口大小和步长
        window_size = self._get_window_size_for_fs(fs)
        step_size = self._get_step_size_for_fs(fs)
        
        signal_length = len(signal_data)
        
        # 添加调试信息
        # print(f"    信号长度: {signal_length}, 采样频率: {fs}Hz, 窗口大小: {window_size}, 步长: {step_size}")
        
        # 计算窗口数量
        if signal_length < window_size:
            print(f"  警告: 信号长度 {signal_length} 小于窗口大小 {window_size}")
            return []
        
        num_windows = (signal_length - window_size) // step_size + 1
        # print(f"    计算得到窗口数量: {num_windows}")
        
        for i in range(num_windows):
            start_idx = i * step_size
            end_idx = start_idx + window_size
            
            if end_idx <= signal_length:
                window = signal_data[start_idx:end_idx]
                windows.append(window)
        
        # print(f"    实际生成窗口数量: {len(windows)}")
        return windows


    def _normalize_data(self):
        """标准化数据 - 处理不同长度的窗口"""
        if len(self.samples) == 0:
            print("警告: 没有样本数据可以标准化")
            return
            
        print("正在标准化数据...")
        
        # 检查窗口长度
        window_lengths = [len(sample) for sample in self.samples]
        unique_lengths = set(window_lengths)
        
        print(f"检测到窗口长度: {unique_lengths}")
        
        # 收集所有数据点用于计算全局统计量
        all_values = []
        for sample in self.samples:
            all_values.extend(sample.flatten())
        
        # 计算全局均值和标准差
        all_values = np.array(all_values)
        self.mean = np.mean(all_values)
        self.std = np.std(all_values)
        
        # 避免除零错误
        if self.std == 0:
            self.std = 1.0
        
        # 标准化每个样本
        for i in range(len(self.samples)):
            self.samples[i] = (self.samples[i] - self.mean) / self.std
        
        print(f"数据标准化完成: 均值={self.mean:.4f}, 标准差={self.std:.4f}")
        
        # 统计窗口长度分布
        length_counts = {}
        for length in window_lengths:
            length_counts[length] = length_counts.get(length, 0) + 1
        print(f"窗口长度分布: {length_counts}")
    
    def _signal_to_2d(self, signal, method='stft'):
        """
        将1D信号转换为2D图像
        
        Args:
            signal: 输入的1D信号
            method: 转换方法 ('stft', 'cwt', 'spectrogram', 'reshape')
        """
        # 移除强制截断！让信号保持原始长度进行2D转换
        # 注释掉这段强制截断的代码：
        # if len(signal) != self.window_size:
        #     if len(signal) > self.window_size:
        #         signal = signal[:self.window_size]  # 删除这个截断！
        #     else:
        #         signal = np.pad(signal, (0, self.window_size - len(signal)), 'constant')
        
        if method == 'stft':
            # 短时傅里叶变换 (STFT) - 根据信号长度动态调整参数
            from scipy import signal as scipy_signal
            
            # 根据信号长度动态调整STFT参数
            signal_length = len(signal)
            if signal_length <= 8192:
                nperseg = 256  # 12kHz数据使用较小窗口
                fs = 12000     # 12kHz采样频率
            else:
                nperseg = 1024  # 48kHz数据使用较大窗口
                fs = 48000      # 48kHz采样频率
            
            noverlap = nperseg // 2  # 50%重叠
            nfft = nperseg  # FFT点数
            
            # 使用汉宁窗
            window = 'hann'
            
            # 计算STFT
            f, t, Zxx = scipy_signal.stft(
                signal, 
                fs=fs,  # 使用正确的采样频率
                window=window,
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
            # 连续小波变换 - 处理不同长度的信号
            try:
                import pywt
                
                # 根据信号长度调整尺度范围
                signal_length = len(signal)
                if signal_length <= 8192:
                    scales = np.arange(1, 65)  # 12kHz数据
                else:
                    scales = np.arange(1, 129)  # 48kHz数据，使用更多尺度
                
                # 选择Morlet小波
                wavelet = 'morl'
                
                # 计算CWT
                coefficients, frequencies = pywt.cwt(signal, scales, wavelet)
                
                # 取幅度
                magnitude = np.abs(coefficients)
                
                # 对数变换
                log_magnitude = 20 * np.log10(magnitude + 1e-10)
                
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
        
        if method == 'spectrogram':
            # 功率谱密度
            from scipy import signal as scipy_signal
            
            # 根据信号长度自适应调整参数
            signal_length = len(signal)
            if signal_length <= 8192:
                # 12kHz数据
                fs = 12000
                nperseg = 256
            else:
                # 48kHz数据  
                fs = 48000
                nperseg = 1024
            
            # 验证时间分辨率一致性（可选的调试信息）
            time_resolution = nperseg / fs
            # print(f"信号长度: {signal_length}, 采样频率: {fs}Hz, 窗口大小: {nperseg}, 时间分辨率: {time_resolution:.4f}s")
                
            f, t, Sxx = scipy_signal.spectrogram(
                signal, 
                fs=fs,  # 使用正确的采样频率
                nperseg=nperseg,
                noverlap=nperseg//2
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
        
        elif method == 'reshape':
            # 直接reshape方法 - 处理不同长度
            signal_length = len(signal)
            
            # 找到最接近的平方数
            sqrt_len = int(np.sqrt(signal_length))
            target_len = sqrt_len * sqrt_len
            
            # 调整信号长度
            if signal_length > target_len:
                signal = signal[:target_len]
            else:
                signal = np.pad(signal, (0, target_len - signal_length), 'constant')
            
            # reshape为方形
            image_2d = signal.reshape(sqrt_len, sqrt_len)
            
            # 调整到64x64
            if sqrt_len != 64:
                from scipy.ndimage import zoom
                zoom_factor = 64 / sqrt_len
                image_2d = zoom(image_2d, zoom_factor)
            
            return image_2d
        
        else:
            raise ValueError(f"不支持的转换方法: {method}")

    def visualize_comprehensive(self, show_raw_signals=True, num_windows_per_class=5, name=''):
        """综合可视化：显示原始信号和2D窗口，在原始信号上标记窗口位置"""
        
        # 收集类别数据
        class_data = self._collect_class_data()
        
        if not class_data:
            print("错误: 没有收集到任何类别数据")
            return
        
        # 获取实际存在的类别
        available_classes = sorted(class_data.keys())
        num_classes = len(available_classes)
        
        print(f"可视化 {num_classes} 个类别的数据")
        
        # 计算布局
        if show_raw_signals:
            cols = 1 + num_windows_per_class  # 1个原始信号 + N个2D窗口
        else:
            cols = num_windows_per_class
        
        rows = num_classes
        
        # 创建图形
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 2.5))
        
        # 确保axes是2D数组
        if rows == 1:
            axes = axes.reshape(1, -1)
        elif cols == 1:
            axes = axes.reshape(-1, 1)
        
        # 获取类别名称
        class_names = self.get_class_names()
        
        # 定义窗口标记的颜色
        window_colors = ['red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive']
        
        # 为每个类别绘制图形
        for row_idx, class_idx in enumerate(available_classes):
            data = class_data[class_idx]
            raw_signal = data['raw_signal']
            windows = data['windows']
            fault_name = data['fault_name']
            fs = data['fs']
            
            col_idx = 0
            
            # 绘制原始信号
            if show_raw_signals:
                ax = axes[row_idx, col_idx]
                time_axis = np.arange(len(raw_signal)) / fs
                ax.plot(time_axis, raw_signal, 'b-', linewidth=0.5)
                
                # 计算窗口参数
                window_size = self._get_window_size_for_fs(fs)
                step_size = self._get_step_size_for_fs(fs)
                
                # 在原始信号上标记窗口位置
                num_windows_to_mark = min(num_windows_per_class, len(windows))
                for window_idx in range(num_windows_to_mark):
                    # 计算窗口在原始信号中的起始和结束位置
                    start_sample = window_idx * step_size
                    end_sample = start_sample + window_size
                    
                    # 转换为时间轴
                    start_time = start_sample / fs
                    end_time = end_sample / fs
                    
                    # 获取窗口对应的信号范围
                    if end_sample <= len(raw_signal):
                        window_signal = raw_signal[start_sample:end_sample]
                        window_min = np.min(window_signal)
                        window_max = np.max(window_signal)
                        
                        # 选择颜色
                        color = window_colors[window_idx % len(window_colors)]
                        
                        # 绘制矩形框标记窗口位置
                        from matplotlib.patches import Rectangle
                        rect = Rectangle((start_time, window_min), 
                                       end_time - start_time, 
                                       window_max - window_min,
                                       linewidth=2, 
                                       edgecolor=color, 
                                       facecolor='none',
                                       alpha=0.8)
                        ax.add_patch(rect)
                        
                        # 添加窗口编号标签
                        ax.text(start_time, window_max, f'W{window_idx+1}', 
                               color=color, fontsize=8, fontweight='bold',
                               verticalalignment='bottom')
                ax.set_title(f'{class_names[class_idx]}\n原始信号 (fs={fs}Hz)')
                ax.set_xlabel('时间 (s)')
                ax.set_ylabel('振幅')
                ax.grid(True, alpha=0.3)
                col_idx += 1
            
            # 绘制2D窗口
            for window_idx in range(min(num_windows_per_class, len(windows))):
                if col_idx >= cols:
                    break
                    
                ax = axes[row_idx, col_idx]
                window_data = windows[window_idx]
                
                # 选择对应的颜色
                color = window_colors[window_idx % len(window_colors)]
                
                # 转换为2D
                if self.transform_to_2d:
                    window_2d = self._signal_to_2d(window_data, method=self.transform_method)
                    
                    # 显示2D图像
                    if window_2d.ndim == 2:
                        im = ax.imshow(window_2d, aspect='auto', cmap='viridis', origin='lower')
                        ax.set_title(f'窗口 {window_idx+1}\n({self.transform_method.upper()})', 
                                   color=color, fontweight='bold')
                        # 添加彩色边框
                        for spine in ax.spines.values():
                            spine.set_edgecolor(color)
                            spine.set_linewidth(2)
                    else:
                        # 如果是1D，显示为时域信号
                        ax.plot(window_2d, color=color)
                        ax.set_title(f'窗口 {window_idx+1}\n(时域)', 
                                   color=color, fontweight='bold')
                else:
                    # 显示时域窗口
                    ax.plot(window_data, color=color)
                    ax.set_title(f'窗口 {window_idx+1}\n(时域)', 
                               color=color, fontweight='bold')
                ax.set_xlabel('样本点')
                ax.set_ylabel('振幅')
                col_idx += 1
            
            # 填充剩余的子图
            while col_idx < cols:
                axes[row_idx, col_idx].axis('off')
                col_idx += 1
        
        plt.tight_layout()
        
        # 保存图像
        if name:
            save_path = f'imgs_pre/class_signals_and_2d_windows_{name}.png'
        else:
            save_path = 'imgs_pre/class_signals_and_2d_windows.png'
        
        os.makedirs('imgs_pre', exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图像已保存到: {save_path}")
        
        plt.show()

    def _collect_class_data(self):
        """收集每个类别的原始信号和窗口样本"""
        class_data = {}
        
        # 遍历每个类别
        for fault_name, class_idx in self.fault_mapping.items():
            print(f"收集 {fault_name} 类别数据...")
            
            # 获取原始信号和采样频率
            result = self._load_class_raw_signal(fault_name)
            if result is None or result[0] is None:
                print(f"警告: 无法加载 {fault_name} 类别数据，跳过...")
                continue
                
            raw_signal, fs = result
            
            # 使用正确的采样频率生成窗口样本
            windows = self._sliding_window_sampling(raw_signal, fs=fs)
            
            if len(windows) == 0:
                print(f"警告: {fault_name} 类别没有生成窗口样本，跳过...")
                continue
            
            class_data[class_idx] = {
                'raw_signal': raw_signal,
                'windows': windows[:5],  # 只取前5个窗口
                'fault_name': fault_name,
                'fs': fs  # 保存采样频率信息
            }
            print(f"成功收集 {fault_name} 类别数据: {len(windows)} 个窗口")
        
        print(f"总共收集到 {len(class_data)} 个类别的数据")
        return class_data

    def _load_class_raw_signal(self, fault_name):
        """加载指定类别的原始信号，返回信号和采样频率"""
        try:
            if fault_name == 'Normal':
                # 优先加载48kHz正常数据
                data_sources = [
                    ('48kHz_Normal_data', 48000),
                    ('Normal', 12000)  # 备用
                ]
                
                for folder_name, fs in data_sources:
                    normal_folder = os.path.join(self.data_path, folder_name)
                    if os.path.exists(normal_folder):
                        for file_name in os.listdir(normal_folder):
                            if file_name.endswith('.mat'):
                                file_path = os.path.join(normal_folder, file_name)
                                try:
                                    mat_data = sio.loadmat(file_path)
                                    for key in mat_data.keys():
                                        if not key.startswith('_') and isinstance(mat_data[key], np.ndarray):
                                            if mat_data[key].ndim == 2 and mat_data[key].shape[0] > mat_data[key].shape[1]:
                                                signal = mat_data[key].flatten()
                                                print(f"加载 {fault_name} 数据: {file_name}, 采样频率: {fs}Hz, 长度: {len(signal)}")
                                                return signal, fs
                                except Exception as e:
                                    continue
            else:
                # 故障数据搜索优先级：48kHz_DE > 12kHz_DE > 12kHz_FE
                # 在4分类模式下，fault_name就是故障类型（IR、B、OR）
                fault_type = fault_name
                
                # 搜索路径列表（按优先级排序）
                search_paths = [
                    ('48kHz_DE_data', 48000),
                    ('12kHz_DE_data', 12000),
                    ('12kHz_FE_data', 12000)
                ]
                
                for folder_name, fs in search_paths:
                    base_folder = os.path.join(self.data_path, folder_name)
                    if not os.path.exists(base_folder):
                        continue
                        
                    # 构建具体路径 - 遍历所有可能的故障尺寸
                    if fault_type == 'OR':
                        # OR故障有多个子文件夹
                        or_subfolders = ['Centered', 'Opposite', 'Orthogonal']
                        fault_sizes = ['0007', '0021']  # OR故障的可用尺寸
                        for subfolder in or_subfolders:
                            for fault_size in fault_sizes:
                                fault_path = os.path.join(base_folder, fault_type, subfolder, fault_size)
                                if os.path.exists(fault_path):
                                    result = self._try_load_from_path(fault_path, fault_name, fs, folder_name)
                                    if result is not None:
                                        return result
                    else:
                        # IR和B故障直接路径 - 遍历所有可能的故障尺寸
                        fault_sizes = ['0007', '0014', '0021', '0028']  # 所有可能的故障尺寸
                        for fault_size in fault_sizes:
                            fault_path = os.path.join(base_folder, fault_type, fault_size)
                            if os.path.exists(fault_path):
                                result = self._try_load_from_path(fault_path, fault_name, fs, folder_name)
                                if result is not None:
                                    return result
                                        
        except Exception as e:
            print(f"加载 {fault_name} 数据时出错: {e}")
            
        return None, None
        
    def _try_load_from_path(self, fault_path, fault_name, fs, data_source):
        """尝试从指定路径加载数据"""
        try:
            for file_name in os.listdir(fault_path):
                if file_name.endswith('.mat'):
                    file_path = os.path.join(fault_path, file_name)
                    try:
                        mat_data = sio.loadmat(file_path)
                        for key in mat_data.keys():
                            if not key.startswith('_') and isinstance(mat_data[key], np.ndarray):
                                if mat_data[key].ndim == 2 and mat_data[key].shape[0] > mat_data[key].shape[1]:
                                    signal = mat_data[key].flatten()
                                    print(f"加载 {fault_name} 数据: {file_name} ({data_source}), 采样频率: {fs}Hz, 长度: {len(signal)}")
                                    return signal, fs
                    except Exception as e:
                        continue
        except Exception as e:
            pass
            return None

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
            # 使用指定的方法转换为2D
            sample = self._signal_to_2d(sample, method=self.transform_method)
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
            # 兼容旧的命名方式（带尺寸的）
            elif fault_name.startswith('IR_'):
                size = fault_name.split('_')[1]
                class_names[label] = f'内圈故障_{size}'
            elif fault_name.startswith('B_'):
                size = fault_name.split('_')[1]
                class_names[label] = f'滚动体故障_{size}'
            elif fault_name.startswith('OR_'):
                size = fault_name.split('_')[1]
                class_names[label] = f'外圈故障_{size}'
        return class_names

    def _print_class_distribution(self):
        """打印详细的类别分布"""
        print("\n详细类别分布:")
        class_names = self.get_class_names()
        label_counts = np.bincount(self.labels)
        
        for i, (name, count) in enumerate(zip(class_names, label_counts)):
            if count > 0:
                print(f"  {i}: {name} - {count} 个样本")

def create_bearing_dataloaders(data_path, batch_size=32, train_ratio=0.8, window_size=4096, step_size=None, overlap_ratio=0.5, transform_to_2d=False, transform_method='stft'):
    """创建训练和测试数据加载器"""
    # 如果没有指定step_size，根据overlap_ratio计算
    if step_size is None:
        step_size = int(window_size * (1 - overlap_ratio))
    
    # 创建数据集
    dataset = BearingDataset(data_path, window_size, step_size, transform_to_2d, transform_method)
    
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
    # 测试不同的2D转换方法
    data_path = r"数据集\数据集\源域数据集"
    methods = ['stft', 'cwt', 'spectrogram', 'reshape']
    
    # 关闭交互模式，避免图像显示阻塞
    plt.ioff()
    
    for method in methods:
        print(f"\n{'='*60}")
        print(f"测试 {method.upper()} 转换方法")
        print(f"{'='*60}")
        
        train_loader, test_loader, dataset = create_bearing_dataloaders(
            data_path, 
            batch_size=Config.BATCH_SIZE, 
            train_ratio=Config.TRAIN_RATIO,
            window_size=Config.WINDOW_SIZE,
            step_size=None,  # 让函数自动根据Config.OVERLAP_RATIO计算
            transform_to_2d=True,
            transform_method=method
        )
        
        # 可视化对比
        print(f"使用 {method} 方法的2D窗口可视化")
        dataset.visualize_comprehensive(show_raw_signals=True,num_windows_per_class=4,name=method)
        
        # 清理内存
        plt.close('all')
    
    # 最后显示所有图像
    print(f"\n{'='*60}")
    print("所有转换方法测试完成！")
    print("生成的图像文件:")
    for method in methods:
        print(f"  - imgs_pre/class_signals_and_2d_windows_{method}.png")
    print(f"{'='*60}")
    
    # 测试数据加载（使用最后一个数据集）
    print("\n测试数据加载:")
    for i, (batch_data, batch_labels) in enumerate(train_loader):
        print(f"批次 {i+1}: 数据形状 {batch_data.shape}, 标签形状 {batch_labels.shape}")
        if i >= 2:  # 只显示前3个批次
            break