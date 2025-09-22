import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

class BearingDataset(Dataset):
    def __init__(self, data_path, window_size=8192, step_size=2048, transform_to_2d=False, transform_method='stft'):
        """
        轴承故障数据集
        
        Args:
            data_path: 数据路径
            window_size: 滑动窗口大小
            step_size: 滑动步长
            transform_to_2d: 是否转换为2D
            transform_method: 2D转换方法 ('stft', 'cwt', 'spectrogram', 'reshape')
        """
        self.data_path = data_path
        self.window_size = window_size
        self.step_size = step_size
        self.transform_to_2d = transform_to_2d
        self.transform_method = transform_method
        
        # 修正后的10种分类故障映射（根据实际数据集）
        self.fault_mapping = {
            'Normal': 0,
            'IR_0007': 1, 'IR_0014': 2, 'IR_0021': 3, 'IR_0028': 4,
            'B_0007': 5, 'B_0014': 6, 'B_0021': 7, 'B_0028': 8,
            'OR_0007': 9, 'OR_0021': 10  # 只有这两种外圈故障尺寸
        }
        
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
        """加载数据"""
        print("开始加载轴承数据...")
        
        # 处理正常数据
        normal_folder = os.path.join(self.data_path, 'Normal')
        if os.path.exists(normal_folder):
            print("处理正常数据...")
            self._process_normal_data(normal_folder, fs=12000)
        
        # 处理故障数据
        fault_folder = os.path.join(self.data_path, '12kHz_DE_data')
        if os.path.exists(fault_folder):
            print("处理故障数据...")
            self._process_fault_data_detailed(fault_folder, fs=12000)
        
        print(f"数据加载完成: 总共 {len(self.samples)} 个样本")
    
    def _process_normal_data(self, normal_folder, fs):
        """处理正常数据"""
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
                        # 滑窗采样
                        windows = self._sliding_window_sampling(vibration_data, fs=fs)
                        
                        # 添加到数据集
                        for window in windows:
                            self.samples.append(window)
                            self.labels.append(self.fault_mapping['Normal'])
                            self.file_info.append({
                                'file': file_name,
                                'fault_type': 'Normal',
                                'fault_size': 'N/A',
                                'fs': fs
                            })
                        
                        print(f"  {file_name}: 生成 {len(windows)} 个正常样本")
                        
                except Exception as e:
                    print(f"  错误: {file_name} - {e}")
    
    def _process_fault_data_detailed(self, fault_folder_path, fs):
        """处理故障数据 - 详细分类"""
        fault_types = ['B', 'IR', 'OR']
        
        for fault_type in fault_types:
            fault_path = os.path.join(fault_folder_path, fault_type)
            if not os.path.exists(fault_path):
                continue
                
            print(f"处理 {fault_type} 故障数据...")
            
            if fault_type == 'OR':
                # 外圈故障只处理Opposite方向
                opposite_path = os.path.join(fault_path, 'Opposite')
                if os.path.exists(opposite_path):
                    self._process_fault_sizes(opposite_path, fault_type, fs)
            else:
                # 内圈和滚动体故障处理所有尺寸
                self._process_fault_sizes(fault_path, fault_type, fs)
    
    def _process_fault_sizes(self, fault_path, fault_type, fs):
        """处理特定故障类型的不同尺寸"""
        # 定义尺寸映射 - 根据实际数据集
        if fault_type == 'OR':
            # 外圈故障只有0007和0021两种尺寸（根据实际数据集）
            size_folders = ['0007', '0021']
        else:
            # 内圈和滚动体故障有四种尺寸
            size_folders = ['0007', '0014', '0021', '0028']
        
        for size_folder in size_folders:
            size_path = os.path.join(fault_path, size_folder)
            if os.path.exists(size_path):
                # 构建详细故障标签
                detailed_fault_label = f"{fault_type}_{size_folder}"
                
                # 检查是否在映射中
                if detailed_fault_label in self.fault_mapping:
                    self._process_size_folder(size_path, detailed_fault_label, fs)
    
    def _process_size_folder(self, size_path, detailed_fault_label, fs):
        """处理特定尺寸文件夹中的所有.mat文件"""
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
                        # 滑窗采样
                        windows = self._sliding_window_sampling(vibration_data, fs=fs)
                        
                        # 添加到数据集
                        for window in windows:
                            self.samples.append(window)
                            self.labels.append(self.fault_mapping[detailed_fault_label])
                            self.file_info.append({
                                'file': file_name,
                                'fault_type': detailed_fault_label.split('_')[0],
                                'fault_size': detailed_fault_label.split('_')[1],
                                'detailed_label': detailed_fault_label,
                                'fs': fs
                            })
                        
                        sample_count += len(windows)
                        
                except Exception as e:
                    print(f"  错误: {file_name} - {e}")
        
        if sample_count > 0:
            print(f"  {detailed_fault_label}: 生成 {sample_count} 个样本")

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
        """根据采样频率计算步长"""
        # 基准: 12kHz采样频率下步长为self.step_size
        base_fs = 12000
        if fs == base_fs:
            return self.step_size
        else:
            # 按采样频率比例调整步长
            adjusted_step_size = int(self.step_size * fs / base_fs)
            return adjusted_step_size

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
        """标准化数据"""
        if len(self.samples) == 0:
            print("警告: 没有样本数据可以标准化")
            return
            
        print("正在标准化数据...")
        
        # 将所有样本合并用于计算统计量
        all_data = np.array(self.samples)
        
        # 计算全局均值和标准差
        self.mean = np.mean(all_data)
        self.std = np.std(all_data)
        
        # 避免除零错误
        if self.std == 0:
            self.std = 1.0
        
        # 标准化每个样本
        for i in range(len(self.samples)):
            self.samples[i] = (self.samples[i] - self.mean) / self.std
        
        print(f"数据标准化完成: 均值={self.mean:.4f}, 标准差={self.std:.4f}")
    
    def _signal_to_2d(self, signal, method='stft'):
        """
        将1D信号转换为2D图像
        
        Args:
            signal: 输入的1D信号
            method: 转换方法 ('stft', 'cwt', 'spectrogram', 'reshape')
        """
        # 确保信号长度为window_size
        if len(signal) != self.window_size:
            if len(signal) > self.window_size:
                signal = signal[:self.window_size]
            else:
                signal = np.pad(signal, (0, self.window_size - len(signal)), 'constant')
        
        if method == 'stft':
            # 短时傅里叶变换 (STFT) - 按照标准公式实现
            from scipy import signal as scipy_signal
            
            # STFT参数设置
            nperseg = 256  # 窗口长度
            noverlap = nperseg // 2  # 50%重叠
            nfft = nperseg  # FFT点数
            
            # 使用汉宁窗（常用的窗函数）
            window = 'hann'
            
            # 计算STFT: F(ω) = ∫ f(t) * w(t-τ) * e^(-jωt) dt
            f, t, Zxx = scipy_signal.stft(
                signal, 
                fs=12000,  # 采样频率
                window=window,
                nperseg=nperseg,
                noverlap=noverlap,
                nfft=nfft,
                return_onesided=True,
                boundary=None,
                padded=False
            )
            
            # 计算幅度谱 |STFT(f,t)|
            magnitude = np.abs(Zxx)
            
            # 对数变换以增强动态范围
            log_magnitude = 20 * np.log10(magnitude + 1e-10)  # 转换为dB
            
            # 调整到目标尺寸 (64x64)
            target_size = 64
            from scipy.ndimage import zoom
            zoom_factor_0 = target_size / log_magnitude.shape[0]
            zoom_factor_1 = target_size / log_magnitude.shape[1]
            image_2d = zoom(log_magnitude, (zoom_factor_0, zoom_factor_1))
            
            return image_2d
            
        elif method == 'cwt':
            # 连续小波变换 (CWT) - 按照标准公式实现
            try:
                import pywt
                
                # 选择Morlet小波作为母小波 φ(t)
                # Morlet小波: φ(t) = π^(-1/4) * e^(jω₀t) * e^(-t²/2)
                wavelet = 'cmor1.5-1.0'  # 复Morlet小波，中心频率1.0，带宽1.5
                
                # 计算尺度参数 a
                # 根据公式 Wf(a,b) = |a|^(-1/2) ∫ f(t) φ*((t-b)/a) dt
                scales = np.logspace(0, 2, 64)  # 对数分布的64个尺度，从1到100
                
                # 计算CWT系数
                coefficients, frequencies = pywt.cwt(
                    signal, 
                    scales, 
                    wavelet, 
                    sampling_period=1/12000  # 采样周期
                )
                
                # 计算幅度 |Wf(a,b)|
                magnitude = np.abs(coefficients)
                
                # 对数变换
                log_magnitude = 20 * np.log10(magnitude + 1e-10)  # 转换为dB
                
                # 确保输出为64x64
                if log_magnitude.shape != (64, 64):
                    from scipy.ndimage import zoom
                    zoom_factor_0 = 64 / log_magnitude.shape[0]
                    zoom_factor_1 = 64 / log_magnitude.shape[1]
                    image_2d = zoom(log_magnitude, (zoom_factor_0, zoom_factor_1))
                else:
                    image_2d = log_magnitude
                
                return image_2d
                
            except ImportError:
                print("警告: pywt未安装，回退到STFT方法")
                return self._signal_to_2d(signal, method='stft')
                
        elif method == 'spectrogram':
            # 功率谱密度图 - 基于STFT的能量分布
            from scipy import signal as scipy_signal
            
            # 计算功率谱密度 |STFT(f,t)|²
            f, t, Sxx = scipy_signal.spectrogram(
                signal, 
                fs=12000,
                window='hann',
                nperseg=256,
                noverlap=128,
                nfft=256,
                return_onesided=True,
                scaling='density'
            )
            
            # 对数变换
            log_Sxx = 10 * np.log10(Sxx + 1e-10)  # 功率谱的dB表示
            
            # 调整到64x64
            target_size = 64
            from scipy.ndimage import zoom
            zoom_factor_0 = target_size / log_Sxx.shape[0]
            zoom_factor_1 = target_size / log_Sxx.shape[1]
            image_2d = zoom(log_Sxx, (zoom_factor_0, zoom_factor_1))
            
            return image_2d
            
        else:
            # 原始reshape方法（保留作为对比）
            sqrt_size = int(np.sqrt(self.window_size))
            if sqrt_size * sqrt_size == self.window_size:
                image_2d = signal.reshape(sqrt_size, sqrt_size)
            else:
                # 寻找最接近正方形的因子分解
                factors = []
                for i in range(1, int(np.sqrt(self.window_size)) + 1):
                    if self.window_size % i == 0:
                        factors.append((i, self.window_size // i))
                
                best_factor = min(factors, key=lambda x: abs(x[0] - x[1]))
                image_2d = signal.reshape(best_factor[0], best_factor[1])
            
            return image_2d

    def visualize_comprehensive(self, show_raw_signals=True, num_windows_per_class=5,name=''):
        """
        综合可视化函数：可以选择是否显示原始信号
        
        Args:
            show_raw_signals: 是否显示原始信号（True=信号+窗口，False=仅窗口对比）
            num_windows_per_class: 每个类别显示的窗口数量
            name: 保存文件名的前缀
        """
        print(f"生成{'信号和窗口' if show_raw_signals else '2D窗口对比'}可视化...")
        
        # 获取每个类别的原始信号和样本（只收集一次）
        class_data = self._collect_class_data()
        class_names = self.get_class_names()
        num_classes = len(class_names)
        
        if show_raw_signals:
            # 原始模式：每个类别一行，包含原始信号和5个2D窗口
            fig = plt.figure(figsize=(24, 4 * num_classes))
            cols = 6  # 1个原始信号 + 5个2D窗口
            
            for class_idx in range(num_classes):
                if class_idx not in class_data:
                    continue
                    
                raw_signal = class_data[class_idx]['raw_signal']
                windows = class_data[class_idx]['windows']
                
                # 每个类别占一行，6个子图
                row_start = class_idx * cols + 1
                
                # 1. 原始振动信号
                ax_signal = plt.subplot(num_classes, cols, row_start)
                ax_signal.plot(raw_signal[:min(50000, len(raw_signal))], 'b-', linewidth=0.5)
                ax_signal.set_title(f'{class_names[class_idx]}\n原始信号', fontsize=10)
                ax_signal.set_xlabel('采样点')
                ax_signal.set_ylabel('振幅')
                ax_signal.grid(True, alpha=0.3)
                
                # 标记前5个窗口的位置
                colors = ['red', 'green', 'orange', 'purple', 'brown']
                for i in range(min(num_windows_per_class, len(windows))):
                    start_idx = i * self.step_size
                    end_idx = start_idx + self.window_size
                    if end_idx <= len(raw_signal):
                        ax_signal.axvspan(start_idx, end_idx, alpha=0.2, color=colors[i])
                
                # 2-6. 2D窗口
                for window_idx in range(num_windows_per_class):
                    ax_2d = plt.subplot(num_classes, cols, row_start + 1 + window_idx)
                    
                    if window_idx < len(windows):
                        # 关键修复：传递正确的transform_method参数
                        window_2d = self._signal_to_2d(windows[window_idx], method=self.transform_method)
                        im = ax_2d.imshow(window_2d, cmap='viridis', aspect='auto')
                        ax_2d.set_title(f'窗口 {window_idx + 1}', fontsize=9)
                        plt.colorbar(im, ax=ax_2d, fraction=0.046, pad=0.04)
                    else:
                        ax_2d.text(0.5, 0.5, '无数据', ha='center', va='center', 
                                 transform=ax_2d.transAxes, fontsize=10)
                    
                    ax_2d.set_xticks([])
                    ax_2d.set_yticks([])
            
            # 添加转换方法信息到图像标题
            method_name = self.transform_method
            fig.suptitle(f'轴承故障信号和2D窗口可视化 - {method_name.upper()}方法', fontsize=16, y=0.98)
            save_path = f'imgs/{name}_class_signals_and_2d_windows.png' if name else f'imgs/class_signals_and_2d_windows_{method_name}.png'
            
        else:
            # 对比模式：只显示2D窗口对比
            fig, axes = plt.subplots(num_classes, num_windows_per_class, figsize=(20, 3 * num_classes))
            
            for class_idx in range(num_classes):
                for window_idx in range(num_windows_per_class):
                    ax = axes[class_idx, window_idx] if num_classes > 1 else axes[window_idx]
                    
                    if class_idx in class_data and window_idx < len(class_data[class_idx]['windows']):
                        # 获取窗口数据并转换为2D
                        window_data = class_data[class_idx]['windows'][window_idx]
                        # 关键修复：传递正确的transform_method参数
                        window_2d = self._signal_to_2d(window_data, method=self.transform_method)
                        
                        # 显示2D图像
                        im = ax.imshow(window_2d, cmap='viridis', aspect='auto')
                        
                        if window_idx == 0:
                            ax.set_ylabel(f'{class_names[class_idx]}', fontsize=10)
                        if class_idx == 0:
                            ax.set_title(f'窗口 {window_idx + 1}', fontsize=10)
                        
                        ax.set_xticks([])
                        ax.set_yticks([])
                        
                        # 添加小的颜色条
                        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                    else:
                        # 无数据时显示空白
                        ax.text(0.5, 0.5, '无数据', ha='center', va='center', 
                               transform=ax.transAxes, fontsize=10)
                        ax.set_xticks([])
                        ax.set_yticks([])
            
            method_name = self.transform_method
            fig.suptitle(f'2D窗口对比 - {method_name.upper()}方法', fontsize=16, y=0.98)
            save_path = f'imgs/{name}_2d_windows_comparison.png' if name else f'imgs/2d_windows_comparison_{method_name}.png'
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        # 移除阻塞的plt.show()
        plt.draw()
        plt.pause(0.1)
        print(f"可视化完成，图像已保存到 {save_path}")

    def _collect_class_data(self):
        """收集每个类别的原始信号和窗口样本"""
        class_data = {}
        
        # 遍历每个类别
        for fault_name, class_idx in self.fault_mapping.items():
            print(f"收集 {fault_name} 类别数据...")
            
            # 获取原始信号
            raw_signal = self._load_class_raw_signal(fault_name)
            if raw_signal is None:
                continue
            
            # 生成窗口样本
            windows = self._sliding_window_sampling(raw_signal)
            
            class_data[class_idx] = {
                'raw_signal': raw_signal,
                'windows': windows[:5],  # 只取前5个窗口
                'fault_name': fault_name
            }
        
        return class_data

    def _load_class_raw_signal(self, fault_name):
        """加载指定类别的原始信号"""
        try:
            if fault_name == 'Normal':
                # 加载正常数据
                normal_folder = os.path.join(self.data_path, '48kHz_Normal_data')
                if os.path.exists(normal_folder):
                    for file_name in os.listdir(normal_folder):
                        if file_name.endswith('.mat'):
                            file_path = os.path.join(normal_folder, file_name)
                            try:
                                mat_data = sio.loadmat(file_path)
                                for key in mat_data.keys():
                                    if not key.startswith('_') and isinstance(mat_data[key], np.ndarray):
                                        if mat_data[key].ndim == 2 and mat_data[key].shape[0] > mat_data[key].shape[1]:
                                            return mat_data[key].flatten()
                            except Exception as e:
                                continue
            else:
                # 加载故障数据
                fault_type = fault_name.split('_')[0]
                fault_size = fault_name.split('_')[1]
                
                # 构建文件路径
                if fault_type == 'OR':
                    fault_path = os.path.join(self.data_path, '12kHz_DE_data', fault_type, 'Opposite', fault_size)
                else:
                    fault_path = os.path.join(self.data_path, '12kHz_DE_data', fault_type, fault_size)
                
                if os.path.exists(fault_path):
                    for file_name in os.listdir(fault_path):
                        if file_name.endswith('.mat'):
                            file_path = os.path.join(fault_path, file_name)
                            try:
                                mat_data = sio.loadmat(file_path)
                                for key in mat_data.keys():
                                    if not key.startswith('_') and isinstance(mat_data[key], np.ndarray):
                                        if mat_data[key].ndim == 2 and mat_data[key].shape[0] > mat_data[key].shape[1]:
                                            return mat_data[key].flatten()
                            except Exception as e:
                                continue
        except Exception as e:
            print(f"加载 {fault_name} 数据时出错: {e}")
        
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

if __name__ == "__main__":
    data_path = "数据集/数据集/源域数据集"
    
    # 测试不同的2D转换方法
    methods = ['stft', 'cwt', 'spectrogram', 'reshape']
    
    # 设置matplotlib为非交互模式
    plt.ioff()
    
    for method in methods:
        print(f"\n{'='*60}")
        print(f"测试 {method.upper()} 转换方法")
        print(f"{'='*60}")
        
        train_loader, test_loader, dataset = create_bearing_dataloaders(
            data_path, 
            batch_size=32, 
            window_size=8192,
            step_size=2048,
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
        print(f"  - imgs/class_signals_and_2d_windows_{method}.png")
    print(f"{'='*60}")
    
    # 测试数据加载（使用最后一个数据集）
    print("\n测试数据加载:")
    for i, (batch_data, batch_labels) in enumerate(train_loader):
        print(f"批次 {i+1}: 数据形状 {batch_data.shape}, 标签形状 {batch_labels.shape}")
        if i >= 2:  # 只显示前3个批次
            break