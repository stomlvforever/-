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
    def __init__(self, data_path, window_size=8192, step_size=2048, transform_to_2d=False):
        """
        轴承故障诊断数据集 - 修正版本
        
        Args:
            data_path: 数据路径
            window_size: 窗口大小 (增大到8192)
            step_size: 步长
            transform_to_2d: 是否转换为2D图像
        """
        self.data_path = data_path
        self.window_size = window_size
        self.step_size = step_size
        self.transform_to_2d = transform_to_2d
        
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
                        windows = self._sliding_window_sampling(vibration_data)
                        
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
                        windows = self._sliding_window_sampling(vibration_data)
                        
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

    def _sliding_window_sampling(self, signal_data):
        """滑动窗口采样"""
        windows = []
        signal_length = len(signal_data)
        
        # 计算窗口数量
        num_windows = (signal_length - self.window_size) // self.step_size + 1
        
        for i in range(num_windows):
            start_idx = i * self.step_size
            end_idx = start_idx + self.window_size
            
            if end_idx <= signal_length:
                window = signal_data[start_idx:end_idx]
                windows.append(window)
        
        return windows
    
    def visualize_sliding_window_sampling(self, signal_data=None, num_windows_to_show=5):
        """
        可视化滑动窗口采样效果
        
        Args:
            signal_data: 输入信号数据，如果为None则使用第一个样本的原始数据
            num_windows_to_show: 显示的窗口数量
        """
        # 如果没有提供信号数据，尝试加载一个示例文件
        if signal_data is None:
            signal_data = self._load_sample_signal()
            if signal_data is None:
                print("无法加载示例信号数据")
                return
        
        # 确保信号长度足够
        if len(signal_data) < self.window_size:
            print(f"信号长度 {len(signal_data)} 小于窗口大小 {self.window_size}")
            return
        
        # 计算窗口信息
        signal_length = len(signal_data)
        num_windows = (signal_length - self.window_size) // self.step_size + 1
        overlap_ratio = 1 - (self.step_size / self.window_size)
        
        print(f"滑动窗口采样参数:")
        print(f"  信号长度: {signal_length}")
        print(f"  窗口大小: {self.window_size}")
        print(f"  步长: {self.step_size}")
        print(f"  重叠率: {overlap_ratio:.2%}")
        print(f"  总窗口数: {num_windows}")
        
        # 创建图形
        fig, axes = plt.subplots(3, 1, figsize=(15, 12))
        
        # 第一个子图：显示原始信号和窗口位置
        axes[0].plot(signal_data, 'b-', alpha=0.7, linewidth=0.8, label='原始信号')
        axes[0].set_title(f'原始信号与滑动窗口位置 (窗口大小={self.window_size}, 步长={self.step_size})', fontsize=12)
        axes[0].set_xlabel('采样点')
        axes[0].set_ylabel('振幅')
        axes[0].grid(True, alpha=0.3)
        
        # 显示前几个窗口的位置
        colors = ['red', 'green', 'orange', 'purple', 'brown']
        for i in range(min(num_windows_to_show, num_windows)):
            start_idx = i * self.step_size
            end_idx = start_idx + self.window_size
            
            # 高亮显示窗口区域
            axes[0].axvspan(start_idx, end_idx, alpha=0.2, color=colors[i % len(colors)], 
                           label=f'窗口 {i+1}')
            
            # 标记窗口边界
            axes[0].axvline(start_idx, color=colors[i % len(colors)], linestyle='--', alpha=0.8)
            axes[0].axvline(end_idx, color=colors[i % len(colors)], linestyle='--', alpha=0.8)
        
        axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # 第二个子图：显示提取的窗口样本
        axes[1].set_title(f'提取的窗口样本 (前{num_windows_to_show}个)', fontsize=12)
        axes[1].set_xlabel('窗口内采样点')
        axes[1].set_ylabel('振幅')
        axes[1].grid(True, alpha=0.3)
        
        for i in range(min(num_windows_to_show, num_windows)):
            start_idx = i * self.step_size
            end_idx = start_idx + self.window_size
            window_data = signal_data[start_idx:end_idx]
            
            axes[1].plot(window_data, color=colors[i % len(colors)], 
                        alpha=0.8, linewidth=1.2, label=f'窗口 {i+1}')
        
        axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # 第三个子图：显示窗口重叠情况
        axes[2].set_title('窗口重叠可视化', fontsize=12)
        axes[2].set_xlabel('采样点')
        axes[2].set_ylabel('窗口编号')
        
        # 创建重叠矩阵
        overlap_matrix = np.zeros((min(num_windows_to_show, num_windows), signal_length))
        for i in range(min(num_windows_to_show, num_windows)):
            start_idx = i * self.step_size
            end_idx = start_idx + self.window_size
            overlap_matrix[i, start_idx:end_idx] = 1
        
        # 显示重叠热力图
        im = axes[2].imshow(overlap_matrix, aspect='auto', cmap='viridis', alpha=0.8)
        axes[2].set_yticks(range(min(num_windows_to_show, num_windows)))
        axes[2].set_yticklabels([f'窗口 {i+1}' for i in range(min(num_windows_to_show, num_windows))])
        
        # 添加颜色条
        plt.colorbar(im, ax=axes[2], label='窗口覆盖')
        
        plt.tight_layout()
        plt.savefig('imgs/sliding_window_sampling_visualization.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 打印统计信息
        print(f"\n采样统计:")
        print(f"  每个窗口包含 {self.window_size} 个采样点")
        print(f"  相邻窗口重叠 {self.window_size - self.step_size} 个采样点")
        print(f"  重叠率: {overlap_ratio:.2%}")
        print(f"  数据利用率: {(num_windows * self.window_size) / signal_length:.2f}x")
    
    def _load_sample_signal(self):
        """加载一个示例信号用于可视化"""
        # 尝试从正常数据中加载一个示例
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
        
        # 如果正常数据不可用，尝试故障数据
        fault_folder = os.path.join(self.data_path, '12kHz_DE_data', 'B', '0007')
        if os.path.exists(fault_folder):
            for file_name in os.listdir(fault_folder):
                if file_name.endswith('.mat'):
                    file_path = os.path.join(fault_folder, file_name)
                    try:
                        mat_data = sio.loadmat(file_path)
                        for key in mat_data.keys():
                            if not key.startswith('_') and isinstance(mat_data[key], np.ndarray):
                                if mat_data[key].ndim == 2 and mat_data[key].shape[0] > mat_data[key].shape[1]:
                                    return mat_data[key].flatten()
                    except Exception as e:
                        continue
        
        return None

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
    
    def _signal_to_2d(self, signal):
        """将1D信号转换为2D图像"""
        # 确保信号长度为window_size
        if len(signal) != self.window_size:
            # 如果长度不匹配，进行截断或填充
            if len(signal) > self.window_size:
                signal = signal[:self.window_size]
            else:
                # 填充零
                signal = np.pad(signal, (0, self.window_size - len(signal)), 'constant')
        
        # 计算合适的2D形状 (尽量接近正方形)
        sqrt_size = int(np.sqrt(self.window_size))
        if sqrt_size * sqrt_size == self.window_size:
            # 完全平方数
            image_2d = signal.reshape(sqrt_size, sqrt_size)
        else:
            # 找到最接近的因子分解
            factors = []
            for i in range(1, int(np.sqrt(self.window_size)) + 1):
                if self.window_size % i == 0:
                    factors.append((i, self.window_size // i))
            
            # 选择最接近正方形的因子对
            best_factor = min(factors, key=lambda x: abs(x[0] - x[1]))
            image_2d = signal.reshape(best_factor[0], best_factor[1])
        
        return image_2d

    def visualize_class_signals_and_windows(self, num_windows_per_class=5):
        """
        可视化十一个类别的初始振动信号和滑窗采样后的2D图像
        
        Args:
            num_windows_per_class: 每个类别显示的窗口数量
        """
        print("开始生成类别信号和窗口可视化...")
        
        # 获取每个类别的原始信号和样本
        class_data = self._collect_class_data()
        class_names = self.get_class_names()
        
        # 创建大图：每个类别一行，包含原始信号和5个2D窗口
        num_classes = len(class_names)
        fig = plt.figure(figsize=(24, 4 * num_classes))
        
        for class_idx in range(num_classes):
            if class_idx not in class_data:
                continue
                
            raw_signal = class_data[class_idx]['raw_signal']
            windows = class_data[class_idx]['windows']
            
            # 每个类别占一行，6个子图（1个原始信号 + 5个2D窗口）
            row_start = class_idx * 6 + 1
            
            # 1. 原始振动信号
            ax_signal = plt.subplot(num_classes, 6, row_start)
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
            
            # 2-6. 前5个窗口的2D可视化
            for i in range(num_windows_per_class):
                ax_window = plt.subplot(num_classes, 6, row_start + 1 + i)
                
                if i < len(windows):
                    # 转换为2D图像
                    window_2d = self._signal_to_2d(windows[i])
                    
                    # 显示2D热力图
                    im = ax_window.imshow(window_2d, cmap='viridis', aspect='auto')
                    ax_window.set_title(f'窗口 {i+1}', fontsize=9)
                    ax_window.set_xticks([])
                    ax_window.set_yticks([])
                    
                    # 添加颜色条
                    plt.colorbar(im, ax=ax_window, fraction=0.046, pad=0.04)
                else:
                    # 如果窗口不足，显示空白
                    ax_window.text(0.5, 0.5, '无数据', ha='center', va='center', 
                                 transform=ax_window.transAxes, fontsize=12)
                    ax_window.set_xticks([])
                    ax_window.set_yticks([])
        
        plt.tight_layout()
        plt.savefig('imgs/class_signals_and_2d_windows.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("可视化完成，图像已保存到 imgs/class_signals_and_2d_windows.png")

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
                            mat_data = sio.loadmat(file_path)
                            for key in mat_data.keys():
                                if not key.startswith('_') and isinstance(mat_data[key], np.ndarray):
                                    if mat_data[key].ndim == 2 and mat_data[key].shape[0] > mat_data[key].shape[1]:
                                        return mat_data[key].flatten()
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
                            mat_data = sio.loadmat(file_path)
                            for key in mat_data.keys():
                                if not key.startswith('_') and isinstance(mat_data[key], np.ndarray):
                                    if mat_data[key].ndim == 2 and mat_data[key].shape[0] > mat_data[key].shape[1]:
                                        return mat_data[key].flatten()
        except Exception as e:
            print(f"加载 {fault_name} 数据时出错: {e}")
        
        return None

    def visualize_2d_window_comparison(self):
        """
        专门可视化不同类别的2D窗口对比
        """
        print("生成2D窗口对比图...")
        
        class_names = self.get_class_names()
        num_classes = len(class_names)
        
        # 创建网格布局：11行5列
        fig, axes = plt.subplots(num_classes, 5, figsize=(20, 3 * num_classes))
        
        # 收集每个类别的数据
        class_data = self._collect_class_data()
        
        for class_idx in range(num_classes):
            for window_idx in range(5):
                ax = axes[class_idx, window_idx] if num_classes > 1 else axes[window_idx]
                
                if class_idx in class_data and window_idx < len(class_data[class_idx]['windows']):
                    # 获取窗口数据并转换为2D
                    window_data = class_data[class_idx]['windows'][window_idx]
                    window_2d = self._signal_to_2d(window_data)
                    
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
        
        plt.tight_layout()
        plt.savefig('imgs/2d_windows_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("2D窗口对比图已保存到 imgs/2d_windows_comparison.png")

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
        """获取单个样本"""
        sample = self.samples[idx]
        label = self.labels[idx]
        
        if self.transform_to_2d:
            # 转换为2D图像格式
            sample = self._signal_to_2d(sample)
            # 添加通道维度 (1, H, W)
            sample = np.expand_dims(sample, axis=0)
        
        # 转换为PyTorch张量
        sample = torch.FloatTensor(sample)
        label = torch.LongTensor([label])[0]
        
        return sample, label
    
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

def create_bearing_dataloaders(data_path, batch_size=32, train_ratio=0.8, window_size=4096, step_size=None, overlap_ratio=0.5, transform_to_2d=False):
    """创建训练和测试数据加载器"""
    # 如果没有指定step_size，根据overlap_ratio计算
    if step_size is None:
        step_size = int(window_size * (1 - overlap_ratio))
    
    # 创建数据集
    dataset = BearingDataset(data_path, window_size, step_size, transform_to_2d)
    
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
    # 数据路径
    data_path = "数据集/数据集/源域数据集"
    
    # 创建数据加载器 - 使用更大的窗口
    train_loader, test_loader, dataset = create_bearing_dataloaders(
        data_path, 
        batch_size=32, 
        window_size=8192,  # 增大窗口大小
        step_size=2048,    # 相应调整步长
        transform_to_2d=True
    )
    
    # 可视化十一个类别的信号和2D窗口
    print("=" * 60)
    print("十一个类别的初始信号和2D窗口可视化")
    print("=" * 60)
    dataset.visualize_class_signals_and_windows(num_windows_per_class=5)
    
    # 可视化2D窗口对比
    print("\n" + "=" * 60)
    print("2D窗口对比可视化")
    print("=" * 60)
    dataset.visualize_2d_window_comparison()
    
    # 测试数据加载
    print("\n测试数据加载:")
    for i, (batch_data, batch_labels) in enumerate(train_loader):
        print(f"批次 {i+1}: 数据形状 {batch_data.shape}, 标签形状 {batch_labels.shape}")
        if i >= 2:  # 只显示前3个批次
            break