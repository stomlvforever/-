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
    def __init__(self, data_path, window_size=4096, step_size=1024, transform_to_2d=False):
        """
        轴承故障诊断数据集 - 修正版本
        
        Args:
            data_path: 数据路径
            window_size: 窗口大小
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
        # 确保信号长度为4096 (64*64)
        if len(signal) != self.window_size:
            # 如果长度不匹配，进行截断或填充
            if len(signal) > self.window_size:
                signal = signal[:self.window_size]
            else:
                # 填充零
                signal = np.pad(signal, (0, self.window_size - len(signal)), 'constant')
        
        # 重塑为64x64的2D图像
        image_2d = signal.reshape(64, 64)
        return image_2d
    
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
            # 添加通道维度 (1, 64, 64)
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
    
    def visualize_samples(self, num_samples=12, transform_to_2d=False):
        """可视化样本 - 修改为显示12个样本"""
        # 尝试每个类别显示一个样本
        fig, axes = plt.subplots(3, 4, figsize=(20, 12))
        axes = axes.flatten()
        
        class_names = self.get_class_names()
        
        # 尝试为每个类别找一个样本
        samples_to_show = []
        labels_to_show = []
        
        for class_idx in range(len(class_names)):
            # 找到该类别的第一个样本
            found = False
            for i, label in enumerate(self.labels):
                if label == class_idx:
                    samples_to_show.append(self.samples[i])
                    labels_to_show.append(label)
                    found = True
                    break
            
            if not found:
                # 如果某个类别没有样本，用其他样本填充
                if len(self.samples) > 0:
                    samples_to_show.append(self.samples[0])
                    labels_to_show.append(0)
        
        # 确保有12个样本
        while len(samples_to_show) < 12:
            if len(self.samples) > 0:
                samples_to_show.append(self.samples[0])
                labels_to_show.append(0)
            else:
                break
        
        # 绘制样本
        for i in range(min(12, len(samples_to_show))):
            sample = samples_to_show[i]
            label = labels_to_show[i]
            
            if transform_to_2d:
                # 2D热力图
                sample_2d = self._signal_to_2d(sample)
                im = axes[i].imshow(sample_2d, cmap='viridis', aspect='auto')
                axes[i].set_title(f'{class_names[label]}', fontsize=10)
                plt.colorbar(im, ax=axes[i])
            else:
                # 1D折线图
                axes[i].plot(sample)
                axes[i].set_title(f'{class_names[label]}', fontsize=10)
                axes[i].set_xlabel('时间点')
                axes[i].set_ylabel('振幅')
        
        # 隐藏多余的子图
        for i in range(len(samples_to_show), 12):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig('imgs/bearing_samples_visualization_12class.png', dpi=300, bbox_inches='tight')
        plt.show()

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
    
    # 创建数据加载器
    train_loader, test_loader, dataset = create_bearing_dataloaders(
        data_path, 
        batch_size=32, 
        window_size=4096, 
        step_size=1024,
        transform_to_2d=True
    )
    
    # 可视化样本
    dataset.visualize_samples(12)
    
    # 测试数据加载
    print("\n测试数据加载:")
    for i, (batch_data, batch_labels) in enumerate(train_loader):
        print(f"批次 {i+1}: 数据形状 {batch_data.shape}, 标签形状 {batch_labels.shape}")
        if i >= 2:  # 只显示前3个批次
            break