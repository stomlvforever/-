import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, adjusted_rand_score
from datetime import datetime
import logging
import matplotlib
from scipy import signal as scipy_signal
from scipy.signal import butter, filtfilt, resample

matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

from cnn_model_pytorch import DenseNetCNN, DenseBlock
from bearing_data_loader import BearingDataset
from config import Config

class TargetDomainDataset(Dataset):
    """目标域数据集（与源域预处理保持一致）"""
    
    def __init__(self, data_path, transform_method='stft', enable_resampling=True, target_fs=12000, for_training=False):
        """
        初始化目标域数据集
        
        Args:
            data_path: 目标域数据路径
            transform_method: 信号转换方法 ('stft', 'cwt', 'spectrogram', 'reshape')
            enable_resampling: 是否启用重采样
            target_fs: 目标采样频率
            for_training: 是否用于训练（如果是，则使用随机标签；否则使用文件标签）
        """
        self.data_path = data_path
        self.transform_method = transform_method
        self.enable_resampling = enable_resampling
        self.target_fs = target_fs
        self.for_training = for_training
        
        # 滑窗参数（与源域保持一致）
        self.window_size = 1024
        self.step_size = 512
        
        # 数据存储
        self.samples = []
        self.file_labels = []  # 用于训练的标签
        self.real_file_labels = []  # 真实的文件标签（用于预测时识别文件）
        self.file_names = []
        self.file_label_to_name = {}
        
        # 加载和预处理数据
        self._load_data()
        self._normalize_data()
        
        print(f"目标域数据集加载完成:")
        print(f"  总样本数: {len(self.samples)}")
        print(f"  文件数: {len(self.file_label_to_name)}")
        print(f"  重采样: {'启用' if enable_resampling else '禁用'}")
        if enable_resampling:
            print(f"  目标频率: {target_fs}Hz")
        print(f"  转换方法: {transform_method}")
        print(f"  训练模式: {'是' if for_training else '否'}")
    
    def _resample_signal(self, signal, original_fs, target_fs):
        """
        重采样信号到目标采样频率（与源域保持一致）
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
    
    def _sliding_window_sampling_unified(self, signal_data, fs):
        """
        统一的滑窗采样（与源域保持一致）
        """
        if self.enable_resampling:
            # 重采样模式：使用统一的窗口大小
            window_size = self.window_size
            step_size = self.step_size
        else:
            # 原始模式：根据采样频率调整窗口大小
            base_fs = 12000  # 基准采样频率
            scale_factor = fs / base_fs
            window_size = int(self.window_size * scale_factor)
            step_size = int(self.step_size * scale_factor)
        
        # 滑窗采样
        windows = []
        for i in range(0, len(signal_data) - window_size + 1, step_size):
            window = signal_data[i:i + window_size]
            windows.append(window)
        
        return windows
    
    def _signal_to_2d_unified(self, signal, fs):
        """
        统一的信号到2D转换（与源域保持一致）
        """
        method = self.transform_method
        
        if method == 'stft':
            # 动态调整STFT参数，确保nperseg不超过信号长度
            signal_length = len(signal)
            nperseg = min(256, signal_length)  # 确保nperseg不超过信号长度
            
            # 确保nperseg是偶数，便于计算noverlap
            if nperseg % 2 != 0:
                nperseg -= 1
                
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
    
    def _detect_sampling_frequency(self, file_path):
        """检测采样频率"""
        try:
            data = sio.loadmat(file_path)
            
            # 查找包含采样频率信息的键
            for key in data.keys():
                if not key.startswith('__'):
                    signal_data = data[key]
                    if isinstance(signal_data, np.ndarray) and signal_data.size > 0:
                        # 根据信号长度估计采样频率
                        signal_length = len(signal_data.flatten())
                        
                        # 基于经验的采样频率估计
                        if signal_length > 400000:  # 长信号，可能是48kHz
                            return 48000
                        elif signal_length > 200000:  # 中等长度，可能是32kHz
                            return 32000
                        else:  # 短信号，可能是12kHz
                            return 12000
            
            # 默认返回32kHz
            return 32000
            
        except Exception as e:
            print(f"检测采样频率失败 {file_path}: {e}")
            return 32000  # 默认采样频率
    
    def _load_data(self):
        """加载目标域数据"""
        print(f"加载目标域数据: {self.data_path}")
        
        # 获取所有.mat文件
        mat_files = []
        for file in os.listdir(self.data_path):
            if file.endswith('.mat'):
                mat_files.append(file)
        
        mat_files.sort()  # 确保文件顺序一致
        print(f"找到 {len(mat_files)} 个.mat文件")
        
        current_file_label = 0
        
        for file_name in mat_files:
            file_path = os.path.join(self.data_path, file_name)
            print(f"处理文件: {file_name}")
            
            try:
                # 加载.mat文件
                data = sio.loadmat(file_path)
                
                # 检测采样频率
                original_fs = self._detect_sampling_frequency(file_path)
                print(f"  检测到采样频率: {original_fs}Hz")
                
                # 查找信号数据
                signal_data = None
                for key in data.keys():
                    if not key.startswith('__'):
                        potential_signal = data[key]
                        if isinstance(potential_signal, np.ndarray) and potential_signal.size > 0:
                            signal_data = potential_signal.flatten()
                            break
                
                if signal_data is None:
                    print(f"  警告: 未找到有效信号数据")
                    continue
                
                print(f"  原始信号长度: {len(signal_data)}")
                
                # 重采样（如果启用）
                if self.enable_resampling:
                    signal_data = self._resample_signal(signal_data, original_fs, self.target_fs)
                    fs = self.target_fs
                    print(f"  重采样后长度: {len(signal_data)}")
                else:
                    fs = original_fs
                
                # 滑窗采样
                windows = self._sliding_window_sampling_unified(signal_data, fs)
                print(f"  生成窗口数: {len(windows)}")
                
                # 转换为2D图像
                for window in windows:
                    try:
                        image_2d = self._signal_to_2d_unified(window, fs)
                        self.samples.append(image_2d)
                        
                        if self.for_training:
                            # 训练模式：使用随机类别标签（0-3）
                            random_label = np.random.randint(0, 4)
                            self.file_labels.append(random_label)
                        else:
                            # 预测模式：使用虚拟标签0
                            self.file_labels.append(0)
                        
                        self.real_file_labels.append(current_file_label)  # 保存真实的文件标签
                        self.file_names.append(file_name)
                    except Exception as e:
                        print(f"    2D转换失败: {e}")
                        continue
                
                # 建立文件标签到文件名的映射
                self.file_label_to_name[current_file_label] = file_name
                current_file_label += 1
                
            except Exception as e:
                print(f"  处理文件失败: {e}")
                continue
        
        print(f"数据加载完成，总样本数: {len(self.samples)}")
    
    def _normalize_data(self):
        """标准化数据"""
        if len(self.samples) == 0:
            return
        
        # 转换为numpy数组
        samples_array = np.array(self.samples)
        
        # 计算均值和标准差
        mean = np.mean(samples_array)
        std = np.std(samples_array)
        
        # 标准化
        if std > 0:
            samples_array = (samples_array - mean) / std
        
        # 转换回列表
        self.samples = samples_array.tolist()
        
        print(f"数据标准化完成 - 均值: {mean:.4f}, 标准差: {std:.4f}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = torch.FloatTensor(self.samples[idx]).unsqueeze(0)  # 添加通道维度
        
        if self.for_training:
            # 训练模式：返回类别标签
            label = torch.tensor(self.file_labels[idx], dtype=torch.long)
        else:
            # 预测模式：返回文件标签
            label = torch.tensor(self.real_file_labels[idx], dtype=torch.long)
        
        return sample, label

class TransferLearningTrainer:
    """迁移学习训练器"""
    
    def __init__(self, source_model_path, target_data_path, transfer_strategy='fine_tune', 
                 enable_resampling=True, target_fs=12000):
        """
        初始化迁移学习训练器
        
        Args:
            source_model_path: 源域预训练模型路径
            target_data_path: 目标域数据路径
            transfer_strategy: 迁移策略 ('feature_extract', 'fine_tune')
            enable_resampling: 是否启用重采样
            target_fs: 重采样目标频率
        """
        self.source_model_path = source_model_path
        self.target_data_path = target_data_path
        self.transfer_strategy = transfer_strategy
        self.enable_resampling = enable_resampling
        self.target_fs = target_fs
        
        # 设备选择和内存管理
        if torch.cuda.is_available():
            # 检查GPU内存
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"GPU内存: {gpu_memory:.1f}GB")
            
            if gpu_memory < 4.0:  # 如果GPU内存小于4GB，使用CPU
                print("GPU内存不足，使用CPU进行训练")
                self.device = torch.device('cpu')
            else:
                self.device = torch.device('cuda')
                # 清理GPU内存
                torch.cuda.empty_cache()
        else:
            self.device = torch.device('cpu')
        
        # 创建结果目录
        resampling_suffix = "_resampled" if enable_resampling else ""
        self.results_dir = f"transfer_learning_training_{Config.TRANSFORM_METHOD}{resampling_suffix}"
        os.makedirs(self.results_dir, exist_ok=True)
        
        # 设置日志
        self._setup_logging()
        
        print(f"使用设备: {self.device}")
        print(f"迁移策略: {self.transfer_strategy}")
        print(f"重采样: {'启用' if enable_resampling else '禁用'}")
        if enable_resampling:
            print(f"重采样目标频率: {target_fs}Hz")
        print(f"结果保存到: {self.results_dir}")
    
    def _setup_logging(self):
        """设置日志记录"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(self.results_dir, f"transfer_training_{timestamp}.log")
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def load_source_model(self):
        """加载源域预训练模型"""
        self.logger.info(f"加载源域模型: {self.source_model_path}")
        
        try:
            # 修复torch.load的安全警告，并添加错误处理
            checkpoint = torch.load(
                self.source_model_path, 
                map_location='cpu',  # 先加载到CPU
                weights_only=False   # 明确设置以消除警告
            )
            
            if isinstance(checkpoint, DenseNetCNN):
                model = checkpoint
                self.logger.info("加载完整模型对象")
            elif isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model = DenseNetCNN(num_classes=Config.NUM_CLASSES)
                model.load_state_dict(checkpoint['model_state_dict'])
                self.logger.info("加载模型状态字典")
            elif isinstance(checkpoint, dict):
                model = DenseNetCNN(num_classes=Config.NUM_CLASSES)
                model.load_state_dict(checkpoint)
                self.logger.info("加载状态字典")
            else:
                raise ValueError(f"无法识别的模型文件格式: {type(checkpoint)}")
            
            # 清理GPU内存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # 将模型移动到目标设备
            model.to(self.device)
            model.eval()  # 设置为评估模式
            
            self.logger.info(f"源域模型加载完成，设备: {self.device}")
            return model
            
        except RuntimeError as e:
            if "CUDA" in str(e):
                self.logger.warning(f"CUDA错误，尝试使用CPU: {e}")
                # 如果CUDA出错，强制使用CPU
                self.device = torch.device('cpu')
                return self.load_source_model()  # 递归调用，但现在使用CPU
            else:
                raise e
        except Exception as e:
            self.logger.error(f"模型加载失败: {e}")
            raise e
    
    def prepare_target_data(self, train_ratio=0.8):
        """准备目标域数据"""
        self.logger.info("准备目标域数据...")
        
        # 创建目标域数据集（训练模式）
        target_dataset = TargetDomainDataset(
            data_path=self.target_data_path,
            transform_method=Config.TRANSFORM_METHOD,
            enable_resampling=self.enable_resampling,
            target_fs=self.target_fs,
            for_training=True  # 启用训练模式
        )
        
        # 分割训练和验证集
        dataset_size = len(target_dataset)
        train_size = int(train_ratio * dataset_size)
        val_size = dataset_size - train_size
        
        train_dataset, val_dataset = random_split(target_dataset, [train_size, val_size])
        
        # 创建数据加载器
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
        
        self.logger.info(f"目标域数据准备完成:")
        self.logger.info(f"  训练样本: {train_size}")
        self.logger.info(f"  验证样本: {val_size}")
        
        return train_loader, val_loader, target_dataset
    
    def setup_transfer_strategy(self, model):
        """设置迁移策略"""
        self.logger.info(f"设置迁移策略: {self.transfer_strategy}")
        
        if self.transfer_strategy == 'feature_extract':
            # 特征提取：冻结所有层，只训练最后的分类器
            for param in model.parameters():
                param.requires_grad = False
            
            # 解冻最后的全连接层
            for param in model.fc2.parameters():
                param.requires_grad = True
            
            self.logger.info("特征提取模式：冻结卷积层，只训练分类器")
            
        elif self.transfer_strategy == 'fine_tune':
            # 微调：训练所有层，但使用较小的学习率
            for param in model.parameters():
                param.requires_grad = True
            
            self.logger.info("微调模式：训练所有层")
        
        return model
    
    def train_epoch(self, model, train_loader, optimizer, criterion, epoch):
        """训练一个epoch"""
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, targets) in enumerate(train_loader):
            try:
                # 添加数据形状调试信息（仅第一个epoch的第一个批次）
                if epoch == 0 and batch_idx == 0:
                    self.logger.info(f"训练数据形状: {data.shape}, 标签形状: {targets.shape}")
                    self.logger.info(f"训练数据类型: {data.dtype}, 标签类型: {targets.dtype}")
                
                # 确保数据是4D的
                if data.dim() == 3:
                    # 如果是3D，添加batch维度
                    data = data.unsqueeze(0)
                    self.logger.warning(f"训练数据维度不正确，从3D调整为4D: {data.shape}")
                elif data.dim() != 4:
                    self.logger.error(f"训练数据维度错误: {data.dim()}D, 形状: {data.shape}")
                    continue
                
                data, targets = data.to(self.device), targets.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(data)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                # 定期清理GPU内存
                if batch_idx % 10 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
            except RuntimeError as e:
                if "CUDA" in str(e):
                    self.logger.error(f"CUDA错误在批次 {batch_idx}: {e}")
                    # 清理内存并跳过这个批次
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue
                else:
                    raise e
        
        train_loss /= len(train_loader)
        train_acc = 100. * correct / total
        
        self.logger.info(f"Epoch {epoch+1} - 训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.2f}%")
        return train_loss, train_acc
    
    def validate_epoch(self, model, val_loader, criterion):
        """验证一个epoch"""
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch_idx, (data, targets) in enumerate(val_loader):
                try:
                    # 确保数据类型正确
                    if not isinstance(data, torch.Tensor):
                        self.logger.error(f"数据类型错误: {type(data)}")
                        continue
                    if not isinstance(targets, torch.Tensor):
                        self.logger.error(f"标签类型错误: {type(targets)}, 值: {targets}")
                        continue
                    
                    # 添加详细的调试信息
                    if batch_idx == 0:  # 只在第一个批次打印
                        self.logger.info(f"验证数据形状: {data.shape}, 标签形状: {targets.shape}")
                        self.logger.info(f"验证数据类型: {data.dtype}, 标签类型: {targets.dtype}")
                        self.logger.info(f"标签值: {targets}")
                    
                    # 检查标签是否为空
                    if targets.numel() == 0:
                        self.logger.warning(f"批次 {batch_idx} 标签为空，跳过")
                        continue
                    
                    # 确保数据是4D的
                    if data.dim() == 3:
                        # 如果是3D，添加batch维度
                        data = data.unsqueeze(0)
                        self.logger.warning(f"验证数据维度不正确，从3D调整为4D: {data.shape}")
                    elif data.dim() != 4:
                        self.logger.error(f"验证数据维度错误: {data.dim()}D, 形状: {data.shape}")
                        continue
                    
                    # 确保数据和标签的batch size匹配
                    if data.size(0) != targets.size(0):
                        self.logger.error(f"批次 {batch_idx} 数据和标签batch size不匹配: 数据{data.size(0)}, 标签{targets.size(0)}")
                        continue
                        
                    data, targets = data.to(self.device), targets.to(self.device)
                    
                    outputs = model(data)
                    loss = criterion(outputs, targets)
                    
                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()
                    
                    all_preds.extend(predicted.cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())
                    
                    # 定期清理GPU内存
                    if batch_idx % 10 == 0 and torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        
                except RuntimeError as e:
                    if "CUDA" in str(e):
                        self.logger.error(f"CUDA错误在验证批次 {batch_idx}: {e}")
                        # 清理内存并跳过这个批次
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        continue
                    else:
                        self.logger.error(f"运行时错误在验证批次 {batch_idx}: {e}")
                        self.logger.error(f"数据形状: {data.shape if isinstance(data, torch.Tensor) else 'N/A'}")
                        self.logger.error(f"标签形状: {targets.shape if isinstance(targets, torch.Tensor) else 'N/A'}")
                        continue
                except Exception as e:
                    self.logger.error(f"验证批次 {batch_idx} 发生错误: {e}")
                    self.logger.error(f"数据类型: {type(data)}, 标签类型: {type(targets)}")
                    if isinstance(data, torch.Tensor):
                        self.logger.error(f"数据形状: {data.shape}")
                    if isinstance(targets, torch.Tensor):
                        self.logger.error(f"标签形状: {targets.shape}")
                    continue
        
        if total == 0:
            self.logger.warning("验证过程中没有有效的批次")
            return 0.0, 0.0, [], []
        
        val_loss /= len(val_loader)
        val_acc = 100. * correct / total
        
        self.logger.info(f"验证损失: {val_loss:.4f}, 验证准确率: {val_acc:.2f}%")
        return val_loss, val_acc, all_preds, all_targets
    
    def train(self, epochs=20):
        """执行迁移学习训练"""
        self.logger.info("开始迁移学习训练...")
        
        # 1. 加载源域模型
        model = self.load_source_model()
        
        # 2. 准备目标域数据
        train_loader, val_loader, target_dataset = self.prepare_target_data()
        
        # 3. 设置迁移策略
        model = self.setup_transfer_strategy(model)
        
        # 4. 定义损失函数和优化器
        criterion = nn.CrossEntropyLoss()
        
        if self.transfer_strategy == 'feature_extract':
            # 特征提取模式：只优化分类器
            optimizer = optim.Adam(model.fc2.parameters(), lr=0.001)
        else:
            # 微调模式：优化所有参数，使用较小学习率
            optimizer = optim.Adam(model.parameters(), lr=0.0001)
        
        # 5. 训练循环
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        best_val_acc = 0.0
        best_model_state = None
        
        for epoch in range(epochs):
            # 训练
            train_loss, train_acc = self.train_epoch(model, train_loader, optimizer, criterion, epoch)
            
            # 验证
            val_loss, val_acc, val_preds, val_targets = self.validate_epoch(model, val_loader, criterion)
            
            # 记录历史
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            # 保存最佳模型
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = model.state_dict().copy()
                self.logger.info(f"新的最佳验证准确率: {best_val_acc:.2f}%")
        
        # 6. 加载最佳模型状态
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        
        # 7. 保存结果
        self.save_results(model, best_model_state, history, val_preds, val_targets)
        
        self.logger.info("迁移学习训练完成")
        return model, history
    
    def save_results(self, model, best_model_state, history, val_preds, val_targets):
        """保存训练结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存模型
        model_path = os.path.join(self.results_dir, f"best_model_{self.transfer_strategy}_{timestamp}.pth")
        torch.save({
            'model_state_dict': best_model_state,
            'transfer_strategy': self.transfer_strategy,
            'enable_resampling': self.enable_resampling,
            'resampling_target_fs': self.target_fs,
            'transform_method': Config.TRANSFORM_METHOD
        }, model_path)
        self.logger.info(f"模型已保存: {model_path}")
        
        # 保存训练历史
        history_df = pd.DataFrame(history)
        history_csv_path = os.path.join(self.results_dir, f"training_history_{self.transfer_strategy}_{timestamp}.csv")
        history_df.to_csv(history_csv_path, index=False)
        
        # 绘制训练曲线
        self.plot_training_history(history, timestamp)
        
        # 绘制混淆矩阵
        self.plot_confusion_matrix(val_preds, val_targets, timestamp)
    
    def plot_training_history(self, history, timestamp):
        """绘制训练历史曲线"""
        plt.figure(figsize=(12, 4))
        
        # 损失曲线
        plt.subplot(1, 2, 1)
        plt.plot(history['train_loss'], label='训练损失')
        plt.plot(history['val_loss'], label='验证损失')
        plt.title('训练和验证损失')
        plt.xlabel('Epoch')
        plt.ylabel('损失')
        plt.legend()
        plt.grid(True)
        
        # 准确率曲线
        plt.subplot(1, 2, 2)
        plt.plot(history['train_acc'], label='训练准确率')
        plt.plot(history['val_acc'], label='验证准确率')
        plt.title('训练和验证准确率')
        plt.xlabel('Epoch')
        plt.ylabel('准确率 (%)')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        
        # 保存图像
        save_path = os.path.join(self.results_dir, f"training_history_{self.transfer_strategy}_{timestamp}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"训练曲线已保存: {save_path}")
    
    def plot_confusion_matrix(self, val_preds, val_targets, timestamp):
        """绘制混淆矩阵"""
        class_names = ['正常', '内圈故障', '滚动体故障', '外圈故障']
        all_labels = list(range(len(class_names)))
        
        # 计算混淆矩阵
        cm = confusion_matrix(val_targets, val_preds, labels=all_labels)
        
        # 绘制混淆矩阵
        plt.figure(figsize=(8, 6))
        
        # 计算百分比
        cm_percent = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-10) * 100
        
        # 创建标注
        annotations = []
        for i in range(cm.shape[0]):
            row = []
            for j in range(cm.shape[1]):
                if cm[i, j] > 0:
                    row.append(f'{cm[i, j]}\n({cm_percent[i, j]:.1f}%)')
                else:
                    row.append('0\n(0.0%)')
            annotations.append(row)
        
        # 绘制热力图
        sns.heatmap(cm, annot=annotations, fmt='', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        
        plt.title(f'混淆矩阵 - {self.transfer_strategy}')
        plt.xlabel('预测标签')
        plt.ylabel('真实标签')
        
        # 保存图像
        save_path = os.path.join(self.results_dir, f"confusion_matrix_{self.transfer_strategy}_{timestamp}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"混淆矩阵已保存: {save_path}")

class UnsupervisedTransferLearning:
    """无监督迁移学习类"""
    
    def __init__(self, source_model_path, target_data_path, enable_resampling=True, target_fs=12000):
        """
        初始化无监督迁移学习
        
        Args:
            source_model_path: 源域预训练模型路径
            target_data_path: 目标域数据路径
            enable_resampling: 是否启用重采样
            target_fs: 重采样目标频率
        """
        self.source_model_path = source_model_path
        self.target_data_path = target_data_path
        self.enable_resampling = enable_resampling
        self.target_fs = target_fs
        
        # 设备选择
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 创建结果目录
        self.results_dir = f"unsupervised_transfer_learning_{Config.TRANSFORM_METHOD}"
        os.makedirs(self.results_dir, exist_ok=True)
        
        # 设置日志
        self._setup_logging()
        
        print(f"无监督迁移学习初始化完成")
        print(f"使用设备: {self.device}")
        print(f"结果保存到: {self.results_dir}")
    
    def _setup_logging(self):
        """设置日志记录"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(self.results_dir, f"unsupervised_transfer_{timestamp}.log")
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def load_source_model(self):
        """加载源域预训练模型"""
        self.logger.info(f"加载源域模型: {self.source_model_path}")
        
        try:
            checkpoint = torch.load(
                self.source_model_path, 
                map_location='cpu',
                weights_only=False
            )
            
            if isinstance(checkpoint, DenseNetCNN):
                model = checkpoint
                self.logger.info("加载完整模型对象")
            elif isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model = DenseNetCNN(num_classes=Config.NUM_CLASSES)
                model.load_state_dict(checkpoint['model_state_dict'])
                self.logger.info("加载模型状态字典")
            elif isinstance(checkpoint, dict):
                model = DenseNetCNN(num_classes=Config.NUM_CLASSES)
                model.load_state_dict(checkpoint)
                self.logger.info("加载状态字典")
            else:
                raise ValueError(f"无法识别的模型文件格式: {type(checkpoint)}")
            
            model.to(self.device)
            model.eval()
            
            self.logger.info(f"源域模型加载完成")
            return model
            
        except Exception as e:
            self.logger.error(f"模型加载失败: {e}")
            raise e
    
    def extract_features(self, model, data_loader):
        """提取特征"""
        self.logger.info("开始提取特征...")
        
        model.eval()
        features = []
        file_names = []
        
        with torch.no_grad():
            for batch_idx, (data, _) in enumerate(data_loader):
                data = data.to(self.device)
                
                # 按照DenseNetCNN的实际结构提取特征
                # Conv1 + BN + ReLU + Pool1
                x = model.pool1(F.relu(model.bn1(model.conv1(data))))
                
                # Dense Block 1
                x = model.dense1(x)
                
                # Conv2 + BN + ReLU + Pool2
                x = model.pool2(F.relu(model.bn2(model.conv2(x))))
                
                # Dense Block 2
                x = model.dense2(x)
                
                # Conv3 + BN + ReLU + Pool3
                x = model.pool3(F.relu(model.bn3(model.conv3(x))))
                
                # Dense Block 3
                x = model.dense3(x)
                
                # Global Average Pooling
                x = model.global_avg_pool(x)
                x = x.view(x.size(0), -1)  # 展平
                
                # 通过第一个全连接层获取特征表示
                x = F.relu(model.fc1(x))
                # 不使用dropout和最后的分类层，保留特征表示
                
                features.append(x.cpu().numpy())
                
                # 生成文件名（基于批次索引）
                for i in range(len(data)):
                    file_names.append(f"batch_{batch_idx}_sample_{i}")
        
        features = np.vstack(features)
        self.logger.info(f"特征提取完成，特征维度: {features.shape}")
        
        return features, file_names
    
    def perform_clustering(self, features, n_clusters=4):
        """执行聚类分析"""
        self.logger.info(f"开始聚类分析，聚类数: {n_clusters}")
        
        # K-means聚类
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(features)
        
        # 计算聚类质量指标
        silhouette_avg = silhouette_score(features, cluster_labels)
        
        self.logger.info(f"聚类完成")
        self.logger.info(f"轮廓系数: {silhouette_avg:.4f}")
        
        return {
            'cluster_labels': cluster_labels,
            'cluster_centers': kmeans.cluster_centers_,
            'silhouette_score': silhouette_avg,
            'kmeans_model': kmeans
        }
    
    def reduce_dimensions(self, features, method='both'):
        """降维可视化"""
        self.logger.info(f"开始降维分析: {method}")
        
        results = {}
        
        if method in ['pca', 'both']:
            # PCA降维
            pca = PCA(n_components=2, random_state=42)
            features_pca = pca.fit_transform(features)
            results['pca'] = {
                'features': features_pca,
                'explained_variance_ratio': pca.explained_variance_ratio_
            }
            self.logger.info(f"PCA降维完成，解释方差比: {pca.explained_variance_ratio_}")
        
        if method in ['tsne', 'both']:
            # t-SNE降维
            tsne = TSNE(n_components=2, random_state=42, perplexity=30)
            features_tsne = tsne.fit_transform(features)
            results['tsne'] = {
                'features': features_tsne
            }
            self.logger.info("t-SNE降维完成")
        
        return results
    
    def analyze_file_level_predictions(self, features, cluster_labels, file_names, target_dataset):
        """分析文件级别的预测结果"""
        self.logger.info("开始文件级别分析...")
        
        # 将窗口级别的结果聚合到文件级别
        file_clusters = {}
        
        # 使用数据集中的真实文件名
        for i, sample_idx in enumerate(range(len(cluster_labels))):
            # 获取真实的文件名
            real_file_name = target_dataset.file_names[sample_idx] if sample_idx < len(target_dataset.file_names) else f"unknown_file_{i}"
            
            if real_file_name not in file_clusters:
                file_clusters[real_file_name] = []
            file_clusters[real_file_name].append(cluster_labels[i])
        
        # 为每个文件确定最终的聚类标签（多数投票）
        file_predictions = {}
        for file_name, clusters in file_clusters.items():
            cluster_counts = np.bincount(clusters)
            predicted_cluster = np.argmax(cluster_counts)
            confidence = cluster_counts[predicted_cluster] / len(clusters)
            
            file_predictions[file_name] = {
                'predicted_cluster': predicted_cluster,
                'confidence': confidence,
                'window_clusters': clusters,
                'cluster_distribution': cluster_counts
            }
        
        self.logger.info(f"文件级别分析完成，共 {len(file_predictions)} 个文件")
        
        return file_predictions
    
    def visualize_results(self, features, cluster_labels, file_names, dim_reduction_results, file_predictions):
        """可视化结果"""
        self.logger.info("开始生成可视化结果...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. PCA聚类可视化
        if 'pca' in dim_reduction_results:
            ax1 = axes[0, 0]
            pca_features = dim_reduction_results['pca']['features']
            scatter = ax1.scatter(pca_features[:, 0], pca_features[:, 1], 
                                c=cluster_labels, cmap='viridis', alpha=0.6)
            ax1.set_title('PCA聚类可视化')
            ax1.set_xlabel('第一主成分')
            ax1.set_ylabel('第二主成分')
            plt.colorbar(scatter, ax=ax1)
        
        # 2. t-SNE聚类可视化
        if 'tsne' in dim_reduction_results:
            ax2 = axes[0, 1]
            tsne_features = dim_reduction_results['tsne']['features']
            scatter = ax2.scatter(tsne_features[:, 0], tsne_features[:, 1], 
                                c=cluster_labels, cmap='viridis', alpha=0.6)
            ax2.set_title('t-SNE聚类可视化')
            ax2.set_xlabel('t-SNE维度1')
            ax2.set_ylabel('t-SNE维度2')
            plt.colorbar(scatter, ax=ax2)
        
        # 3. 文件级别预测分布
        ax3 = axes[1, 0]
        file_cluster_labels = [pred['predicted_cluster'] for pred in file_predictions.values()]
        cluster_counts = np.bincount(file_cluster_labels)
        
        bars = ax3.bar(range(len(cluster_counts)), cluster_counts, 
                      color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'][:len(cluster_counts)])
        ax3.set_title('文件级别聚类分布')
        ax3.set_xlabel('聚类标签')
        ax3.set_ylabel('文件数量')
        ax3.set_xticks(range(len(cluster_counts)))
        ax3.set_xticklabels([f'聚类{i}' for i in range(len(cluster_counts))])
        
        # 添加数值标签
        for bar, count in zip(bars, cluster_counts):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    str(count), ha='center', va='bottom', fontweight='bold')
        
        # 4. 置信度分布
        ax4 = axes[1, 1]
        confidences = [pred['confidence'] for pred in file_predictions.values()]
        ax4.hist(confidences, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax4.set_title('预测置信度分布')
        ax4.set_xlabel('置信度')
        ax4.set_ylabel('文件数量')
        ax4.axvline(np.mean(confidences), color='red', linestyle='--', 
                   label=f'平均值: {np.mean(confidences):.3f}')
        ax4.legend()
        
        plt.tight_layout()
        
        # 保存图像
        save_path = os.path.join(self.results_dir, "unsupervised_transfer_learning_results.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        self.logger.info(f"可视化结果已保存: {save_path}")
        plt.show()
        
        return save_path
    
    def save_results(self, file_predictions, features, cluster_labels, clustering_results):
        """保存结果"""
        self.logger.info("保存分析结果...")
        
        # 1. 保存文件级别预测结果
        file_results = []
        for file_name, pred in file_predictions.items():
            file_results.append({
                'file_name': file_name,
                'predicted_cluster': pred['predicted_cluster'],
                'confidence': pred['confidence'],
                'window_count': len(pred['window_clusters'])
            })
        
        df_files = pd.DataFrame(file_results)
        file_csv_path = os.path.join(self.results_dir, "file_level_predictions.csv")
        df_files.to_csv(file_csv_path, index=False, encoding='utf-8-sig')
        
        # 2. 保存聚类统计信息
        stats_data = []
        
        # 整体统计
        stats_data.append({
            '指标': '总样本数',
            '值': len(cluster_labels),
            '说明': '目标域总样本数量'
        })
        
        stats_data.append({
            '指标': '特征维度',
            '值': features.shape[1],
            '说明': '提取的特征维度'
        })
        
        stats_data.append({
            '指标': '聚类数量',
            '值': len(np.unique(cluster_labels)),
            '说明': 'K-means聚类数量'
        })
        
        stats_data.append({
            '指标': '轮廓系数',
            '值': f"{clustering_results['silhouette_score']:.4f}",
            '说明': '聚类质量指标 (越高越好)'
        })
        
        # 各聚类样本分布
        unique_clusters, counts = np.unique(cluster_labels, return_counts=True)
        for cluster, count in zip(unique_clusters, counts):
            percentage = count / len(cluster_labels) * 100
            stats_data.append({
                '指标': f'聚类{cluster}_样本数',
                '值': count,
                '说明': f'占总样本的 {percentage:.1f}%'
            })
        
        df_stats = pd.DataFrame(stats_data)
        stats_csv_path = os.path.join(self.results_dir, "clustering_statistics.csv")
        df_stats.to_csv(stats_csv_path, index=False, encoding='utf-8-sig')
        
        self.logger.info(f"文件级别预测结果已保存: {file_csv_path}")
        self.logger.info(f"聚类统计信息已保存: {stats_csv_path}")
        
        return file_csv_path, stats_csv_path
    
    def run_unsupervised_transfer_learning(self):
        """运行完整的无监督迁移学习流程"""
        self.logger.info("开始无监督迁移学习...")
        
        try:
            # 1. 加载源域模型
            source_model = self.load_source_model()
            
            # 2. 准备目标域数据
            target_dataset = TargetDomainDataset(
                data_path=self.target_data_path,
                transform_method=Config.TRANSFORM_METHOD,
                enable_resampling=self.enable_resampling,
                target_fs=self.target_fs,
                for_training=False  # 不使用随机标签
            )
            
            data_loader = DataLoader(target_dataset, batch_size=32, shuffle=False)
            
            # 3. 提取特征
            features, file_names = self.extract_features(source_model, data_loader)
            
            # 4. 聚类分析
            clustering_results = self.perform_clustering(features, n_clusters=Config.NUM_CLASSES)
            
            # 5. 降维可视化
            dim_reduction_results = self.reduce_dimensions(features, method='both')
            
            # 6. 文件级别分析
            file_predictions = self.analyze_file_level_predictions(
                features, clustering_results['cluster_labels'], file_names, target_dataset
            )
            
            # 7. 可视化结果
            self.visualize_results(
                features, clustering_results['cluster_labels'], file_names,
                dim_reduction_results, file_predictions
            )
            
            # 8. 保存结果
            self.save_results(file_predictions, features, clustering_results['cluster_labels'], clustering_results)
            
            # 9. 打印摘要
            self._print_summary(file_predictions, clustering_results)
            
            self.logger.info("无监督迁移学习完成！")
            
            return file_predictions, features, clustering_results, dim_reduction_results
            
        except Exception as e:
            self.logger.error(f"无监督迁移学习失败: {e}")
            raise e
    
    def _print_summary(self, file_predictions, clustering_results):
        """打印结果摘要"""
        print(f"\n=== 无监督迁移学习结果摘要 ===")
        print("-" * 60)
        
        print(f"聚类质量:")
        print(f"  轮廓系数: {clustering_results['silhouette_score']:.4f}")
        
        print(f"\n文件级别预测结果:")
        cluster_counts = {}
        confidences = []
        
        for file_name, pred in file_predictions.items():
            cluster = pred['predicted_cluster']
            confidence = pred['confidence']
            cluster_counts[cluster] = cluster_counts.get(cluster, 0) + 1
            confidences.append(confidence)
            print(f"  {file_name}: 聚类{cluster} (置信度: {confidence:.3f})")
        
        print(f"\n聚类分布:")
        for cluster, count in sorted(cluster_counts.items()):
            percentage = (count / len(file_predictions)) * 100
            print(f"  聚类{cluster}: {count}个文件 ({percentage:.1f}%)")
        
        avg_confidence = np.mean(confidences)
        print(f"\n平均置信度: {avg_confidence:.3f}")
        print(f"置信度范围: {min(confidences):.3f} - {max(confidences):.3f}")

def load_source_model(model_path, device):
    """
    加载源域预训练模型
    
    Args:
        model_path: 模型文件路径
        device: 计算设备
    
    Returns:
        加载的模型
    """
    print(f"加载源域模型: {model_path}")
    
    try:
        # 先加载到CPU，避免CUDA内存问题
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        
        # 根据checkpoint类型创建模型
        if hasattr(checkpoint, 'state_dict'):
            # 如果是完整模型对象
            model = checkpoint
        elif isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            # 如果是包含state_dict的字典
            model = DenseNetCNN(num_classes=Config.NUM_CLASSES)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"模型信息:")
            print(f"  训练轮次: {checkpoint.get('epoch', 'unknown')}")
            print(f"  验证准确率: {checkpoint.get('val_acc', 'unknown')}")
            print(f"  转换方法: {checkpoint.get('transform_method', 'unknown')}")
        elif isinstance(checkpoint, dict):
            # 如果是纯state_dict
            model = DenseNetCNN(num_classes=Config.NUM_CLASSES)
            model.load_state_dict(checkpoint)
        else:
            raise ValueError(f"无法识别的模型文件格式: {type(checkpoint)}")
        
        # 清理GPU内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        model.to(device)
        model.eval()
        print("源域模型加载完成")
        return model
        
    except RuntimeError as e:
        if "CUDA" in str(e):
            print(f"CUDA错误，尝试使用CPU: {e}")
            # 如果CUDA出错，强制使用CPU
            device = torch.device('cpu')
            return load_source_model(model_path, device)  # 递归调用，但现在使用CPU
        else:
            raise e
    except Exception as e:
        print(f"源域模型加载失败: {e}")
        raise e

def predict_target_domain(model, target_loader, device, class_names, target_dataset):
    """对目标域数据进行预测"""
    model.eval()
    
    all_predictions = []
    all_confidences = []
    all_real_file_labels = []
    
    with torch.no_grad():
        for batch_idx, (data, _) in enumerate(target_loader):  # 忽略虚拟标签
            data = data.to(device)
            outputs = model(data)
            
            # 计算softmax概率
            probabilities = F.softmax(outputs, dim=1)
            confidences, predictions = torch.max(probabilities, 1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_confidences.extend(confidences.cpu().numpy())
            
            # 获取对应的真实文件标签
            batch_size = data.size(0)
            start_idx = batch_idx * target_loader.batch_size
            end_idx = start_idx + batch_size
            batch_real_labels = target_dataset.real_file_labels[start_idx:end_idx]
            all_real_file_labels.extend(batch_real_labels)
    
    # 按文件聚合预测结果
    file_predictions = {}
    
    for pred, conf, real_file_label in zip(all_predictions, all_confidences, all_real_file_labels):
        file_name = target_dataset.file_label_to_name[real_file_label]
        
        if file_name not in file_predictions:
            file_predictions[file_name] = {
                'predictions': [],
                'confidences': []
            }
        
        file_predictions[file_name]['predictions'].append(pred)
        file_predictions[file_name]['confidences'].append(conf)
    
    # 计算每个文件的最终预测结果
    final_predictions = {}
    
    for file_name, data in file_predictions.items():
        predictions = np.array(data['predictions'])
        confidences = np.array(data['confidences'])
        
        # 使用投票机制确定最终预测
        unique_preds, counts = np.unique(predictions, return_counts=True)
        final_pred = unique_preds[np.argmax(counts)]
        
        # 计算平均置信度
        avg_confidence = np.mean(confidences)
        
        final_predictions[file_name] = {
            'prediction': final_pred,
            'predicted_class': class_names[final_pred],
            'confidence': avg_confidence,
            'total_windows': len(predictions),
            'class_distribution': dict(zip(unique_preds, counts))
        }
    
    return final_predictions

def save_prediction_results(file_predictions, class_names, save_path):
    """保存预测结果"""
    results = []
    
    for file_name, pred_data in file_predictions.items():
        results.append({
            'file_name': file_name,
            'predicted_class': pred_data['predicted_class'],
            'confidence': pred_data['confidence'],
            'total_windows': pred_data['total_windows']
        })
    
    # 保存为CSV
    df = pd.DataFrame(results)
    df.to_csv(save_path, index=False, encoding='utf-8-sig')
    
    return df

def plot_prediction_results(file_predictions, class_names, save_path):
    """绘制预测结果可视化"""
    # 统计预测类别分布
    pred_counts = {class_name: 0 for class_name in class_names}
    confidences = []
    
    for file_name, pred_data in file_predictions.items():
        pred_counts[pred_data['predicted_class']] += 1
        confidences.append(pred_data['confidence'])
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. 预测类别分布
    axes[0, 0].bar(pred_counts.keys(), pred_counts.values())
    axes[0, 0].set_title('预测类别分布')
    axes[0, 0].set_xlabel('类别')
    axes[0, 0].set_ylabel('文件数量')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # 2. 置信度分布
    axes[0, 1].hist(confidences, bins=20, alpha=0.7)
    axes[0, 1].set_title('预测置信度分布')
    axes[0, 1].set_xlabel('置信度')
    axes[0, 1].set_ylabel('频次')
    
    # 3. 每个文件的预测结果
    file_names = list(file_predictions.keys())
    predictions = [file_predictions[f]['prediction'] for f in file_names]
    confidences_per_file = [file_predictions[f]['confidence'] for f in file_names]
    
    scatter = axes[1, 0].scatter(range(len(file_names)), predictions, 
                                c=confidences_per_file, cmap='viridis', alpha=0.7)
    axes[1, 0].set_title('每个文件的预测结果')
    axes[1, 0].set_xlabel('文件索引')
    axes[1, 0].set_ylabel('预测类别')
    axes[1, 0].set_yticks(range(len(class_names)))
    axes[1, 0].set_yticklabels(class_names)
    plt.colorbar(scatter, ax=axes[1, 0], label='置信度')
    
    # 4. 置信度统计
    class_confidences = {class_name: [] for class_name in class_names}
    for file_name, pred_data in file_predictions.items():
        class_confidences[pred_data['predicted_class']].append(pred_data['confidence'])
    
    box_data = [class_confidences[class_name] for class_name in class_names if class_confidences[class_name]]
    box_labels = [class_name for class_name in class_names if class_confidences[class_name]]
    
    if box_data:
        axes[1, 1].boxplot(box_data, tick_labels=box_labels)
        axes[1, 1].set_title('各类别置信度分布')
        axes[1, 1].set_xlabel('预测类别')
        axes[1, 1].set_ylabel('置信度')
        axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()



def main_unsupervised():
    """无监督迁移学习主函数"""
    print("=== 无监督迁移学习 ===")
    print("这种方法不需要标签，通过特征提取和聚类分析来理解目标域数据")
    print("-" * 60)
    
    try:
        # 配置参数
        source_model_path = Config.get_full_model_path()
        target_data_path = r"数据集\数据集\目标域数据集"
        
        if not os.path.exists(source_model_path):
            print(f"源域模型文件不存在: {source_model_path}")
            return
        
        if not os.path.exists(target_data_path):
            print(f"目标域数据路径不存在: {target_data_path}")
            return
        
        # 创建无监督迁移学习实例
        unsupervised_tl = UnsupervisedTransferLearning(
            source_model_path=source_model_path,
            target_data_path=target_data_path,
            enable_resampling=True,
            target_fs=12000
        )
        
        # 运行无监督迁移学习
        results = unsupervised_tl.run_unsupervised_transfer_learning()
        
        print(f"\n无监督迁移学习完成！")
        print(f"结果保存在: {unsupervised_tl.results_dir}")
        
        return results
        
    except KeyboardInterrupt:
        print("\n用户中断操作")
    except Exception as e:
        print(f"无监督迁移学习失败: {e}")
        import traceback
        traceback.print_exc()

def main_feature_extract():
    """有监督迁移学习主函数（原有功能）"""
    print("=== 目标域故障诊断系统 (特征提取迁移学习) ===")
    print("使用特征提取方法进行迁移学习")
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 数据路径
    target_data_path = "数据集/数据集/目标域数据集"
    
    # 直接使用config中的TRANSFORM_METHOD
    transform_method = Config.TRANSFORM_METHOD
    print(f"使用转换方法: {transform_method}")
    
    # 自动选择对应的模型文件
    # 优先选择full版本，如果不存在则选择普通版本
    full_model_path = Config.get_full_model_path()
    regular_model_path = Config.get_model_path()
    
    if os.path.exists(full_model_path):
        source_model_path = full_model_path
        print(f"使用完整模型: {os.path.basename(full_model_path)}")
    elif os.path.exists(regular_model_path):
        source_model_path = regular_model_path
        print(f"使用模型: {os.path.basename(regular_model_path)}")
    else:
        print(f"错误: 未找到对应的{transform_method}模型文件")
        print(f"期望的模型文件: {full_model_path} 或 {regular_model_path}")
        return
    
    # 询问是否使用重采样
    print("\n请选择数据处理方式:")
    print("1. 原始方式 (根据采样频率调整窗口大小)")
    print("2. 重采样方式 (统一重采样到12kHz)")
    
    choice = input("请输入选择 (1 或 2): ").strip()
    enable_resampling = (choice == '2')
    
    if enable_resampling:
        print("使用重采样方式: 统一重采样到12kHz")
        results_dir = f"transfer_learning_results_{transform_method}_feature_extract_resampled"
    else:
        print("使用原始方式: 根据采样频率调整窗口大小")
        results_dir = f"transfer_learning_results_{transform_method}_feature_extract"
    
    os.makedirs(results_dir, exist_ok=True)
    
    # 类别名称
    class_names = ['正常', '内圈故障', '滚动体故障', '外圈故障']
    
    try:
        # 创建迁移学习训练器
        print(f"\n创建迁移学习训练器...")
        trainer = TransferLearningTrainer(
            source_model_path=source_model_path,
            target_data_path=target_data_path,
            transfer_strategy='feature_extract',  # 使用特征提取策略
            enable_resampling=enable_resampling,
            target_fs=12000
        )
        
        # 询问是否进行训练
        print("\n请选择操作模式:")
        print("1. 进行迁移学习训练")
        print("2. 直接使用预训练模型预测")
        
        mode_choice = input("请输入选择 (1 或 2): ").strip()
        
        if mode_choice == '1':
            # 进行迁移学习训练
            print("\n开始特征提取迁移学习训练...")
            
            # 询问训练轮数
            epochs_input = input("请输入训练轮数 (默认20): ").strip()
            epochs = int(epochs_input) if epochs_input else 20
            
            # 执行训练
            trainer.train(epochs=epochs)
            
            # 训练完成后，使用训练好的模型进行预测
            # 查找最新的训练模型文件
            model_files = [f for f in os.listdir(trainer.results_dir) if f.startswith('best_model_feature_extract_') and f.endswith('.pth')]
            if model_files:
                # 选择最新的模型文件
                latest_model = max(model_files, key=lambda x: os.path.getctime(os.path.join(trainer.results_dir, x)))
                trained_model_path = os.path.join(trainer.results_dir, latest_model)
                print(f"\n使用训练好的模型进行预测: {latest_model}")
                model = load_source_model(trained_model_path, device)
            else:
                print("训练好的模型不存在，使用原始预训练模型")
                model = load_source_model(source_model_path, device)
        else:
            # 直接使用预训练模型，但冻结特征提取层
            print("\n加载预训练模型并冻结特征提取层...")
            model = load_source_model(source_model_path, device)
            
            # 冻结除分类器外的所有层
            for name, param in model.named_parameters():
                if 'fc' not in name:  # 不冻结全连接层
                    param.requires_grad = False
            
            print("已冻结特征提取层，只使用分类器进行预测")
        
        # 创建目标域数据集
        print("创建目标域数据集...")
        target_dataset = TargetDomainDataset(
            data_path=target_data_path,
            transform_method=transform_method,
            enable_resampling=enable_resampling,
            target_fs=12000,
            for_training=False
        )
        
        # 创建数据加载器
        target_loader = DataLoader(target_dataset, batch_size=32, shuffle=False)
        
        # 进行预测
        print("开始预测...")
        file_predictions = predict_target_domain(
            model, target_loader, device, class_names, target_dataset
        )
        
        # 保存预测结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = os.path.join(results_dir, f"{transform_method}_target_domain_predictions_feature_extract_{timestamp}.csv")
        save_prediction_results(file_predictions, class_names, csv_path)
        print(f"预测结果已保存: {csv_path}")
        
        # 绘制可视化结果
        plot_path = os.path.join(results_dir, f"{transform_method}_prediction_visualization_feature_extract_{timestamp}.png")
        plot_prediction_results(file_predictions, class_names, plot_path)
        print(f"可视化结果已保存: {plot_path}")
        
        # 显示预测结果摘要
        print(f"\n=== 特征提取迁移学习预测结果摘要 ===")
        print("-" * 60)
        
        for i, (file_name, pred_data) in enumerate(file_predictions.items()):
            print(f"{file_name}: {pred_data['predicted_class']} "
                  f"(置信度: {pred_data['confidence']:.3f})")
        
        # 统计分析
        pred_counts = {}
        confidences = []
        
        for pred_data in file_predictions.values():
            pred_class = pred_data['predicted_class']
            pred_counts[pred_class] = pred_counts.get(pred_class, 0) + 1
            confidences.append(pred_data['confidence'])
        
        print(f"\n统计分析:")
        print("-" * 30)
        for class_name, count in pred_counts.items():
            percentage = (count / len(file_predictions)) * 100
            print(f"{class_name}: {count}个文件 ({percentage:.1f}%)")
        
        avg_confidence = np.mean(confidences)
        print(f"\n平均置信度: {avg_confidence:.3f}")
        print(f"置信度范围: {min(confidences):.3f} - {max(confidences):.3f}")
        
        print(f"\n特征提取迁移学习完成！结果已保存到 {results_dir} 目录")
        
        # 如果进行了训练，显示训练信息
        if mode_choice == '1':
            print(f"\n训练相关文件保存在: {trainer.results_dir}")
            print("包含:")
            print("- 训练历史记录")
            print("- 混淆矩阵")
            print("- 训练好的模型")
            print("- 训练日志")
        
        print(f"\n所有结果文件保存在: {results_dir}")
        
    except KeyboardInterrupt:
        print("\n用户中断操作")
    except Exception as e:
        print(f"特征提取迁移学习失败: {e}")
        import traceback
        traceback.print_exc()

def main():
    """主函数 - 添加无监督选项"""
    print("=== 轴承故障诊断 - 迁移学习 ===")
    print("1. 有监督迁移学习（特征提取）")
    print("2. 无监督迁移学习（聚类分析）")
    print("3. 退出")
    
    while True:
        choice = input("\n请选择模式 (1-3): ").strip()
        
        if choice == '1':
            # 原有的有监督迁移学习
            main_feature_extract()
            break
        elif choice == '2':
            # 新的无监督迁移学习
            main_unsupervised()
            break
        elif choice == '3':
            print("退出程序")
            break
        else:
            print("无效选择，请输入 1-3")

if __name__ == "__main__":
    main()