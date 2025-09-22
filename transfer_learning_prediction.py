import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 导入模型和配置
from cnn_model_pytorch import DenseNetCNN, Config
from bearing_data_loader import BearingDataset

class TargetDomainDataset(Dataset):
    """目标域数据集类"""
    def __init__(self, data_path, window_size=4096, step_size=2048, transform_to_2d=True):
        """
        目标域数据集
        
        Args:
            data_path: 目标域数据路径
            window_size: 窗口大小
            step_size: 步长
            transform_to_2d: 是否转换为2D图像
        """
        self.data_path = data_path
        self.window_size = window_size
        self.step_size = step_size
        self.transform_to_2d = transform_to_2d
        
        self.samples = []
        self.file_labels = []  # 记录每个样本来自哪个文件
        self.file_names = []   # 文件名列表
        
        # 加载数据
        self._load_data()
        
        # 标准化数据（使用源域的统计量）
        self._normalize_data()
    
    def _load_data(self):
        """加载目标域数据"""
        print("开始加载目标域数据...")
        
        # 获取所有.mat文件
        mat_files = [f for f in os.listdir(self.data_path) if f.endswith('.mat')]
        mat_files.sort()  # 按字母顺序排序
        
        for file_idx, file_name in enumerate(mat_files):
            file_path = os.path.join(self.data_path, file_name)
            
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
                        elif mat_data[key].ndim == 1:
                            vibration_data = mat_data[key]
                            break
                
                if vibration_data is not None:
                    # 滑窗采样
                    windows = self._sliding_window_sampling(vibration_data)
                    
                    # 添加到数据集
                    for window in windows:
                        self.samples.append(window)
                        self.file_labels.append(file_idx)
                    
                    self.file_names.append(file_name)
                    print(f"  {file_name}: 生成 {len(windows)} 个样本")
                    
            except Exception as e:
                print(f"  错误: {file_name} - {e}")
        
        print(f"目标域数据加载完成: 总共 {len(self.samples)} 个样本，来自 {len(self.file_names)} 个文件")
    
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
        """标准化数据 - 使用源域的统计量"""
        if len(self.samples) == 0:
            print("警告: 没有样本数据可以标准化")
            return
            
        print("正在标准化目标域数据...")
        
        # 这里可以使用源域的均值和标准差，或者计算目标域的统计量
        # 为了简化，我们计算目标域的统计量
        all_data = np.array(self.samples)
        self.mean = np.mean(all_data)
        self.std = np.std(all_data)
        
        # 避免除零错误
        if self.std == 0:
            self.std = 1.0
        
        # 标准化每个样本
        for i in range(len(self.samples)):
            self.samples[i] = (self.samples[i] - self.mean) / self.std
        
        print(f"目标域数据标准化完成: 均值={self.mean:.4f}, 标准差={self.std:.4f}")
    
    def _signal_to_2d(self, signal):
        """将1D信号转换为2D图像"""
        # 确保信号长度为4096 (64*64)
        if len(signal) != self.window_size:
            if len(signal) > self.window_size:
                signal = signal[:self.window_size]
            else:
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
        file_label = self.file_labels[idx]
        
        if self.transform_to_2d:
            # 转换为2D图像格式
            sample = self._signal_to_2d(sample)
            # 添加通道维度 (1, 64, 64)
            sample = np.expand_dims(sample, axis=0)
        
        # 转换为PyTorch张量
        sample = torch.FloatTensor(sample)
        
        return sample, file_label

def load_trained_model(model_path, device):
    """加载训练好的模型"""
    print(f"加载训练好的模型: {model_path}")
    
    # 创建模型
    model = DenseNetCNN(num_classes=Config.NUM_CLASSES)
    
    # 加载模型权重
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    print("模型加载完成")
    return model

def predict_target_domain(model, target_loader, device, class_names, file_names):
    """预测目标域数据"""
    print("开始预测目标域数据...")
    
    model.eval()
    all_predictions = []
    all_file_labels = []
    all_probabilities = []
    
    with torch.no_grad():
        for data, file_labels in target_loader:
            data = data.to(device)
            
            # 前向传播
            outputs = model(data)
            probabilities = F.softmax(outputs, dim=1)
            predictions = torch.argmax(outputs, dim=1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_file_labels.extend(file_labels.numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    # 整理预测结果
    results = {
        'predictions': np.array(all_predictions),
        'file_labels': np.array(all_file_labels),
        'probabilities': np.array(all_probabilities)
    }
    
    # 按文件汇总预测结果
    file_predictions = {}
    for file_idx, file_name in enumerate(file_names):
        # 找到属于该文件的所有样本
        file_mask = results['file_labels'] == file_idx
        file_preds = results['predictions'][file_mask]
        file_probs = results['probabilities'][file_mask]
        
        if len(file_preds) > 0:
            # 使用投票机制确定文件的最终预测
            unique_preds, counts = np.unique(file_preds, return_counts=True)
            final_pred = unique_preds[np.argmax(counts)]
            
            # 计算平均概率
            avg_probs = np.mean(file_probs, axis=0)
            confidence = np.max(avg_probs)
            
            file_predictions[file_name] = {
                'predicted_class': final_pred,
                'predicted_class_name': class_names[final_pred],
                'confidence': confidence,
                'sample_count': len(file_preds),
                'class_distribution': dict(zip(unique_preds, counts)),
                'avg_probabilities': avg_probs
            }
    
    return file_predictions, results

def save_prediction_results(file_predictions, class_names, save_path):
    """保存预测结果到CSV文件"""
    results_data = []
    
    for file_name, pred_info in file_predictions.items():
        row = {
            '文件名': file_name,
            '预测类别': pred_info['predicted_class'],
            '预测类别名称': pred_info['predicted_class_name'],
            '置信度': f"{pred_info['confidence']:.4f}",
            '样本数量': pred_info['sample_count']
        }
        
        # 添加各类别的概率
        for i, class_name in enumerate(class_names):
            row[f'{class_name}_概率'] = f"{pred_info['avg_probabilities'][i]:.4f}"
        
        results_data.append(row)
    
    # 创建DataFrame并保存
    df = pd.DataFrame(results_data)
    df.to_csv(save_path, index=False, encoding='utf-8-sig')
    print(f"预测结果已保存到: {save_path}")
    
    return df

def plot_prediction_results(file_predictions, class_names, save_path):
    """绘制预测结果可视化图表"""
    # 准备数据
    file_names = list(file_predictions.keys())
    predicted_classes = [file_predictions[f]['predicted_class'] for f in file_names]
    confidences = [file_predictions[f]['confidence'] for f in file_names]
    
    # 创建图表
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 11))
    
    # 1. 预测类别分布
    class_counts = np.bincount(predicted_classes, minlength=len(class_names))
    bars1 = ax1.bar(range(len(class_names)), class_counts, color='skyblue', alpha=0.7)
    ax1.set_xlabel('故障类别', fontsize=11)
    ax1.set_ylabel('文件数量', fontsize=11)
    ax1.set_title('目标域数据预测类别分布', fontsize=14, fontweight='bold')
    ax1.set_xticks(range(len(class_names)))
    ax1.set_xticklabels(class_names, rotation=45, ha='right')
    ax1.grid(True, alpha=0.3)
    
    # 在柱状图上添加数值标签
    for bar, count in zip(bars1, class_counts):
        if count > 0:
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    str(count), ha='center', va='bottom', fontweight='bold')
    
    # 2. 各文件的预测置信度
    colors = plt.cm.viridis(np.linspace(0, 1, len(file_names)))
    bars2 = ax2.bar(range(len(file_names)), confidences, color=colors, alpha=0.7)
    ax2.set_xlabel('文件', fontsize=11)
    ax2.set_ylabel('预测置信度', fontsize=11)
    ax2.set_title('各文件预测置信度', fontsize=14, fontweight='bold')
    ax2.set_xticks(range(len(file_names)))
    ax2.set_xticklabels(file_names, rotation=45, ha='right')
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3)
    
    # 添加置信度阈值线
    ax2.axhline(y=0.8, color='red', linestyle='--', alpha=0.7, label='高置信度阈值(0.8)')
    ax2.axhline(y=0.6, color='orange', linestyle='--', alpha=0.7, label='中等置信度阈值(0.6)')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"预测结果可视化图表已保存到: {save_path}")

def main():
    """主函数"""
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 路径配置
    target_data_path = "数据集/数据集/目标域数据集"
    model_path = "model/bearing_fault_cnn_11class.pth"  # 使用11分类模型
    results_dir = "transfer_learning_results"
    
    # 创建结果目录
    os.makedirs(results_dir, exist_ok=True)
    
    # 获取类别名称
    class_names = [
        '正常', '内圈故障_0007', '内圈故障_0014', '内圈故障_0021', '内圈故障_0028',
        '滚动体故障_0007', '滚动体故障_0014', '滚动体故障_0021', '滚动体故障_0028',
        '外圈故障_0007', '外圈故障_0021'
    ]
    
    # 加载训练好的模型
    model = load_trained_model(model_path, device)
    
    # 创建目标域数据集
    target_dataset = TargetDomainDataset(
        data_path=target_data_path,
        window_size=Config.WINDOW_SIZE,
        step_size=Config.WINDOW_SIZE // 2,  # 50% 重叠
        transform_to_2d=True
    )
    
    # 创建数据加载器
    target_loader = DataLoader(target_dataset, batch_size=32, shuffle=False)
    
    # 进行预测
    file_predictions, detailed_results = predict_target_domain(
        model, target_loader, device, class_names, target_dataset.file_names
    )
    
    # 打印预测结果
    print("\n=== 目标域数据预测结果 ===")
    for file_name, pred_info in file_predictions.items():
        print(f"{file_name}: {pred_info['predicted_class_name']} "
              f"(置信度: {pred_info['confidence']:.4f}, "
              f"样本数: {pred_info['sample_count']})")
    
    # 保存预测结果
    csv_path = os.path.join(results_dir, "target_domain_predictions.csv")
    results_df = save_prediction_results(file_predictions, class_names, csv_path)
    
    # 绘制预测结果可视化
    plot_path = os.path.join(results_dir, "prediction_visualization.png")
    plot_prediction_results(file_predictions, class_names, plot_path)
    
    # 统计分析
    print("\n=== 预测统计分析 ===")
    predicted_classes = [pred['predicted_class'] for pred in file_predictions.values()]
    class_counts = np.bincount(predicted_classes, minlength=len(class_names))
    
    for i, (class_name, count) in enumerate(zip(class_names, class_counts)):
        if count > 0:
            percentage = count / len(file_predictions) * 100
            print(f"{class_name}: {count} 个文件 ({percentage:.1f}%)")
    
    # 置信度分析
    confidences = [pred['confidence'] for pred in file_predictions.values()]
    print(f"\n置信度统计:")
    print(f"  平均置信度: {np.mean(confidences):.4f}")
    print(f"  最高置信度: {np.max(confidences):.4f}")
    print(f"  最低置信度: {np.min(confidences):.4f}")
    print(f"  高置信度(>0.8)文件数: {sum(1 for c in confidences if c > 0.8)}")
    print(f"  中等置信度(0.6-0.8)文件数: {sum(1 for c in confidences if 0.6 <= c <= 0.8)}")
    print(f"  低置信度(<0.6)文件数: {sum(1 for c in confidences if c < 0.6)}")

if __name__ == "__main__":
    main()