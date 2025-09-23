import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from torchsummary import summary
import os
import logging
import sys
from datetime import datetime
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from tqdm import tqdm
from bearing_data_loader import create_bearing_dataloaders
from config import Config

# 设置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# ==================== 日志设置 ====================
def setup_logging():
    """设置日志记录"""
    # 确保log目录存在
    os.makedirs(Config.LOG_DIR, exist_ok=True)
    
    # 创建带时间戳的日志文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f'{Config.LOG_DIR}/training_{timestamp}.log'
    
    # 清除之前的日志配置
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)  # 同时输出到控制台
        ]
    )
    
    return log_filename

def log_and_print(message, level='info'):
    """同时记录日志和打印到控制台"""
    if level == 'info':
        logging.info(message)
    elif level == 'warning':
        logging.warning(message)
    elif level == 'error':
        logging.error(message)
    else:
        logging.info(message)

# ==================== 模型定义 ====================
class DenseBlock(nn.Module):
    """Dense Block层 - 实现密集连接"""
    def __init__(self, in_channels, growth_rate=None):
        super(DenseBlock, self).__init__()
        if growth_rate is None:
            growth_rate = Config.GROWTH_RATE
        self.growth_rate = growth_rate
        
        # 1x1卷积用于降维
        self.conv1x1 = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, growth_rate, kernel_size=1, padding=0, bias=False)
        )
        
        # 3x3卷积用于特征提取
        self.conv3x3 = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, growth_rate, kernel_size=3, padding=1, bias=False)
        )
    
    def forward(self, x):
        # 1x1卷积分支
        out1 = self.conv1x1(x)
        
        # 3x3卷积分支
        out2 = self.conv3x3(x)
        
        # 密集连接：将输入和两个卷积输出连接
        out = torch.cat([x, out1, out2], dim=1)
        return out

class DenseNetCNN(nn.Module):
    """
    根据表格架构创建CNN模型 - 适用于轴承故障诊断
    """
    
    def __init__(self, num_classes=None):
        super(DenseNetCNN, self).__init__()
        if num_classes is None:
            num_classes = Config.NUM_CLASSES
        
        # Conv1: 5×5×16, dilation_rate=1
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, padding=2, dilation=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        
        # Pooling1: 2×2 max pool, stride=2
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Dense Block 1: growth_rate=6
        self.dense1 = DenseBlock(16, Config.GROWTH_RATE)
        # 输出通道数: 16 + 6 + 6 = 28
        
        # Conv2: 3×3×32, dilation_rate=2
        self.conv2 = nn.Conv2d(28, 32, kernel_size=3, padding=2, dilation=2, bias=False)
        self.bn2 = nn.BatchNorm2d(32)
        
        # Pooling2: 2×2 max pool, stride=2
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Dense Block 2: growth_rate=6
        self.dense2 = DenseBlock(32, Config.GROWTH_RATE)
        # 输出通道数: 32 + 6 + 6 = 44
        
        # Conv3: 3×3×64, dilation_rate=3
        self.conv3 = nn.Conv2d(44, 64, kernel_size=3, padding=3, dilation=3, bias=False)
        self.bn3 = nn.BatchNorm2d(64)
        
        # Pooling3: 2×2 max pool, stride=2
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Dense Block 3: growth_rate=6
        self.dense3 = DenseBlock(64, Config.GROWTH_RATE)
        # 输出通道数: 64 + 6 + 6 = 76
        
        # Global Average Pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 全连接层
        self.fc1 = nn.Linear(76, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)
        
        # 初始化权重
        self._initialize_weights()
    
    def forward(self, x):
        # Conv1 + BN + ReLU + Pool1
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        
        # Dense Block 1
        x = self.dense1(x)
        
        # Conv2 + BN + ReLU + Pool2
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        
        # Dense Block 2
        x = self.dense2(x)
        
        # Conv3 + BN + ReLU + Pool3
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        
        # Dense Block 3
        x = self.dense3(x)
        
        # Global Average Pooling
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)  # 展平
        
        # 全连接层
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
    
    def _initialize_weights(self):
        """初始化模型权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

def train_model(model, train_loader, val_loader=None, epochs=10, learning_rate=0.001, device='cpu'):
    """训练模型并记录日志，使用tqdm显示进度"""
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    log_and_print(f"开始训练，共 {epochs} 个epoch")
    log_and_print(f"训练批次数: {len(train_loader)}")
    if val_loader:
        log_and_print(f"验证批次数: {len(val_loader)}")
    
    # 使用tqdm显示epoch进度
    epoch_pbar = tqdm(range(epochs), desc="训练进度", unit="epoch")
    
    for epoch in epoch_pbar:
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        # 使用tqdm显示训练批次进度
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} 训练", 
                         leave=False, unit="batch")
        
        for batch_idx, (data, target) in enumerate(train_pbar):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = output.max(1)
            train_total += target.size(0)
            train_correct += predicted.eq(target).sum().item()
            
            # 更新训练进度条显示
            current_acc = 100. * train_correct / train_total
            train_pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{current_acc:.2f}%'
            })
        
        # 计算训练指标
        avg_train_loss = train_loss / len(train_loader)
        train_accuracy = 100. * train_correct / train_total
        
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(train_accuracy)
        
        # 验证阶段
        val_info = ""
        if val_loader:
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            # 使用tqdm显示验证批次进度
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} 验证", 
                           leave=False, unit="batch")
            
            with torch.no_grad():
                for data, target in val_pbar:
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    loss = criterion(output, target)
                    
                    val_loss += loss.item()
                    _, predicted = output.max(1)
                    val_total += target.size(0)
                    val_correct += predicted.eq(target).sum().item()
                    
                    # 更新验证进度条显示
                    current_val_acc = 100. * val_correct / val_total
                    val_pbar.set_postfix({
                        'Loss': f'{loss.item():.4f}',
                        'Acc': f'{current_val_acc:.2f}%'
                    })
            
            avg_val_loss = val_loss / len(val_loader)
            val_accuracy = 100. * val_correct / val_total
            
            history['val_loss'].append(avg_val_loss)
            history['val_acc'].append(val_accuracy)
            
            val_info = f", 验证损失: {avg_val_loss:.4f}, 验证准确率: {val_accuracy:.2f}%"
        
        # 更新epoch进度条显示
        epoch_info = f"训练损失: {avg_train_loss:.4f}, 训练准确率: {train_accuracy:.2f}%{val_info}"
        epoch_pbar.set_postfix_str(epoch_info)
        
        # 记录到日志
        log_and_print(f'Epoch {epoch+1}/{epochs} - {epoch_info}')
    
    epoch_pbar.close()
    log_and_print("训练完成！")
    return history

def print_classification_results(eval_results, class_names, dataset_name=""):
    """打印分类结果和指标"""
    print(f"\n{'='*50}")
    print(f"{dataset_name}分类结果")
    print(f"{'='*50}")
    
    print(f"总体准确率: {eval_results['accuracy']:.2f}%")
    print(f"总体损失: {eval_results['loss']:.4f}")
    
    # 检查验证集中实际存在的类别
    unique_targets = sorted(list(set(eval_results['targets'])))
    unique_predictions = sorted(list(set(eval_results['predictions'])))
    all_unique_labels = sorted(list(set(unique_targets + unique_predictions)))
    
    print(f"\n数据集中存在的类别: {all_unique_labels}")
    print(f"对应类别名称: {[class_names[i] for i in all_unique_labels]}")
    
    # 打印sklearn分类报告 - 只包含实际存在的类别
    try:
        actual_class_names = [class_names[i] for i in all_unique_labels]
        print(f"\n详细分类报告:")
        print(classification_report(
            eval_results['targets'], 
            eval_results['predictions'], 
            labels=all_unique_labels,  # 指定实际存在的标签
            target_names=actual_class_names,  # 对应的类别名称
            digits=4
        ))
    except Exception as e:
        print(f"分类报告生成失败: {e}")
        # 手动计算每个类别的指标
        print(f"\n手动计算的分类指标:")
        for label in all_unique_labels:
            class_name = class_names[label]
            true_count = sum(1 for t in eval_results['targets'] if t == label)
            pred_count = sum(1 for p in eval_results['predictions'] if p == label)
            correct_count = sum(1 for t, p in zip(eval_results['targets'], eval_results['predictions']) if t == label and p == label)
            
            if true_count > 0:
                precision = correct_count / pred_count if pred_count > 0 else 0
                recall = correct_count / true_count
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                
                print(f"  {class_name}: 精确率={precision:.4f}, 召回率={recall:.4f}, F1={f1:.4f}, 支持数={true_count}")
    
    # 打印混淆矩阵
    print(f"\n混淆矩阵:")
    cm = eval_results['confusion_matrix']
    
    # 只显示实际存在的类别
    print(f"{'实际/预测':<12}", end="")
    for label in all_unique_labels:
        print(f"{class_names[label]:<12}", end="")
    print()
    
    for i, true_label in enumerate(all_unique_labels):
        print(f"{class_names[true_label]:<12}", end="")
        for j, pred_label in enumerate(all_unique_labels):
            # 找到对应的混淆矩阵位置
            cm_value = 0
            if true_label < cm.shape[0] and pred_label < cm.shape[1]:
                cm_value = cm[true_label][pred_label]
            print(f"{cm_value:<12}", end="")
        print()
    
    # 分析每个类别的分类情况
    print(f"\n各类别分类详情:")
    for label in all_unique_labels:
        class_name = class_names[label]
        true_count = sum(1 for t in eval_results['targets'] if t == label)
        pred_count = sum(1 for p in eval_results['predictions'] if p == label)
        correct_count = sum(1 for t, p in zip(eval_results['targets'], eval_results['predictions']) if t == label and p == label)
        
        if true_count > 0:
            class_accuracy = 100. * correct_count / true_count
            print(f"  {class_name}: 实际{true_count}个, 预测{pred_count}个, 正确{correct_count}个, 准确率{class_accuracy:.2f}%")
            
            # 显示该类别被误分类到哪些类别
            if correct_count < true_count:
                print(f"    误分类情况:", end="")
                for other_label in all_unique_labels:
                    if label != other_label:
                        misclass_count = sum(1 for t, p in zip(eval_results['targets'], eval_results['predictions']) 
                                           if t == label and p == other_label)
                        if misclass_count > 0:
                            print(f" {misclass_count}个被分为{class_names[other_label]}", end="")
                print()

def evaluate_model(model, data_loader, device, class_names):
    """评估模型性能，使用tqdm显示进度"""
    model.eval()
    all_predictions = []
    all_targets = []
    total_loss = 0.0
    correct = 0
    total = 0
    
    criterion = nn.CrossEntropyLoss()
    
    # 使用tqdm显示评估进度
    eval_pbar = tqdm(data_loader, desc="模型评估", unit="batch")
    
    with torch.no_grad():
        for data, target in eval_pbar:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            
            total_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            # 收集预测结果和真实标签
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            
            # 更新进度条显示
            current_acc = 100. * correct / total
            eval_pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{current_acc:.2f}%'
            })
    
    eval_pbar.close()
    
    # 计算最终指标
    avg_loss = total_loss / len(data_loader)
    accuracy = 100. * correct / total
    
    # 计算混淆矩阵
    cm = confusion_matrix(all_targets, all_predictions)
    
    # 获取实际存在的类别
    unique_labels = sorted(list(set(all_targets)))
    
    log_and_print(f"评估完成 - 损失: {avg_loss:.4f}, 准确率: {accuracy:.2f}%")
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'predictions': all_predictions,
        'targets': all_targets,
        'confusion_matrix': cm,
        'unique_labels': unique_labels
    }

def model_summary(model, input_size=(1, 64, 64)):
    """显示模型摘要"""
    print("=" * 60)
    print("轴承故障诊断CNN模型架构摘要")
    print("=" * 60)
    
    try:
        # 创建模型的副本用于summary，避免影响原模型
        import copy
        model_copy = copy.deepcopy(model)
        summary(model_copy, input_size)
        # 删除副本，释放内存
        del model_copy
    except:
        print("无法使用torchsummary，显示基本信息：")
        print(model)
        
        # 计算参数数量
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n总参数数量: {total_params:,}")
        print(f"可训练参数数量: {trainable_params:,}")

def plot_training_history(history):
    """绘制训练历史"""
    if not history:
        print("没有训练历史数据")
        return
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    plt.figure(figsize=(15, 5))
    
    # 损失曲线
    plt.subplot(1, 3, 1)
    plt.plot(epochs, history['train_loss'], 'b-', label='训练损失', linewidth=2)
    plt.plot(epochs, history['val_loss'], 'r-', label='验证损失', linewidth=2)
    plt.title('损失曲线', fontsize=14, fontweight='bold')
    plt.xlabel('轮次')
    plt.ylabel('损失')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 准确率曲线
    plt.subplot(1, 3, 2)
    plt.plot(epochs, history['train_acc'], 'b-', label='训练准确率', linewidth=2)
    plt.plot(epochs, history['val_acc'], 'r-', label='验证准确率', linewidth=2)
    plt.title('准确率曲线', fontsize=14, fontweight='bold')
    plt.xlabel('轮次')
    plt.ylabel('准确率')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 学习率曲线（如果有的话）
    plt.subplot(1, 3, 3)
    if 'learning_rate' in history and history['learning_rate']:
        plt.plot(epochs, history['learning_rate'], 'g-', label='学习率', linewidth=2)
        plt.title('学习率曲线', fontsize=14, fontweight='bold')
        plt.xlabel('轮次')
        plt.ylabel('学习率')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.yscale('log')  # 使用对数刻度
    else:
        # 如果没有学习率历史，显示训练时间
        plt.text(0.5, 0.5, '训练完成', ha='center', va='center', 
                transform=plt.gca().transAxes, fontsize=16, fontweight='bold')
        plt.title('训练状态', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    # 保存图片 - 使用类方法获取路径字符串
    os.makedirs(Config.IMG_DIR, exist_ok=True)
    training_history_path = Config.get_training_history_path()  # 修改这里
    plt.savefig(training_history_path, dpi=300, bbox_inches='tight')
    log_and_print(f"训练历史图表已保存到: {training_history_path}")
    
    plt.show()

def plot_confusion_matrix(cm, class_names, title="混淆矩阵", save_path=None):
    """绘制混淆矩阵 - 优化12分类显示"""
    # 计算百分比
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    # 为12分类调整图像大小
    fig_width = max(16, len(class_names) * 1.2)
    fig_height = max(12, len(class_names) * 0.8)
    
    # 创建子图
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(fig_width, fig_height))
    
    # 绘制数量混淆矩阵
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names, ax=ax1,
                cbar_kws={'shrink': 0.8})
    ax1.set_title('混淆矩阵 (数量)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('预测类别', fontsize=10)
    ax1.set_ylabel('真实类别', fontsize=10)
    ax1.tick_params(axis='x', rotation=45, labelsize=8)
    ax1.tick_params(axis='y', rotation=0, labelsize=8)
    
    # 绘制百分比混淆矩阵
    sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax2,
                cbar_kws={'shrink': 0.8})
    ax2.set_title('混淆矩阵 (百分比)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('预测类别', fontsize=10)
    ax2.set_ylabel('真实类别', fontsize=10)
    ax2.tick_params(axis='x', rotation=45, labelsize=8)
    ax2.tick_params(axis='y', rotation=0, labelsize=8)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        log_and_print(f"混淆矩阵已保存到: {save_path}")
    
    plt.show()

# 使用示例
if __name__ == "__main__":
    # 设置日志记录
    log_file = setup_logging()
    log_and_print("开始轴承故障诊断CNN模型训练 - 4分类")
    log_and_print(f"配置参数:")
    log_and_print(f"  数据路径: {Config.DATA_PATH}")
    log_and_print(f"  窗口大小: {Config.WINDOW_SIZE}")
    log_and_print(f"  重叠比例: {Config.OVERLAP_RATIO}")
    log_and_print(f"  批次大小: {Config.BATCH_SIZE}")
    log_and_print(f"  训练轮数: {Config.EPOCHS}")
    log_and_print(f"  学习率: {Config.LEARNING_RATE}")
    log_and_print(f"  类别数量: {Config.NUM_CLASSES}")
    log_and_print(f"  类别名称: {Config.CLASS_NAMES}")
    
    try:
        # 检查CUDA可用性
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        log_and_print(f"使用设备: {device}")

        log_and_print("创建轴承故障诊断CNN模型...")
        model = DenseNetCNN(num_classes=Config.NUM_CLASSES)
        
        # 显示模型结构
        model_summary(model, Config.INPUT_SIZE)

        log_and_print("加载轴承振动信号数据...")
        train_loader, val_loader, dataset = create_bearing_dataloaders(
            data_path=Config.DATA_PATH,
            window_size=Config.WINDOW_SIZE,
            overlap_ratio=Config.OVERLAP_RATIO,
            batch_size=Config.BATCH_SIZE,
            train_ratio=Config.TRAIN_RATIO,
            transform_to_2d=True,
            transform_method=Config.TRANSFORM_METHOD  # 新增这一行
        )
        
        log_and_print(f"训练批次数: {len(train_loader)}")
        log_and_print(f"验证批次数: {len(val_loader)}")
        class_names = dataset.get_class_names()
        log_and_print(f"类别名称: {class_names}")

        log_and_print("开始训练轴承故障诊断模型...")
        history = train_model(
            model, train_loader, val_loader, 
            epochs=Config.EPOCHS, 
            learning_rate=Config.LEARNING_RATE, 
            device=device
        )
        
        # 绘制训练历史
        plot_training_history(history)
        
        # 在验证集上评估模型
        if val_loader:
            log_and_print("开始验证集详细分类评估")
            
            val_results = evaluate_model(model, val_loader, device, class_names)
            print_classification_results(val_results, class_names, "验证集")
            
            # 绘制混淆矩阵
            # 绘制混淆矩阵
            if val_results and 'confusion_matrix' in val_results:
                os.makedirs(Config.IMG_DIR, exist_ok=True)
                confusion_matrix_path = Config.get_confusion_matrix_path()  # 修改这里
                plot_confusion_matrix(val_results['confusion_matrix'], class_names, 
                                    title="验证集混淆矩阵", save_path=confusion_matrix_path)
            
            # 保存模型
            os.makedirs(Config.MODEL_DIR, exist_ok=True)
            
            # 获取动态生成的文件路径
            model_state_path = Config.get_model_state_path()  # 修改这里
            model_full_path = Config.get_full_model_path()    # 修复：正确的方法名
            
            # 只保存模型参数，避免pickle问题
            torch.save(model.state_dict(), model_state_path)
            log_and_print(f"模型参数已保存为 '{model_state_path}'")
            
            # 尝试保存完整模型，如果失败则跳过
            try:
                # 清理模型，移除可能的钩子函数
                model.eval()
                torch.save(model, model_full_path)
                log_and_print(f"完整模型已保存为 '{model_full_path}'")
            except Exception as e:
                log_and_print(f"保存完整模型失败: {e}")
                log_and_print("仅保存了模型参数，可以通过加载参数到新模型实例来使用")
        
        log_and_print("轴承故障诊断CNN模型训练完成！")
        log_and_print(f"完整日志已保存到: {log_file}")
        
    except Exception as e:
        log_and_print(f"训练过程中发生错误: {e}", level='error')
        import traceback
        log_and_print(f"错误详情: {traceback.format_exc()}", level='error')
        raise