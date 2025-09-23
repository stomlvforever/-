"""
CNN架构对比脚本
比较DenseNetCNN、StandardCNN和DilatedCNN三种架构的性能
"""

import torch
import torch.nn as nn
import time
import numpy as np
from config import Config
from bearing_data_loader import create_bearing_dataloaders
from cnn_model_pytorch import DenseNetCNN, StandardCNN, DilatedCNN, train_model, evaluate_model
import matplotlib.pyplot as plt
import seaborn as sns

def count_parameters(model):
    """计算模型参数数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def compare_architectures():
    """对比三种CNN架构"""
    print("=" * 60)
    print("CNN架构对比实验")
    print("=" * 60)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载数据
    print("\n加载数据...")
    train_loader, val_loader, dataset = create_bearing_dataloaders(
        data_path=Config.DATA_PATH,
        window_size=Config.WINDOW_SIZE,
        overlap_ratio=Config.OVERLAP_RATIO,
        batch_size=Config.BATCH_SIZE,
        train_ratio=Config.TRAIN_RATIO,
        transform_to_2d=True,
        transform_method=Config.TRANSFORM_METHOD
    )
    
    class_names = dataset.get_class_names()
    print(f"类别数量: {len(class_names)}")
    print(f"训练批次: {len(train_loader)}, 验证批次: {len(val_loader)}")
    
    # 定义模型
    models = {
        'DenseNetCNN': DenseNetCNN(num_classes=Config.NUM_CLASSES),
        'StandardCNN': StandardCNN(num_classes=Config.NUM_CLASSES),
        'DilatedCNN': DilatedCNN(num_classes=Config.NUM_CLASSES)
    }
    
    results = {}
    
    # 对每个模型进行训练和评估
    for model_name, model in models.items():
        print(f"\n{'='*20} {model_name} {'='*20}")
        
        # 计算参数数量
        param_count = count_parameters(model)
        print(f"参数数量: {param_count:,}")
        
        # 训练模型
        start_time = time.time()
        history = train_model(
            model, train_loader, val_loader,
            epochs=Config.EPOCHS,
            learning_rate=Config.LEARNING_RATE,
            device=device
        )
        training_time = time.time() - start_time
        
        # 评估模型
        val_results = evaluate_model(model, val_loader, device, class_names)
        
        # 保存结果
        results[model_name] = {
            'history': history,
            'val_results': val_results,
            'param_count': param_count,
            'training_time': training_time,
            'final_val_acc': history['val_acc'][-1] if history['val_acc'] else 0,
            'final_val_loss': history['val_loss'][-1] if history['val_loss'] else float('inf')
        }
        
        print(f"训练时间: {training_time:.2f}秒")
        print(f"最终验证准确率: {results[model_name]['final_val_acc']:.4f}")
        print(f"最终验证损失: {results[model_name]['final_val_loss']:.4f}")
    
    # 绘制对比结果
    plot_comparison_results(results)
    
    # 打印总结
    print_summary(results)
    
    return results

def plot_comparison_results(results):
    """绘制对比结果"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. 训练损失对比
    ax = axes[0, 0]
    for model_name, result in results.items():
        epochs = range(1, len(result['history']['train_loss']) + 1)
        ax.plot(epochs, result['history']['train_loss'], label=model_name, linewidth=2)
    ax.set_title('训练损失对比', fontsize=14, fontweight='bold')
    ax.set_xlabel('轮次')
    ax.set_ylabel('损失')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. 验证损失对比
    ax = axes[0, 1]
    for model_name, result in results.items():
        epochs = range(1, len(result['history']['val_loss']) + 1)
        ax.plot(epochs, result['history']['val_loss'], label=model_name, linewidth=2)
    ax.set_title('验证损失对比', fontsize=14, fontweight='bold')
    ax.set_xlabel('轮次')
    ax.set_ylabel('损失')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. 训练准确率对比
    ax = axes[0, 2]
    for model_name, result in results.items():
        epochs = range(1, len(result['history']['train_acc']) + 1)
        ax.plot(epochs, result['history']['train_acc'], label=model_name, linewidth=2)
    ax.set_title('训练准确率对比', fontsize=14, fontweight='bold')
    ax.set_xlabel('轮次')
    ax.set_ylabel('准确率')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. 验证准确率对比
    ax = axes[1, 0]
    for model_name, result in results.items():
        epochs = range(1, len(result['history']['val_acc']) + 1)
        ax.plot(epochs, result['history']['val_acc'], label=model_name, linewidth=2)
    ax.set_title('验证准确率对比', fontsize=14, fontweight='bold')
    ax.set_xlabel('轮次')
    ax.set_ylabel('准确率')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 5. 参数数量对比
    ax = axes[1, 1]
    model_names = list(results.keys())
    param_counts = [results[name]['param_count'] for name in model_names]
    bars = ax.bar(model_names, param_counts, color=['skyblue', 'lightgreen', 'lightcoral'])
    ax.set_title('参数数量对比', fontsize=14, fontweight='bold')
    ax.set_ylabel('参数数量')
    ax.tick_params(axis='x', rotation=45)
    
    # 在柱状图上添加数值标签
    for bar, count in zip(bars, param_counts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{count:,}', ha='center', va='bottom')
    
    # 6. 训练时间对比
    ax = axes[1, 2]
    training_times = [results[name]['training_time'] for name in model_names]
    bars = ax.bar(model_names, training_times, color=['skyblue', 'lightgreen', 'lightcoral'])
    ax.set_title('训练时间对比', fontsize=14, fontweight='bold')
    ax.set_ylabel('训练时间 (秒)')
    ax.tick_params(axis='x', rotation=45)
    
    # 在柱状图上添加数值标签
    for bar, time_val in zip(bars, training_times):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{time_val:.1f}s', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('cnn_architecture_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def print_summary(results):
    """打印对比总结"""
    print("\n" + "="*80)
    print("CNN架构对比总结")
    print("="*80)
    
    # 创建对比表格
    print(f"{'模型':<15} {'参数数量':<12} {'训练时间(s)':<12} {'验证准确率':<12} {'验证损失':<12}")
    print("-" * 80)
    
    for model_name, result in results.items():
        print(f"{model_name:<15} {result['param_count']:<12,} "
              f"{result['training_time']:<12.2f} {result['final_val_acc']:<12.4f} "
              f"{result['final_val_loss']:<12.4f}")
    
    # 找出最佳模型
    best_acc_model = max(results.keys(), key=lambda x: results[x]['final_val_acc'])
    best_loss_model = min(results.keys(), key=lambda x: results[x]['final_val_loss'])
    fastest_model = min(results.keys(), key=lambda x: results[x]['training_time'])
    smallest_model = min(results.keys(), key=lambda x: results[x]['param_count'])
    
    print("\n" + "="*50)
    print("最佳性能:")
    print(f"  最高验证准确率: {best_acc_model} ({results[best_acc_model]['final_val_acc']:.4f})")
    print(f"  最低验证损失:   {best_loss_model} ({results[best_loss_model]['final_val_loss']:.4f})")
    print(f"  最快训练速度:   {fastest_model} ({results[fastest_model]['training_time']:.2f}s)")
    print(f"  最少参数数量:   {smallest_model} ({results[smallest_model]['param_count']:,})")
    
    # 架构特点分析
    print("\n" + "="*50)
    print("架构特点分析:")
    print("  DenseNetCNN (密集连接):")
    print("    - 特征重用，梯度流动好")
    print("    - 参数效率高，性能通常较好")
    print("    - 适合深层网络，缓解梯度消失")
    
    print("  StandardCNN (普通卷积):")
    print("    - 结构简单，易于理解和调试")
    print("    - 计算效率高，训练速度快")
    print("    - 适合作为基线模型")
    
    print("  DilatedCNN (空洞卷积):")
    print("    - 扩大感受野，不增加参数")
    print("    - 保持空间分辨率")
    print("    - 适合需要大感受野的任务")

if __name__ == "__main__":
    results = compare_architectures()