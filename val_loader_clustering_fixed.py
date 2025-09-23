import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import seaborn as sns
from sklearn.cluster import KMeans  # K-means聚类算法
from sklearn.decomposition import PCA  # 主成分分析降维
from sklearn.manifold import TSNE  # t-SNE非线性降维
from sklearn.metrics import silhouette_score, adjusted_rand_score  # 聚类评估指标
import pandas as pd

# 导入必要的模块
from cnn_model_pytorch import DenseNetCNN, DenseBlock, create_bearing_dataloaders
from config import Config

# 设置中文字体，确保matplotlib能正确显示中文
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def load_trained_model(model_path, device):
    """
    加载训练好的模型
    
    Args:
        model_path: 模型文件路径
        device: 计算设备 (CPU或GPU)
    
    Returns:
        model: 加载好的模型对象
    """
    print(f"加载模型: {model_path}")
    
    try:
        # 加载模型文件，weights_only=False允许加载完整模型对象
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        # 判断加载的内容类型并相应处理
        if isinstance(checkpoint, DenseNetCNN):
            # 情况1: 直接保存的完整模型对象
            model = checkpoint
            print("成功加载完整模型对象")
        elif isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            # 情况2: 包含状态字典的字典格式 (通常还包含优化器状态等)
            model = DenseNetCNN(num_classes=Config.NUM_CLASSES)
            model.load_state_dict(checkpoint['model_state_dict'])
            print("成功加载模型状态字典")
        elif isinstance(checkpoint, dict):
            # 情况3: 纯状态字典格式
            model = DenseNetCNN(num_classes=Config.NUM_CLASSES)
            model.load_state_dict(checkpoint)
            print("成功加载状态字典")
        else:
            raise ValueError(f"无法识别的模型文件格式: {type(checkpoint)}")
        
        # 将模型移动到指定设备并设置为评估模式
        model.to(device)
        model.eval()  # 关闭dropout和batch normalization的训练模式
        return model
        
    except Exception as e:
        print(f"加载模型失败: {e}")
        raise

def extract_features_from_val_loader(model, val_loader, device):
    """
    从验证集加载器中提取深度特征
    
    这个函数通过在CNN模型的全连接层前注册hook来提取特征向量，
    这些特征向量包含了模型学习到的高级语义信息
    
    Args:
        model: 训练好的CNN模型
        val_loader: 验证集数据加载器
        device: 计算设备
    
    Returns:
        features: 提取的特征矩阵 (n_samples, feature_dim)
        labels: 对应的真实标签 (n_samples,)
    """
    print("从验证集提取特征...")
    
    model.eval()  # 确保模型处于评估模式
    features_list = []  # 存储所有批次的特征
    labels_list = []    # 存储所有批次的标签
    
    # 用于存储hook捕获的特征
    features_hook = []
    
    def hook_fn(module, input, output):
        """
        Hook函数：在前向传播时捕获指定层的输出
        
        Args:
            module: 被hook的模块
            input: 模块的输入
            output: 模块的输出
        """
        # 将输出从GPU移到CPU并转换为numpy数组
        features_hook.append(output.detach().cpu().numpy())
    
    # 在全连接层fc1前注册hook，提取卷积特征
    # fc1是分类前的最后一个特征层，包含最丰富的语义信息
    handle = model.fc1.register_forward_hook(hook_fn)
    
    try:
        with torch.no_grad():  # 禁用梯度计算，节省内存和计算
            for batch_idx, (data, labels) in enumerate(val_loader):
                print(f"处理批次 {batch_idx + 1}/{len(val_loader)}")
                
                # 将数据移动到指定设备
                data = data.to(device)
                
                # 前向传播，触发hook函数
                _ = model(data)
                
                # 获取hook捕获的特征
                if features_hook:
                    batch_features = features_hook[-1]  # 获取最新的特征
                    features_list.append(batch_features)
                    labels_list.append(labels.numpy())  # 标签保持在CPU上
                    features_hook.clear()  # 清空hook列表
    
    finally:
        # 移除hook，避免内存泄漏
        handle.remove()
    
    # 将所有批次的特征和标签合并
    features = np.vstack(features_list)  # 垂直堆叠特征矩阵
    labels = np.concatenate(labels_list)  # 连接标签数组
    
    print(f"特征提取完成: {features.shape}, 标签: {labels.shape}")
    return features, labels

def perform_clustering_analysis(features, labels, n_clusters=4):
    """
    执行K-means聚类分析
    
    K-means算法将数据点分为k个簇，使得簇内数据点尽可能相似，
    簇间数据点尽可能不同
    
    Args:
        features: 特征矩阵
        labels: 真实标签 (用于评估聚类质量)
        n_clusters: 聚类数量
    
    Returns:
        dict: 包含聚类结果和评估指标的字典
    """
    print("执行聚类分析...")
    
    # 初始化K-means聚类器
    # n_init=10: 运行10次不同的初始化，选择最好的结果
    # random_state=42: 设置随机种子，确保结果可重现
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    
    # 执行聚类，返回每个样本的聚类标签
    cluster_labels = kmeans.fit_predict(features)
    
    # 计算聚类质量指标
    
    # 轮廓系数 (Silhouette Score): 衡量聚类的紧密度和分离度
    # 取值范围[-1, 1]，越接近1表示聚类效果越好
    silhouette = silhouette_score(features, cluster_labels)
    
    # 调整兰德指数 (Adjusted Rand Index): 衡量聚类结果与真实标签的一致性
    # 取值范围[-1, 1]，1表示完全一致，0表示随机分配
    ari = adjusted_rand_score(labels, cluster_labels)
    
    print(f"聚类质量指标:")
    print(f"  轮廓系数: {silhouette:.4f}")
    print(f"  调整兰德指数: {ari:.4f}")
    
    return {
        'cluster_labels': cluster_labels,      # 聚类标签
        'silhouette_score': silhouette,        # 轮廓系数
        'ari_score': ari,                      # 调整兰德指数
        'kmeans_model': kmeans                 # 训练好的K-means模型
    }

def reduce_dimensions(features, method='both'):
    """
    降维处理：将高维特征降到2D用于可视化
    
    Args:
        features: 高维特征矩阵
        method: 降维方法 ('pca', 'tsne', 'both')
    
    Returns:
        dict: 包含降维结果的字典
    """
    print("执行降维处理...")
    
    results = {}
    
    if method in ['pca', 'both']:
        # PCA (主成分分析) 降维
        # 线性降维方法，保留数据中方差最大的主成分
        pca = PCA(n_components=2, random_state=42)  # 降到2维用于可视化
        pca_features = pca.fit_transform(features)
        
        results['pca'] = {
            'features': pca_features,
            'explained_variance': pca.explained_variance_ratio_  # 每个主成分解释的方差比例
        }
        print(f"PCA解释方差比: {pca.explained_variance_ratio_}")
        print(f"PCA累计解释方差: {np.sum(pca.explained_variance_ratio_):.4f}")
    
    if method in ['tsne', 'both']:
        # t-SNE (t-分布随机邻域嵌入) 降维
        # 非线性降维方法，更好地保持局部邻域结构
        # perplexity: 控制局部和全局结构的平衡，通常设为5-50
        # max_iter: 最大迭代次数，减少以加快速度
        tsne = TSNE(n_components=2, random_state=42, max_iter=1000, perplexity=30)
        tsne_features = tsne.fit_transform(features)
        
        results['tsne'] = {
            'features': tsne_features
        }
        print("t-SNE降维完成")
    
    return results

def plot_clustering_results(features_2d, true_labels, cluster_labels, class_names, method_name, save_dir):
    """
    绘制聚类结果的可视化图表
    
    创建两个子图：左图显示真实标签的分布，右图显示聚类结果
    
    Args:
        features_2d: 2D降维后的特征
        true_labels: 真实标签
        cluster_labels: 聚类标签
        class_names: 类别名称列表
        method_name: 降维方法名称
        save_dir: 保存目录
    """
    # 创建包含两个子图的画布
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # 左图：真实标签散点图
    scatter1 = axes[0].scatter(features_2d[:, 0], features_2d[:, 1], 
                              c=true_labels,        # 颜色按真实标签分配
                              cmap='tab10',          # 使用tab10颜色映射
                              alpha=0.7,             # 透明度
                              s=20)                  # 点的大小
    axes[0].set_title(f'真实标签 ({method_name})', fontsize=14, fontweight='bold')
    axes[0].set_xlabel(f'{method_name} 第1维')
    axes[0].set_ylabel(f'{method_name} 第2维')
    
    # 为真实标签添加图例
    for i, class_name in enumerate(class_names):
        if i in true_labels:  # 只为存在的类别添加图例
            axes[0].scatter([], [], c=plt.cm.tab10(i), label=class_name, s=50)
    axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 右图：聚类结果散点图
    scatter2 = axes[1].scatter(features_2d[:, 0], features_2d[:, 1], 
                              c=cluster_labels,      # 颜色按聚类标签分配
                              cmap='viridis',        # 使用viridis颜色映射
                              alpha=0.7, 
                              s=20)
    axes[1].set_title(f'K-means聚类结果 ({method_name})', fontsize=14, fontweight='bold')
    axes[1].set_xlabel(f'{method_name} 第1维')
    axes[1].set_ylabel(f'{method_name} 第2维')
    
    plt.tight_layout()  # 自动调整子图间距
    
    # 保存图像
    save_path = os.path.join(save_dir, f'validation_clustering_{method_name.lower()}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"聚类可视化已保存: {save_path}")
    
    plt.show()

def analyze_cluster_composition(true_labels, cluster_labels, class_names):
    """
    分析每个聚类的组成成分
    
    统计每个聚类中包含哪些真实类别，以及各类别的比例
    这有助于理解聚类算法是否能够正确识别不同的故障类型
    
    Args:
        true_labels: 真实标签
        cluster_labels: 聚类标签
        class_names: 类别名称列表
    """
    print("\n=== 聚类组成分析 ===")
    
    n_clusters = len(np.unique(cluster_labels))
    
    # 遍历每个聚类
    for cluster_id in range(n_clusters):
        # 获取属于当前聚类的所有样本
        cluster_mask = cluster_labels == cluster_id
        cluster_true_labels = true_labels[cluster_mask]
        
        print(f"\n聚类 {cluster_id} (共 {np.sum(cluster_mask)} 个样本):")
        
        # 统计各类别在该聚类中的分布
        unique_labels, counts = np.unique(cluster_true_labels, return_counts=True)
        for label, count in zip(unique_labels, counts):
            percentage = count / np.sum(cluster_mask) * 100
            print(f"  {class_names[label]}: {count} 样本 ({percentage:.1f}%)")

def save_clustering_statistics(features, true_labels, cluster_results, class_names, save_dir):
    """
    保存聚类统计信息到CSV文件
    
    Args:
        features: 特征矩阵
        true_labels: 真实标签
        cluster_results: 聚类结果字典
        class_names: 类别名称列表
        save_dir: 保存目录
    
    Returns:
        DataFrame: 统计信息表格
    """
    stats_data = []
    
    # 整体统计信息
    stats_data.append({
        '指标': '总样本数',
        '值': len(true_labels),
        '说明': '验证集总样本数量'
    })
    
    stats_data.append({
        '指标': '特征维度',
        '值': features.shape[1],
        '说明': '提取的特征维度'
    })
    
    stats_data.append({
        '指标': '聚类数量',
        '值': len(np.unique(cluster_results['cluster_labels'])),
        '说明': 'K-means聚类数量'
    })
    
    stats_data.append({
        '指标': '轮廓系数',
        '值': f"{cluster_results['silhouette_score']:.4f}",
        '说明': '聚类质量指标 (越高越好)'
    })
    
    stats_data.append({
        '指标': '调整兰德指数',
        '值': f"{cluster_results['ari_score']:.4f}",
        '说明': '聚类与真实标签的一致性 (越高越好)'
    })
    
    # 各类别样本分布统计
    unique_labels, counts = np.unique(true_labels, return_counts=True)
    for label, count in zip(unique_labels, counts):
        percentage = count / len(true_labels) * 100
        stats_data.append({
            '指标': f'{class_names[label]}_样本数',
            '值': count,
            '说明': f'占总样本的 {percentage:.1f}%'
        })
    
    # 保存为CSV文件
    df = pd.DataFrame(stats_data)
    csv_path = os.path.join(save_dir, 'validation_clustering_statistics.csv')
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"聚类统计信息已保存: {csv_path}")
    
    return df

def main():
    """
    主函数：执行完整的验证集聚类可视化分析流程
    
    流程包括：
    1. 创建数据加载器
    2. 加载训练好的模型
    3. 提取深度特征
    4. 执行聚类分析
    5. 降维可视化
    6. 分析聚类组成
    7. 保存统计结果
    """
    print("=== 验证集聚类可视化分析 ===")
    
    # 设置计算设备 (优先使用GPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建结果保存目录
    results_dir = f"validation_clustering_results_{Config.TRANSFORM_METHOD}"
    os.makedirs(results_dir, exist_ok=True)
    
    try:
        # 1. 创建数据加载器 (与训练时使用相同的配置)
        print("创建数据加载器...")
        train_loader, val_loader, dataset = create_bearing_dataloaders(
            data_path=Config.DATA_PATH,           # 数据路径
            window_size=Config.WINDOW_SIZE,       # 窗口大小
            overlap_ratio=Config.OVERLAP_RATIO,   # 重叠比例
            batch_size=32,                        # 使用较小的批次大小以节省内存
            train_ratio=Config.TRAIN_RATIO,       # 训练集比例
            transform_to_2d=True,                 # 转换为2D图像格式
            transform_method=Config.TRANSFORM_METHOD  # 转换方法 (STFT/CWT等)
        )
        
        print(f"验证集批次数: {len(val_loader)}")
        class_names = dataset.get_class_names()
        print(f"类别名称: {class_names}")
        
        # 2. 加载训练好的模型
        # 获取模型文件路径
        full_model_path = Config.get_full_model_path()      # 完整模型路径
        state_dict_path = Config.get_model_state_path()     # 状态字典路径
        
        model = None
        
        # 优先尝试加载完整模型
        if os.path.exists(full_model_path):
            try:
                model = load_trained_model(full_model_path, device)
                print(f"成功加载完整模型: {full_model_path}")
            except Exception as e:
                print(f"加载完整模型失败: {e}")
        
        # 如果完整模型加载失败，尝试加载状态字典
        if model is None and os.path.exists(state_dict_path):
            try:
                model = DenseNetCNN(num_classes=Config.NUM_CLASSES)
                model.load_state_dict(torch.load(state_dict_path, map_location=device, weights_only=False))
                model.to(device)
                model.eval()
                print(f"成功加载状态字典: {state_dict_path}")
            except Exception as e:
                print(f"加载状态字典失败: {e}")
        
        if model is None:
            raise ValueError("无法加载任何模型文件")
        
        # 3. 从验证集提取深度特征
        features, labels = extract_features_from_val_loader(model, val_loader, device)
        
        # 4. 执行K-means聚类分析
        cluster_results = perform_clustering_analysis(features, labels, n_clusters=len(class_names))
        
        # 5. 降维处理 (PCA和t-SNE)
        dim_reduction_results = reduce_dimensions(features, method='both')
        
        # 6. 绘制聚类可视化结果
        for method_name, dim_data in dim_reduction_results.items():
            plot_clustering_results(
                dim_data['features'],              # 降维后的2D特征
                labels,                           # 真实标签
                cluster_results['cluster_labels'], # 聚类标签
                class_names,                      # 类别名称
                method_name.upper(),              # 方法名称 (PCA/TSNE)
                results_dir                       # 保存目录
            )
        
        # 7. 分析聚类组成
        analyze_cluster_composition(labels, cluster_results['cluster_labels'], class_names)
        
        # 8. 保存详细统计信息
        save_clustering_statistics(features, labels, cluster_results, class_names, results_dir)
        
        print(f"\n验证集聚类分析完成！结果已保存到: {results_dir}")
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()