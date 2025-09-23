"""
轴承故障诊断 - 随机森林分类器
使用特征工程方法提取信号特征，然后使用随机森林进行分类
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from scipy import stats
from scipy.signal import welch, find_peaks, hilbert
from scipy.fft import fft, fftfreq
import os
import logging
from datetime import datetime
from tqdm import tqdm
import joblib
from bearing_data_loader import BearingDataset
from config import Config

# 设置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class BearingFeatureExtractor:
    """轴承振动信号特征提取器"""
    
    def __init__(self, fs=12000):
        self.fs = fs
    
    def extract_time_domain_features(self, signal):
        """提取时域特征"""
        features = {}
        
        try:
            # 确保信号是数值类型
            signal = np.asarray(signal, dtype=np.float64)
            
            # 基本统计特征
            features['mean'] = np.mean(signal)
            features['std'] = np.std(signal)
            features['var'] = np.var(signal)
            features['rms'] = np.sqrt(np.mean(signal**2))
            features['max'] = np.max(signal)
            features['min'] = np.min(signal)
            features['peak_to_peak'] = features['max'] - features['min']
            features['energy'] = np.sum(signal**2)
            
            # 形状特征
            features['skewness'] = stats.skew(signal)
            features['kurtosis'] = stats.kurtosis(signal)
            
            # 峰值因子和裕度因子
            if features['rms'] > 0:
                features['crest_factor'] = features['max'] / features['rms']
            else:
                features['crest_factor'] = 0
                
            if features['mean'] > 0:
                features['clearance_factor'] = features['max'] / features['mean']
            else:
                features['clearance_factor'] = 0
            
            # 脉冲因子
            if features['mean'] > 0:
                features['impulse_factor'] = features['max'] / features['mean']
            else:
                features['impulse_factor'] = 0
                
        except Exception as e:
            logging.warning(f"时域特征提取失败: {e}")
            # 返回默认值
            feature_names = ['mean', 'std', 'var', 'rms', 'max', 'min', 'peak_to_peak', 
                           'energy', 'skewness', 'kurtosis', 'crest_factor', 
                           'clearance_factor', 'impulse_factor']
            features = {name: 0.0 for name in feature_names}
        
        return features
    
    def extract_frequency_domain_features(self, signal):
        """提取频域特征"""
        features = {}
        
        try:
            # 确保信号是数值类型
            signal = np.asarray(signal, dtype=np.float64)
            
            # FFT
            fft_signal = fft(signal)
            freqs = fftfreq(len(signal), 1/self.fs)
            magnitude = np.abs(fft_signal)
            
            # 只取正频率部分
            positive_freqs = freqs[:len(freqs)//2]
            positive_magnitude = magnitude[:len(magnitude)//2]
            
            if len(positive_magnitude) > 0:
                # 频域统计特征
                features['freq_mean'] = np.mean(positive_magnitude)
                features['freq_std'] = np.std(positive_magnitude)
                features['freq_max'] = np.max(positive_magnitude)
                features['freq_energy'] = np.sum(positive_magnitude**2)
                
                # 主频率
                max_freq_idx = np.argmax(positive_magnitude)
                features['dominant_freq'] = positive_freqs[max_freq_idx] if max_freq_idx < len(positive_freqs) else 0
                
                # 频率重心
                if np.sum(positive_magnitude) > 0:
                    features['freq_centroid'] = np.sum(positive_freqs * positive_magnitude) / np.sum(positive_magnitude)
                else:
                    features['freq_centroid'] = 0
            else:
                features.update({
                    'freq_mean': 0, 'freq_std': 0, 'freq_max': 0, 
                    'freq_energy': 0, 'dominant_freq': 0, 'freq_centroid': 0
                })
                
        except Exception as e:
            logging.warning(f"频域特征提取失败: {e}")
            features = {
                'freq_mean': 0, 'freq_std': 0, 'freq_max': 0, 
                'freq_energy': 0, 'dominant_freq': 0, 'freq_centroid': 0
            }
        
        return features
    
    def extract_envelope_features(self, signal):
        """提取包络特征"""
        features = {}
        
        try:
            # 确保信号是数值类型
            signal = np.asarray(signal, dtype=np.float64)
            
            # 希尔伯特变换获取包络
            analytic_signal = np.abs(hilbert(signal))
            
            # 包络统计特征
            features['envelope_mean'] = np.mean(analytic_signal)
            features['envelope_std'] = np.std(analytic_signal)
            features['envelope_max'] = np.max(analytic_signal)
            features['envelope_energy'] = np.sum(analytic_signal**2)
            
        except Exception as e:
            logging.warning(f"包络特征提取失败: {e}")
            features = {
                'envelope_mean': 0, 'envelope_std': 0, 
                'envelope_max': 0, 'envelope_energy': 0
            }
        
        return features
    
    def extract_all_features(self, signal):
        """提取所有特征"""
        all_features = {}
        
        # 时域特征
        time_features = self.extract_time_domain_features(signal)
        all_features.update(time_features)
        
        # 频域特征
        freq_features = self.extract_frequency_domain_features(signal)
        all_features.update(freq_features)
        
        # 包络特征
        envelope_features = self.extract_envelope_features(signal)
        all_features.update(envelope_features)
        
        return all_features

class RandomForestBearingClassifier:
    """基于随机森林的轴承故障分类器"""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_extractor = BearingFeatureExtractor()
        self.feature_names = None
        self.class_names = ['正常', '内圈故障', '滚动体故障', '外圈故障']
        
        # 设置日志
        self.setup_logging()
    
    def setup_logging(self):
        """设置日志"""
        log_dir = 'log'
        os.makedirs(log_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = os.path.join(log_dir, f'random_forest_{timestamp}.log')
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_filename, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        
    def prepare_data(self, transform_method='stft'):
        """准备训练数据"""
        logging.info("开始加载数据...")
        
        # 计算step_size
        step_size = int(Config.WINDOW_SIZE * (1 - Config.OVERLAP_RATIO))
        
        # 加载数据
        dataset = BearingDataset(
            data_path=Config.DATA_PATH,
            window_size=Config.WINDOW_SIZE,
            step_size=step_size,
            transform_to_2d=False,  # 不需要2D转换，我们要原始信号
            transform_method=transform_method
        )
        
        # 获取样本数据和标签
        samples = dataset.samples  # 这是窗口化的样本数据
        labels = dataset.labels
        
        logging.info(f"数据加载完成，共 {len(samples)} 个样本")
        
        # 提取特征
        features_list = []
        valid_labels = []
        logging.info("开始提取特征...")
        
        for i, sample in enumerate(tqdm(samples, desc="特征提取")):
            try:
                # 确保sample是numpy数组且为数值类型
                if isinstance(sample, (list, tuple)):
                    sample = np.array(sample, dtype=np.float64)
                elif isinstance(sample, np.ndarray):
                    sample = sample.astype(np.float64)
                else:
                    logging.warning(f"样本 {i} 类型异常: {type(sample)}")
                    continue
                
                # 检查是否为有效的数值数据
                if sample.size == 0 or not np.isfinite(sample).all():
                    logging.warning(f"样本 {i} 包含无效数据")
                    continue
                
                features = self.feature_extractor.extract_all_features(sample)
                features_list.append(features)
                valid_labels.append(labels[i])
                
            except Exception as e:
                logging.warning(f"样本 {i} 特征提取失败: {e}")
                continue
        
        # 转换为DataFrame
        features_df = pd.DataFrame(features_list)
        self.feature_names = features_df.columns.tolist()
        
        logging.info(f"特征提取完成，共提取 {len(self.feature_names)} 个特征，有效样本 {len(features_list)} 个")
        
        return features_df.values, np.array(valid_labels)
    
    def train(self, X, y, test_size=0.2, random_state=42):
        """训练随机森林模型"""
        logging.info("开始训练随机森林模型...")
        
        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # 特征标准化
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # 简化的参数网格搜索（减少计算时间）
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [10, 20],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
        
        # 网格搜索
        logging.info("开始网格搜索最优参数...")
        rf = RandomForestClassifier(random_state=random_state, n_jobs=-1)
        grid_search = GridSearchCV(
            rf, param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X_train_scaled, y_train)
        
        # 使用最优参数训练模型
        self.model = grid_search.best_estimator_
        logging.info(f"最优参数: {grid_search.best_params_}")
        logging.info(f"最优交叉验证分数: {grid_search.best_score_:.4f}")
        
        # 在测试集上评估
        y_pred = self.model.predict(X_test_scaled)
        test_accuracy = accuracy_score(y_test, y_pred)
        
        logging.info(f"测试集准确率: {test_accuracy:.4f}")
        
        # 打印分类报告
        report = classification_report(y_test, y_pred, target_names=self.class_names)
        logging.info(f"分类报告:\n{report}")
        
        return {
            'train_accuracy': grid_search.best_score_,
            'test_accuracy': test_accuracy,
            'best_params': grid_search.best_params_,
            'classification_report': report,
            'y_test': y_test,
            'y_pred': y_pred
        }
    
    def plot_feature_importance(self, top_n=20):
        """绘制特征重要性"""
        if self.model is None:
            logging.error("模型未训练，无法绘制特征重要性")
            return
        
        # 获取特征重要性
        importance = self.model.feature_importances_
        feature_importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        # 绘制前top_n个重要特征
        plt.figure(figsize=(12, 8))
        top_features = feature_importance_df.head(top_n)
        
        sns.barplot(data=top_features, x='importance', y='feature')
        plt.title(f'前{top_n}个最重要特征')
        plt.xlabel('特征重要性')
        plt.ylabel('特征名称')
        plt.tight_layout()
        
        # 保存图片
        os.makedirs('imgs', exist_ok=True)
        plt.savefig('imgs/random_forest_feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 打印重要特征
        logging.info(f"前{top_n}个最重要特征:")
        for i, row in top_features.iterrows():
            logging.info(f"{row['feature']}: {row['importance']:.4f}")
    
    def plot_confusion_matrix(self, y_test, y_pred):
        """绘制混淆矩阵"""
        plt.figure(figsize=(10, 8))
        
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names)
        
        plt.title('随机森林分类器混淆矩阵')
        plt.xlabel('预测标签')
        plt.ylabel('真实标签')
        plt.tight_layout()
        
        # 保存图片
        os.makedirs('imgs', exist_ok=True)
        plt.savefig('imgs/random_forest_confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_model(self, filename='model/random_forest_bearing_classifier.pkl'):
        """保存模型"""
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'class_names': self.class_names
        }
        
        joblib.dump(model_data, filename)
        logging.info(f"模型已保存到: {filename}")
    
    def load_model(self, filename='model/random_forest_bearing_classifier.pkl'):
        """加载模型"""
        model_data = joblib.load(filename)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.class_names = model_data['class_names']
        
        logging.info(f"模型已从 {filename} 加载")
    
    def predict(self, signal):
        """预测单个信号"""
        if self.model is None:
            raise ValueError("模型未训练或加载")
        
        # 提取特征
        features = self.feature_extractor.extract_all_features(signal)
        features_array = np.array([list(features.values())])
        
        # 标准化
        features_scaled = self.scaler.transform(features_array)
        
        # 预测
        prediction = self.model.predict(features_scaled)[0]
        probability = self.model.predict_proba(features_scaled)[0]
        
        return {
            'prediction': prediction,
            'class_name': self.class_names[prediction],
            'probabilities': dict(zip(self.class_names, probability))
        }

def main():
    """主函数"""
    # 创建分类器
    rf_classifier = RandomForestBearingClassifier()
    
    # 准备数据
    logging.info("开始准备数据...")
    X, y = rf_classifier.prepare_data(transform_method='stft')
    
    # 训练模型
    results = rf_classifier.train(X, y)
    
    # 绘制特征重要性
    rf_classifier.plot_feature_importance()
    
    # 绘制混淆矩阵
    rf_classifier.plot_confusion_matrix(results['y_test'], results['y_pred'])
    
    # 保存模型
    rf_classifier.save_model()
    
    logging.info("随机森林训练完成！")

if __name__ == "__main__":
    main()