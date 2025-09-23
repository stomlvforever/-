"""
轴承故障诊断项目配置文件
"""

class Config:
    """统一配置参数管理"""
    
    # 数据相关参数
    DATA_PATH = r"数据集\数据集\源域数据集"
    WINDOW_SIZE = 8192  # 64*64 = 4096，匹配CNN输入
    OVERLAP_RATIO = 0.3  # 重叠
    TRAIN_RATIO = 0.8  # 训练集比例
    
    # 2D转换方法配置
    TRANSFORM_METHOD = 'cwt'  # 可选: 'stft', 'cwt', 'spectrogram', 'reshape'
    
    # 模型相关参数
    NUM_CLASSES = 4
    INPUT_SIZE = (1, 64, 64)  # 输入图像尺寸
    GROWTH_RATE = 6  # DenseBlock增长率
    
    # 训练相关参数
    BATCH_SIZE = 32
    EPOCHS = 50
    LEARNING_RATE = 0.0003
    
    # 基础目录参数
    MODEL_DIR = 'model'
    LOG_DIR = 'log'
    IMG_DIR = 'imgs'
    
    # # 11分类标签名称
    # CLASS_NAMES = [
    #     '正常', 
    #     '内圈故障_0007', '内圈故障_0014', '内圈故障_0021', '内圈故障_0028',
    #     '滚动体故障_0007', '滚动体故障_0014', '滚动体故障_0021', '滚动体故障_0028',
    #     '外圈故障_0007', '外圈故障_0021'
    # ]
    
    # 4分类标签名称
    CLASS_NAMES = [
        '正常',        # N
        '内圈故障',    # IR
        '滚动体故障',  # B  
        '外圈故障'     # OR
    ]
    
    @classmethod
    def get_model_path(cls, transform_method=None):
        """获取模型文件路径"""
        method = transform_method or cls.TRANSFORM_METHOD
        return f"{cls.MODEL_DIR}/bearing_fault_cnn_4class_{method}.pth"
    
    @classmethod
    def get_full_model_path(cls, transform_method=None):
        """获取完整模型文件路径"""
        method = transform_method or cls.TRANSFORM_METHOD
        return f"{cls.MODEL_DIR}/bearing_fault_cnn_4class_{method}_full.pth"
    
    @classmethod
    def get_model_state_path(cls):
        """根据TRANSFORM_METHOD生成模型状态文件路径"""
        return f'model/bearing_fault_cnn_4class_{cls.TRANSFORM_METHOD}.pth'
    
    @classmethod
    def get_confusion_matrix_path(cls):
        """根据TRANSFORM_METHOD生成混淆矩阵图片路径"""
        return f'imgs/confusion_matrix_4class_{cls.TRANSFORM_METHOD}.png'
    
    @classmethod
    def get_training_history_path(cls):
        """根据TRANSFORM_METHOD生成训练历史图片路径"""
        return f'imgs/bearing_cnn_training_history_4class_{cls.TRANSFORM_METHOD}.png'
    
    @classmethod
    def get_log_path(cls):
        """获取日志文件路径"""
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{cls.LOG_DIR}/training_{timestamp}.log"