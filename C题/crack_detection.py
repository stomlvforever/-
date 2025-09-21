import cv2
import numpy as np
import os
from sklearn.cluster import KMeans
from skimage import filters, morphology, measure
from skimage.feature import local_binary_pattern
import matplotlib.pyplot as plt
from scipy import ndimage
import warnings
warnings.filterwarnings('ignore')

class CrackDetector:
    def __init__(self):
        """
        钻孔成像展开图裂隙识别器
        考虑岩石纹理、钻进痕迹及泥浆污染等干扰因素
        """
        self.kernel_size = 3
        self.threshold_ratio = 0.15
        
    def preprocess_image(self, image):
        """
        图像预处理：去噪、增强对比度
        """
        # 转换为灰度图
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            
        # 高斯滤波去噪
        denoised = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # CLAHE对比度增强
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)
        
        return enhanced
    
    def remove_vertical_artifacts(self, image):
        """
        去除垂直方向的泥浆干扰和拼接线
        """
        # 检测垂直线条
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))
        vertical_lines = cv2.morphologyEx(image, cv2.MORPH_OPEN, vertical_kernel)
        
        # 创建掩码去除垂直干扰
        mask = cv2.threshold(vertical_lines, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        
        # 膨胀掩码以覆盖更多干扰区域
        mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=2)
        
        # 使用inpainting修复被掩码覆盖的区域
        result = cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)
        
        return result
    
    def texture_analysis(self, image):
        """
        基于局部二值模式的纹理分析，区分岩石纹理和裂隙
        """
        # 计算LBP特征
        radius = 3
        n_points = 8 * radius
        lbp = local_binary_pattern(image, n_points, radius, method='uniform')
        
        # 计算LBP直方图
        n_bins = n_points + 2
        hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
        hist = hist.astype(float)
        hist /= (hist.sum() + 1e-7)
        
        # 基于纹理复杂度判断
        texture_complexity = np.std(hist)
        
        return lbp, texture_complexity
    
    def detect_cracks_multiscale(self, image):
        """
        多尺度裂隙检测
        """
        # 去除垂直干扰
        cleaned = self.remove_vertical_artifacts(image)
        
        # 多尺度边缘检测
        edges_list = []
        scales = [1, 2, 3]
        
        for scale in scales:
            # 高斯滤波
            blurred = cv2.GaussianBlur(cleaned, (scale*2+1, scale*2+1), scale)
            
            # Canny边缘检测
            edges = cv2.Canny(blurred, 50, 150)
            edges_list.append(edges)
        
        # 融合多尺度边缘
        combined_edges = np.zeros_like(edges_list[0])
        for edges in edges_list:
            combined_edges = cv2.bitwise_or(combined_edges, edges)
        
        return combined_edges
    
    def morphological_filtering(self, binary_image):
        """
        形态学滤波，去除噪声并连接断裂的裂隙
        """
        # 去除小的噪声点
        kernel_small = np.ones((3, 3), np.uint8)
        cleaned = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel_small)
        
        # 连接断裂的裂隙
        kernel_connect = np.ones((5, 5), np.uint8)
        connected = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_connect)
        
        # 细化操作
        thinned = cv2.ximgproc.thinning(connected)
        
        return thinned
    
    def filter_by_geometry(self, binary_image, min_length=20, max_width=10):
        """
        基于几何特征过滤，保留符合裂隙特征的区域
        """
        # 连通组件分析
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_image, connectivity=8)
        
        # 创建输出图像
        filtered = np.zeros_like(binary_image)
        
        for i in range(1, num_labels):  # 跳过背景
            # 获取组件的统计信息
            area = stats[i, cv2.CC_STAT_AREA]
            width = stats[i, cv2.CC_STAT_WIDTH]
            height = stats[i, cv2.CC_STAT_HEIGHT]
            
            # 计算长宽比
            aspect_ratio = max(width, height) / (min(width, height) + 1e-7)
            
            # 过滤条件：面积适中，长宽比大（线性特征）
            if (area > min_length and 
                aspect_ratio > 3 and 
                min(width, height) <= max_width):
                
                # 保留该组件
                component_mask = (labels == i).astype(np.uint8) * 255
                filtered = cv2.bitwise_or(filtered, component_mask)
        
        return filtered
    
    def detect_cracks(self, image_path):
        """
        主要的裂隙检测函数
        """
        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法读取图像: {image_path}")
        
        # 预处理
        preprocessed = self.preprocess_image(image)
        
        # 纹理分析
        lbp, texture_complexity = self.texture_analysis(preprocessed)
        
        # 多尺度裂隙检测
        edges = self.detect_cracks_multiscale(preprocessed)
        
        # 自适应阈值处理
        adaptive_thresh = cv2.adaptiveThreshold(
            preprocessed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # 结合边缘和阈值结果
        combined = cv2.bitwise_and(edges, adaptive_thresh)
        
        # 形态学滤波
        morphed = self.morphological_filtering(combined)
        
        # 几何特征过滤
        final_result = self.filter_by_geometry(morphed)
        
        return final_result, preprocessed
    
    def process_batch(self, input_dir, output_dir):
        """
        批量处理图像
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        results = {}
        
        # 获取所有jpg文件
        image_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.jpg')]
        image_files.sort()
        
        for filename in image_files:
            print(f"正在处理: {filename}")
            
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, f"result_{filename}")
            
            try:
                # 检测裂隙
                crack_mask, preprocessed = self.detect_cracks(input_path)
                
                # 保存结果
                cv2.imwrite(output_path, crack_mask)
                
                # 保存分析结果
                results[filename] = {
                    'crack_pixels': np.sum(crack_mask > 0),
                    'total_pixels': crack_mask.shape[0] * crack_mask.shape[1],
                    'crack_ratio': np.sum(crack_mask > 0) / (crack_mask.shape[0] * crack_mask.shape[1])
                }
                
                print(f"完成处理: {filename}, 裂隙像素比例: {results[filename]['crack_ratio']:.4f}")
                
            except Exception as e:
                print(f"处理 {filename} 时出错: {str(e)}")
                results[filename] = {'error': str(e)}
        
        return results
    
    def visualize_results(self, image_path, save_path=None):
        """
        可视化检测结果
        """
        # 读取原图
        original = cv2.imread(image_path)
        original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        
        # 检测裂隙
        crack_mask, preprocessed = self.detect_cracks(image_path)
        
        # 创建彩色叠加图
        overlay = original_rgb.copy()
        overlay[crack_mask > 0] = [255, 0, 0]  # 红色标记裂隙
        
        # 创建子图
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        axes[0, 0].imshow(original_rgb)
        axes[0, 0].set_title('原始图像')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(preprocessed, cmap='gray')
        axes[0, 1].set_title('预处理后图像')
        axes[0, 1].axis('off')
        
        axes[1, 0].imshow(crack_mask, cmap='gray')
        axes[1, 0].set_title('裂隙检测结果（二值图）')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(overlay)
        axes[1, 1].set_title('裂隙标记叠加图')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        
        return crack_mask

def main():
    """
    主函数：处理附件1中的所有图像
    """
    # 创建检测器实例
    detector = CrackDetector()
    
    # 设置路径
    input_dir = "附件1"
    output_dir = "裂隙识别结果"
    
    # 批量处理
    print("开始批量处理附件1中的图像...")
    results = detector.process_batch(input_dir, output_dir)
    
    # 重点分析图1-1, 图1-2, 图1-3
    key_images = ["图1-1.jpg", "图1-2.jpg", "图1-3.jpg"]
    
    print("\n重点分析结果:")
    for img_name in key_images:
        if img_name in results and 'error' not in results[img_name]:
            print(f"\n{img_name}:")
            print(f"  - 总像素数: {results[img_name]['total_pixels']}")
            print(f"  - 裂隙像素数: {results[img_name]['crack_pixels']}")
            print(f"  - 裂隙像素比例: {results[img_name]['crack_ratio']:.4f}")
            
            # 生成可视化结果
            img_path = os.path.join(input_dir, img_name)
            vis_path = os.path.join(output_dir, f"analysis_{img_name.replace('.jpg', '.png')}")
            detector.visualize_results(img_path, vis_path)
    
    # 保存统计结果
    import json
    with open(os.path.join(output_dir, "detection_statistics.json"), 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n所有结果已保存到 {output_dir} 目录")

if __name__ == "__main__":
    main()