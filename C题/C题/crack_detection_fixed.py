"""
最终版钻孔成像裂隙识别系统
修复JSON序列化问题
"""

import numpy as np
import os
import json
from PIL import Image, ImageFilter, ImageEnhance
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from scipy import ndimage
from scipy.ndimage import binary_opening, binary_closing, label

# 设置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class FinalCrackDetector:
    def __init__(self):
        """最终版裂隙检测器"""
        self.threshold = 0.3
        self.setup_chinese_font()
        
    def setup_chinese_font(self):
        """设置中文字体"""
        try:
            font_list = fm.findSystemFonts()
            chinese_fonts = []
            
            for font_path in font_list:
                try:
                    font_prop = fm.FontProperties(fname=font_path)
                    font_name = font_prop.get_name()
                    if any(chinese in font_name.lower() for chinese in ['simhei', 'microsoft', 'yahei', 'simsun']):
                        chinese_fonts.append(font_name)
                except:
                    continue
            
            if chinese_fonts:
                plt.rcParams['font.sans-serif'] = chinese_fonts[:3] + ['DejaVu Sans']
                print(f"✓ 找到中文字体: {chinese_fonts[0]}")
                self.use_english = False
            else:
                print("⚠️  未找到中文字体，将使用英文标题")
                self.use_english = True
                
        except Exception as e:
            print(f"⚠️  字体设置失败: {str(e)}，将使用英文标题")
            self.use_english = True
    
    def convert_to_serializable(self, obj):
        """转换numpy类型为Python原生类型，使其可JSON序列化"""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self.convert_to_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self.convert_to_serializable(item) for item in obj]
        else:
            return obj
    
    def load_image(self, image_path):
        """加载图像"""
        try:
            img = Image.open(image_path)
            if img.mode != 'L':
                img = img.convert('L')
            return np.array(img)
        except Exception as e:
            raise ValueError(f"无法加载图像 {image_path}: {str(e)}")
    
    def preprocess(self, image):
        """预处理图像"""
        pil_img = Image.fromarray(image)
        blurred = pil_img.filter(ImageFilter.GaussianBlur(radius=1))
        enhancer = ImageEnhance.Contrast(blurred)
        enhanced = enhancer.enhance(1.5)
        return np.array(enhanced)
    
    def detect_edges_sobel(self, image):
        """使用Sobel算子检测边缘"""
        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        
        grad_x = ndimage.convolve(image.astype(float), sobel_x)
        grad_y = ndimage.convolve(image.astype(float), sobel_y)
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        if magnitude.max() > 0:
            magnitude = (magnitude / magnitude.max() * 255).astype(np.uint8)
        else:
            magnitude = magnitude.astype(np.uint8)
        
        return magnitude
    
    def remove_vertical_lines(self, image):
        """去除垂直线条干扰"""
        vertical_kernel = np.ones((15, 1))
        binary_img = image > np.mean(image)
        vertical_lines = binary_opening(binary_img, vertical_kernel)
        
        mask = vertical_lines.astype(bool)
        result = image.copy()
        
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                if mask[i, j]:
                    row = image[i, :]
                    row_mask = mask[i, :]
                    valid_pixels = row[~row_mask]
                    if len(valid_pixels) > 0:
                        result[i, j] = np.mean(valid_pixels)
        
        return result
    
    def morphological_filter(self, binary_image):
        """形态学滤波"""
        cleaned = binary_opening(binary_image, np.ones((3, 3)))
        connected = binary_closing(cleaned, np.ones((5, 5)))
        return connected
    
    def filter_components(self, binary_image, min_area=20):
        """过滤连通组件"""
        labeled, num_features = label(binary_image)
        result = np.zeros_like(binary_image)
        
        for i in range(1, num_features + 1):
            component = (labeled == i)
            area = np.sum(component)
            
            if area >= min_area:
                coords = np.where(component)
                if len(coords[0]) > 0 and len(coords[1]) > 0:
                    min_row, max_row = coords[0].min(), coords[0].max()
                    min_col, max_col = coords[1].min(), coords[1].max()
                    
                    height = max_row - min_row + 1
                    width = max_col - min_col + 1
                    aspect_ratio = max(height, width) / (min(height, width) + 1e-7)
                    
                    if aspect_ratio > 2:
                        result[component] = 1
        
        return result
    
    def detect_cracks(self, image_path):
        """主检测函数"""
        image = self.load_image(image_path)
        preprocessed = self.preprocess(image)
        cleaned = self.remove_vertical_lines(preprocessed)
        edges = self.detect_edges_sobel(cleaned)
        
        threshold_value = np.percentile(edges, 85)
        binary = edges > threshold_value
        morphed = self.morphological_filter(binary)
        final_result = self.filter_components(morphed)
        
        output = np.ones_like(final_result, dtype=np.uint8) * 255
        output[final_result > 0] = 0
        
        return output, preprocessed
    
    def create_detailed_analysis(self, input_dir, output_dir, key_images):
        """创建详细分析报告"""
        analysis_results = {}
        
        for img_name in key_images:
            img_path = os.path.join(input_dir, img_name)
            if not os.path.exists(img_path):
                print(f"⚠️  文件不存在: {img_path}")
                continue
                
            try:
                print(f"🔍 分析 {img_name}...")
                
                # 检测裂隙
                crack_mask, preprocessed = self.detect_cracks(img_path)
                original = self.load_image(img_path)
                
                # 统计信息 - 确保转换为Python原生类型
                crack_pixels = int(np.sum(crack_mask == 0))
                total_pixels = int(crack_mask.shape[0] * crack_mask.shape[1])
                crack_ratio = float(crack_pixels / total_pixels)
                
                # 创建详细分析图
                fig, axes = plt.subplots(2, 2, figsize=(12, 10))
                
                if self.use_english:
                    titles = [
                        f'Original Image - {img_name}',
                        'Preprocessed Image',
                        'Crack Detection Result',
                        'Overlay Result'
                    ]
                    suptitle = f'Detailed Analysis - {img_name}'
                else:
                    titles = [
                        f'原始图像 - {img_name}',
                        '预处理图像',
                        '裂隙检测结果',
                        '叠加结果'
                    ]
                    suptitle = f'详细分析 - {img_name}'
                
                # 原始图像
                axes[0, 0].imshow(original, cmap='gray')
                axes[0, 0].set_title(titles[0])
                axes[0, 0].axis('off')
                
                # 预处理图像
                axes[0, 1].imshow(preprocessed, cmap='gray')
                axes[0, 1].set_title(titles[1])
                axes[0, 1].axis('off')
                
                # 检测结果
                axes[1, 0].imshow(crack_mask, cmap='gray')
                axes[1, 0].set_title(titles[2])
                axes[1, 0].axis('off')
                
                # 叠加结果
                overlay = np.stack([original, original, original], axis=-1)
                overlay[crack_mask == 0] = [255, 0, 0]  # 红色标记裂隙
                axes[1, 1].imshow(overlay)
                axes[1, 1].set_title(titles[3])
                axes[1, 1].axis('off')
                
                # 添加统计信息
                stats_text = f'Total Pixels: {total_pixels:,}\nCrack Pixels: {crack_pixels:,}\nCrack Ratio: {crack_ratio:.4f} ({crack_ratio*100:.2f}%)'
                fig.text(0.02, 0.02, stats_text, fontsize=10, 
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
                
                plt.suptitle(suptitle, fontsize=14)
                plt.tight_layout()
                
                # 保存详细分析图
                detail_path = os.path.join(output_dir, f"detailed_analysis_{img_name.replace('.jpg', '.png')}")
                plt.savefig(detail_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                # 保存结果 - 确保所有数据都是可序列化的
                analysis_results[img_name] = {
                    'crack_pixels': crack_pixels,
                    'total_pixels': total_pixels,
                    'crack_ratio': crack_ratio,
                    'detail_image': os.path.basename(detail_path)
                }
                
                print(f"✓ 完成分析: {img_name}")
                print(f"  裂隙像素: {crack_pixels:,} / {total_pixels:,}")
                print(f"  裂隙比例: {crack_ratio:.4f} ({crack_ratio*100:.2f}%)")
                
            except Exception as e:
                print(f"✗ 分析 {img_name} 失败: {str(e)}")
                analysis_results[img_name] = {'error': str(e)}
        
        return analysis_results
    
    def save_binary_results(self, input_dir, output_dir, all_images=True):
        """保存所有图像的二值化结果"""
        binary_dir = os.path.join(output_dir, "binary_results")
        if not os.path.exists(binary_dir):
            os.makedirs(binary_dir)
        
        if all_images:
            # 处理所有图像
            image_files = [f for f in os.listdir(input_dir) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        else:
            # 只处理重点图像
            image_files = ["图1-1.jpg", "图1-2.jpg", "图1-3.jpg"]
        
        results = {}
        
        for filename in image_files:
            img_path = os.path.join(input_dir, filename)
            if not os.path.exists(img_path):
                continue
                
            try:
                crack_mask, _ = self.detect_cracks(img_path)
                
                # 保存二值图像
                output_path = os.path.join(binary_dir, f"result_{filename}")
                Image.fromarray(crack_mask).save(output_path)
                
                # 统计信息
                crack_pixels = int(np.sum(crack_mask == 0))
                total_pixels = int(crack_mask.shape[0] * crack_mask.shape[1])
                crack_ratio = float(crack_pixels / total_pixels)
                
                results[filename] = {
                    'crack_pixels': crack_pixels,
                    'total_pixels': total_pixels,
                    'crack_ratio': crack_ratio,
                    'binary_image': f"result_{filename}"
                }
                
                print(f"✓ 保存二值结果: {filename}")
                
            except Exception as e:
                print(f"✗ 处理 {filename} 失败: {str(e)}")
                results[filename] = {'error': str(e)}
        
        return results

def main():
    """主函数"""
    print("最终版钻孔成像裂隙识别系统")
    print("=" * 60)
    
    detector = FinalCrackDetector()
    
    # 查找输入目录
    input_dir = "C题/附件1"
    if not os.path.exists(input_dir):
        input_dir = "附件1"
    
    if not os.path.exists(input_dir):
        print("❌ 错误: 找不到附件1目录")
        return
    
    output_dir = "裂隙识别结果_最终版"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print(f"📁 输入目录: {input_dir}")
    print(f"📁 输出目录: {output_dir}")
    
    # 重点分析图像
    key_images = ["图1-1.jpg", "图1-2.jpg", "图1-3.jpg"]
    
    print("\n🔍 开始详细分析重点图像...")
    analysis_results = detector.create_detailed_analysis(input_dir, output_dir, key_images)
    
    print("\n💾 保存所有图像的二值化结果...")
    binary_results = detector.save_binary_results(input_dir, output_dir, all_images=True)
    
    # 显示分析结果
    print("\n" + "="*60)
    print("📊 重点图像详细分析结果")
    print("="*60)
    
    for img_name in key_images:
        if img_name in analysis_results and 'error' not in analysis_results[img_name]:
            result = analysis_results[img_name]
            print(f"\n📷 {img_name}:")
            print(f"   总像素数: {result['total_pixels']:,}")
            print(f"   裂隙像素数: {result['crack_pixels']:,}")
            print(f"   裂隙像素比例: {result['crack_ratio']:.4f} ({result['crack_ratio']*100:.2f}%)")
            print(f"   详细分析图: {result['detail_image']}")
        elif img_name in analysis_results:
            print(f"\n❌ {img_name}: {analysis_results[img_name]['error']}")
        else:
            print(f"\n⚠️  {img_name}: 文件不存在")
    
    # 转换为可序列化格式并保存
    try:
        serializable_analysis = detector.convert_to_serializable(analysis_results)
        serializable_binary = detector.convert_to_serializable(binary_results)
        
        # 保存详细分析结果
        analysis_path = os.path.join(output_dir, "detailed_analysis_results.json")
        with open(analysis_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_analysis, f, ensure_ascii=False, indent=2)
        
        # 保存二值化结果统计
        binary_path = os.path.join(output_dir, "binary_results_statistics.json")
        with open(binary_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_binary, f, ensure_ascii=False, indent=2)
        
        print(f"\n✅ 所有结果保存成功！")
        print(f"📁 结果目录: {output_dir}")
        print(f"📊 详细分析: {analysis_path}")
        print(f"📊 二值化统计: {binary_path}")
        
    except Exception as e:
        print(f"❌ 保存结果时出错: {str(e)}")
    
    # 显示处理总结
    total_success = len([k for k in binary_results.keys() if 'error' not in binary_results[k]])
    total_error = len([k for k in binary_results.keys() if 'error' in binary_results[k]])
    
    print(f"\n📈 处理总结:")
    print(f"   ✅ 成功处理: {total_success} 张图像")
    print(f"   ❌ 处理失败: {total_error} 张图像")
    print(f"   🎯 重点分析: {len(key_images)} 张图像")
    
    print(f"\n🎉 钻孔成像裂隙识别完成！")
    print(f"   📋 所有识别结果以二值图像形式保存")
    print(f"   🖤 裂隙像素为黑色(0)，背景像素为白色(255)")
    print(f"   📊 包含详细的统计分析和可视化结果")

if __name__ == "__main__":
    main()