"""
安装所需的Python包
"""

import subprocess
import sys
import os

def install_package(package):
    """安装单个包"""
    try:
        print(f"正在安装 {package}...")
        result = subprocess.run([sys.executable, "-m", "pip", "install", package], 
                              capture_output=True, text=True, check=True)
        print(f"✓ 成功安装 {package}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ 安装 {package} 失败")
        print(f"错误信息: {e.stderr}")
        return False

def check_package(package_name, import_name=None):
    """检查包是否已安装"""
    if import_name is None:
        import_name = package_name
    
    try:
        __import__(import_name)
        print(f"✓ {package_name} 已安装")
        return True
    except ImportError:
        print(f"✗ {package_name} 未安装")
        return False

def main():
    """安装所有必需的包"""
    print("钻孔成像裂隙识别系统 - 依赖包安装")
    print("=" * 50)
    
    # 定义需要安装的包
    packages = [
        ("opencv-python", "cv2"),
        ("numpy", "numpy"),
        ("scikit-image", "skimage"),
        ("matplotlib", "matplotlib"),
        ("scipy", "scipy")
    ]
    
    # 检查已安装的包
    print("\n检查已安装的包...")
    installed_packages = []
    missing_packages = []
    
    for package_name, import_name in packages:
        if check_package(package_name, import_name):
            installed_packages.append(package_name)
        else:
            missing_packages.append(package_name)
    
    # 安装缺失的包
    if missing_packages:
        print(f"\n需要安装 {len(missing_packages)} 个包...")
        
        for package in missing_packages:
            success = install_package(package)
            if not success:
                print(f"\n警告: {package} 安装失败，可能会影响程序运行")
    else:
        print("\n所有依赖包都已安装！")
    
    # 最终检查
    print("\n最终检查...")
    all_installed = True
    for package_name, import_name in packages:
        if not check_package(package_name, import_name):
            all_installed = False
    
    if all_installed:
        print("\n🎉 所有依赖包安装完成！可以运行裂隙识别程序了。")
    else:
        print("\n⚠️  部分包安装失败，请手动安装或检查网络连接。")
    
    print("\n按任意键继续...")
    input()

if __name__ == "__main__":
    main()