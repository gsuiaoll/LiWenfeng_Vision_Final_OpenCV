"""
任务一：图像基础预处理
实现功能：
1. 图片读取
2. 灰度转换
3. 高斯模糊去噪
4. 直方图均衡化
5. 输出预处理前后对比图
"""

import cv2
import os
import sys

# 将项目根目录加入模块搜索路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.utils import read_image, save_image, create_comparison_image


def grayscale(image):
    """灰度转换：将BGR彩色图像转为单通道灰度图"""
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def gaussian_blur(image, kernel_size=(5, 5), sigma=1.0):
    """
    高斯模糊去噪
    :param image: 输入图像（灰度或彩色）
    :param kernel_size: 高斯核大小，必须为奇数
    :param sigma: 标准差
    :return: 去噪后的图像
    """
    return cv2.GaussianBlur(image, kernel_size, sigma)


def histogram_equalization(image):
    """
    直方图均衡化，增强对比度
    :param image: 输入图像，彩色BGR或灰度
    :return: 均衡化后的图像
    """
    if len(image.shape) == 3:
        # 彩色图像：转换到YUV空间，对Y通道均衡化，再转回BGR
        yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
        return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
    else:
        return cv2.equalizeHist(image)


def preprocess_pipeline(image_path, output_dir):
    """
    预处理主流程：依次进行灰度、高斯模糊、直方图均衡化，并保存对比图
    :param image_path: 输入图片路径
    :param output_dir: 结果输出目录
    """
    original = read_image(image_path)
    if original is None:
        return

    gray = grayscale(original)
    blurred = gaussian_blur(gray, (5, 5), 1.0)
    # 彩色图均衡化，保留更多视觉信息
    equalized_color = histogram_equalization(original)
    # 同时保留灰度均衡化
    equalized_gray = histogram_equalization(gray)

    # 单独保存各个处理结果
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    save_image(os.path.join(output_dir, f"{base_name}_gray.jpg"), gray)
    save_image(os.path.join(output_dir, f"{base_name}_blur.jpg"), blurred)
    save_image(os.path.join(output_dir, f"{base_name}_equalize.jpg"), equalized_gray)

    # 生成对比图：原图、灰度、模糊、均衡化（彩色）
    comparison = create_comparison_image(
        [original, gray, blurred, equalized_color],
        ["Original", "Grayscale", "Gaussian Blur", "Histogram Equalization"],
        cols=2
    )
    save_image(os.path.join(output_dir, f"{base_name}_comparison.jpg"), comparison)


if __name__ == "__main__":
    # 基于当前文件位置计算项目根目录，支持从任意目录运行
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    input_image = os.path.join(project_root, "test_images", "original", "images", "basic_test.jpg")
    output_dir = os.path.join(project_root, "test_images", "results", "task1")
    preprocess_pipeline(input_image, output_dir)
