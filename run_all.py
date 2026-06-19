"""
项目总运行脚本
一键执行所有基础任务和进阶任务，生成测试效果图和分析报告

运行方式：
    python run_all.py
"""

import os
import sys

# 确保项目根目录在模块搜索路径中
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from src.task1_preprocessing.preprocessing import preprocess_pipeline
from src.task2_color_detection.color_detection import color_detection_pipeline
from src.task3_shape_digit_recognition.shape_recognition import shape_recognition_pipeline
from src.task3_shape_digit_recognition.digit_recognition import digit_recognition_pipeline
from src.task4_advanced.armor_detection import armor_detection_pipeline
from src.task4_advanced.robustness_test import robustness_test_pipeline


def main():
    print("=" * 60)
    print("OpenCV 视觉算法最终考核 - 自动化测试脚本")
    print("=" * 60)

    # 使用绝对路径，避免依赖运行目录
    base_image_dir = os.path.join(project_root, "test_images", "original", "images")
    results_dir = os.path.join(project_root, "test_images", "results")

    # 任务一：图像基础预处理
    print("\n[任务一] 图像基础预处理...")
    preprocess_pipeline(
        os.path.join(base_image_dir, "basic_test.jpg"),
        os.path.join(results_dir, "task1")
    )

    # 任务二：颜色阈值色块识别
    print("\n[任务二] 颜色阈值色块识别...")
    color_detection_pipeline(
        os.path.join(base_image_dir, "color_test.jpg"),
        os.path.join(results_dir, "task2")
    )

    # 任务三：几何图形识别
    print("\n[任务三] 几何图形识别...")
    shape_recognition_pipeline(
        os.path.join(base_image_dir, "shape_number_test.jpg"),
        os.path.join(results_dir, "task3")
    )

    # 任务三：数字识别
    print("\n[任务三] 印刷体数字识别...")
    digit_recognition_pipeline(
        os.path.join(base_image_dir, "shape_number_test.jpg"),
        os.path.join(results_dir, "task3")
    )

    # 进阶任务一：装甲板目标定位
    print("\n[进阶任务一] 装甲板目标定位...")
    armor_detection_pipeline(
        os.path.join(base_image_dir, "armor_test.jpg"),
        os.path.join(results_dir, "task4")
    )

    # 进阶任务二：鲁棒性测试
    print("\n[进阶任务二] 算法鲁棒性测试...")
    robustness_test_pipeline(base_image_dir, os.path.join(results_dir, "task4"))

    print("\n" + "=" * 60)
    print("所有任务执行完成！")
    print(f"测试结果保存在：{results_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
