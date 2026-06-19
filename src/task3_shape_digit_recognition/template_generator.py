"""
数字模板生成器
从已知的0-9印刷体数字图片中自动提取模板，保存为本地模板文件
供后续数字识别模块加载使用，提高识别准确率
"""

import cv2
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.utils import read_image, save_image


def extract_digit_templates(image_path, output_dir, digit_region_ratio=0.6):
    """
    从包含0-9印刷体数字的图片中提取模板
    假设数字按0-9顺序从左到右排列在图片下方区域
    :param image_path: 输入图片路径
    :param output_dir: 模板输出目录
    :param digit_region_ratio: 数字区域位于图片下方的比例
    :return: 是否成功提取
    """
    image = read_image(image_path)
    if image is None:
        return False

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    region_y = int(h * digit_region_ratio)
    digit_region = gray[region_y:h, :]

    # 二值化
    blurred = cv2.GaussianBlur(digit_region, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 闭运算连接断裂笔画
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 过滤并排序候选轮廓
    candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 80 or area > 3000:
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        aspect_ratio = float(bw) / bh
        if not (0.15 <= aspect_ratio <= 1.0):
            continue
        candidates.append((x, cnt, (x, y, bw, bh)))

    # 按x坐标排序
    candidates.sort(key=lambda item: item[0])

    if len(candidates) < 10:
        print(f"[警告] 仅提取到 {len(candidates)} 个数字候选，需要10个")
        return False

    # 取前10个作为0-9模板
    os.makedirs(output_dir, exist_ok=True)
    for i, (_, cnt, (x, y, bw, bh)) in enumerate(candidates[:10]):
        roi = binary[y:y+bh, x:x+bw]
        # 归一化到28x28画布中央
        template = np.zeros((28, 28), dtype=np.uint8)
        scale = min(24.0 / bw, 24.0 / bh)
        new_w, new_h = int(bw * scale), int(bh * scale)
        resized = cv2.resize(roi, (new_w, new_h), interpolation=cv2.INTER_AREA)
        y_off = (28 - new_h) // 2
        x_off = (28 - new_w) // 2
        template[y_off:y_off+new_h, x_off:x_off+new_w] = resized
        save_image(os.path.join(output_dir, f"{i}.png"), template)
        print(f"[模板生成] 数字 {i} 模板已保存")

    return True


if __name__ == "__main__":
    # 基于当前文件位置计算项目根目录，支持从任意目录运行
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    input_image = os.path.join(project_root, "test_images", "original", "images", "shape_number_test.jpg")
    output_dir = os.path.join(project_root, "src", "task3_shape_digit_recognition", "templates")
    extract_digit_templates(input_image, output_dir)
