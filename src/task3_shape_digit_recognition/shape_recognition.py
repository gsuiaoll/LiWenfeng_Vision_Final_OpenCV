"""
任务三（上）：简单几何图形识别
实现功能：
1. 识别图片中的矩形、圆形等基础几何图形
2. 在图形上标注识别结果
3. 输出原图与识别结果对比图
"""

import cv2
import os
import sys
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.utils import read_image, save_image, create_comparison_image


def preprocess_for_contours(image):
    """
    对图像进行预处理，便于后续轮廓检测
    :param image: BGR彩色图像
    :return: 边缘检测后的二值图
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 120)
    # 膨胀连接断裂边缘
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.dilate(edges, kernel, iterations=1)
    return edges


def detect_shapes(image, min_area=150, shape_region_ratio=0.75):
    """
    基于轮廓近似识别矩形、三角形、圆形、多边形
    仅检测图片上方区域（避免下方数字干扰），扩大检测范围至75%
    :param image: BGR彩色图像
    :param min_area: 最小有效轮廓面积
    :param shape_region_ratio: 形状区域占图片高度的比例（上方部分）
    :return: 标注后的图像和图形信息列表
    """
    h, w = image.shape[:2]
    # 仅在图片上方区域检测形状（避免下方数字干扰）
    shape_region = image[:int(h * shape_region_ratio), :]

    edges = preprocess_for_contours(shape_region)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    result = image.copy()
    shapes = []

    # 按面积从大到小排序，先处理大轮廓
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for i, cnt in enumerate(sorted_contours):
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue

        # 计算轮廓周长
        peri = cv2.arcLength(cnt, True)
        # 多边形近似
        approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
        vertices = len(approx)

        # 计算外接矩形、长宽比
        x, y, bw, bh = cv2.boundingRect(approx)
        aspect_ratio = float(bw) / bh if bh > 0 else 0
        center_x, center_y = x + bw // 2, y + bh // 2

        # 圆度判断：面积与周长关系
        circularity = 4 * np.pi * area / (peri * peri) if peri > 0 else 0

        shape_name = "unknown"
        if vertices == 3:
            shape_name = "triangle"
        elif vertices == 4:
            shape_name = "square" if 0.9 <= aspect_ratio <= 1.1 else "rectangle"
        elif vertices == 5:
            shape_name = "pentagon"
        elif vertices == 6:
            shape_name = "hexagon"
        elif vertices >= 7:
            if circularity > 0.7:
                shape_name = "circle"
            else:
                shape_name = "polygon"

        shapes.append({
            'id': len(shapes) + 1,
            'shape': shape_name,
            'vertices': vertices,
            'area': int(area),
            'center': (center_x, center_y),
            'circularity': round(circularity, 3)
        })

        # 绘制识别结果（使用不同颜色区分形状类型）
        color_map = {
            'triangle': (0, 255, 0),
            'square': (255, 255, 0),
            'rectangle': (255, 165, 0),
            'pentagon': (0, 255, 255),
            'hexagon': (255, 0, 255),
            'circle': (0, 255, 0),
            'polygon': (128, 128, 128)
        }
        color = color_map.get(shape_name, (0, 255, 0))
        cv2.drawContours(result, [approx], -1, color, 2)
        cv2.circle(result, (center_x, center_y), 5, (0, 0, 255), -1)

        # 标签放在图形上方，增加深色背景条确保文字在任何背景色上都清晰
        label = f"{shape_name}"
        label_font = cv2.FONT_HERSHEY_SIMPLEX
        label_size = cv2.getTextSize(label, label_font, 0.5, 2)[0]
        label_x = max(5, center_x - label_size[0] // 2)
        label_y = max(20, y - 5)
        # 绘制深色背景条
        bg_x1 = max(0, label_x - 3)
        bg_y1 = max(0, label_y - label_size[1] - 5)
        bg_x2 = min(result.shape[1], label_x + label_size[0] + 3)
        bg_y2 = min(result.shape[0], label_y + 5)
        cv2.rectangle(result, (bg_x1, bg_y1), (bg_x2, bg_y2), (30, 30, 30), -1)
        cv2.putText(result, label, (label_x, label_y),
                    label_font, 0.5, color, 2)

    return result, shapes


def shape_recognition_pipeline(image_path, output_dir, min_area=300):
    """
    几何图形识别主流程
    :param image_path: 输入图片路径
    :param output_dir: 结果输出目录
    :param min_area: 最小有效面积
    """
    original = read_image(image_path)
    if original is None:
        return

    result, shapes = detect_shapes(original, min_area)
    base_name = os.path.splitext(os.path.basename(image_path))[0]

    save_image(os.path.join(output_dir, f"{base_name}_edges.jpg"), preprocess_for_contours(original))
    save_image(os.path.join(output_dir, f"{base_name}_shapes.jpg"), result)

    comparison = create_comparison_image(
        [original, result],
        ["Original", "Shape Recognition"],
        cols=2
    )
    save_image(os.path.join(output_dir, f"{base_name}_shape_comparison.jpg"), comparison)

    print(f"\n===== {base_name} 几何图形识别结果 =====")
    print(f"共识别到 {len(shapes)} 个图形：")
    for s in shapes:
        print(f"  #{s['id']} {s['shape']} 顶点数={s['vertices']} 面积={s['area']} 中心={s['center']}")


if __name__ == "__main__":
    input_image = os.path.join("test_images", "original", "images", "shape_number_test.jpg")
    output_dir = os.path.join("test_images", "results", "task3")
    shape_recognition_pipeline(input_image, output_dir)
