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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.utils import read_image, save_image, create_comparison_image


def _is_white_background(hsv):
    """判断是否为纯白/近白背景（如 shape_number_test.jpg）"""
    white_mask = ((hsv[:, :, 1] < 40) & (hsv[:, :, 2] > 200)).astype(np.uint8) * 255
    total = hsv.shape[0] * hsv.shape[1]
    return cv2.countNonZero(white_mask) / total > 0.6


def _preprocess_hsv(image):
    """基于 HSV 饱和度/亮度阈值分离彩色/深色形状与白色背景"""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # 彩色形状饱和度较高；深色形状亮度较低
    mask = ((hsv[:, :, 1] > 20) | (hsv[:, :, 2] < 200)).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    return mask


def _preprocess_canny(image):
    """基于多通道 Canny 边缘检测，适用于非白色背景"""
    edges = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    for i in range(3):
        blurred = cv2.GaussianBlur(image[:, :, i], (5, 5), 0)
        edges = cv2.bitwise_or(edges, cv2.Canny(blurred, 30, 120))

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edges = cv2.dilate(edges, kernel, iterations=2)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(edges)
    cv2.drawContours(filled, contours, -1, 255, -1)
    filled = cv2.erode(filled, np.ones((3, 3), np.uint8), iterations=1)
    return filled


def preprocess_for_contours(image):
    """
    对图像进行预处理，分离前景形状与背景。
    若背景为白色，使用 HSV 饱和度阈值；否则使用多通道 Canny 边缘检测并填充。
    :param image: BGR彩色图像
    :return: 二值化后的实心形状图
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    if _is_white_background(hsv):
        return _preprocess_hsv(image)
    return _preprocess_canny(image)


def detect_shapes(image, min_area=150, shape_region_ratio=1.0):
    """
    基于轮廓近似识别三角形、正方形、矩形、五边形、六边形、圆形、椭圆
    仅检测图片上方区域（避免下方数字干扰），默认检测范围为全图
    :param image: BGR彩色图像
    :param min_area: 最小有效轮廓面积
    :param shape_region_ratio: 形状区域占图片高度的比例（上方部分），默认1.0表示全图
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

        # 计算轮廓周长，使用原始轮廓保留凹顶点（如五边形的侧边）
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.015 * peri, True)
        vertices = len(approx)

        # 计算外接矩形、中心点
        x, y, bw, bh = cv2.boundingRect(cnt)
        center_x, center_y = x + bw // 2, y + bh // 2
        # 计算长宽比（使用外接矩形，长边/短边，始终>=1）
        aspect_ratio = max(bw, bh) / min(bw, bh) if min(bw, bh) > 0 else 1.0

        # 圆度判断：面积与周长关系
        circularity = 4 * np.pi * area / (peri * peri) if peri > 0 else 0

        # 拟合椭圆的长短轴之比：对倾斜椭圆比圆度/长宽比更鲁棒
        fit_axes_ratio = 1.0
        if len(cnt) >= 5:
            try:
                (_, _), (minor, major), _ = cv2.fitEllipse(cnt)
                fit_axes_ratio = major / minor if minor > 0 else 1.0
            except cv2.error:
                fit_axes_ratio = 1.0

        # 基于顶点数、圆度和长宽比的简洁分类规则
        shape_name = "unknown"
        if vertices == 3:
            shape_name = "triangle"
        elif vertices == 4:
            # 接近正方形的长宽比阈值放宽到1.2，适应轻微透视畸变
            shape_name = "square" if aspect_ratio < 1.2 else "rectangle"
        elif vertices == 5:
            shape_name = "pentagon"
        elif vertices == 6:
            shape_name = "hexagon"
        elif vertices >= 7:
            # 多顶点图形：先按圆度区分圆，再按长短轴比区分椭圆
            if circularity > 0.7 and aspect_ratio < 1.3:
                shape_name = "circle"
            elif fit_axes_ratio > 1.15 or aspect_ratio > 1.25:
                shape_name = "ellipse"
            else:
                shape_name = "hexagon"

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
            'ellipse': (128, 0, 128),
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


def shape_recognition_pipeline(image_path, output_dir, min_area=150, shape_region_ratio=1.0):
    """
    几何图形识别主流程
    :param image_path: 输入图片路径
    :param output_dir: 结果输出目录
    :param min_area: 最小有效面积
    :param shape_region_ratio: 形状检测区域占图片高度的比例
    """
    original = read_image(image_path)
    if original is None:
        return

    result, shapes = detect_shapes(original, min_area, shape_region_ratio)
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
    # 基于当前文件位置计算项目根目录，支持从任意目录运行
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    input_image = os.path.join(project_root, "test_images", "original", "images", "shape_number_test.jpg")
    output_dir = os.path.join(project_root, "test_images", "results", "task3")
    # shape_number_test 下方有数字，限制检测区域避免误识别
    shape_recognition_pipeline(input_image, output_dir, shape_region_ratio=0.82)
