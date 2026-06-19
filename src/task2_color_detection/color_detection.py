"""
任务二：颜色阈值色块识别
实现功能：
1. 基于HSV颜色空间实现红/蓝色目标分割
2. 使用形态学腐蚀、膨胀去除噪点
3. 筛选有效目标轮廓并标注坐标、面积信息
4. 输出处理过程和最终识别结果对比图
"""

import cv2
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.utils import read_image, save_image, create_comparison_image


# HSV颜色空间中红色和蓝色的阈值范围
# 参考：OpenCV官方HSV颜色空间文档 https://docs.opencv.org/4.x/df/d9d/tutorial_py_colorspaces.html
COLOR_RANGES = {
    'red': [
        # 红色在HSV中跨越0度/360度边界，需要两段范围
        {'lower': np.array([0, 120, 100]), 'upper': np.array([10, 255, 255])},
        {'lower': np.array([160, 120, 100]), 'upper': np.array([180, 255, 255])}
    ],
    'blue': [
        {'lower': np.array([100, 120, 100]), 'upper': np.array([130, 255, 255])}
    ]
}


def hsv_color_threshold(image, color_name):
    """
    在HSV空间中对指定颜色进行阈值分割
    :param image: BGR格式彩色图像
    :param color_name: 颜色名称 'red' 或 'blue'
    :return: 二值化掩码图，目标区域为白色255
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for rng in COLOR_RANGES.get(color_name, []):
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, rng['lower'], rng['upper']))
    return mask


def morphological_process(mask, kernel_size=5, iterations=2):
    """
    形态学处理：先腐蚀去除小噪点，再膨胀恢复目标区域
    :param mask: 二值掩码图
    :param kernel_size: 结构元素大小
    :param iterations: 迭代次数
    :return: 处理后的掩码图
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    # 开运算：先腐蚀后膨胀，去除噪点并平滑边界
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=iterations)
    # 闭运算：先膨胀后腐蚀，填补小孔洞
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=iterations)
    return closed


def detect_color_targets(image, color_name, min_area=200):
    """
    检测图像中指定颜色的目标，返回标注后的图像和目标信息
    :param image: BGR格式彩色图像
    :param color_name: 'red' 或 'blue'
    :param min_area: 最小有效面积，过滤噪点
    :return: (标注图, 目标信息列表)
    """
    mask = hsv_color_threshold(image, color_name)
    processed_mask = morphological_process(mask)

    # 查找轮廓
    contours, _ = cv2.findContours(processed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    result_image = image.copy()
    targets = []
    color_bgr = (0, 0, 255) if color_name == 'red' else (255, 0, 0)

    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue

        # 计算外接矩形和最小外接矩形
        x, y, w, h = cv2.boundingRect(cnt)
        center_x, center_y = x + w // 2, y + h // 2

        # 拟合旋转矩形，获取角度信息
        if len(cnt) >= 5:
            rect = cv2.minAreaRect(cnt)
            angle = rect[2]
        else:
            angle = 0.0

        targets.append({
            'id': len(targets) + 1,
            'color': color_name,
            'area': int(area),
            'center': (center_x, center_y),
            'bbox': (x, y, w, h),
            'angle': round(angle, 2)
        })

        # 绘制轮廓、外接矩形和中心点
        cv2.drawContours(result_image, [cnt], -1, color_bgr, 2)
        cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.circle(result_image, (center_x, center_y), 5, (0, 255, 255), -1)

        # 标注信息（增加深色背景条，确保文字在任何背景色上都清晰可读）
        h_img, w_img = result_image.shape[:2]
        label = f"{color_name.upper()}#{targets[-1]['id']} A:{int(area)} C:({center_x},{center_y})"
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size, _ = cv2.getTextSize(label, font, 0.45, 2)
        tw, th = text_size
        # 默认放在外接矩形上方
        label_x = max(2, x)
        label_y = max(th + 5, y - 5)
        # 如果上方空间不足，则放在矩形内部
        if label_y < th + 10:
            label_y = y + th + 5
        # 避免超出右边界
        if label_x + tw > w_img:
            label_x = max(2, w_img - tw - 2)
        # 绘制深色背景条（确保文字在任何背景色上都清晰）
        bg_x1 = max(0, label_x - 3)
        bg_y1 = max(0, label_y - th - 5)
        bg_x2 = min(w_img, label_x + tw + 3)
        bg_y2 = min(h_img, label_y + 5)
        cv2.rectangle(result_image, (bg_x1, bg_y1), (bg_x2, bg_y2), (30, 30, 30), -1)
        cv2.putText(result_image, label, (label_x, label_y),
                    font, 0.45, color_bgr, 2)

    return result_image, targets, mask, processed_mask


def color_detection_pipeline(image_path, output_dir, min_area=200):
    """
    颜色检测主流程：分别识别红色和蓝色目标，保存处理过程和结果
    :param image_path: 输入图片路径
    :param output_dir: 结果输出目录
    :param min_area: 最小有效面积
    """
    original = read_image(image_path)
    if original is None:
        return

    red_result, red_targets, red_mask, red_processed = detect_color_targets(original, 'red', min_area)
    blue_result, blue_targets, blue_mask, blue_processed = detect_color_targets(original, 'blue', min_area)

    # 合并红蓝识别结果
    combined = original.copy()
    cv2.addWeighted(red_result, 0.5, blue_result, 0.5, 0, combined)
    combined = cv2.addWeighted(original, 0.3, combined, 0.7, 0)

    base_name = os.path.splitext(os.path.basename(image_path))[0]

    # 保存中间过程和最终结果
    save_image(os.path.join(output_dir, f"{base_name}_red_mask.jpg"), red_mask)
    save_image(os.path.join(output_dir, f"{base_name}_red_processed.jpg"), red_processed)
    save_image(os.path.join(output_dir, f"{base_name}_red_result.jpg"), red_result)
    save_image(os.path.join(output_dir, f"{base_name}_blue_mask.jpg"), blue_mask)
    save_image(os.path.join(output_dir, f"{base_name}_blue_processed.jpg"), blue_processed)
    save_image(os.path.join(output_dir, f"{base_name}_blue_result.jpg"), blue_result)
    save_image(os.path.join(output_dir, f"{base_name}_combined.jpg"), combined)

    # 生成红蓝检测对比图
    red_comparison = create_comparison_image(
        [original, red_mask, red_processed, red_result],
        ["Original", "Red Mask", "Red Morphology", "Red Result"],
        cols=2
    )
    save_image(os.path.join(output_dir, f"{base_name}_red_comparison.jpg"), red_comparison)

    blue_comparison = create_comparison_image(
        [original, blue_mask, blue_processed, blue_result],
        ["Original", "Blue Mask", "Blue Morphology", "Blue Result"],
        cols=2
    )
    save_image(os.path.join(output_dir, f"{base_name}_blue_comparison.jpg"), blue_comparison)

    # 打印目标信息
    print(f"\n===== {base_name} 颜色检测结果 =====")
    print(f"红色目标数量：{len(red_targets)}")
    for t in red_targets:
        print(f"  #{t['id']} 面积={t['area']} 中心={t['center']} 角度={t['angle']}°")
    print(f"蓝色目标数量：{len(blue_targets)}")
    for t in blue_targets:
        print(f"  #{t['id']} 面积={t['area']} 中心={t['center']} 角度={t['angle']}°")


if __name__ == "__main__":
    # 基于当前文件位置计算项目根目录，支持从任意目录运行
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    input_image = os.path.join(project_root, "test_images", "original", "images", "color_test.jpg")
    output_dir = os.path.join(project_root, "test_images", "results", "task2")
    color_detection_pipeline(input_image, output_dir)
