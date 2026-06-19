"""
任务三（下）：印刷体数字识别（0-9）
实现方式：基于轮廓匹配/模板匹配，不使用OCR库

核心思路：
1. 生成或加载标准数字模板
2. 对输入图像中的数字区域进行二值化和轮廓提取
3. 将每个候选数字区域与模板进行匹配
4. 输出识别结果
"""

import cv2
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.utils import read_image, save_image, create_comparison_image


def load_digit_templates(template_dir=None):
    """
    加载数字模板
    优先从本地模板目录加载（如果存在），否则生成默认Hershey字体模板
    :param template_dir: 本地模板目录路径
    :return: {数字字符串: (模板二值图, 模板轮廓)}
    """
    if template_dir is None:
        # 默认模板目录位于本文件同级目录下的 templates 文件夹
        template_dir = os.path.join(os.path.dirname(__file__), 'templates')

    templates = {}
    if os.path.exists(template_dir):
        for digit in range(10):
            template_path = os.path.join(template_dir, f"{digit}.png")
            if os.path.exists(template_path):
                # OpenCV的cv2.imread对中文路径支持不佳，使用numpy读取字节流解码
                binary = cv2.imdecode(np.fromfile(template_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
                if binary is None:
                    continue
                contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cnt = max(contours, key=cv2.contourArea) if contours else None
                templates[str(digit)] = (binary, cnt)

    if len(templates) == 10:
        print(f"[模板加载] 已从 {template_dir} 加载10个数字模板")
        return templates

    # 如果本地模板不完整，生成默认Hershey字体模板
    print("[模板生成] 本地模板不存在或不完整，使用默认Hershey字体模板")
    return generate_default_templates()


def generate_default_templates(size=(28, 28)):
    """
    生成默认Hershey字体数字0-9模板
    :param size: 模板图像尺寸
    :return: {数字字符串: (模板二值图, 模板轮廓)}
    """
    templates = {}
    for digit in range(10):
        img = np.ones((size[1], size[0]), dtype=np.uint8) * 255
        text = str(digit)
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.9
        thickness = 2
        text_size = cv2.getTextSize(text, font, scale, thickness)[0]
        x = (size[0] - text_size[0]) // 2
        y = (size[1] + text_size[1]) // 2
        cv2.putText(img, text, (x, y), font, scale, 0, thickness)
        _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnt = max(contours, key=cv2.contourArea) if contours else None
        templates[text] = (binary, cnt)
    return templates


def preprocess_digit_roi(roi):
    """
    预处理数字ROI区域，便于轮廓匹配
    :param roi: 输入的数字区域图像（灰度或彩色）
    :return: 归一化尺寸后的二值图和对应轮廓
    """
    if len(roi.shape) == 3:
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    # 二值化
    _, binary = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # 去除边缘空白
    coords = cv2.findNonZero(binary)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        cropped = binary[y:y+h, x:x+w]
    else:
        cropped = binary

    # 保持长宽比，缩放到28x28画布中央
    canvas = np.zeros((28, 28), dtype=np.uint8)
    h, w = cropped.shape
    scale = min(24.0 / w, 24.0 / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_AREA)
    y_off = (28 - new_h) // 2
    x_off = (28 - new_w) // 2
    canvas[y_off:y_off+new_h, x_off:x_off+new_w] = resized

    # 提取轮廓
    contours, _ = cv2.findContours(canvas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = max(contours, key=cv2.contourArea) if contours else None
    return canvas, cnt


def match_digit(roi, cnt, templates, aspect_ratio=0.6):
    """
    综合模板匹配、Hu矩轮廓匹配和几何特征识别单个数字
    模板匹配对同字体数字准确率高，Hu矩辅助区分形近字，
    宽高比几何特征对“1”等瘦长数字有显著区分度
    :param roi: 预处理后的数字二值图
    :param cnt: 数字轮廓
    :param templates: 数字模板字典 {数字: (模板图, 模板轮廓)}
    :param aspect_ratio: 原始数字区域的宽高比（w/h）
    :return: 最佳匹配数字和匹配得分
    """
    # 各数字的典型宽高比，用于几何特征评分
    typical_ratios = {
        '0': 0.65, '1': 0.20, '2': 0.60, '3': 0.60, '4': 0.65,
        '5': 0.60, '6': 0.60, '7': 0.55, '8': 0.65, '9': 0.60
    }
    scores = {}

    # 1. 模板相关系数匹配（主要依据）
    for digit, (template_img, _) in templates.items():
        # 确保尺寸一致
        if roi.shape != template_img.shape:
            template_img = cv2.resize(template_img, (roi.shape[1], roi.shape[0]))
        result = cv2.matchTemplate(roi, template_img, cv2.TM_CCOEFF_NORMED)
        template_score = np.max(result)
        scores[digit] = {'template': template_score, 'shape': 0, 'geometry': 0}

    # 2. Hu矩形状匹配（辅助依据）
    if cnt is not None and len(cnt) >= 3:
        for digit, (_, template_cnt) in templates.items():
            if template_cnt is None or len(template_cnt) < 3:
                continue
            shape_distance = cv2.matchShapes(cnt, template_cnt, cv2.CONTOURS_MATCH_I1, 0.0)
            # 将距离转换为相似度（0-1之间）
            shape_score = max(0, min(1, 1.0 - shape_distance))
            scores[digit]['shape'] = shape_score

    # 3. 几何特征：利用原始宽高比辅助区分“1”等瘦长数字
    for digit in scores:
        ratio_diff = abs(aspect_ratio - typical_ratios.get(digit, 0.6))
        geometry_score = max(0, 1.0 - ratio_diff / 0.4)
        scores[digit]['geometry'] = geometry_score

    # 4. 综合评分：模板匹配55% + 形状匹配25% + 几何特征20%
    best_digit = '?'
    best_score = -1
    for digit, s in scores.items():
        combined = 0.55 * s['template'] + 0.25 * s['shape'] + 0.20 * s['geometry']
        if combined > best_score:
            best_score = combined
            best_digit = digit

    return best_digit, round(best_score, 3)


def detect_digits(image, min_area=80, max_area=2000, digit_region_ratio=0.6):
    """
    检测并识别图像中的印刷体数字
    :param image: BGR彩色图像
    :param min_area: 最小数字轮廓面积
    :param max_area: 最大数字轮廓面积
    :param digit_region_ratio: 数字区域位于图片下方的比例，默认下方60%为数字区域
    :return: 标注后的图像、识别结果列表、二值图
    """
    templates = load_digit_templates()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # 截取图片下方区域作为数字识别ROI（避免上方图形干扰）
    region_y = int(h * digit_region_ratio)
    digit_region_gray = gray[region_y:h, :]

    # 高斯模糊后使用Otsu自动二值化（背景为白，数字为黑）
    blurred = cv2.GaussianBlur(digit_region_gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 形态学闭运算：连接数字"1"等可能断裂的笔画
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    # 开运算去除细小噪点
    kernel_small = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_small, iterations=1)

    # 把彩色形状的内部镂空区域也填黑，避免形状内的浅色像素被误识为数字笔画
    # 限制：仅当形状 y 范围与数字 ROI 重叠时排除，避免误删真实数字
    bgr_region = image[region_y:h, :, :]
    hsv_region = cv2.cvtColor(bgr_region, cv2.COLOR_BGR2HSV)
    # 仅排除亮色非白色像素（彩色形状），不排除纯白背景
    colored_mask = ((hsv_region[:, :, 1] > 60) & (hsv_region[:, :, 2] > 100)).astype(np.uint8) * 255
    # 闭运算填实形状内部
    kernel_fill = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    colored_mask = cv2.morphologyEx(colored_mask, cv2.MORPH_CLOSE, kernel_fill, iterations=2)
    binary = cv2.bitwise_and(binary, cv2.bitwise_not(colored_mask))

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    result = image.copy()
    detections = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue

        x, y, bw, bh = cv2.boundingRect(cnt)
        # 转换回原图坐标
        original_y = y + region_y
        aspect_ratio = float(bw) / bh
        # 数字一般具有一定宽高比
        if not (0.15 <= aspect_ratio <= 1.0):
            continue

        roi = digit_region_gray[y:y+bh, x:x+bw]
        processed_roi, digit_cnt = preprocess_digit_roi(roi)
        digit, score = match_digit(processed_roi, digit_cnt, templates, aspect_ratio)

        detections.append({
            'digit': digit,
            'score': round(float(score), 3),
            'bbox': (x, original_y, bw, bh),
            'area': int(area)
        })

        # 绘制识别结果（使用原图坐标）
        cv2.rectangle(result, (x, original_y), (x+bw, original_y+bh), (0, 255, 0), 2)
        # 标签放在数字框内部底部，增加深色背景条确保清晰
        label = f"{digit}"
        label_font = cv2.FONT_HERSHEY_SIMPLEX
        label_size, _ = cv2.getTextSize(label, label_font, 0.5, 2)
        lw, lh = label_size
        label_x = x + 2
        label_y = original_y + bh - 5
        h_img, w_img = result.shape[:2]
        bg_x1 = max(0, label_x - 2)
        bg_y1 = max(0, label_y - lh - 2)
        bg_x2 = min(w_img, label_x + lw + 2)
        bg_y2 = min(h_img, label_y + 2)
        cv2.rectangle(result, (bg_x1, bg_y1), (bg_x2, bg_y2), (30, 30, 30), -1)
        cv2.putText(result, label, (label_x, label_y),
                    label_font, 0.5, (0, 255, 0), 2)

    # 按x坐标排序，模拟从左到右的阅读顺序
    detections.sort(key=lambda d: d['bbox'][0])

    # 还原完整尺寸的二值图（上方补黑）
    full_binary = np.zeros((h, w), dtype=np.uint8)
    full_binary[region_y:h, :] = binary

    return result, detections, full_binary


def digit_recognition_pipeline(image_path, output_dir, min_area=80, max_area=2000):
    """
    数字识别主流程
    :param image_path: 输入图片路径
    :param output_dir: 结果输出目录
    :param min_area: 最小数字面积
    :param max_area: 最大数字面积
    """
    original = read_image(image_path)
    if original is None:
        return

    result, detections, binary = detect_digits(original, min_area, max_area)
    base_name = os.path.splitext(os.path.basename(image_path))[0]

    save_image(os.path.join(output_dir, f"{base_name}_digit_binary.jpg"), binary)
    save_image(os.path.join(output_dir, f"{base_name}_digits.jpg"), result)

    comparison = create_comparison_image(
        [original, result],
        ["Original", "Digit Recognition"],
        cols=2
    )
    save_image(os.path.join(output_dir, f"{base_name}_digit_comparison.jpg"), comparison)

    recognized_text = ''.join([d['digit'] for d in detections])
    print(f"\n===== {base_name} 数字识别结果 =====")
    print(f"识别到的数字序列：{recognized_text}")
    print(f"共识别 {len(detections)} 个数字：")
    for d in detections:
        print(f"  数字={d['digit']} 置信度={d['score']} 位置={d['bbox']}")


if __name__ == "__main__":
    # 基于当前文件位置计算项目根目录，支持从任意目录运行
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    input_image = os.path.join(project_root, "test_images", "original", "images", "shape_number_test.jpg")
    output_dir = os.path.join(project_root, "test_images", "results", "task3")
    digit_recognition_pipeline(input_image, output_dir)
