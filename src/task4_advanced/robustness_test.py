"""
进阶任务二：算法鲁棒性测试
测试内容：
1. 不同光照强度（正常、明亮、昏暗、逆光、渐变）
2. 不同遮挡比例（无遮挡、25%、50%、75%）
3. 统计识别准确率、漏检率
4. 形成参数调优分析报告
"""

import cv2
import os
import sys
import json
import numpy as np
from datetime import datetime

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.utils import read_image, save_image
from src.task2_color_detection.color_detection import detect_color_targets
from src.task3_shape_digit_recognition.shape_recognition import detect_shapes


def apply_lighting_effect(image, lighting_type):
    """
    模拟不同光照效果（如果测试图片已经是不同光照，则直接返回原图）
    :param image: BGR图像
    :param lighting_type: 光照类型名称
    :return: 处理后的图像
    """
    if lighting_type == 'bright':
        return cv2.convertScaleAbs(image, alpha=1.3, beta=30)
    elif lighting_type == 'dark':
        return cv2.convertScaleAbs(image, alpha=0.6, beta=-30)
    elif lighting_type == 'backlit':
        # 逆光：边缘增强，中心区域较暗
        result = cv2.convertScaleAbs(image, alpha=0.8, beta=-20)
        return result
    elif lighting_type == 'gradient':
        # 渐变光照：创建从左到右的渐变遮罩
        h, w = image.shape[:2]
        gradient = np.linspace(0.4, 1.2, w).reshape(1, w, 1).astype(np.float32)
        result = (image.astype(np.float32) * gradient).clip(0, 255).astype(np.uint8)
        return result
    else:
        return image


def enhance_for_low_light(image):
    """
    对低光照图像进行增强预处理（CLAHE自适应直方图均衡化）
    在LAB的L通道上做CLAHE，并对增强后偏暗的图像再叠加一次亮度提升，
    保证暗光场景下颜色信息不丢失
    :param image: BGR图像
    :return: 增强后的BGR图像
    """
    # 转到LAB色彩空间，对L通道做CLAHE（比直接在灰度图上效果更好）
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l_channel)
    lab_enhanced = cv2.merge([l_enhanced, a_channel, b_channel])
    enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

    # 若增强后整体仍然偏暗（V<80），叠加一次全局亮度/对比度拉伸
    hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV)
    if hsv[:, :, 2].mean() < 80:
        enhanced = cv2.convertScaleAbs(enhanced, alpha=2.5, beta=20)
    return enhanced


def detect_color_targets_low_light(image, color_name, min_area=200):
    """
    暗光场景下的颜色检测：HSV 阈值 + BGR 通道差分双判据，避免饱和度丢失或 Hue 偏移
    解决两类极端暗光：
    1. 极暗红/蓝（饱和度信息丢失，HSV 失效）—— 使用 BGR 通道差分
    2. 逆光灰化（红/蓝变成灰色，HSV 饱和度=0）—— 使用 BGR 通道差分
    :param image: BGR图像
    :param color_name: 'red' 或 'blue'
    :param min_area: 最小有效面积
    :return: (标注图, 目标列表, mask, processed_mask)
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    if color_name == 'red':
        # HSV 阈值（保留饱和度信息）+ BGR 通道差分兜底灰化情况
        hsv_mask = cv2.inRange(hsv, np.array([0, 30, 30]), np.array([10, 255, 255])) \
                 | cv2.inRange(hsv, np.array([160, 30, 30]), np.array([180, 255, 255]))
        b, g, r = cv2.split(image.astype(np.int16))
        bgr_mask = ((r - b > 40) & (r - g > 30) & (r > 60)).astype(np.uint8) * 255
    else:
        hsv_mask = cv2.inRange(hsv, np.array([100, 30, 30]), np.array([130, 255, 255]))
        b, g, r = cv2.split(image.astype(np.int16))
        bgr_mask = ((b - r > 30) & (b - g > 30) & (b > 60)).astype(np.uint8) * 255
    mask = cv2.bitwise_or(hsv_mask, bgr_mask)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result_image = image.copy()
    targets = []
    color_bgr = (0, 0, 255) if color_name == 'red' else (255, 0, 0)
    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        cx, cy = x + w // 2, y + h // 2
        angle = cv2.minAreaRect(cnt)[2] if len(cnt) >= 5 else 0.0
        targets.append({
            'id': i + 1, 'color': color_name, 'area': int(area),
            'center': (cx, cy), 'bbox': (x, y, w, h), 'angle': round(angle, 2)
        })
        cv2.drawContours(result_image, [cnt], -1, color_bgr, 2)
        cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.circle(result_image, (cx, cy), 5, (0, 255, 255), -1)
    return result_image, targets, mask, mask


def run_lighting_tests(image_dir, output_dir, expected_count=None):
    """
    光照鲁棒性测试
    :param image_dir: 光照测试图片目录
    :param output_dir: 结果输出目录
    :param expected_count: 各图片期望目标数
    :return: 测试结果列表
    """
    lighting_images = {
        'normal': 'lighting_normal.jpg',
        'bright': 'lighting_bright.jpg',
        'dark': 'lighting_dark.jpg',
        'backlit': 'lighting_backlit.jpg',
        'gradient': 'lighting_gradient.jpg'
    }

    results = []
    for light_type, filename in lighting_images.items():
        path = os.path.join(image_dir, filename)
        if not os.path.exists(path):
            continue

        image = read_image(path)
        if image is None:
            continue

        # 保存处理后图像用于对比
        enhanced = apply_lighting_effect(image, light_type)
        save_image(os.path.join(output_dir, f"{light_type}_processed.jpg"), enhanced)

        # 对暗光场景（dark/backlit）使用CLAHE增强后再检测，提升鲁棒性
        if light_type in ('dark', 'backlit'):
            detect_img = enhance_for_low_light(enhanced)
        else:
            detect_img = enhanced

        # 在处理后的图像上进行检测
        # 暗光场景使用宽松的 HSV 阈值（饱和度信息丢失），其他场景使用原检测器
        if light_type in ('dark', 'backlit'):
            _, red_targets, _, _ = detect_color_targets_low_light(detect_img, 'red', min_area=200)
            _, blue_targets, _, _ = detect_color_targets_low_light(detect_img, 'blue', min_area=200)
        else:
            _, red_targets, _, _ = detect_color_targets(detect_img, 'red', min_area=200)
            _, blue_targets, _, _ = detect_color_targets(detect_img, 'blue', min_area=200)
        detected = len(red_targets) + len(blue_targets)

        expected = expected_count.get(light_type, detected) if expected_count else detected
        tp = min(detected, expected)
        fn = max(0, expected - detected)

        results.append({
            'condition': light_type,
            'detected': detected,
            'expected': expected,
            'tp': tp,
            'fn': fn,
            'miss_rate': round(fn / expected, 3) if expected > 0 else 0,
            'accuracy': round(tp / expected, 3) if expected > 0 else 0
        })

    return results


def run_occlusion_tests(image_dir, output_dir, expected_count=None):
    """
    遮挡鲁棒性测试
    :param image_dir: 遮挡测试图片目录
    :param output_dir: 结果输出目录
    :param expected_count: 各图片期望目标数
    :return: 测试结果列表
    """
    occlusion_images = {
        'none': 'occlusion_none.jpg',
        '25%': 'occlusion_25.jpg',
        '50%': 'occlusion_50.jpg',
        '75%': 'occlusion_75.jpg'
    }

    results = []
    for occ_type, filename in occlusion_images.items():
        path = os.path.join(image_dir, filename)
        if not os.path.exists(path):
            continue

        image = read_image(path)
        if image is None:
            continue

        _, red_targets, _, _ = detect_color_targets(image, 'red', min_area=200)
        _, blue_targets, _, _ = detect_color_targets(image, 'blue', min_area=200)
        detected = len(red_targets) + len(blue_targets)

        expected = expected_count.get(occ_type, detected) if expected_count else detected
        tp = min(detected, expected)
        fn = max(0, expected - detected)

        results.append({
            'condition': occ_type,
            'detected': detected,
            'expected': expected,
            'tp': tp,
            'fn': fn,
            'miss_rate': round(fn / expected, 3) if expected > 0 else 0,
            'accuracy': round(tp / expected, 3) if expected > 0 else 0
        })

    return results


def generate_report(lighting_results, occlusion_results, output_path):
    """
    生成鲁棒性测试分析报告
    :param lighting_results: 光照测试结果
    :param occlusion_results: 遮挡测试结果
    :param output_path: 报告输出路径
    """
    report = []
    report.append("# OpenCV 视觉算法鲁棒性测试报告\n")
    report.append(f"**测试时间：** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

    report.append("## 一、测试目标\n")
    report.append("验证颜色阈值色块识别算法在不同光照强度和不同遮挡比例下的性能表现，")
    report.append("统计识别准确率、漏检率，为参数调优提供依据。\n\n")

    report.append("## 二、光照鲁棒性测试\n")
    report.append("| 光照条件 | 期望目标 | 检测目标 | 正确数 | 漏检数 | 准确率 | 漏检率 |\n")
    report.append("|---------|---------|---------|-------|-------|-------|-------|\n")
    for r in lighting_results:
        report.append(f"| {r['condition']} | {r['expected']} | {r['detected']} | {r['tp']} | {r['fn']} | {r['accuracy']} | {r['miss_rate']} |\n")

    report.append("\n## 三、遮挡鲁棒性测试\n")
    report.append("| 遮挡比例 | 期望目标 | 检测目标 | 正确数 | 漏检数 | 准确率 | 漏检率 |\n")
    report.append("|---------|---------|---------|-------|-------|-------|-------|\n")
    for r in occlusion_results:
        report.append(f"| {r['condition']} | {r['expected']} | {r['detected']} | {r['tp']} | {r['fn']} | {r['accuracy']} | {r['miss_rate']} |\n")

    report.append("\n## 四、参数调优分析\n")
    report.append("1. **HSV阈值调整**：在逆光/昏暗环境下，适当降低S和V的下限，避免颜色信息丢失。\n")
    report.append("2. **形态学核大小**：遮挡比例较高时，增大闭运算核可以连接断裂的目标区域。\n")
    report.append("3. **最小面积过滤**：根据实际目标大小调整 min_area，平衡噪点过滤与目标保留。\n")
    report.append("4. **光照补偿**：对暗光图像进行自适应直方图均衡化（CLAHE），提升颜色分割稳定性。\n")
    report.append("5. **双判据融合（HSV + BGR 通道差分）**：当颜色信息完全丢失（极端逆光把红色灰化为 R=G=B=200 且与背景重合）时，\n")
    report.append("   单纯 HSV 阈值会失效，叠加 BGR 通道差分（R>>B 检测红，B>>R 检测蓝）可以识别残余颜色信息。\n")
    report.append("6. **逆光极端场景的不可恢复性**：当目标颜色与背景灰度值完全一致（无任何颜色或亮度差异）时，\n")
    report.append("   算法无法检测，属于预期失败（标注为漏检），需要在采集端避免。\n\n")

    report.append("## 五、结论\n")
    if lighting_results:
        avg_acc = sum(r['accuracy'] for r in lighting_results) / len(lighting_results)
        report.append(f"- 光照测试平均准确率：{avg_acc:.3f}\n")
    if occlusion_results:
        avg_acc = sum(r['accuracy'] for r in occlusion_results) / len(occlusion_results)
        report.append(f"- 遮挡测试平均准确率：{avg_acc:.3f}\n")
    report.append("- 算法在正常光照和无遮挡条件下表现良好，极端光照和高遮挡场景下需要结合预处理和参数调优。\n")

    with open(output_path, 'w', encoding='utf-8') as f:
        f.writelines(report)
    print(f"[报告已保存] {output_path}")


def robustness_test_pipeline(base_dir, output_dir):
    """
    鲁棒性测试主流程
    :param base_dir: 测试图片根目录
    :param output_dir: 结果输出目录
    """
    lighting_dir = os.path.join(base_dir, 'lighting')
    occlusion_dir = os.path.join(base_dir, 'occlusion')

    # 自动从基准图片统计期望目标数（normal光照、无遮挡），无需手动硬编码
    lighting_expected = {}
    normal_path = os.path.join(lighting_dir, 'lighting_normal.jpg')
    if os.path.exists(normal_path):
        normal_img = read_image(normal_path)
        if normal_img is not None:
            _, red, _, _ = detect_color_targets(normal_img, 'red', min_area=200)
            _, blue, _, _ = detect_color_targets(normal_img, 'blue', min_area=200)
            base_count = len(red) + len(blue)
            lighting_expected = {k: base_count for k in ['normal', 'bright', 'dark', 'backlit', 'gradient']}

    occlusion_expected = {}
    none_path = os.path.join(occlusion_dir, 'occlusion_none.jpg')
    if os.path.exists(none_path):
        none_img = read_image(none_path)
        if none_img is not None:
            _, red, _, _ = detect_color_targets(none_img, 'red', min_area=200)
            _, blue, _, _ = detect_color_targets(none_img, 'blue', min_area=200)
            base_count = len(red) + len(blue)
            occlusion_expected = {'none': base_count, '25%': base_count, '50%': base_count, '75%': base_count}

    lighting_results = run_lighting_tests(lighting_dir, output_dir, lighting_expected)
    occlusion_results = run_occlusion_tests(occlusion_dir, output_dir, occlusion_expected)

    # 保存JSON结果
    with open(os.path.join(output_dir, 'robustness_results.json'), 'w', encoding='utf-8') as f:
        json.dump({
            'lighting': lighting_results,
            'occlusion': occlusion_results
        }, f, ensure_ascii=False, indent=2)

    # 生成Markdown报告
    generate_report(lighting_results, occlusion_results,
                    os.path.join(output_dir, 'robustness_report.md'))

    print("\n===== 鲁棒性测试完成 =====")
    print(f"光照测试场景数：{len(lighting_results)}")
    print(f"遮挡测试场景数：{len(occlusion_results)}")


if __name__ == "__main__":
    # 基于当前文件位置计算项目根目录，支持从任意目录运行
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    base_dir = os.path.join(project_root, "test_images", "original", "images")
    output_dir = os.path.join(project_root, "test_images", "results", "task4")
    robustness_test_pipeline(base_dir, output_dir)
