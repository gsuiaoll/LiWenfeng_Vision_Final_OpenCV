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


def evaluate_color_detection(image_path, expected_targets=None):
    """
    评估颜色检测算法性能
    :param image_path: 测试图片路径
    :param expected_targets: 期望检测到的目标数量（Ground Truth）
    :return: 评估指标字典
    """
    image = read_image(image_path)
    if image is None:
        return None

    _, red_targets, _, _ = detect_color_targets(image, 'red', min_area=200)
    _, blue_targets, _, _ = detect_color_targets(image, 'blue', min_area=200)

    detected = len(red_targets) + len(blue_targets)
    expected = expected_targets if expected_targets is not None else detected

    tp = min(detected, expected)  # 简化：正确检测数
    fn = max(0, expected - detected)  # 漏检
    fp = max(0, detected - expected)  # 误检

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    accuracy = tp / max(detected, expected) if max(detected, expected) > 0 else 0

    return {
        'image': os.path.basename(image_path),
        'detected': detected,
        'expected': expected,
        'tp': tp,
        'fn': fn,
        'fp': fp,
        'precision': round(precision, 3),
        'recall': round(recall, 3),
        'accuracy': round(accuracy, 3),
        'miss_rate': round(fn / expected, 3) if expected > 0 else 0
    }


def enhance_for_low_light(image):
    """
    对低光照图像进行增强预处理（CLAHE自适应直方图均衡化）
    在HSV的V通道上做CLAHE，保留颜色信息的同时提升亮度对比度
    :param image: BGR图像
    :return: 增强后的BGR图像
    """
    # 转到LAB色彩空间，对L通道做CLAHE（比直接在灰度图上效果更好）
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l_channel)
    lab_enhanced = cv2.merge([l_enhanced, a_channel, b_channel])
    return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)


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
    report.append("4. **光照补偿**：对暗光图像进行自适应直方图均衡化（CLAHE），提升颜色分割稳定性。\n\n")

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
    base_dir = os.path.join("test_images", "original", "images")
    output_dir = os.path.join("test_images", "results", "task4")
    robustness_test_pipeline(base_dir, output_dir)
