"""
进阶任务一：多条件目标精定位（模拟装甲板目标）
实现功能：
1. 融合颜色阈值（红蓝灯条）、轮廓特征、角点检测
2. 识别模拟装甲板目标
3. 输出目标旋转角度、中心像素坐标等关键信息
"""

import cv2
import os
import sys
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.utils import read_image, save_image, create_comparison_image


# 装甲板常用颜色：红色和蓝色灯条
ARMOR_COLOR_RANGES = {
    'red': [
        {'lower': np.array([0, 120, 100]), 'upper': np.array([10, 255, 255])},
        {'lower': np.array([160, 120, 100]), 'upper': np.array([180, 255, 255])}
    ],
    'blue': [
        {'lower': np.array([100, 120, 100]), 'upper': np.array([130, 255, 255])}
    ]
}


def detect_light_bars(image, color_name, min_area=100, max_area=5000):
    """
    检测指定颜色的灯条轮廓
    :param image: BGR彩色图像
    :param color_name: 'red' 或 'blue'
    :param min_area: 最小面积
    :param max_area: 最大面积
    :return: 灯条轮廓列表
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for rng in ARMOR_COLOR_RANGES.get(color_name, []):
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, rng['lower'], rng['upper']))

    # 形态学处理
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    light_bars = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue
        rect = cv2.minAreaRect(cnt)
        (cx, cy), (w, h), angle = rect
        # 灯条通常是细长的矩形，长宽比大
        aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else 0
        if aspect_ratio < 1.5:
            continue
        light_bars.append({
            'center': (int(cx), int(cy)),
            'rect': rect,
            'area': area,
            'angle': angle,
            'aspect_ratio': aspect_ratio
        })
    return light_bars


def find_armor_candidates(light_bars, max_distance_ratio=3.5, max_angle_diff=15):
    """
    根据灯条配对找到候选装甲板，每个灯条最多使用一次
    条件：两个灯条颜色相同、角度相近、距离适中、大致平行
    :param light_bars: 灯条列表
    :param max_distance_ratio: 两灯条中心距离与灯条长度的最大比值
    :param max_angle_diff: 两灯条角度最大差值
    :return: 装甲板候选列表
    """
    # 先计算所有可能的配对及其得分（距离越接近2倍灯条长度，得分越高）
    pairs = []
    n = len(light_bars)
    for i in range(n):
        for j in range(i + 1, n):
            bar1 = light_bars[i]
            bar2 = light_bars[j]

            cx1, cy1 = bar1['center']
            cx2, cy2 = bar2['center']
            distance = np.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2)

            (_, (w1, h1), _) = bar1['rect']
            (_, (w2, h2), _) = bar2['rect']
            length1 = max(w1, h1)
            length2 = max(w2, h2)
            avg_length = (length1 + length2) / 2

            if distance < avg_length * 0.5 or distance > avg_length * max_distance_ratio:
                continue

            angle_diff = abs(bar1['angle'] - bar2['angle'])
            if angle_diff > max_angle_diff:
                continue

            # 得分：距离在1.5~2.5倍灯条长度之间最佳
            score = 1.0 / (1.0 + abs(distance - 2.0 * avg_length) / avg_length)
            pairs.append({
                'score': score,
                'idx': (i, j),
                'bar1': bar1,
                'bar2': bar2
            })

    # 按得分降序排序，贪心选择不冲突的配对
    pairs.sort(key=lambda x: x['score'], reverse=True)
    used = set()
    armors = []
    for p in pairs:
        i, j = p['idx']
        if i in used or j in used:
            continue
        used.add(i)
        used.add(j)

        bar1, bar2 = p['bar1'], p['bar2']
        cx1, cy1 = bar1['center']
        cx2, cy2 = bar2['center']
        distance = np.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2)
        (_, (w1, h1), _) = bar1['rect']
        (_, (w2, h2), _) = bar2['rect']
        avg_length = (max(w1, h1) + max(w2, h2)) / 2

        armor_cx = (cx1 + cx2) // 2
        armor_cy = (cy1 + cy2) // 2
        armor_angle = np.degrees(np.arctan2(cy2 - cy1, cx2 - cx1))
        armor_width = int(distance)
        armor_height = int(max(avg_length * 1.5, 30))

        armors.append({
            'center': (armor_cx, armor_cy),
            'angle': round(armor_angle, 2),
            'width': armor_width,
            'height': armor_height,
            'light_bars': [bar1, bar2]
        })

    return armors


def detect_corners(image, armor, max_corners=10):
    """
    在装甲板候选区域内进行Shi-Tomasi角点检测
    :param image: 灰度图像
    :param armor: 装甲板信息
    :param max_corners: 最大角点数
    :return: 检测到的角点列表
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    cx, cy = armor['center']
    w, h = armor['width'], armor['height']
    x1 = max(0, cx - w // 2)
    y1 = max(0, cy - h // 2)
    x2 = min(gray.shape[1], cx + w // 2)
    y2 = min(gray.shape[0], cy + h // 2)
    roi = gray[y1:y2, x1:x2]

    corners = cv2.goodFeaturesToTrack(roi, maxCorners=max_corners,
                                       qualityLevel=0.1, minDistance=10)
    if corners is not None:
        corners = corners.reshape(-1, 2)
        corners[:, 0] += x1
        corners[:, 1] += y1
        return corners.astype(int).tolist()
    return []


def detect_armor(image, min_area=100, max_area=5000):
    """
    装甲板检测主函数
    :param image: BGR彩色图像
    :return: 标注后的图像和装甲板信息列表
    """
    result = image.copy()
    all_armors = []

    for color_name in ['red', 'blue']:
        light_bars = detect_light_bars(image, color_name, min_area, max_area)
        armors = find_armor_candidates(light_bars)

        for armor in armors:
            armor['color'] = color_name
            all_armors.append(armor)

            cx, cy = armor['center']
            color_bgr = (0, 0, 255) if color_name == 'red' else (255, 0, 0)

            # 绘制装甲板中心点和外接矩形
            cv2.circle(result, (cx, cy), 6, (0, 255, 255), -1)
            box = cv2.boxPoints(((cx, cy), (armor['width'], armor['height']), armor['angle']))
            box = np.intp(box)
            cv2.drawContours(result, [box], 0, color_bgr, 2)

            # 绘制角点
            corners = detect_corners(image, armor)
            for corner in corners:
                cv2.circle(result, tuple(corner), 3, (0, 255, 0), -1)

            # 标注信息（自适应位置，增加深色背景条确保文字在任何背景下都清晰）
            h_img, w_img = result.shape[:2]
            label1 = f"ARMOR {color_name.upper()}"
            label2 = f"C:({cx},{cy}) A:{round(armor['angle'], 1)}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            tw1, th1 = cv2.getTextSize(label1, font, 0.45, 2)[0]
            tw2, th2 = cv2.getTextSize(label2, font, 0.4, 2)[0]
            max_tw = max(tw1, tw2) + 8  # 增加内边距防止截断
            # 标签放在装甲板框下方，避免超出图片边界
            lx = max(5, min(cx - max_tw // 2 + 4, w_img - max_tw - 5))
            # 放在框下方，如果下方空间不够则放上方
            box_bottom = cy + armor['height'] // 2 + 5
            box_top = cy - armor['height'] // 2 - 5
            if box_bottom + 45 < h_img:
                ly1 = box_bottom + 15
            else:
                ly1 = max(20, box_top - 40)
            ly2 = ly1 + 20
            # 绘制背景条（覆盖两行文字，增加padding）
            bg_y1 = max(0, ly1 - th1 - 4)
            bg_y2 = min(h_img, ly2 + 4)
            bg_x1 = max(0, lx - 4)
            bg_x2 = min(w_img, lx + max_tw)
            cv2.rectangle(result, (bg_x1, bg_y1), (bg_x2, bg_y2), (30, 30, 30), -1)
            cv2.putText(result, label1, (lx, ly1), font, 0.45, color_bgr, 2)
            cv2.putText(result, label2, (lx, ly2), font, 0.4, color_bgr, 2)

    return result, all_armors


def armor_detection_pipeline(image_path, output_dir):
    """
    装甲板检测主流程
    :param image_path: 输入图片路径
    :param output_dir: 结果输出目录
    """
    original = read_image(image_path)
    if original is None:
        return

    result, armors = detect_armor(original)
    base_name = os.path.splitext(os.path.basename(image_path))[0]

    save_image(os.path.join(output_dir, f"{base_name}_armor_result.jpg"), result)

    comparison = create_comparison_image(
        [original, result],
        ["Original", "Armor Detection"],
        cols=2
    )
    save_image(os.path.join(output_dir, f"{base_name}_armor_comparison.jpg"), comparison)

    print(f"\n===== {base_name} 装甲板定位结果 =====")
    print(f"检测到 {len(armors)} 个装甲板目标：")
    for i, a in enumerate(armors, 1):
        print(f"  #{i} 颜色={a['color']} 中心={a['center']} 角度={a['angle']}° 宽={a['width']} 高={a['height']}")


if __name__ == "__main__":
    input_image = os.path.join("test_images", "original", "images", "armor_test.jpg")
    output_dir = os.path.join("test_images", "results", "task4")
    armor_detection_pipeline(input_image, output_dir)
