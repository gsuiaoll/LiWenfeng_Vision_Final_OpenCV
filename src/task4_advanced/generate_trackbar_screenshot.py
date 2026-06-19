"""
自动生成Trackbar调参工具的截图
无需手动操作，直接生成调参效果图保存到results目录
"""

import cv2
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.utils import read_image, save_image


def generate_trackbar_screenshot(image_path, output_dir):
    """
    使用预设参数生成Trackbar调参效果图
    模拟Trackbar调参后的效果，保存为截图
    """
    image = read_image(image_path)
    if image is None:
        return

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 预设参数组合：蓝色检测与红色双段范围检测
    param_sets = [
        {
            'name': 'Blue Default',
            'h_ranges': [(100, 130)],
            's_lower': 120, 's_upper': 255,
            'v_lower': 100, 'v_upper': 255,
            'blur_k': 5, 'morph_k': 5, 'min_area': 200
        },
        {
            'name': 'Blue Sensitive',
            'h_ranges': [(90, 140)],
            's_lower': 80, 's_upper': 255,
            'v_lower': 80, 'v_upper': 255,
            'blur_k': 3, 'morph_k': 3, 'min_area': 100
        },
        {
            'name': 'Red Default',
            # 红色在HSV中跨越0°/180°边界，使用两段范围合并
            'h_ranges': [(0, 10), (160, 180)],
            's_lower': 120, 's_upper': 255,
            'v_lower': 100, 'v_upper': 255,
            'blur_k': 5, 'morph_k': 5, 'min_area': 200
        }
    ]

    os.makedirs(output_dir, exist_ok=True)

    for params in param_sets:
        # 高斯模糊
        if params['blur_k'] > 1:
            blurred_hsv = cv2.GaussianBlur(hsv, (params['blur_k'], params['blur_k']), 0)
        else:
            blurred_hsv = hsv

        # HSV阈值分割（支持多段H范围，如红色）
        mask = np.zeros(blurred_hsv.shape[:2], dtype=np.uint8)
        for h_lower, h_upper in params['h_ranges']:
            lower = np.array([h_lower, params['s_lower'], params['v_lower']])
            upper = np.array([h_upper, params['s_upper'], params['v_upper']])
            mask = cv2.bitwise_or(mask, cv2.inRange(blurred_hsv, lower, upper))

        # 形态学处理
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,
                                           (params['morph_k'], params['morph_k']))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        # 查找轮廓并标注
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        result = image.copy()
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < params['min_area']:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.drawContours(result, [cnt], -1, (0, 255, 0), 2)
            cv2.rectangle(result, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(result, f"A:{int(area)}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # 合并原图、掩码、结果
        mask_color = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        combined = np.hstack([image, mask_color, result])

        # 添加参数信息
        h_info = " U ".join([f"[{h_l},{h_u}]" for h_l, h_u in params['h_ranges']])
        info = (f"{params['name']} | H:{h_info} "
                f"S:[{params['s_lower']},{params['s_upper']}] "
                f"V:[{params['v_lower']},{params['v_upper']}] "
                f"Blur:{params['blur_k']} Morph:{params['morph_k']} "
                f"MinArea:{params['min_area']}")
        cv2.putText(combined, info, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        save_image(os.path.join(output_dir, f"trackbar_{params['name'].replace(' ', '_')}.jpg"), combined)

    print(f"[Trackbar截图] 已生成到 {output_dir}")


if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    input_image = os.path.join(project_root, "test_images", "original", "images", "color_test.jpg")
    output_dir = os.path.join(project_root, "test_images", "results", "task4", "trackbar")
    generate_trackbar_screenshot(input_image, output_dir)
