"""
进阶任务三：交互调参工具
基于OpenCV的Trackbar组件创建可视化调参界面
支持实时调节HSV阈值、模糊核大小等关键参数
"""

import cv2
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.utils import read_image, save_image


class TrackbarTool:
    """Trackbar可视化调参工具类"""

    WINDOW_NAME = "HSV Parameter Tuning Tool"

    def __init__(self, image_path):
        self.image = read_image(image_path)
        if self.image is None:
            raise FileNotFoundError(f"无法加载图片: {image_path}")

        self.hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        cv2.namedWindow(self.WINDOW_NAME, cv2.WINDOW_AUTOSIZE)

        # 创建Trackbar
        cv2.createTrackbar("H Lower", self.WINDOW_NAME, 0, 180, self.nothing)
        cv2.createTrackbar("H Upper", self.WINDOW_NAME, 180, 180, self.nothing)
        cv2.createTrackbar("H Lower2", self.WINDOW_NAME, 0, 180, self.nothing)
        cv2.createTrackbar("H Upper2", self.WINDOW_NAME, 0, 180, self.nothing)
        cv2.createTrackbar("S Lower", self.WINDOW_NAME, 0, 255, self.nothing)
        cv2.createTrackbar("S Upper", self.WINDOW_NAME, 255, 255, self.nothing)
        cv2.createTrackbar("V Lower", self.WINDOW_NAME, 0, 255, self.nothing)
        cv2.createTrackbar("V Upper", self.WINDOW_NAME, 255, 255, self.nothing)
        cv2.createTrackbar("Blur Kernel", self.WINDOW_NAME, 1, 21, self.nothing)
        cv2.createTrackbar("Morph Kernel", self.WINDOW_NAME, 1, 21, self.nothing)
        cv2.createTrackbar("Min Area", self.WINDOW_NAME, 100, 2000, self.nothing)

        # 设置初始值（蓝色示例，H2 范围默认关闭）
        cv2.setTrackbarPos("H Lower", self.WINDOW_NAME, 100)
        cv2.setTrackbarPos("H Upper", self.WINDOW_NAME, 130)
        cv2.setTrackbarPos("S Lower", self.WINDOW_NAME, 120)
        cv2.setTrackbarPos("V Lower", self.WINDOW_NAME, 100)

    def nothing(self, x):
        """Trackbar回调函数，不需要额外操作"""
        pass

    def get_trackbar_values(self):
        """读取当前Trackbar的值"""
        h_lower = cv2.getTrackbarPos("H Lower", self.WINDOW_NAME)
        h_upper = cv2.getTrackbarPos("H Upper", self.WINDOW_NAME)
        h_lower2 = cv2.getTrackbarPos("H Lower2", self.WINDOW_NAME)
        h_upper2 = cv2.getTrackbarPos("H Upper2", self.WINDOW_NAME)
        s_lower = cv2.getTrackbarPos("S Lower", self.WINDOW_NAME)
        s_upper = cv2.getTrackbarPos("S Upper", self.WINDOW_NAME)
        v_lower = cv2.getTrackbarPos("V Lower", self.WINDOW_NAME)
        v_upper = cv2.getTrackbarPos("V Upper", self.WINDOW_NAME)
        blur_k = cv2.getTrackbarPos("Blur Kernel", self.WINDOW_NAME)
        morph_k = cv2.getTrackbarPos("Morph Kernel", self.WINDOW_NAME)
        min_area = cv2.getTrackbarPos("Min Area", self.WINDOW_NAME)

        # 模糊核和形态学核必须为奇数
        blur_k = max(1, blur_k)
        if blur_k % 2 == 0:
            blur_k += 1
        morph_k = max(1, morph_k)
        if morph_k % 2 == 0:
            morph_k += 1

        return {
            'h_lower': h_lower, 'h_upper': h_upper,
            'h_lower2': h_lower2, 'h_upper2': h_upper2,
            's_lower': s_lower, 's_upper': s_upper,
            'v_lower': v_lower, 'v_upper': v_upper,
            'blur_k': blur_k, 'morph_k': morph_k,
            'min_area': min_area
        }

    def process(self, params):
        """
        根据当前参数处理图像
        :param params: Trackbar参数字典
        :return: 处理结果图、掩码图
        """
        # 高斯模糊
        if params['blur_k'] > 1:
            blurred_hsv = cv2.GaussianBlur(self.hsv, (params['blur_k'], params['blur_k']), 0)
        else:
            blurred_hsv = self.hsv

        # HSV阈值分割（支持双段H范围，如红色跨越0°/180°）
        lower1 = np.array([params['h_lower'], params['s_lower'], params['v_lower']])
        upper1 = np.array([params['h_upper'], params['s_upper'], params['v_upper']])
        mask = cv2.inRange(blurred_hsv, lower1, upper1)
        if params['h_upper2'] > params['h_lower2']:
            lower2 = np.array([params['h_lower2'], params['s_lower'], params['v_lower']])
            upper2 = np.array([params['h_upper2'], params['s_upper'], params['v_upper']])
            mask = cv2.bitwise_or(mask, cv2.inRange(blurred_hsv, lower2, upper2))

        # 形态学处理
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,
                                           (params['morph_k'], params['morph_k']))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # 查找轮廓并过滤
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        result = self.image.copy()
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < params['min_area']:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.drawContours(result, [cnt], -1, (0, 255, 0), 2)
            cv2.rectangle(result, (x, y), (x+w, y+h), (0, 0, 255), 2)
            cv2.putText(result, f"A:{int(area)}", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # 合并原图、掩码、结果
        mask_color = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        combined = np.hstack([self.image, mask_color, result])

        # 显示当前参数
        h2_info = f" H2:[{params['h_lower2']},{params['h_upper2']}]" if params['h_upper2'] > params['h_lower2'] else ""
        info = (f"H:[{params['h_lower']},{params['h_upper']}]{h2_info} "
                f"S:[{params['s_lower']},{params['s_upper']}] "
                f"V:[{params['v_lower']},{params['v_upper']}] "
                f"Blur:{params['blur_k']} Morph:{params['morph_k']} MinArea:{params['min_area']}")
        cv2.putText(combined, info, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        return combined, mask

    def run(self):
        """启动调参界面主循环"""
        print("=" * 50)
        print("OpenCV HSV 交互调参工具")
        print("窗口标题:", self.WINDOW_NAME)
        print("按 'q' 退出")
        print("按 's' 保存当前参数和结果到 test_images/results/task4/trackbar/")
        print("=" * 50)
        while True:
            params = self.get_trackbar_values()
            combined, mask = self.process(params)
            cv2.imshow(self.WINDOW_NAME, combined)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == ord('s'):
                self.save_current_state(params, combined, mask)

            # 检测窗口是否被鼠标关闭，避免程序在后台空转
            if cv2.getWindowProperty(self.WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break

        cv2.destroyAllWindows()
        print("调参工具已关闭")

    def save_current_state(self, params, combined, mask):
        """保存当前参数和结果图"""
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        output_dir = os.path.join(project_root, "test_images", "results", "task4", "trackbar")
        os.makedirs(output_dir, exist_ok=True)

        # 保存结果图（使用save_image支持中文路径）
        save_image(os.path.join(output_dir, "trackbar_result.jpg"), combined)
        save_image(os.path.join(output_dir, "trackbar_mask.jpg"), mask)

        # 保存参数
        with open(os.path.join(output_dir, "trackbar_params.txt"), 'w', encoding='utf-8') as f:
            f.write("# Trackbar 调参结果\n")
            for k, v in params.items():
                f.write(f"{k}: {v}\n")
        print(f"[保存] 参数和结果已保存到 {output_dir}")


def trackbar_tool_pipeline(image_path):
    """
    Trackbar调参工具入口
    :param image_path: 输入图片路径
    """
    tool = TrackbarTool(image_path)
    tool.run()


if __name__ == "__main__":
    # 基于当前文件位置计算项目根目录，支持从任意目录运行
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    input_image = os.path.join(project_root, "test_images", "original", "images", "color_test.jpg")
    trackbar_tool_pipeline(input_image)
