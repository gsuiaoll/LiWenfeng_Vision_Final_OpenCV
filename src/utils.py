"""
公共工具模块
提供图像读取、保存、创建对比图等通用功能
"""

import cv2
import os
import numpy as np


def read_image(path):
    """
    安全读取图片文件（支持中文路径）
    :param path: 图片路径
    :return: BGR格式的numpy数组，读取失败返回None
    """
    if not os.path.exists(path):
        print(f"[错误] 图片不存在: {path}")
        return None
    # 使用numpy读取字节流并通过cv2.imdecode解码，避免中文路径问题
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        print(f"[错误] 无法读取图片: {path}")
        return None
    return image


def save_image(path, image):
    """
    保存图片到指定路径（支持中文路径）
    :param path: 保存路径
    :param image: 要保存的图像
    :return: 是否保存成功
    """
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    # 使用imencode+tofile方式保存，兼容中文路径
    ext = os.path.splitext(path)[1].lower()
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, 95] if ext in ('.jpg', '.jpeg') else []
    success, encoded = cv2.imencode(ext, image, encode_params)
    if success:
        encoded.tofile(path)
        print(f"[保存成功] {path}")
        return True
    else:
        print(f"[保存失败] {path}")
        return False


def create_comparison_image(images, titles, cols=2):
    """
    将多张图片按网格拼接成对比图
    :param images: 图像列表
    :param titles: 每张图像的标题列表
    :param cols: 每行显示的列数
    :return: 拼接后的对比图
    """
    n = len(images)
    rows = (n + cols - 1) // cols

    # 统一图像尺寸
    target_h, target_w = 0, 0
    for img in images:
        h, w = img.shape[:2]
        target_h = max(target_h, h)
        target_w = max(target_w, w)

    canvas = np.zeros((rows * target_h, cols * target_w, 3), dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX

    for i, (img, title) in enumerate(zip(images, titles)):
        row = i // cols
        col = i % cols

        # 统一为3通道
        if len(img.shape) == 2:
            img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            img_color = img.copy()

        # 等比例缩放
        ih, iw = img_color.shape[:2]
        scale = min(target_w / iw, target_h / ih)
        new_w, new_h = int(iw * scale), int(ih * scale)
        resized = cv2.resize(img_color, (new_w, new_h))

        # 居中放置
        y_offset = (target_h - new_h) // 2
        x_offset = (target_w - new_w) // 2
        canvas[row * target_h + y_offset: row * target_h + y_offset + new_h,
               col * target_w + x_offset: col * target_w + x_offset + new_w] = resized

        # 绘制标题背景条（深灰色半透明），保证任何背景上标题都清晰
        text_size, _ = cv2.getTextSize(title, font, 0.7, 2)
        tw, th = text_size
        bar_h = th + 16
        bar_x1 = col * target_w
        bar_y1 = row * target_h
        bar_x2 = min(canvas.shape[1], bar_x1 + tw + 20)
        bar_y2 = min(canvas.shape[0], bar_y1 + bar_h)
        cv2.rectangle(canvas, (bar_x1, bar_y1), (bar_x2, bar_y2), (40, 40, 40), -1)
        # 添加标题（黄色字体+黑色描边）
        title_pos = (bar_x1 + 10, bar_y1 + th + 5)
        cv2.putText(canvas, title, title_pos, font, 0.7, (0, 0, 0), 4)  # 黑色描边
        cv2.putText(canvas, title, title_pos, font, 0.7, (0, 255, 255), 2)  # 黄色字体

    return canvas
