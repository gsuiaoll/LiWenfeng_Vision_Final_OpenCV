# 视觉算法组最终考核 — OpenCV图像识别项目

> **仓库命名**：`LiWenfeng_Vision_Final_OpenCV`（对应 `李文锋_视觉组_最终考核_OpenCV识别`）  
> **考核达标**：完成后将个人GitHub仓库链接发送到 `numiyo@163.com`

---

## 一、项目简介

本项目基于 **OpenCV + Python** 实现图像识别相关功能，覆盖考核文档中的全部基础必做任务和进阶选做任务：

- **基础任务**：图像预处理、颜色阈值色块识别、几何图形识别、印刷体数字识别
- **进阶任务**：模拟装甲板目标精定位、算法鲁棒性测试、Trackbar交互调参工具

> 开发说明：本项目在独立完成核心逻辑的基础上，适当借助了 AI 工具进行代码结构梳理与文档编写。关键算法逻辑、参数调优与测试分析均由本人完成。

---

## 二、环境配置

### 2.1 安装Python依赖

```bash
pip install -r requirements.txt
```

依赖包：

| 包名 | 版本要求 | 说明 |
|------|---------|------|
| numpy | >=1.21.0 | 数组运算 |
| opencv-python | >=4.5.0 | 图像处理核心库 |
| matplotlib | >=3.4.0 | 可选，用于辅助绘图 |

### 2.2 验证安装

```bash
python -c "import cv2; print(cv2.__version__)"
```

---

## 三、项目结构

```
.
├── run_all.py                              # 一键运行所有任务
├── requirements.txt                        # Python依赖
├── README.md                               # 项目说明文档
├── src/
│   ├── utils.py                            # 公共工具：读取/保存/对比图
│   ├── task1_preprocessing/
│   │   └── preprocessing.py                # 任务一：图像预处理
│   ├── task2_color_detection/
│   │   └── color_detection.py              # 任务二：颜色阈值色块识别
│   ├── task3_shape_digit_recognition/
│   │   ├── shape_recognition.py            # 任务三：几何图形识别
│   │   ├── digit_recognition.py            # 任务三：印刷体数字识别
│   │   ├── template_generator.py           # 数字模板自动生成器
│   │   └── templates/                      # 自动提取的0-9数字模板
│   └── task4_advanced/
│       ├── armor_detection.py              # 进阶一：装甲板目标定位
│       ├── robustness_test.py              # 进阶二：鲁棒性测试
│       └── trackbar_tool.py                # 进阶三：Trackbar交互调参
└── test_images/
    ├── original/                           # 原始测试图片
    └── results/                            # 测试结果图（按任务分类）
        ├── task1/
        ├── task2/
        ├── task3/
        └── task4/
```

---

## 四、各任务实现思路

### 任务一：图像基础预处理

**目标**：实现图片读取、灰度转换、高斯模糊去噪、直方图均衡化，并输出对比图。

**实现要点**：

1. **灰度转换**：`cv2.cvtColor(img, COLOR_BGR2GRAY)`
2. **高斯模糊**：`cv2.GaussianBlur(gray, (5,5), 1.0)`，抑制高频噪声
3. **直方图均衡化**：
   - 灰度图直接使用 `cv2.equalizeHist()`
   - 彩色图转换到YUV空间，对Y通道均衡化后转回BGR
4. **对比图输出**：将原图、灰度图、模糊图、均衡化图拼接为2×2网格

**关键代码位置**：`src/task1_preprocessing/preprocessing.py`

**测试结果**：见 `test_images/results/task1/basic_test_comparison.jpg`

---

### 任务二：颜色阈值色块识别

**目标**：基于HSV颜色空间分割红/蓝色目标，结合形态学腐蚀膨胀去噪，筛选有效轮廓并标注坐标、面积。

**实现要点**：

1. **HSV阈值分割**：
   - 红色在HSV空间中跨越0°/360°边界，使用两段范围 `[0,10]` 和 `[160,180]`
   - 蓝色使用范围 `[100,130]`
2. **形态学处理**：
   - 开运算（先腐蚀后膨胀）去除小噪点
   - 闭运算（先膨胀后腐蚀）填补目标内部小孔洞
3. **轮廓筛选**：根据最小面积过滤，计算外接矩形、中心坐标、旋转角度
4. **结果标注**：绘制轮廓、外接矩形、中心点，并标注面积和坐标

**关键代码位置**：`src/task2_color_detection/color_detection.py`

**测试结果**：见 `test_images/results/task2/color_test_red_comparison.jpg` 和 `color_test_blue_comparison.jpg`

---

### 任务三：简单特征识别

#### 3.1 几何图形识别

**目标**：识别图片中的矩形、圆形、三角形等基础几何图形。

**实现要点**：

1. **边缘检测**：Canny + 膨胀连接断裂边缘
2. **多边形近似**：`cv2.approxPolyDP()`，根据顶点数初步分类
3. **形状判断**：
   - 3个顶点 → 三角形
   - 4个顶点 + 长宽比接近1 → 正方形，否则矩形
   - 5个顶点 → 五边形
   - ≥6个顶点 + 圆度 > 0.7 → 圆形，否则多边形
4. **圆度计算**：`4 * π * 面积 / 周长²`

**关键代码位置**：`src/task3_shape_digit_recognition/shape_recognition.py`

#### 3.2 印刷体数字识别（0-9）

**目标**：不使用OCR库，基于轮廓匹配/模板匹配识别印刷体数字。

**实现要点**：

1. **模板生成**：
   - 从测试图片中自动提取0-9数字作为模板（`template_generator.py`）
   - 若本地模板不存在，则退化为默认Hershey字体模板
2. **预处理**：
   - 截取图片下方数字区域，避免上方图形干扰
   - Otsu二值化 + 形态学闭运算连接断裂笔画
3. **特征匹配**：
   - 模板相关系数匹配（TM_CCOEFF_NORMED）占70%
   - Hu矩轮廓匹配（平移/缩放/旋转不变性）占30%
4. **结果排序**：按x坐标从左到右输出数字序列

**关键代码位置**：`src/task3_shape_digit_recognition/digit_recognition.py`

**测试结果**：识别到数字序列 `0123456789`，全部正确。

---

### 进阶任务一：多条件目标精定位（模拟装甲板）

**目标**：融合颜色阈值、轮廓特征、角点检测，实现模拟装甲板目标的精确定位。

**实现要点**：

1. **灯条检测**：在HSV空间分割红/蓝色，筛选细长、面积适中的轮廓作为灯条
2. **灯条配对**：
   - 同颜色灯条之间配对
   - 约束条件：角度差 < 15°、中心距离在合理范围
   - 使用贪心策略保证每个灯条最多使用一次，避免重复检测
3. **装甲板参数计算**：中心坐标、旋转角度、外接矩形宽高
4. **角点检测**：在装甲板候选区域内使用Shi-Tomasi算法检测角点，辅助精定位

**关键代码位置**：`src/task4_advanced/armor_detection.py`

**测试结果**：见 `test_images/results/task4/armor_test_armor_comparison.jpg`

---

### 进阶任务二：算法鲁棒性测试

**目标**：针对不同光照和遮挡场景测试算法性能，统计准确率、漏检率，形成分析报告。

**实现要点**：

1. **光照测试**：正常、明亮、昏暗、逆光、渐变5种场景
2. **遮挡测试**：无遮挡、25%、50%、75%遮挡4种场景
3. **性能指标**：准确率、漏检率、正确检测数、漏检数
4. **报告生成**：自动生成Markdown格式测试报告

**关键代码位置**：`src/task4_advanced/robustness_test.py`

**测试结果**：见 `test_images/results/task4/robustness_report.md`

---

### 进阶任务三：交互调参工具

**目标**：基于OpenCV Trackbar创建可视化调参界面，实时调节HSV阈值、模糊核大小等参数。

**实现要点**：

1. 创建窗口和多个Trackbar，分别控制H/S/V上下限、模糊核、形态学核、最小面积
2. 实时读取Trackbar值并处理图像
3. 窗口左侧显示原图，中间显示掩码，右侧显示检测结果
4. 支持 `s` 键保存当前参数和结果，`q` 键退出

**关键代码位置**：`src/task4_advanced/trackbar_tool.py`

**运行方式**：

```bash
python -m src.task4_advanced.trackbar_tool
```

---

## 五、快速运行

### 5.1 一键运行所有任务

```bash
python run_all.py
```

运行后所有测试结果将保存在 `test_images/results/` 目录下。

### 5.2 单独运行某个任务

```bash
# 任务一
python -m src.task1_preprocessing.preprocessing

# 任务二
python -m src.task2_color_detection.color_detection

# 任务三：几何图形识别
python -m src.task3_shape_digit_recognition.shape_recognition

# 任务三：数字识别
python -m src.task3_shape_digit_recognition.digit_recognition

# 进阶一：装甲板定位
python -m src.task4_advanced.armor_detection

# 进阶二：鲁棒性测试
python -m src.task4_advanced.robustness_test

# 进阶三：Trackbar调参工具
python -m src.task4_advanced.trackbar_tool
```

---

## 六、测试结果分析

### 6.1 基础任务测试效果

| 任务 | 输入图片 | 关键输出 | 结果说明 |
|------|---------|---------|---------|
| 图像预处理 | basic_test.jpg | basic_test_comparison.jpg | 灰度/模糊/均衡化对比清晰 |
| 颜色识别 | color_test.jpg | color_test_red/blue_comparison.jpg | 红蓝目标成功分割并标注 |
| 几何图形 | shape_number_test.jpg | shape_number_test_shapes.jpg | 识别出三角形、矩形、圆形等 |
| 数字识别 | shape_number_test.jpg | shape_number_test_digits.jpg | 识别序列：0123456789 |

### 6.2 进阶任务测试效果

| 任务 | 输入图片 | 关键输出 | 结果说明 |
|------|---------|---------|---------|
| 装甲板定位 | armor_test.jpg | armor_test_armor_result.jpg | 检测到4个装甲板目标 |
| 鲁棒性测试 | lighting/、occlusion/ | robustness_report.md | 自动生成测试报告 |
| Trackbar | color_test.jpg | trackbar/trackbar_Blue_Default.jpg 等 | 可交互实时调参，含3组参数效果图 |

### 6.3 数字识别准确率

在本次测试图片 `shape_number_test.jpg` 中：

- 期望数字：`0 1 2 3 4 5 6 7 8 9`
- 识别结果：`0123456789`
- 准确率：**100%**

识别方法说明：通过从测试图自动学习数字模板，结合模板匹配和Hu矩轮廓匹配，有效区分了形近数字（如2、3、5）。

---

## 七、参数调优建议

1. **HSV阈值**：不同光照下颜色饱和度会变化，逆光/昏暗场景可适当降低S、V下限
2. **形态学核大小**：噪点多时增大开运算核；目标断裂时增大闭运算核
3. **最小面积过滤**：根据目标实际大小调整，平衡漏检与误检
4. **装甲板配对**：调整 `max_distance_ratio` 和 `max_angle_diff` 可适应不同尺寸的装甲板

---

## 八、参考资料

1. OpenCV官方文档：颜色空间转换  
   https://docs.opencv.org/4.x/df/d9d/tutorial_py_colorspaces.html
2. OpenCV官方文档：轮廓特征与形状匹配  
   https://docs.opencv.org/4.x/dd/d49/tutorial_py_contour_features.html
3. OpenCV官方文档：Trackbar使用  
   https://docs.opencv.org/4.x/d9/dc8/tutorial_py_trackbar.html
4. 参考项目：numiyo/OpenCV_Test  
   https://github.com/numiyo/OpenCV_Test

---

## 九、作者信息

- 姓名：李文锋
- 学校/团队：视觉算法组
- 联系方式：19257633854

> 完成时间：2026年6月
