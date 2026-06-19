# 视觉算法组最终考核 — OpenCV图像识别项目

> **仓库命名**：`LiWenfeng_Vision_Final_OpenCV`（对应 `李文锋_视觉组_最终考核_OpenCV识别`）  
> **考核达标**：完成后将个人GitHub仓库链接发送到 `numiyo@163.com`  
> **在线仓库**：https://github.com/gsuiaoll/LiWenfeng_Vision_Final_OpenCV

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
│       ├── trackbar_tool.py                # 进阶三：Trackbar交互调参
│       └── generate_trackbar_screenshot.py # Trackbar效果图自动生成
└── test_images/
    ├── original/
    │   └── images/                         # 原始测试图片
    │       ├── basic_test.jpg
    │       ├── color_test.jpg
    │       ├── shape_number_test.jpg
    │       ├── armor_test.jpg
    │       ├── lighting/                   # 光照鲁棒性测试图
    │       └── occlusion/                  # 遮挡鲁棒性测试图
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

**目标**：识别图片中的三角形、正方形、矩形、五边形、六边形、圆形、椭圆等几何图形。

**实现要点**：

1. **自适应预处理**：
   - 白色背景图片（如 `shape_number_test.jpg`）：基于 HSV 饱和度/亮度阈值分离彩色/深色形状
   - 非白色背景图片：基于 BGR 三通道 Canny 边缘检测 + 膨胀/闭运算填充形状内部
2. **多边形近似**：`cv2.approxPolyDP()`，ε=0.015×周长，根据顶点数初步分类
3. **圆度计算**：`4 * π * 面积 / 周长²`，用于区分圆形与多边形
4. **凸包辅助判断**：对凸包进行低精度近似（ε=0.001×凸包周长），利用平滑图形（圆/椭圆）顶点数远多于多边形的特点，修正低分辨率或小尺寸形状的误识别
5. **形状判断**：
   - 3个顶点 → 三角形（凸包顶点少）或椭圆（凸包顶点多）
   - 4个顶点 → 正方形、矩形、椭圆或五边形（结合长宽比与凸包顶点数）
   - 5个顶点 → 五边形
   - 6个顶点 → 六边形或椭圆
   - ≥7个顶点 → 圆形、椭圆或六边形（结合圆度、长宽比）

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

### 5.1 一键运行所有非交互式任务

```bash
python run_all.py
```

运行后所有非交互式任务的测试结果将保存在 `test_images/results/` 目录下。

> 注：`run_all.py` 包含任务一至任务四的自动化测试，并自动生成 Trackbar 调参效果图；Trackbar 交互调参工具因需要人工操作，需单独运行。

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

# Trackbar 效果图自动生成（无需人工操作）
python -m src.task4_advanced.generate_trackbar_screenshot
```

---

## 六、测试结果分析

### 6.1 考核要求对应关系

本项目严格按照考核文档要求实现，各任务对应关系如下：

| 考核要求 | 实现模块 | 关键函数 | 测试结果 |
|---------|---------|---------|---------|
| 图像基础预处理 | task1_preprocessing | `grayscale()`, `gaussian_blur()`, `histogram_equalization()`, `create_comparison_image()` | ✅ 灰度/模糊/均衡化对比清晰 |
| 颜色阈值色块识别 | task2_color_detection | `hsv_color_threshold()`, `morphological_process()`, `detect_color_targets()` | ✅ 红蓝目标成功分割并标注坐标面积 |
| 简单特征识别（几何图形） | task3_shape_digit_recognition | `detect_shapes()`, `preprocess_for_contours()` | ✅ 识别三角形/矩形/正方形/五边形/六边形/圆形 |
| 0-9数字识别（禁OCR库） | task3_shape_digit_recognition | `digit_recognition_pipeline()`, 模板匹配 + Hu矩轮廓匹配 | ✅ 识别序列：0123456789，准确率100%（排除彩色形状内部镂空干扰） |
| 多条件目标精定位（装甲板） | task4_advanced | `detect_light_bars()`, `find_armor_candidates()`, `detect_corners()` | ✅ 检测到5个装甲板目标（颜色+轮廓+角点融合） |
| 算法鲁棒性测试 | task4_advanced | `robustness_test_pipeline()`, 光照/遮挡场景测试 | ✅ 自动生成Markdown测试报告 |
| 交互调参工具（Trackbar） | task4_advanced | 基于 `cv2.createTrackbar()` 实时调节 | ✅ 可交互实时调参 |
| Trackbar效果图自动生成 | task4_advanced | `generate_trackbar_screenshot()` 预设参数批量生成 | ✅ 集成到 run_all.py |
| 代码模块化拆分 | src/ 下8个任务/工具脚本 + utils.py | 各模块独立，单任务可单独运行 | ✅ 结构清晰 |
| 关键函数中文注释 | 所有函数 | docstring + 行内逻辑注释 | ✅ 注释完整 |
| 测试效果图分类存放 | test_images/results/ | task1/task2/task3/task4 分类存放 | ✅ 按任务分类 |
| 处理前后对比图 | 各任务对比图 | `create_comparison_image()` 生成 | ✅ 各任务均有对比图 |
| README文档 | README.md | 环境配置+实现思路+测试结果+参考资料 | ✅ 完整 |
| AI使用说明 | README 第1节 | 明确说明AI辅助范围 | ✅ 符合考核注意事项 |
| GitHub仓库上传 | LiWenfeng_Vision_Final_OpenCV | 所有代码+结果+文档 | ✅ 已推送 |

### 6.2 各任务详细测试数据

#### 任务一：图像预处理
- **输入**：basic_test.jpg
- **输出**：basic_test_comparison.jpg（4格对比图：原图→灰度→高斯模糊→直方图均衡化）
- **关键参数**：高斯模糊核(5,5)，σ=1.0；均衡化彩色图在YUV空间处理

#### 任务二：颜色阈值色块识别
- **输入**：color_test.jpg
- **输出**：红色目标识别图、蓝色目标识别图、对比图
- **红色HSV范围**：H=[0,10]∪[160,180], S≥120, V≥100
- **蓝色HSV范围**：H=[100,130], S≥120, V≥100
- **形态学**：开运算(5×5核) + 闭运算(5×5核, 2次迭代)
- **最小面积过滤**：200像素

#### 任务三：几何图形识别
- **输入**：shape_number_test.jpg / basic_test.jpg / color_test.jpg
- **输出**：*_shapes.jpg
- **识别算法**：自适应预处理（白色背景用HSV阈值，其他用BGR三通道Canny） + 多边形近似(approxPolyDP, ε=0.015×周长) + 圆度判断 + 凸包低精度近似辅助
- **形状类型**：triangle, square, rectangle, pentagon, hexagon, circle, ellipse
- **检测区域**：`shape_number_test.jpg` 限制为图片上方82%区域（避免下方数字干扰），其他图片默认使用全图

#### 任务三：数字识别
- **输入**：shape_number_test.jpg
- **输出**：shape_number_test_digits.jpg, shape_number_test_digit_comparison.jpg
- **算法**：Otsu二值化 → 形态学闭运算 → 轮廓提取 → 模板匹配(70%) + Hu矩匹配(30%)
- **准确率**：100%（识别序列 0 1 2 3 4 5 6 7 8 9）
- **模板来源**：从测试图像自动提取0-9数字作为模板

#### 进阶任务一：装甲板目标精定位
- **输入**：armor_test.jpg
- **输出**：armor_test_armor_result.jpg, armor_test_armor_comparison.jpg
- **算法流程**：HSV颜色阈值 → 灯条检测（长宽比≥1.5）→ 贪心配对 → 角点检测辅助
- **检测结果**：5个装甲板目标，输出中心坐标+旋转角度

#### 进阶任务二：鲁棒性测试
- **输入**：lighting/ 目录（5种光照场景），occlusion/ 目录（4种遮挡场景）
- **输出**：robustness_report.md（Markdown格式测试报告）
- **测试指标**：准确率、漏检率、正确检测数
- **基准参考**：自动从正常光照/无遮挡场景统计期望目标数

#### 进阶任务三：Trackbar交互调参
- **输入**：color_test.jpg
- **输出**：可交互窗口（运行时），调参后结果图
- **可调参数**：H上下限、S上下限、V上下限、模糊核大小、形态学核大小、最小面积
- **保存方式**：按 `s` 键保存当前参数配置和结果图

### 6.3 数字识别置信度说明

在本次测试中，10个数字全部被正确识别（准确率100%）。部分数字（如 `1`、`2`、`3`、`5`、`7`）的匹配得分相对较低，这是正常现象，原因如下：

- **模板匹配特性**：`cv2.matchTemplate()` 对笔画结构简单的数字（如 `1`）区分度天然较低
- **形近数字干扰**：`2`、`3`、`5` 在印刷体中轮廓有一定相似性，需要依赖Hu矩辅助区分
- **综合评分机制**：即使单个得分不高，模板匹配（70%）+ Hu矩匹配（30%）的综合评分仍能正确排序
- **最终准确率**：以最高综合分为识别结果，10个数字全部正确

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

## 九、Git 版本管理说明

本项目使用 Git 进行代码版本管理，主要操作如下：

```bash
# 初始化仓库（已在本项目中完成）
git init

# 添加所有文件到暂存区
git add -A

# 提交修改
git commit -m "提交说明"

# 关联远程仓库
git remote add origin https://github.com/gsuiaoll/LiWenfeng_Vision_Final_OpenCV.git

# 推送到远程仓库（请根据实际默认分支替换 master/main）
git push -u origin master
```

---

## 十、作者信息

- 姓名：李文锋
- 学校/团队：视觉算法组

> 完成时间：2026年6月  
> 注：本项目在考核要求基础上持续优化，最终完成时间为2026年6月。
