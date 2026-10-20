from pathlib import Path

import cv2
from PIL import Image

# 根据脚本所在位置寻找图片，不受终端运行目录影响
task_dir = Path(__file__).resolve().parent
image_path = task_dir / "assets" / "example1.jpg"

# 使用 OpenCV 读取图片
image_bgr = cv2.imread(str(image_path))

if image_bgr is None:
    raise FileNotFoundError(f"无法读取图片，请检查：{image_path}")

height, width, channels = image_bgr.shape
print(f"图片宽度：{width} 像素")
print(f"图片高度：{height} 像素")
print(f"通道数量：{channels}")

# OpenCV 使用 BGR 顺序，转换成 Pillow 使用的 RGB 顺序
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

# 使用 Pillow 显示图片
image = Image.fromarray(image_rgb)
image.show()
# 创建结果文件夹；如果已经存在，就直接使用
output_dir = task_dir / "outputs"
output_dir.mkdir(parents=True, exist_ok=True)

# 按 OpenCV 的 BGR 顺序分离三个通道
blue, green, red = cv2.split(image_bgr)

# 使用 Pillow 将每个通道保存为灰度图片
Image.fromarray(blue).save(output_dir / "channel_blue.png")
Image.fromarray(green).save(output_dir / "channel_green.png")
Image.fromarray(red).save(output_dir / "channel_red.png")

print("单个通道的形状：", blue.shape)
print("三个通道已保存到：", output_dir)
# 将宽、高都缩小为原来的一半
new_width = max(1, width // 2)
new_height = max(1, height // 2)

small_bgr = cv2.resize(
    image_bgr,
    (new_width, new_height),
    interpolation=cv2.INTER_AREA,
)

# 转换成 RGB，再使用 Pillow 保存
small_rgb = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2RGB)
Image.fromarray(small_rgb).save(output_dir / "resized_half.png")

print(f"原图尺寸：{width} × {height}")
print(f"缩放尺寸：{new_width} × {new_height}")
import math

# 设置旋转角度与旋转中心
angle = 45
center = (width / 2, height / 2)

# 生成旋转矩阵：旋转 45°，保持原有比例
matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

# 计算旋转后需要的画布大小，避免裁掉图片边角
cos_value = abs(matrix[0, 0])
sin_value = abs(matrix[0, 1])

canvas_width = math.ceil(width * cos_value + height * sin_value)
canvas_height = math.ceil(height * cos_value + width * sin_value)

# 把旋转后的图片移动到新画布中心
matrix[0, 2] += canvas_width / 2 - center[0]
matrix[1, 2] += canvas_height / 2 - center[1]

# 执行旋转，空白区域填充白色
rotated_bgr = cv2.warpAffine(
    image_bgr,
    matrix,
    (canvas_width, canvas_height),
    flags=cv2.INTER_LINEAR,
    borderValue=(255, 255, 255),
)

rotated_rgb = cv2.cvtColor(rotated_bgr, cv2.COLOR_BGR2RGB)
Image.fromarray(rotated_rgb).save(output_dir / "rotated_45.png")

print(f"旋转完成，新画布尺寸：{canvas_width} × {canvas_height}")
import numpy as np

# 水平和垂直方向的移动距离
tx = 80
ty = 50

# 平移矩阵，使用浮点数类型
translation_matrix = np.float32([
    [1, 0, tx],
    [0, 1, ty],
])

# 保持原画布大小，空白区域填充白色
translated_bgr = cv2.warpAffine(
    image_bgr,
    translation_matrix,
    (width, height),
    borderValue=(255, 255, 255),
)

# 转换颜色顺序并保存
translated_rgb = cv2.cvtColor(translated_bgr, cv2.COLOR_BGR2RGB)
Image.fromarray(translated_rgb).save(output_dir / "translated.png")

print(f"平移完成：向右 {tx} 像素，向下 {ty} 像素")
# 三种翻转方式：名称与对应的翻转参数
flip_modes = {
    "horizontal": 1,   # 左右翻转
    "vertical": 0,     # 上下翻转
    "both": -1,        # 同时上下、左右翻转
}

for name, flip_code in flip_modes.items():
    flipped_bgr = cv2.flip(image_bgr, flip_code)

    flipped_rgb = cv2.cvtColor(flipped_bgr, cv2.COLOR_BGR2RGB)
    save_path = output_dir / f"flipped_{name}.png"
    Image.fromarray(flipped_rgb).save(save_path)

    print(f"翻转结果已保存：{save_path.name}")
    from pathlib import Path

import cv2
from PIL import Image

task_dir = Path(__file__).resolve().parent
output_dir = task_dir / "outputs"
output_dir.mkdir(parents=True, exist_ok=True)

image_path = task_dir / "assets" / "example1.jpg"
image = cv2.imread(str(image_path))

if image is None:
    raise FileNotFoundError(f"无法读取图片：{image_path}")

# 1. 转为灰度图
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 2. 高斯模糊，减少细小噪声
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# 3. 检测边缘
edges = cv2.Canny(blurred, 80, 160)

# 4. 闭运算，连接部分细小的边缘缺口
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

# 5. 查找最外层轮廓
contours, _ = cv2.findContours(
    closed_edges,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE,
)

# 6. 绘制轮廓并统计
result = image.copy()
total_area = 0.0
total_perimeter = 0.0

print(f"检测到 {len(contours)} 个外轮廓")

for index, contour in enumerate(contours, start=1):
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)

    total_area += area
    total_perimeter += perimeter

    cv2.drawContours(result, [contour], -1, (0, 255, 0), 2)
        # 在每个轮廓旁标注编号，与终端输出对应
    x, y, w, h = cv2.boundingRect(contour)

    cv2.putText(
        result,
        f"#{index}",
        (x, max(y - 8, 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 255),
        2,
    )
    print(
        f"轮廓 {index}："
        f"面积 = {area:.2f}，周长 = {perimeter:.2f}"
    )

# 7. 保存中间步骤与最终结果
Image.fromarray(gray).save(output_dir / "contour_gray.png")
Image.fromarray(closed_edges).save(output_dir / "contour_edges.png")

result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
Image.fromarray(result_rgb).save(output_dir / "contour_result.png")

# 将统计结果保存成文本，方便写报告
summary = (
    f"外轮廓数量：{len(contours)}\n"
    f"总面积：{total_area:.2f} 像素平方\n"
    f"总周长：{total_perimeter:.2f} 像素\n"
)
(output_dir / "contour_stats.txt").write_text(summary, encoding="utf-8")

print("\n" + summary)