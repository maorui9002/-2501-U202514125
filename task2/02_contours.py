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
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

closed_edges = cv2.morphologyEx(
    edges,
    cv2.MORPH_CLOSE,
    kernel,
    iterations=1,
)

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