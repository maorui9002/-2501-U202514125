from pathlib import Path

import cv2

task_dir = Path(__file__).resolve().parent
video_path = task_dir / "assets" / "example2.mp4"
output_dir = task_dir / "outputs"
output_dir.mkdir(parents=True, exist_ok=True)

cap = cv2.VideoCapture(str(video_path))

if not cap.isOpened():
    raise RuntimeError(f"无法打开视频：{video_path}")

fps = cap.get(cv2.CAP_PROP_FPS)
if fps <= 0:
    fps = 30.0

# 读取第一帧，为后面的比较准备参考
success, first_frame = cap.read()

if not success:
    cap.release()
    raise RuntimeError("无法读取第一帧")

height, width = first_frame.shape[:2]

output_path = output_dir / "motion_mog2.mp4"
writer = cv2.VideoWriter(
    str(output_path),
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height),
)

if not writer.isOpened():
    cap.release()
    raise RuntimeError("无法创建输出视频")

# 第一帧转换成灰度图，并进行模糊去噪
# 创建背景模型
background_model = cv2.createBackgroundSubtractorMOG2(
    history=300,
    varThreshold=16,
    detectShadows=True,
)

# 用第一帧初始化背景模型
background_model.apply(first_frame)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
min_area = 500

# 保存第一帧，保持完整播放时长
writer.write(first_frame)
processed_frames = 1

print("开始检测，在视频窗口中按 Q 提前结束")

try:
    while True:
        success, frame = cap.read()

        if not success:
            break

               # 1. 更新背景模型，得到前景掩码
        foreground = background_model.apply(frame)

        # 2. 排除阴影，只保留前景
        _, mask = cv2.threshold(
            foreground, 200, 255, cv2.THRESH_BINARY
        )

        # 3. 去除孤立噪声并连接小缺口
        mask = cv2.morphologyEx(
            mask, cv2.MORPH_OPEN, kernel
        )

        connect_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (11, 11)
        )
        mask = cv2.morphologyEx(
            mask, cv2.MORPH_CLOSE, connect_kernel
        )

        # 5. 找出变化区域并绘制矩形框
        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        result = frame.copy()

        for contour in contours:
            if cv2.contourArea(contour) < min_area:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(
                result, (x, y), (x + w, y + h), (0, 255, 0), 2
            )

        # 6. 保存标注结果，显示结果和运动掩码
        writer.write(result)
        processed_frames += 1

        cv2.imshow("MOG2 Detection", result)
        cv2.imshow("Motion Mask", mask)


        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
finally:
    cap.release()
    writer.release()
    cv2.destroyAllWindows()

print(f"已保存 {processed_frames} 帧")
print(f"输出位置：{output_path}")