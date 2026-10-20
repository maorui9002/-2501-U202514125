from pathlib import Path

import cv2

task_dir = Path(__file__).resolve().parent
video_path = task_dir / "assets" / "example3.mp4"
output_dir = task_dir / "outputs"
output_dir.mkdir(parents=True, exist_ok=True)

output_path = output_dir / "ball_tracking.mp4"

cap = cv2.VideoCapture(str(video_path))
writer = None
saved_frames = 0

try:
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频：{video_path}")

    # 1. 读取第一帧
    success, first_frame = cap.read()

    if not success:
        raise RuntimeError("无法读取视频第一帧")

    height, width = first_frame.shape[:2]
    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        fps = 30.0

        # 2. 先播放视频，等小球出现后按 S 框选
    print("小球出现后，按 S 暂停并框选")
    print("按 Q 退出")

    cv2.namedWindow("Select Ball", cv2.WINDOW_NORMAL)
    preview_delay = max(1, round(1000 / fps))

    while True:
        preview = first_frame.copy()

        cv2.putText(
            preview, "Press S to select ball",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8, (0, 255, 255), 2,
        )

        cv2.imshow("Select Ball", preview)
        key = cv2.waitKey(preview_delay) & 0xFF

        if key == ord("s"):
            # 在干净的原始画面上框选
            bbox = cv2.selectROI(
                "Select Ball",
                first_frame,
                showCrosshair=True,
                fromCenter=False,
            )

            if bbox[2] > 0 and bbox[3] > 0:
                break

            print("未选中目标，继续播放")

        elif key == ord("q"):
            raise RuntimeError("已取消框选")

        success, first_frame = cap.read()

        if not success:
            raise RuntimeError("视频已结束，请重新运行并提前按 S")

    cv2.destroyWindow("Select Ball")

    x, y, w, h = map(int, bbox)

    if w <= 0 or h <= 0:
        raise RuntimeError("没有选择有效的小球区域，请重新运行")

    # 3. 创建追踪器，用第一帧中的小球初始化
    tracker = cv2.TrackerCSRT_create()
    tracker.init(first_frame, (x, y, w, h))

    # 4. 创建输出视频
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    if not writer.isOpened():
        raise RuntimeError("无法创建输出视频")
        # 当前选中帧的索引，第一帧的索引为 0
    selected_index = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1

    # 重新打开视频，补存框选之前的画面
    prefix_cap = cv2.VideoCapture(str(video_path))

    try:
        if not prefix_cap.isOpened():
            raise RuntimeError("无法重新打开视频以保存前段画面")

        for _ in range(selected_index):
            success, prefix_frame = prefix_cap.read()

            if not success:
                raise RuntimeError("前段视频读取失败")

            writer.write(prefix_frame)
            saved_frames += 1
    finally:
        prefix_cap.release()
    # 标注并保存第一帧；追踪器初始化使用的是未画框的原图
    first_result = first_frame.copy()
    cv2.rectangle(
        first_result,
        (x, y),
        (x + w, y + h),
        (0, 255, 0),
        2,
    )
    writer.write(first_result)
    saved_frames += 1

    cv2.namedWindow("Ball Tracking", cv2.WINDOW_NORMAL)
    print("开始追踪，在播放窗口中按 Q 提前结束")

    # 5. 逐帧更新小球位置
    while True:
        success, frame = cap.read()

        if not success:
            break

        tracked, bbox = tracker.update(frame)
        result = frame.copy()

        if tracked:
            x, y, w, h = map(int, bbox)

            cv2.rectangle(
                result,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2,
            )

            cv2.putText(
                result, "Tracking", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (0, 255, 0), 2,
            )
        else:
            cv2.putText(
                result, "Tracking lost", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (0, 0, 255), 2,
            )

        writer.write(result)
        saved_frames += 1

        cv2.imshow("Ball Tracking", result)

                # 刷新窗口，持续处理到视频结束
        cv2.waitKey(1)

finally:
    cap.release()

    if writer is not None:
        writer.release()

    cv2.destroyAllWindows()

print(f"已保存 {saved_frames} 帧")
print(f"结果位置：{output_path}")