"""用平滑後的關鍵點重繪骨架疊圖影片。"""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

# 骨架連線（用平滑後數據自己畫，不依賴 mediapipe 的原始結果）
SKELETON = [
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"), ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"), ("right_knee", "right_ankle"),
    ("left_ankle", "left_heel"), ("left_heel", "left_foot_index"),
    ("right_ankle", "right_heel"), ("right_heel", "right_foot_index"),
]

LEFT_COLOR = (255, 160, 0)   # BGR 藍橘區分左右
RIGHT_COLOR = (0, 160, 255)


def render(video_path: Path, keypoints: pd.DataFrame, out_path: Path, landing_frame: int = -1) -> Path:
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    kp = keypoints.set_index("frame")
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx in kp.index:
            row = kp.loc[idx]
            for a, b in SKELETON:
                ax, ay = row.get(f"{a}_x", np.nan), row.get(f"{a}_y", np.nan)
                bx, by = row.get(f"{b}_x", np.nan), row.get(f"{b}_y", np.nan)
                if np.isnan(ax) or np.isnan(bx):
                    continue
                color = LEFT_COLOR if a.startswith("left") else RIGHT_COLOR if a.startswith("right") else (0, 255, 0)
                cv2.line(frame, (int(ax * w), int(ay * h)), (int(bx * w), int(by * h)), color, 2)
            if idx == landing_frame:
                cv2.putText(frame, "LANDING", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        writer.write(frame)
        idx += 1
    cap.release()
    writer.release()
    return out_path
