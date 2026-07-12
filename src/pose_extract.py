"""影片 → 逐幀 33 個關鍵點原始座標。

輸出 DataFrame 欄位:
    frame, time_s, 以及每個 landmark 的 {name}_x, {name}_y, {name}_z, {name}_v
座標為正規化影像座標 (0-1)，y 向下。v = visibility。
"""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import mediapipe as mp

mp_pose = mp.solutions.pose

LANDMARK_NAMES = [lm.name.lower() for lm in mp_pose.PoseLandmark]


@dataclass
class VideoMeta:
    fps: float
    width: int
    height: int
    n_frames: int
    detect_rate: float  # 姿態偵測成功率 0-1


def extract(video_path: Path, model_complexity: int = 1) -> tuple[pd.DataFrame, VideoMeta]:
    """跑 MediaPipe Pose，回傳原始關鍵點 DataFrame 與影片資訊。"""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"無法開啟影片: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    rows = []
    detected = 0
    idx = 0
    with mp_pose.Pose(
        model_complexity=model_complexity,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            # 4K 影片先降到 1280 寬再餵模型：關鍵點是正規化座標，
            # 精度幾乎不受影響，速度快數倍，也降低原生層崩潰風險
            if frame.shape[1] > 1280:
                scale = 1280 / frame.shape[1]
                frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            result = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            row = {"frame": idx, "time_s": idx / fps}
            if result.pose_landmarks:
                detected += 1
                for name, lm in zip(LANDMARK_NAMES, result.pose_landmarks.landmark):
                    row[f"{name}_x"] = lm.x
                    row[f"{name}_y"] = lm.y
                    row[f"{name}_z"] = lm.z
                    row[f"{name}_v"] = lm.visibility
            else:
                for name in LANDMARK_NAMES:
                    row[f"{name}_x"] = np.nan
                    row[f"{name}_y"] = np.nan
                    row[f"{name}_z"] = np.nan
                    row[f"{name}_v"] = 0.0
            rows.append(row)
            idx += 1
    cap.release()

    meta = VideoMeta(fps, width, height, idx, detected / idx if idx else 0.0)
    return pd.DataFrame(rows), meta
