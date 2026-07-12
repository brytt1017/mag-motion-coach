"""從平滑後的關鍵點計算關節角度與身體指標。"""

import numpy as np
import pandas as pd

# 關節角度定義: 名稱 -> (端點A, 頂點, 端點B)
JOINTS = {
    "left_elbow": ("left_shoulder", "left_elbow", "left_wrist"),
    "right_elbow": ("right_shoulder", "right_elbow", "right_wrist"),
    "left_shoulder": ("left_elbow", "left_shoulder", "left_hip"),
    "right_shoulder": ("right_elbow", "right_shoulder", "right_hip"),
    "left_hip": ("left_shoulder", "left_hip", "left_knee"),
    "right_hip": ("right_shoulder", "right_hip", "right_knee"),
    "left_knee": ("left_hip", "left_knee", "left_ankle"),
    "right_knee": ("right_hip", "right_knee", "right_ankle"),
}


def _angle(ax, ay, bx, by, cx, cy):
    """向量化三點夾角（頂點 b），輸入為 numpy array，回傳角度。"""
    bax, bay = ax - bx, ay - by
    bcx, bcy = cx - bx, cy - by
    dot = bax * bcx + bay * bcy
    norm = np.sqrt(bax**2 + bay**2) * np.sqrt(bcx**2 + bcy**2)
    with np.errstate(invalid="ignore", divide="ignore"):
        cos = np.clip(dot / norm, -1.0, 1.0)
    return np.degrees(np.arccos(cos))


def compute_angles(df: pd.DataFrame) -> pd.DataFrame:
    """回傳每幀關節角度 + 身體指標 DataFrame。"""
    out = pd.DataFrame({"frame": df["frame"], "time_s": df["time_s"]})

    for name, (a, b, c) in JOINTS.items():
        out[name] = _angle(
            df[f"{a}_x"].to_numpy(), df[f"{a}_y"].to_numpy(),
            df[f"{b}_x"].to_numpy(), df[f"{b}_y"].to_numpy(),
            df[f"{c}_x"].to_numpy(), df[f"{c}_y"].to_numpy(),
        )

    # 髖部中心高度（1 - y，值越大越高；正規化座標）
    hip_y = (df["left_hip_y"] + df["right_hip_y"]) / 2
    out["hip_height"] = 1.0 - hip_y

    # 軀幹傾角：髖中心 -> 肩中心向量與鉛直線的夾角（0 = 直立）
    sh_x = (df["left_shoulder_x"] + df["right_shoulder_x"]) / 2
    sh_y = (df["left_shoulder_y"] + df["right_shoulder_y"]) / 2
    hip_x = (df["left_hip_x"] + df["right_hip_x"]) / 2
    vx, vy = sh_x - hip_x, sh_y - hip_y  # y 向下
    out["trunk_lean"] = np.degrees(np.arctan2(np.abs(vx), -vy))

    # 踝部中心（落地分析用）
    out["ankle_x"] = (df["left_ankle_x"] + df["right_ankle_x"]) / 2
    out["ankle_y"] = (df["left_ankle_y"] + df["right_ankle_y"]) / 2

    return out
