"""關鍵點時序後處理：左右互換修正 → 低可見度剔除 → 插值 → 平滑。

MediaPipe 在快速旋轉/遮擋時常見兩種毛病:
1. 左右手腳互換（left_ankle 跟 right_ankle 突然交換）
2. 關鍵點短暫丟失或亂跳

這裡用「與前一幀連續性」的簡單啟發式修左右互換，
用 visibility 門檻剔除垃圾點，再線性插值 + Savitzky-Golay 平滑。
"""

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

# 左右成對的 landmark（不含臉部，臉部對分析無用）
LR_PAIRS = [
    ("left_shoulder", "right_shoulder"),
    ("left_elbow", "right_elbow"),
    ("left_wrist", "right_wrist"),
    ("left_hip", "right_hip"),
    ("left_knee", "right_knee"),
    ("left_ankle", "right_ankle"),
    ("left_heel", "right_heel"),
    ("left_foot_index", "right_foot_index"),
]

BODY_LANDMARKS = [n for pair in LR_PAIRS for n in pair] + ["nose"]


def _xy(row: pd.Series, name: str) -> np.ndarray:
    return np.array([row[f"{name}_x"], row[f"{name}_y"]])


def fix_lr_swaps(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """逐幀檢查：若整組左右互換後與前一幀位移更小，就換回來。

    以所有左右對的總位移比較（整組換，不單點換，避免把真實交叉動作改壞）。
    回傳 (修正後 df, 修正幀數)。
    """
    df = df.copy()
    n_fixed = 0
    prev = None
    for i in df.index:
        row = df.loc[i]
        if np.isnan(row.get("left_ankle_x", np.nan)):
            continue
        if prev is not None:
            d_keep, d_swap = 0.0, 0.0
            for l, r in LR_PAIRS:
                pl, pr = _xy(prev, l), _xy(prev, r)
                cl, cr = _xy(row, l), _xy(row, r)
                if np.isnan(pl).any() or np.isnan(cl).any():
                    continue
                d_keep += np.linalg.norm(cl - pl) + np.linalg.norm(cr - pr)
                d_swap += np.linalg.norm(cr - pl) + np.linalg.norm(cl - pr)
            if d_swap < d_keep * 0.8:  # 明顯更小才換，避免抖動誤判
                for l, r in LR_PAIRS:
                    for s in ("x", "y", "z", "v"):
                        a, b = f"{l}_{s}", f"{r}_{s}"
                        df.loc[i, [a, b]] = df.loc[i, [b, a]].values
                n_fixed += 1
        prev = df.loc[i]
    return df, n_fixed


def clean_and_smooth(
    df: pd.DataFrame,
    fps: float,
    min_visibility: float = 0.5,
    max_gap_s: float = 0.3,
) -> pd.DataFrame:
    """剔除低可見度點 → 插值（限最大缺口）→ Savitzky-Golay 平滑。"""
    df = df.copy()
    max_gap = max(1, int(max_gap_s * fps))

    for name in BODY_LANDMARKS:
        vis = df[f"{name}_v"]
        for s in ("x", "y", "z"):
            col = f"{name}_{s}"
            series = df[col].where(vis >= min_visibility)
            series = series.interpolate(limit=max_gap, limit_direction="both")
            # 平滑窗口約 0.2 秒，必須是奇數且 >= 5
            win = max(5, int(0.2 * fps) | 1)
            valid = series.notna()
            if valid.sum() > win:
                # .copy(): 新版 pandas (Copy-on-Write) 的 to_numpy 回傳唯讀 view
                arr = series.to_numpy(dtype=float).copy()
                idx = np.where(valid.to_numpy())[0]
                seg = arr[idx]
                if len(seg) > win:
                    arr[idx] = savgol_filter(seg, win, polyorder=2)
                series = pd.Series(arr, index=series.index)
            df[col] = series
    return df


def postprocess(df: pd.DataFrame, fps: float, min_visibility: float = 0.5) -> tuple[pd.DataFrame, dict]:
    """完整後處理流程。回傳 (處理後 df, 統計資訊)。"""
    df2, n_fixed = fix_lr_swaps(df)
    df3 = clean_and_smooth(df2, fps, min_visibility)
    stats = {"lr_swaps_fixed": n_fixed}
    return df3, stats
