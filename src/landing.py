"""落地偵測與落地品質分析（啟發式，v0.2）。

邏輯：
1. 找騰空段：髖部高度明顯高於全片中位數的連續區間
2. 落地幀：騰空段結束後，踝部下降速度由大轉零的那一幀
3. 品質指標：
   - 緩衝深度：落地後 0.5 秒內的最小膝角（越小代表蹲越深）
   - 移步距離：落地後 0.8 秒內踝部水平位移（正規化座標）
   - 穩定時間：踝部速度降到接近零所需秒數

注意：單鏡頭 2D 啟發式，鏡頭晃動會影響結果。固定機位、側拍最準。
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class LandingResult:
    found: bool
    landing_time_s: float = np.nan
    landing_frame: int = -1
    min_knee_angle: float = np.nan      # 落地後最小膝角（度）
    ankle_drift: float = np.nan         # 落地後水平移動（正規化, ~0.05 以上可能有移步）
    settle_time_s: float = np.nan       # 穩定所需時間
    flight_peak_time_s: float = np.nan  # 騰空最高點時間
    notes: list = field(default_factory=list)


def detect_landing(angles_df: pd.DataFrame, fps: float) -> LandingResult:
    df = angles_df.dropna(subset=["hip_height", "ankle_y"]).reset_index(drop=True)
    if len(df) < int(fps):  # 少於 1 秒有效數據
        return LandingResult(found=False, notes=["有效姿態數據不足，無法分析落地"])

    hip = df["hip_height"].to_numpy()
    ankle_y = df["ankle_y"].to_numpy()
    t = df["time_s"].to_numpy()

    # 1. 騰空段：髖高 > 中位數 + 0.6 * (max - median)
    med, peak = np.median(hip), hip.max()
    if peak - med < 0.05:
        return LandingResult(found=False, notes=["未偵測到明顯騰空（髖部高度變化太小）"])
    thresh = med + 0.6 * (peak - med)
    airborne = hip > thresh
    if not airborne.any():
        return LandingResult(found=False, notes=["未偵測到騰空段"])

    # 取最後一個騰空段（成套結尾的下法）
    idx = np.where(airborne)[0]
    seg_end = idx[-1]
    peak_i = idx[np.argmax(hip[idx])]

    # 2. 落地幀：騰空結束後，踝部 y 速度（向下為正）由峰值回落到近零
    vel = np.gradient(ankle_y, t)
    search = slice(seg_end, min(seg_end + int(1.0 * fps), len(df) - 1))
    v_seg = vel[search]
    if len(v_seg) < 3:
        return LandingResult(found=False, notes=["騰空段太靠近影片結尾，看不到落地"])
    land_local = int(np.argmax(v_seg < np.maximum(v_seg.max() * 0.2, 0.01)))
    land_i = seg_end + land_local

    # 3. 品質指標
    after = df.iloc[land_i : min(land_i + int(0.5 * fps), len(df))]
    knees = pd.concat([after["left_knee"], after["right_knee"]])
    min_knee = float(knees.min()) if knees.notna().any() else np.nan

    after8 = df.iloc[land_i : min(land_i + int(0.8 * fps), len(df))]
    drift = float(after8["ankle_x"].max() - after8["ankle_x"].min()) if len(after8) else np.nan

    speed = np.sqrt(np.gradient(df["ankle_x"].to_numpy(), t) ** 2 + vel**2)
    settle = np.nan
    for j in range(land_i, len(df)):
        if speed[j] < 0.03:
            settle = float(t[j] - t[land_i])
            break

    notes = []
    if not np.isnan(drift) and drift > 0.05:
        notes.append("落地後踝部水平位移偏大，可能有移步或跳步")
    if not np.isnan(min_knee) and min_knee < 90:
        notes.append("落地緩衝深蹲明顯（最小膝角 < 90°），扣分風險")

    return LandingResult(
        found=True,
        landing_time_s=float(t[land_i]),
        landing_frame=int(df["frame"].iloc[land_i]),
        min_knee_angle=min_knee,
        ankle_drift=drift,
        settle_time_s=settle,
        flight_peak_time_s=float(t[peak_i]),
        notes=notes,
    )
