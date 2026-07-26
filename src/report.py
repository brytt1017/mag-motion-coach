"""輸出圖表（PNG）與 HTML 分析報告。"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from angles import JOINTS
from landing import LandingResult

plt.rcParams["font.sans-serif"] = ["PingFang TC", "Microsoft JhengHei", "Noto Sans CJK TC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

JOINT_LABELS = {
    "left_elbow": "左肘", "right_elbow": "右肘",
    "left_shoulder": "左肩", "right_shoulder": "右肩",
    "left_hip": "左髖", "right_hip": "右髖",
    "left_knee": "左膝", "right_knee": "右膝",
}


def plot_angles(angles: pd.DataFrame, out_dir: Path, landing: LandingResult) -> list[Path]:
    """畫關節角度曲線（2x4）與髖高曲線，回傳 PNG 路徑。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    fig, axes = plt.subplots(2, 4, figsize=(16, 7), sharex=True)
    for ax, (name, _) in zip(axes.flat, JOINTS.items()):
        ax.plot(angles["time_s"], angles[name], lw=1.2)
        ax.set_title(JOINT_LABELS.get(name, name), fontsize=11)
        ax.set_ylim(0, 190)
        ax.grid(alpha=0.3)
        if landing.found:
            ax.axvline(landing.landing_time_s, color="red", ls="--", lw=0.8)
    fig.supxlabel("時間 (秒)")
    fig.supylabel("角度 (度)")
    fig.suptitle("關節角度曲線（紅線 = 落地）", fontsize=13)
    fig.tight_layout()
    p = out_dir / "joint_angles.png"
    fig.savefig(p, dpi=110)
    plt.close(fig)
    paths.append(p)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(angles["time_s"], angles["hip_height"], lw=1.5, label="髖部高度")
    ax.plot(angles["time_s"], angles["trunk_lean"] / 180, lw=1.0, alpha=0.6, label="軀幹傾角（/180）")
    if landing.found:
        ax.axvline(landing.landing_time_s, color="red", ls="--", label="落地")
        if not np.isnan(landing.flight_peak_time_s):
            ax.axvline(landing.flight_peak_time_s, color="orange", ls=":", label="騰空最高點")
    ax.set_xlabel("時間 (秒)")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_title("髖部高度與軀幹傾角")
    fig.tight_layout()
    p = out_dir / "hip_height.png"
    fig.savefig(p, dpi=110)
    plt.close(fig)
    paths.append(p)

    return paths


def _fmt(x, unit="", nd=1):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}{unit}"


def write_html(
    out_path: Path,
    video_name: str,
    meta,
    stats: dict,
    angles: pd.DataFrame,
    landing: LandingResult,
    chart_paths: list[Path],
    deductions=None,
) -> Path:
    """輸出單檔 HTML 報告（圖表以相對路徑引用）。"""
    rows = ""
    for name in JOINTS:
        s = angles[name].dropna()
        if len(s):
            rows += f"<tr><td>{JOINT_LABELS.get(name, name)}</td><td>{s.min():.1f}°</td><td>{s.max():.1f}°</td></tr>"

    landing_html = "<p>未偵測到落地。</p>"
    if landing.found:
        notes = "".join(f"<li>{n}</li>" for n in landing.notes) or "<li>無明顯扣分警示</li>"
        landing_html = f"""
        <table>
        <tr><td>落地時間</td><td>{_fmt(landing.landing_time_s, ' s', 2)}</td></tr>
        <tr><td>落地後最小膝角</td><td>{_fmt(landing.min_knee_angle, '°')}</td></tr>
        <tr><td>落地後水平位移</td><td>{_fmt(landing.ankle_drift, '', 3)}（>0.05 疑似移步）</td></tr>
        <tr><td>穩定時間</td><td>{_fmt(landing.settle_time_s, ' s', 2)}</td></tr>
        </table>
        <ul>{notes}</ul>"""
    elif landing.notes:
        landing_html = "<p>" + "；".join(landing.notes) + "</p>"

    ded_html = ""
    if deductions is not None and landing.found:
        conf = {"low": "低", "medium": "中"}.get(deductions.confidence, deductions.confidence)
        if deductions.items:
            items = "".join(
                f"<tr><td>{i.category}</td><td>−{i.value:.2f}</td><td>{i.reason}</td></tr>"
                for i in deductions.items
            )
            body = f"""<table>
        <tr><th>項目</th><th>估算扣分</th><th>依據</th></tr>
        {items}
        <tr><td><strong>合計</strong></td><td><strong>−{deductions.total:.2f}</strong></td><td>可信度：{conf}</td></tr>
        </table>"""
        else:
            body = f"<p>未偵測到明顯落地扣分項（可信度：{conf}）。</p>"
        caveats = "".join(f"<li>{c}</li>" for c in deductions.caveats)
        ded_html = f"""
<h2>落地扣分估算</h2>
<p class="warn">⚠ 這不是評分系統，不能取代裁判。以下僅是「量到的數字落在哪個扣分區間」的參考。</p>
{body}
<ul class="meta">{caveats}</ul>"""

    imgs = "".join(f'<img src="{p.name}" style="max-width:100%">' for p in chart_paths)

    html = f"""<!DOCTYPE html>
<html lang="zh-TW"><head><meta charset="utf-8">
<title>分析報告 - {video_name}</title>
<style>
body {{ font-family: "PingFang TC", "Microsoft JhengHei", sans-serif; max-width: 1100px; margin: 2rem auto; padding: 0 1rem; color: #222; }}
table {{ border-collapse: collapse; margin: .5rem 0 1.5rem; }}
td, th {{ border: 1px solid #ccc; padding: .35rem .8rem; }}
h1 {{ font-size: 1.5rem; }} h2 {{ font-size: 1.15rem; margin-top: 2rem; }}
.meta {{ color: #666; font-size: .9rem; }}
.warn {{ background: #fff8e1; border-left: 3px solid #f0ad4e; padding: .5rem .8rem; font-size: .9rem; }}
</style></head><body>
<h1>MAG Motion Coach 分析報告</h1>
<p class="meta">影片：{video_name} ｜ {meta.n_frames} 幀 / {meta.n_frames / meta.fps:.1f} 秒 @ {meta.fps:.0f} fps
｜ 姿態偵測率 {meta.detect_rate * 100:.0f}% ｜ 左右互換修正 {stats.get('lr_swaps_fixed', 0)} 幀</p>

<h2>落地分析</h2>
{landing_html}
{ded_html}

<h2>關節活動範圍</h2>
<table><tr><th>關節</th><th>最小</th><th>最大</th></tr>{rows}</table>

<h2>曲線圖</h2>
{imgs}

<p class="meta">單鏡頭 2D 估計，深度方向角度僅供參考。側面固定機位拍攝結果最準。</p>
</body></html>"""
    out_path.write_text(html, encoding="utf-8")
    return out_path
