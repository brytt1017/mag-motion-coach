"""MAG Motion Coach — 分析 pipeline 入口。

用法:
    python src/analyze.py path/to/video.mp4 [--no-video]

輸出（<影片名>_analysis/ 目錄）:
    report.html         分析報告（落地品質、關節範圍、曲線圖）
    joint_angles.png    8 關節角度曲線
    hip_height.png      髖高/軀幹傾角曲線
    angles.csv          逐幀角度數據
    keypoints_raw.csv   原始關鍵點
    keypoints_clean.csv 後處理關鍵點
    annotated.mp4       骨架疊圖影片（--no-video 可跳過）
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from pose_extract import extract
from smoothing import postprocess
from angles import compute_angles
from landing import detect_landing
from report import plot_angles, write_html
from annotate import render


def run(video_path: Path, out_dir: Path | None = None, render_video: bool = True) -> Path:
    """跑完整 pipeline，回傳報告路徑。"""
    out_dir = out_dir or video_path.with_name(video_path.stem + "_analysis")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/5] 姿態估計: {video_path.name} ...")
    raw, meta = extract(video_path)
    raw.to_csv(out_dir / "keypoints_raw.csv", index=False)
    print(f"      {meta.n_frames} 幀, 偵測率 {meta.detect_rate * 100:.0f}%")
    if meta.detect_rate < 0.3:
        print("      ⚠ 偵測率過低，結果可能不可靠（人太小/太模糊/多人入鏡）")

    print("[2/5] 時序後處理（左右修正 + 平滑）...")
    clean, stats = postprocess(raw, meta.fps)
    clean.to_csv(out_dir / "keypoints_clean.csv", index=False)
    print(f"      修正左右互換 {stats['lr_swaps_fixed']} 幀")

    print("[3/5] 關節角度計算 ...")
    ang = compute_angles(clean)
    ang.to_csv(out_dir / "angles.csv", index=False)

    print("[4/5] 落地分析 ...")
    landing = detect_landing(ang, meta.fps)
    if landing.found:
        print(f"      落地 @ {landing.landing_time_s:.2f}s, 最小膝角 {landing.min_knee_angle:.0f}°")
        for n in landing.notes:
            print(f"      ⚠ {n}")
    else:
        print(f"      {'; '.join(landing.notes)}")

    print("[5/5] 產出報告 ...")
    charts = plot_angles(ang, out_dir, landing)
    report = write_html(out_dir / "report.html", video_path.name, meta, stats, ang, landing, charts)
    if render_video:
        render(video_path, clean, out_dir / "annotated.mp4", landing.landing_frame)

    print(f"\n完成 → {report}")
    return report


def main():
    p = argparse.ArgumentParser(description="MAG 影片動作分析")
    p.add_argument("video", type=Path)
    p.add_argument("--no-video", action="store_true", help="跳過骨架疊圖影片（較快）")
    args = p.parse_args()
    if not args.video.exists():
        sys.exit(f"找不到檔案: {args.video}")
    run(args.video, render_video=not args.no_video)


if __name__ == "__main__":
    main()
