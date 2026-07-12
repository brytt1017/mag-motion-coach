"""MAG Motion Coach — 教練用網頁介面。

啟動:  streamlit run app.py
"""

import sys
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))

from pose_extract import extract
from smoothing import postprocess
from angles import compute_angles
from landing import detect_landing
from report import plot_angles, JOINT_LABELS
from annotate import render

st.set_page_config(page_title="MAG Motion Coach", page_icon="🤸", layout="wide")
st.title("🤸 MAG Motion Coach")
st.caption("上傳訓練影片 → 自動姿態分析。側面、固定機位拍攝效果最好。")

uploaded = st.file_uploader("上傳影片 (mp4/mov)", type=["mp4", "mov", "m4v"])
render_video = st.checkbox("產出骨架疊圖影片（較慢）", value=True)

if uploaded and st.button("開始分析", type="primary"):
    with tempfile.TemporaryDirectory() as tmp:
        video_path = Path(tmp) / uploaded.name
        video_path.write_bytes(uploaded.read())

        with st.status("分析中...", expanded=True) as status:
            st.write("1/4 姿態估計（最花時間）...")
            raw, meta = extract(video_path)
            st.write(f"　{meta.n_frames} 幀，偵測率 {meta.detect_rate*100:.0f}%")
            if meta.detect_rate < 0.3:
                st.warning("偵測率過低，結果可能不可靠（人太小、太模糊或多人入鏡）")

            st.write("2/4 時序後處理...")
            clean, stats = postprocess(raw, meta.fps)

            st.write("3/4 角度與落地分析...")
            ang = compute_angles(clean)
            landing = detect_landing(ang, meta.fps)

            st.write("4/4 產出圖表...")
            out_dir = Path(tmp) / "out"
            charts = plot_angles(ang, out_dir, landing)
            status.update(label="分析完成", state="complete")

        # === 結果 ===
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("落地分析")
            if landing.found:
                st.metric("落地時間", f"{landing.landing_time_s:.2f} s")
                c1, c2, c3 = st.columns(3)
                c1.metric("最小膝角", f"{landing.min_knee_angle:.0f}°" if landing.min_knee_angle == landing.min_knee_angle else "—")
                c2.metric("水平位移", f"{landing.ankle_drift:.3f}" if landing.ankle_drift == landing.ankle_drift else "—")
                c3.metric("穩定時間", f"{landing.settle_time_s:.2f} s" if landing.settle_time_s == landing.settle_time_s else "—")
                for n in landing.notes:
                    st.warning(n)
            else:
                st.info("；".join(landing.notes) or "未偵測到落地")

        with col2:
            st.subheader("關節活動範圍")
            rows = []
            from angles import JOINTS
            for name in JOINTS:
                s = ang[name].dropna()
                if len(s):
                    rows.append({"關節": JOINT_LABELS.get(name, name), "最小": f"{s.min():.1f}°", "最大": f"{s.max():.1f}°"})
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

        st.subheader("曲線圖")
        for p in charts:
            st.image(str(p))

        st.download_button("下載角度數據 (CSV)", ang.to_csv(index=False), "angles.csv", "text/csv")

        if render_video:
            st.subheader("骨架疊圖影片")
            out_video = Path(tmp) / "annotated.mp4"
            render(video_path, clean, out_video, landing.landing_frame)
            # 轉 H.264 才能在瀏覽器播放；失敗就提供下載
            import subprocess
            h264 = Path(tmp) / "annotated_h264.mp4"
            try:
                subprocess.run(
                    ["ffmpeg", "-y", "-i", str(out_video), "-c:v", "libx264", "-preset", "fast", str(h264)],
                    check=True, capture_output=True,
                )
                st.video(str(h264))
            except Exception:
                st.download_button("下載骨架影片 (mp4)", out_video.read_bytes(), "annotated.mp4")
