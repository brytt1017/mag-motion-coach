# MAG Motion Coach 🤸

男子競技體操（MAG）訓練影片分析工具。給教練與選手用：上傳影片，自動輸出姿態分析報告——關節角度曲線、落地品質評估、骨架疊圖影片。

靈感來自 FIG × Fujitsu 的 [Judging Support System](https://www.gymnastics.sport/site/pages/judges-support.php)，這是它的開源、單鏡頭、輕量版。

## 功能（v0.2）

**分析 pipeline**（`src/`，模組化）：

| 模組 | 功能 |
|---|---|
| `pose_extract.py` | MediaPipe Pose 逐幀抽 33 個關鍵點 |
| `smoothing.py` | 左右互換修正（連續性啟發式）＋ 低可見度剔除 ＋ 插值 ＋ Savitzky-Golay 平滑 |
| `angles.py` | 8 個關節角度、髖部高度、軀幹傾角（向量化計算） |
| `landing.py` | 落地偵測：騰空段 → 落地幀 → 緩衝膝角 / 移步距離 / 穩定時間 |
| `report.py` | 曲線圖 PNG ＋ 單檔 HTML 報告 |
| `annotate.py` | 用平滑後關鍵點重繪骨架疊圖影片（左右不同色） |

**兩種用法**：

> ⚠️ **需要 Python 3.10–3.12**。mediapipe 不支援 3.13+，太新的版本會裝到殘缺套件，出現 `module 'mediapipe' has no attribute 'solutions'`。建議 `python3.12 -m venv .venv` 建虛擬環境。

```bash
pip install -r requirements.txt

# 命令列
python src/analyze.py 影片.mp4          # 完整分析
python src/analyze.py 影片.mp4 --no-video  # 跳過疊圖影片，較快

# 網頁介面（給教練）
streamlit run app.py
```

CLI 輸出到 `<影片名>_analysis/`：`report.html`、曲線圖、逐幀 CSV、`annotated.mp4`。

## 拍攝建議（直接影響精度）

- 側面、固定機位（腳架），選手佔畫面 1/3 以上
- 避免多人入鏡，MediaPipe 只追蹤一人
- 燈光充足、快門夠快，減少動態模糊

## 已知限制（誠實聲明）

- 單鏡頭 2D：朝鏡頭方向的轉體度數測不準，深度方向角度誤差大
- 快速空翻轉體時關鍵點可能丟失——後處理能救一部分，救不了全部
- 落地偵測是啟發式，鏡頭晃動會誤判
- 這是訓練輔助工具，不是評分系統

## Roadmap

- [x] **v0.1** 骨架疊圖 + 關節角度 CSV
- [x] **v0.2** 時序後處理、落地分析、HTML 報告、Streamlit 介面
- [ ] **v0.3** 動作比對：兩段影片並排、角度曲線疊圖（選手 vs 示範）
- [ ] **v0.4** 換強一點的姿態模型（RTMPose）＋ 旋轉計數
- [ ] **v0.5** 器械模組：吊環/鞍馬靜止力量動作的角度判定
- [ ] **v1.0** 動作要素辨識（需自建標註資料集，時序模型）

## 專案結構

```
mag-motion-coach/
├── README.md
├── requirements.txt
├── app.py               # Streamlit 網頁介面
└── src/
    ├── analyze.py       # CLI 入口
    ├── pose_extract.py
    ├── smoothing.py
    ├── angles.py
    ├── landing.py
    ├── report.py
    └── annotate.py
```

## 授權

[MIT License](LICENSE)
