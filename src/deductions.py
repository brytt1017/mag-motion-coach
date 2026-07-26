"""落地扣分估算（v0.2.1）。

把 landing.py 算出的客觀指標，對應到 FIG MAG 落地類扣分的「參考估值」。

⚠ 重要聲明
    這不是評分系統，也不能取代裁判。單鏡頭 2D 量測有誤差，
    FIG 扣分還牽涉裁判對動作意圖與整體印象的判斷。
    本模組只回答一個問題：「以量到的數字來看，這次落地大概落在哪個扣分區間？」

門檻值放在 THRESHOLDS，方便使用者用自己的影片校準
（例如拍 10 次已知扣分的落地，回頭調整 ankle_drift 的公尺換算）。
"""

from dataclasses import dataclass, field

import numpy as np

from landing import LandingResult

# FIG MAG 一般錯誤扣分級距
SMALL, MEDIUM, LARGE = 0.10, 0.30, 0.50

THRESHOLDS = {
    # 踝部水平位移（正規化畫面寬度）。需依拍攝距離校準：
    # 選手佔畫面 1/3、側拍約 3m 遠時，0.03 ≈ 一個腳掌寬
    "drift_small": 0.03,   # 小幅調整腳步
    "drift_medium": 0.08,  # 明顯一步
    "drift_large": 0.15,   # 大步或跳步
    # 落地緩衝最小膝角（度）。越小代表蹲越深
    "knee_deep": 90.0,     # 蹲到大腿接近水平
    "knee_very_deep": 60.0,
    # 穩定所需時間（秒）
    "settle_slow": 0.6,
    "settle_very_slow": 1.2,
}


@dataclass
class DeductionItem:
    category: str
    value: float
    reason: str


@dataclass
class DeductionEstimate:
    total: float = 0.0
    items: list = field(default_factory=list)
    confidence: str = "low"   # low / medium — 永遠不宣稱 high
    caveats: list = field(default_factory=list)

    def summary(self) -> str:
        if not self.items:
            return "未偵測到明顯落地扣分項（不代表零扣分）"
        lines = [f"估算總扣分 −{self.total:.2f}（僅供參考）"]
        lines += [f"  · {i.category} −{i.value:.2f}：{i.reason}" for i in self.items]
        return "\n".join(lines)


def estimate(landing: LandingResult, detect_rate: float = 1.0) -> DeductionEstimate:
    """由落地指標估算扣分區間。detect_rate 用於判斷結果可信度。"""
    est = DeductionEstimate()

    if not landing.found:
        est.caveats.append("未偵測到落地，無法估算")
        return est

    t = THRESHOLDS

    # 1. 移步 / 跳步
    drift = landing.ankle_drift
    if not np.isnan(drift):
        if drift >= t["drift_large"]:
            est.items.append(DeductionItem(
                "移步", LARGE, f"踝部水平位移 {drift:.3f}，達大步或跳步等級"))
        elif drift >= t["drift_medium"]:
            est.items.append(DeductionItem(
                "移步", MEDIUM, f"踝部水平位移 {drift:.3f}，約一個明顯步伐"))
        elif drift >= t["drift_small"]:
            est.items.append(DeductionItem(
                "移步", SMALL, f"踝部水平位移 {drift:.3f}，小幅調整腳步"))

    # 2. 緩衝過深
    knee = landing.min_knee_angle
    if not np.isnan(knee):
        if knee < t["knee_very_deep"]:
            est.items.append(DeductionItem(
                "緩衝過深", MEDIUM, f"最小膝角 {knee:.0f}°，蹲得很深"))
        elif knee < t["knee_deep"]:
            est.items.append(DeductionItem(
                "緩衝過深", SMALL, f"最小膝角 {knee:.0f}°，緩衝略深"))

    # 3. 穩定過慢（通常伴隨上肢擺動找平衡）
    settle = landing.settle_time_s
    if not np.isnan(settle):
        if settle >= t["settle_very_slow"]:
            est.items.append(DeductionItem(
                "穩定過慢", MEDIUM, f"落地後 {settle:.2f}s 才穩定，明顯找平衡"))
        elif settle >= t["settle_slow"]:
            est.items.append(DeductionItem(
                "穩定過慢", SMALL, f"落地後 {settle:.2f}s 才穩定"))

    est.total = round(sum(i.value for i in est.items), 2)

    # 可信度：多數指標齊全且姿態偵測率高時才給 medium
    have = sum(not np.isnan(v) for v in (drift, knee, settle))
    est.confidence = "medium" if (have == 3 and detect_rate >= 0.7) else "low"

    est.caveats.append("移步門檻依畫面比例設定，換拍攝距離請重新校準 THRESHOLDS")
    if detect_rate < 0.7:
        est.caveats.append(f"姿態偵測率僅 {detect_rate * 100:.0f}%，數值可靠度下降")
    if est.total == 0:
        est.caveats.append("零扣分僅代表本工具量到的三項指標都在門檻內")

    return est
