"""
수급 팩터 원재료 로더 (한국 특화) — 외국인·기관 순매수.

데이터: korea-market-data (james-brand, CC BY 4.0) — 코스피/코스닥 전종목
        일일 외국인·기관 순매수. 매 거래일 갱신, CSV/JSON.
반환: DataFrame(index=ticker, columns=[net5, net20, consec])
  net5/net20 = 최근 5/20거래일 순매수 합(외국인+기관), consec = 연속 순매수 일수.

미국은 무료로 동급 수급 데이터가 없어 supply를 중립(0.5)으로 둔다.
"""
from __future__ import annotations
import pandas as pd

SCOLS = ["net5", "net20", "consec"]


def load_supply_kr(tickers: list[str]) -> pd.DataFrame:
    """korea-market-data에서 순매수 시계열을 받아 net5/net20/consec 계산.
    네트워크 없으면 빈 프레임(→ supply 중립 처리)."""
    # TODO(Phase2): 최신 CSV/JSON fetch → 종목별 최근 20거래일 외국인+기관 순매수 →
    #   net5, net20 합계, consec(연속 순매수 일수) 산출
    return pd.DataFrame(columns=SCOLS)


def supply_z(sup: pd.DataFrame) -> pd.Series:
    """스냅샷 → 수급 z-score(횡단면). 순매수 강할수록 높음."""
    if sup is None or sup.empty:
        return pd.Series(dtype=float)
    s = sup.apply(pd.to_numeric, errors="coerce")

    def z(x):
        x = x.astype(float)
        sd = x.std(ddof=0)
        return (x - x.mean()) / sd if sd and sd == sd else x * 0.0

    return pd.concat([z(s["net5"]), z(s["net20"]), z(s["consec"])], axis=1).mean(axis=1)
