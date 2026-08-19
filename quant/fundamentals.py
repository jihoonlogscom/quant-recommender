"""
재무 팩터 원재료 로더 (가치·퀄리티) — 실행 시에만 네트워크 사용.

반환: DataFrame(index=ticker, columns=[per, pbr, ev_ebitda, roe, op_margin, debt_ratio, earn_stability])
연산부는 이 스냅샷만 받으면 되므로 테스트는 합성 프레임을 주입해 검증한다.

주의: 현재 스냅샷이다. 과거 백테스트에 그대로 쓰면 look-ahead가 되므로
파이프라인은 재무 팩터를 '현재 랭킹/점수'에만 반영하고 확률 백테스트에서는 제외한다.
Phase 2.5에서 point-in-time(공시일 정렬) 재무로 확장.
"""
from __future__ import annotations
import os
import pandas as pd

FCOLS = ["per", "pbr", "ev_ebitda", "roe", "op_margin", "debt_ratio", "earn_stability"]


def load_fundamentals_kr(tickers: list[str]) -> pd.DataFrame:
    """DART OpenAPI(무료, DART_API_KEY 필요) 기반. 키 없으면 빈 프레임 반환.
    실제 구현은 재무제표 API에서 최근 4~8분기를 받아 지표를 계산한다."""
    key = os.getenv("DART_API_KEY")
    if not key:
        return pd.DataFrame(columns=FCOLS)
    # TODO(Phase2): OpenDartReader 등으로 재무 수집 → 아래 지표 계산
    #   per/pbr: 가격 대비 EPS/BPS, ev_ebitda, roe=순이익/자본, op_margin, debt_ratio,
    #   earn_stability=최근 분기 EPS 변동성의 역수
    return pd.DataFrame(columns=FCOLS)


def load_fundamentals_us(tickers: list[str]) -> pd.DataFrame:
    """SEC EDGAR companyfacts(무료, 무키) 또는 yfinance 기반. 여기선 yfinance 예시 골격."""
    rows = {}
    try:
        import yfinance as yf
    except Exception:
        return pd.DataFrame(columns=FCOLS)
    for tk in tickers:
        try:
            info = yf.Ticker(tk).info
            rows[tk] = {
                "per": info.get("trailingPE"),
                "pbr": info.get("priceToBook"),
                "ev_ebitda": info.get("enterpriseToEbitda"),
                "roe": info.get("returnOnEquity"),
                "op_margin": info.get("operatingMargins"),
                "debt_ratio": info.get("debtToEquity"),
                "earn_stability": None,
            }
        except Exception:
            continue
    return pd.DataFrame(rows).T.reindex(columns=FCOLS) if rows else pd.DataFrame(columns=FCOLS)


def value_quality_z(fund: pd.DataFrame) -> pd.DataFrame:
    """스냅샷 → 가치/퀄리티 z-score (횡단면). 싼 게 좋은 팩터는 역수화."""
    if fund is None or fund.empty:
        return pd.DataFrame(columns=["value_z", "quality_z"])
    f = fund.apply(pd.to_numeric, errors="coerce")

    def z(s):
        s = s.astype(float)
        sd = s.std(ddof=0)
        return (s - s.mean()) / sd if sd and sd == sd else s * 0.0

    inv = lambda s: z(1.0 / s.where(s > 0))               # 낮을수록 좋음 → 역수
    value = pd.concat([inv(f["per"]), inv(f["pbr"]), inv(f["ev_ebitda"])], axis=1).mean(axis=1)
    quality = pd.concat([z(f["roe"]), z(f["op_margin"]),
                         -z(f["debt_ratio"]), z(f["earn_stability"])], axis=1).mean(axis=1)
    out = pd.DataFrame({"value_z": value, "quality_z": quality})
    return out
