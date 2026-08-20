"""
일봉 증분 캐시 (parquet) — 유니버스 확장을 무료 시간 예산 안에 넣기 위한 계층 갱신.

- 종목별 parquet 파일에 OHLCV를 누적 저장.
- 매 실행 시 캐시의 마지막 날짜 이후 구간만 새로 받아 append → 매일 전체 재수집 방지.
- fetch_fn(ticker, start) 을 주입받아 데이터 소스(FDR/yfinance)와 분리한다(테스트 가능).
"""
from __future__ import annotations
import os
import pandas as pd

COLS = ["Open", "High", "Low", "Close", "Volume"]


def _path(cachedir, market, ticker):
    d = os.path.join(cachedir, market)
    os.makedirs(d, exist_ok=True)
    safe = str(ticker).replace("/", "_")
    return os.path.join(d, f"{safe}.parquet")


def load_one(cachedir, market, ticker) -> pd.DataFrame | None:
    p = _path(cachedir, market, ticker)
    if os.path.exists(p):
        try:
            return pd.read_parquet(p)
        except Exception:
            return None
    return None


def save_one(cachedir, market, ticker, df: pd.DataFrame):
    try:
        df[COLS].to_parquet(_path(cachedir, market, ticker))
    except Exception:
        pass


def update_panel(market, tickers, start, fetch_fn, cachedir=".cache/prices",
                 min_history=260, stale_days=3) -> dict[str, pd.DataFrame]:
    """캐시를 읽고 부족분만 fetch_fn으로 채워 최신 패널을 반환·저장한다.

    fetch_fn(ticker, start_str) -> OHLCV DataFrame (index=Datetime).
    """
    today = pd.Timestamp.today().normalize()
    panel = {}
    for tk in tickers:
        cached = load_one(cachedir, market, tk)
        if cached is not None and len(cached):
            last = pd.to_datetime(cached.index.max())
            if (today - last).days <= stale_days:
                df = cached                                   # 충분히 최신 → 그대로
            else:
                fresh = fetch_fn(tk, (last + pd.Timedelta(days=1)).strftime("%Y-%m-%d"))
                df = cached if fresh is None or fresh.empty else \
                    pd.concat([cached, fresh[COLS]]).sort_index()
                df = df[~df.index.duplicated(keep="last")]
                save_one(cachedir, market, tk, df)
        else:
            fresh = fetch_fn(tk, start)
            if fresh is None or fresh.empty:
                continue
            df = fresh[COLS].sort_index()
            save_one(cachedir, market, tk, df)
        if len(df) >= min_history:
            panel[tk] = df
    return panel
