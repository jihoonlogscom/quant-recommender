"""
데일리 퀀트 추천 — Phase 2 파이프라인
(모멘텀·가치·퀄리티·수급·기술 멀티팩터 + IC 가중치 + PBO/DSR 검증)

정직성 원칙
- 확률·검증 지표(prob_up, PBO, DSR, IC)는 **가격 기반 엔진**(모멘텀+추세)으로만 산출한다.
  과거 재무/수급 시계열이 없어 오늘 스냅샷을 과거에 적용하면 look-ahead이므로,
  재무·수급 팩터는 **현재 랭킹/점수와 표시**에만 반영한다(Phase 2.5에서 PIT 재무로 확장).
- 확률은 과거 히트레이트로 미래 수익 보장이 아니다. 본 도구는 투자 자문이 아니다.
"""
from __future__ import annotations
import json, math
from datetime import datetime, timezone
import numpy as np
import pandas as pd

from quant import validate as V
from quant import fundamentals as F
from quant import supply as SUP

CFG = dict(
    horizons=[5, 20, 60],
    min_history=260, liquidity_keep=0.6,
    bt_step=5, bt_lookback=504, n_deciles=10,
    buy_score=72, buy_prob=0.55, sell_score=42, sell_prob=0.50,
    verify_prob=0.55, gate_pbo=0.5, gate_dsr=0.5,   # 룰셋 검증 관문
    atr_entry_lo=0.5, atr_entry_hi=0.3, atr_stop=2.0,
    atr_bear=1.0, atr_base=2.5, atr_bull=5.0,
)
DEFAULT_W = dict(momentum=0.30, value=0.20, quality=0.20, supply=0.15, tech=0.15)
# "균형" 성향: 팩터 그룹별 예산. 가격 예산은 IC 비율로 모멘텀/추세에 분할.
GROUP_BUDGET = dict(price=0.40, value=0.20, quality=0.20, supply=0.20)
SECTORS = {}


# ============================ 데이터 로드 (네트워크) ============================
def load_universe(market: str) -> list[str]:
    import FinanceDataReader as fdr
    if market == "KR":
        df = fdr.StockListing("KRX"); df = df[df["Market"].isin(["KOSPI", "KOSDAQ"])]
        for _, r in df.iterrows():
            SECTORS[("KR", str(r["Code"]))] = {"name": r.get("Name", r["Code"]),
                                               "sector": r.get("Sector", "") or r.get("Industry", "")}
        return df["Code"].astype(str).tolist()
    if market == "US":
        df = fdr.StockListing("S&P500")
        for _, r in df.iterrows():
            SECTORS[("US", str(r["Symbol"]))] = {"name": r.get("Name", r["Symbol"]), "sector": r.get("Sector", "")}
        return df["Symbol"].astype(str).tolist()
    raise ValueError(market)


def load_prices(market: str, tickers: list[str], start: str) -> dict[str, pd.DataFrame]:
    panel = {}
    if market == "KR":
        import FinanceDataReader as fdr
        for tk in tickers:
            try:
                d = fdr.DataReader(tk, start)[["Open", "High", "Low", "Close", "Volume"]].dropna()
                if len(d) >= CFG["min_history"]:
                    panel[tk] = d
            except Exception:
                continue
    else:
        import yfinance as yf
        raw = yf.download(tickers, start=start, group_by="ticker", auto_adjust=True, threads=True, progress=False)
        for tk in tickers:
            try:
                d = raw[tk][["Open", "High", "Low", "Close", "Volume"]].dropna()
                if len(d) >= CFG["min_history"]:
                    panel[tk] = d
            except Exception:
                continue
    return panel


def load_fundamentals(market, tickers):
    return F.load_fundamentals_kr(tickers) if market == "KR" else F.load_fundamentals_us(tickers)


def load_supply(market, tickers):
    return SUP.load_supply_kr(tickers) if market == "KR" else pd.DataFrame(columns=SUP.SCOLS)


def load_regime(market: str) -> dict:
    try:
        if market == "KR":
            import FinanceDataReader as fdr
            idx = fdr.DataReader("KS11")["Close"].dropna()
        else:
            import yfinance as yf
            idx = yf.download("^GSPC", period="2y", auto_adjust=True, progress=False)["Close"].dropna()
        return regime_from_index(idx)
    except Exception:
        return dict(state="중립", note="지수 데이터 없음", led="neutral")


# ============================ 지표 ============================
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy(); c = d["Close"]
    d["ret_12_1"] = c.shift(21) / c.shift(252) - 1
    d["rs_63"] = c / c.shift(63) - 1
    d["hi_252"] = c / c.rolling(252).max()
    ma20, ma50, ma200 = c.rolling(20).mean(), c.rolling(50).mean(), c.rolling(200).mean()
    d["ma_align"] = ((c > ma20).astype(int) + (ma20 > ma50).astype(int) + (ma50 > ma200).astype(int)) / 3.0
    pc = c.shift(1)
    tr = pd.concat([d["High"] - d["Low"], (d["High"] - pc).abs(), (d["Low"] - pc).abs()], axis=1).max(axis=1)
    d["atr14"] = tr.rolling(14).mean()
    d["turnover"] = (c * d["Volume"]).rolling(20).mean()
    return d


def _z(s):
    s = pd.to_numeric(s, errors="coerce").astype(float)
    sd = s.std(ddof=0)
    return (s - s.mean()) / sd if sd and sd == sd else s * 0.0


def _fwd(d, i, h):
    return d["Close"].iloc[i + h] / d["Close"].iloc[i] - 1 if i + h < len(d) else np.nan


def _price_cross_section(panel, date):
    """특정 날짜의 가격 팩터 z + forward return (백테스트/현재 공용, look-ahead 없음)."""
    rows, fwd = {}, {}
    for tk, d in panel.items():
        if date not in d.index:
            continue
        r = d.loc[date]
        if pd.isna(r.get("ret_12_1")) or pd.isna(r.get("ma_align")):
            continue
        rows[tk] = dict(ret_12_1=r["ret_12_1"], rs_63=r["rs_63"], hi_252=r["hi_252"],
                        ma_align=r["ma_align"], turnover=r.get("turnover"),
                        close=r["Close"], atr14=r.get("atr14"))
        i = d.index.get_loc(date)
        fwd[tk] = {f"fwd{h}": _fwd(d, i, h) for h in CFG["horizons"]}
        fwd[tk]["ret_step"] = _fwd(d, i, CFG["bt_step"])
    if len(rows) < 5:
        return pd.DataFrame()
    f = pd.DataFrame(rows).T.join(pd.DataFrame(fwd).T)
    f["mom_z"] = (_z(f["ret_12_1"]) + _z(f["rs_63"]) + _z(f["hi_252"])) / 3.0
    f["trend_z"] = _z(f["ma_align"])
    return f


# ============================ 백테스트 (가격 엔진) ============================
def _sampled(panel):
    all_dates = sorted({dt for d in panel.values() for dt in d.index})
    hz = max(CFG["horizons"])
    if len(all_dates) < CFG["min_history"] + hz:
        return []
    window = all_dates[-(CFG["bt_lookback"] + hz):-hz]
    return [_price_cross_section(panel, dt) for dt in window[::CFG["bt_step"]]]


def _evaluate(frames, wm, wt):
    hit = {h: {dc: [0, 0] for dc in range(CFG["n_deciles"])} for h in CFG["horizons"]}
    series = []
    for f in frames:
        comp = wm * f["mom_z"] + wt * f["trend_z"]
        dec = (comp.rank(pct=True) * CFG["n_deciles"]).clip(upper=CFG["n_deciles"] - 1e-9).astype(int)
        for tk in f.index:
            dc = int(dec[tk])
            for h in CFG["horizons"]:
                v = f.loc[tk, f"fwd{h}"]
                if pd.notna(v):
                    hit[h][dc][1] += 1
                    if v > 0:
                        hit[h][dc][0] += 1
        top = f["ret_step"][dec == CFG["n_deciles"] - 1]
        series.append(top.mean() if len(top) and top.notna().any() else np.nan)
    decile = {dc: {f"d{h}": (round(hit[h][dc][0] / hit[h][dc][1], 3) if hit[h][dc][1] >= 20 else None)
                   for h in CFG["horizons"]} for dc in range(CFG["n_deciles"])}
    return decile, pd.Series(series, dtype=float)


def backtest_engine(panel):
    frames = [f for f in _sampled(panel) if not f.empty]
    if len(frames) < 6:
        return dict(decile={}, weights=(0.6, 0.4), pbo=None, dsr=None, ic={},
                    top_hit={f"d{h}": None for h in CFG["horizons"]})

    ic_mom = V.information_coefficient([(f["mom_z"], f["fwd20"]) for f in frames])
    ic_trd = V.information_coefficient([(f["trend_z"], f["fwd20"]) for f in frames])
    im, it = max(ic_mom["ic"] or 0, 0), max(ic_trd["ic"] or 0, 0)
    wm, wt = (0.6, 0.4) if im + it <= 0 else (im / (im + it), it / (im + it))

    grid = [(0.5, 0.5), (0.6, 0.4), (0.7, 0.3), (0.4, 0.6), (0.8, 0.2), (round(wm, 2), round(wt, 2))]
    cols, sr_trials = {}, []
    for (a, b) in grid:
        _, s = _evaluate(frames, a, b)
        cols[f"{a}_{b}"] = s.reset_index(drop=True)
        sd = s.std(ddof=1)
        sr_trials.append(float(s.mean() / sd) if sd and sd == sd else 0.0)
    pbo = V.pbo_cscv(pd.DataFrame(cols), n_splits=8)

    decile, chosen = _evaluate(frames, wm, wt)
    dsr = V.deflated_sharpe_ratio(chosen.dropna(), sr_trials)
    top = CFG["n_deciles"] - 1
    return dict(decile=decile, weights=(wm, wt), pbo=pbo,
                dsr=(round(dsr, 3) if dsr == dsr else None),
                ic={"momentum": ic_mom, "tech": ic_trd},
                top_hit={f"d{h}": decile[top].get(f"d{h}") for h in CFG["horizons"]})


# ============================ 레짐 · 신호 · 가격 ============================
def regime_from_index(idx: pd.Series) -> dict:
    idx = idx.dropna(); ma200 = idx.rolling(200).mean()
    above = bool(idx.iloc[-1] > ma200.iloc[-1]) if len(idx) >= 200 else True
    vol = idx.pct_change().rolling(20).std().iloc[-1] * math.sqrt(252)
    if above and not (vol > 0.25):
        return dict(state="위험선호", note="지수 200MA 상단·변동성 안정", led="on")
    if not above:
        return dict(state="위험회피", note="지수 200MA 하회", led="off")
    return dict(state="중립", note="추세 혼조·변동성 확대", led="neutral")


def make_signal(score, prob_d20, regime_off):
    if score <= CFG["sell_score"] or (prob_d20 is not None and prob_d20 < CFG["sell_prob"] and score < 55):
        return "sell"
    if not regime_off and score >= CFG["buy_score"] and (prob_d20 or 0) >= CFG["buy_prob"]:
        return "buy"
    return "watch"


def price_levels(close, atr, market, signal):
    if signal == "sell" or not atr or math.isnan(atr):
        return dict(entry={"low": 0, "high": 0}, targets={"bear": 0, "base": 0, "bull": 0}, stop=0)
    rnd = (lambda x: round(float(x), 2)) if market == "US" else (lambda x: int(round(float(x))))
    return dict(entry={"low": rnd(close - CFG["atr_entry_lo"] * atr), "high": rnd(close + CFG["atr_entry_hi"] * atr)},
                targets={"bear": rnd(close - CFG["atr_bear"] * atr), "base": rnd(close + CFG["atr_base"] * atr),
                         "bull": rnd(close + CFG["atr_bull"] * atr)},
                stop=rnd(close - CFG["atr_stop"] * atr))


# ============================ 조립 ============================
def build_market(market, panel, regime, fundamentals=None, supply=None, top_n=40):
    for tk in list(panel):
        panel[tk] = add_indicators(panel[tk])

    bt = backtest_engine(panel)
    wm, wt = bt["weights"]
    ruleset_ok = (bt["pbo"] is not None and bt["pbo"] <= CFG["gate_pbo"]
                  and bt["dsr"] is not None and bt["dsr"] >= CFG["gate_dsr"])

    date = sorted({dt for d in panel.values() for dt in d.index})[-1]
    f = _price_cross_section(panel, date)
    if f.empty:
        return [], dict(hit={}, pbo=bt["pbo"], dsr=bt["dsr"], ic=bt["ic"], weights={}), 0

    vq = F.value_quality_z(fundamentals) if fundamentals is not None else pd.DataFrame()
    sz = SUP.supply_z(supply) if supply is not None else pd.Series(dtype=float)
    f["value_z"] = vq["value_z"].reindex(f.index) if "value_z" in vq else np.nan
    f["quality_z"] = vq["quality_z"].reindex(f.index) if "quality_z" in vq else np.nan
    f["supply_z"] = sz.reindex(f.index) if len(sz) else np.nan
    has_val = f["value_z"].notna().any(); has_sup = f["supply_z"].notna().any()

    # 그룹 예산 방식(척도 혼입 방지): 가격/가치/퀄리티/수급에 예산 배분 후,
    # 가격 예산은 IC 비율로 모멘텀/추세에 분할한다. 데이터 없는 그룹은 예산 0 후 재정규화.
    budget = {"price": GROUP_BUDGET["price"],
              "value": GROUP_BUDGET["value"] if has_val else 0.0,
              "quality": GROUP_BUDGET["quality"] if has_val else 0.0,
              "supply": GROUP_BUDGET["supply"] if has_sup else 0.0}
    tb = sum(budget.values()) or 1.0
    budget = {k: v / tb for k, v in budget.items()}
    W = {"momentum": round(budget["price"] * wm, 4), "tech": round(budget["price"] * wt, 4),
         "value": round(budget["value"], 4), "quality": round(budget["quality"], 4),
         "supply": round(budget["supply"], 4)}

    zc = lambda col: f[col].fillna(0.0)
    full = (W["momentum"] * zc("mom_z") + W["tech"] * zc("trend_z")
            + W["value"] * zc("value_z") + W["quality"] * zc("quality_z") + W["supply"] * zc("supply_z"))
    f["score"] = (full.rank(pct=True) * 100).round().astype(int)
    pcomp = wm * f["mom_z"] + wt * f["trend_z"]
    f["pdecile"] = (pcomp.rank(pct=True) * CFG["n_deciles"]).clip(upper=CFG["n_deciles"] - 1e-9).astype(int)

    disp = lambda col: (f[col].rank(pct=True) if f[col].notna().any() else pd.Series(0.5, index=f.index))
    f["m_disp"], f["t_disp"] = f["mom_z"].rank(pct=True), f["trend_z"].rank(pct=True)
    f["v_disp"], f["q_disp"], f["s_disp"] = disp("value_z"), disp("quality_z"), disp("supply_z")

    if f["turnover"].notna().any():
        f = f[f["turnover"] >= f["turnover"].quantile(1 - CFG["liquidity_keep"])]

    regime_off = regime.get("led") == "off"
    recs = []
    for tk, row in f.sort_values("score", ascending=False).head(top_n).iterrows():
        probs = bt["decile"].get(int(row["pdecile"]), {})
        prob_up = {f"d{h}": (probs.get(f"d{h}") if probs.get(f"d{h}") is not None else 0.5) for h in CFG["horizons"]}
        close, atr = float(row["close"]), float(row["atr14"])
        signal = make_signal(int(row["score"]), prob_up["d20"], regime_off)
        px = price_levels(close, atr, market, signal)
        verified = ruleset_ok and (probs.get("d20") is not None) and (probs["d20"] >= CFG["verify_prob"])
        recs.append({
            "ticker": tk, "name": SECTORS.get((market, tk), {}).get("name", tk),
            "market": market, "sector": SECTORS.get((market, tk), {}).get("sector", ""),
            "score": int(row["score"]),
            "factors": {"momentum": round(float(row["m_disp"]), 3), "value": round(float(row["v_disp"]), 3),
                        "quality": round(float(row["q_disp"]), 3), "supply": round(float(row["s_disp"]), 3),
                        "tech": round(float(row["t_disp"]), 3)},
            "prob_up": {k: round(v, 3) for k, v in prob_up.items()},
            "signal": signal, **px, "verified": bool(verified), "note": "",
        })
    stats = dict(hit=bt["top_hit"], pbo=bt["pbo"], dsr=bt["dsr"], ic=bt["ic"], weights=W)
    return recs, stats, len(f)


def build_payload(markets_data: dict) -> dict:
    recs, regimes, usize, meta = [], {}, {}, {}
    for mk, v in markets_data.items():
        panel, regime = v[0], v[1]
        fund = v[2] if len(v) > 2 else None
        sup = v[3] if len(v) > 3 else None
        r, stats, n = build_market(mk, panel, regime, fund, sup)
        recs += r; regimes[mk.lower()] = regime; usize[mk.lower()] = n; meta[mk] = stats

    def avg(key):
        vals = [meta[m]["hit"].get(key) for m in meta if meta[m]["hit"].get(key) is not None]
        return round(sum(vals) / len(vals), 3) if vals else None

    def avgm(key):
        vals = [meta[m][key] for m in meta if meta[m].get(key) is not None]
        return round(sum(vals) / len(vals), 3) if vals else None

    return {
        "as_of": datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d"),
        "ruleset": "balanced_v2",
        "market_regime": regimes, "universe_size": usize,
        "backtest": {"hit_d5": avg("d5"), "hit_d20": avg("d20"), "hit_d60": avg("d60"),
                     "deflated_sharpe": avgm("dsr"), "pbo": avgm("pbo")},
        "factor_ic": {m: meta[m]["ic"] for m in meta},
        "weights": {m: meta[m]["weights"] for m in meta},
        "recommendations": recs,
    }


def run(markets=("KR", "US"), start="2022-01-01", out="latest.json") -> dict:
    md = {}
    for mk in markets:
        tickers = load_universe(mk)
        panel = load_prices(mk, tickers, start)
        md[mk] = (panel, load_regime(mk), load_fundamentals(mk, tickers), load_supply(mk, tickers))
    payload = build_payload(md)
    with open(out, "w", encoding="utf-8") as fp:
        json.dump(payload, fp, ensure_ascii=False, indent=2)
    return payload
