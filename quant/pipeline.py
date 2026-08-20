"""
데일리 퀀트 추천 — Phase 2.5 파이프라인

Phase 2.5 핵심: Point-in-Time(PIT) 정합으로 재무·수급을 과거 백테스트에 편입.
- attach_pit로 재무(공시일 ffill)·수급(일별 롤링)을 일별 컬럼으로 부착 → look-ahead 없음.
- 이제 5개 팩터(모멘텀·가치·퀄리티·수급·기술) 전부가 과거 횡단면에 들어가
  IC·가중치·확률·검증(PBO/DSR)에 정식 참여한다.
- 데이터 없는 팩터(예: 미국 수급, 재무 키 미설정)는 자동으로 중립(0)·예산 0 처리.

확률은 과거 히트레이트로 미래 수익 보장이 아니다. 본 도구는 투자 자문이 아니다.
"""
from __future__ import annotations
import json, math, os
from datetime import datetime, timezone
import numpy as np
import pandas as pd

from quant import validate as V
from quant import pit as PIT
from quant import cache as CACHE

CFG = dict(
    horizons=[5, 20, 60],
    min_history=260, liquidity_keep=0.6,
    bt_step=5, bt_lookback=504, n_deciles=10,
    buy_score=72, buy_prob=0.55, sell_score=42, sell_prob=0.50,
    verify_prob=0.55, gate_pbo=0.5, gate_dsr=0.5,
    atr_entry_lo=0.5, atr_entry_hi=0.3, atr_stop=2.0,
    atr_bear=1.0, atr_base=2.5, atr_bull=5.0,
)
GROUP_BUDGET = dict(price=0.40, value=0.20, quality=0.20, supply=0.20)  # "균형" 성향
FACTORS = ["momentum", "value", "quality", "supply", "tech"]
ZCOL = {"momentum": "mom_z", "value": "value_z", "quality": "quality_z",
        "supply": "supply_z", "tech": "trend_z"}
# 유니버스 프리셋: 넓힐수록 무료 데이터 수집 시간↑ (증분 캐시로 완화)
US_UNIVERSE = os.getenv("US_UNIVERSE", "S&P500")   # "S&P500" | "NASDAQ" | "NYSE" | "AMEX"
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
        df = fdr.StockListing(US_UNIVERSE)                    # S&P500 기본, 환경변수로 확장
        sym = "Symbol" if "Symbol" in df.columns else df.columns[0]
        for _, r in df.iterrows():
            SECTORS[("US", str(r[sym]))] = {"name": r.get("Name", r[sym]), "sector": r.get("Sector", "") or r.get("Industry", "")}
        return df[sym].astype(str).tolist()
    raise ValueError(market)


def _fetch_one(market):
    """cache.update_panel에 넘길 종목 단위 fetch 함수(소스 분리)."""
    if market == "KR":
        import FinanceDataReader as fdr
        def f(tk, start):
            try:
                return fdr.DataReader(tk, start)[["Open", "High", "Low", "Close", "Volume"]].dropna()
            except Exception:
                return None
        return f
    import yfinance as yf
    def f(tk, start):
        try:
            d = yf.download(tk, start=start, auto_adjust=True, progress=False)
            if d is None or d.empty:
                return None
            if isinstance(d.columns, pd.MultiIndex):
                d.columns = d.columns.get_level_values(0)
            return d[["Open", "High", "Low", "Close", "Volume"]].dropna()
        except Exception:
            return None
    return f


def load_prices(market: str, tickers: list[str], start: str, cachedir=".cache/prices") -> dict[str, pd.DataFrame]:
    """증분 캐시 기반 수집(최초 벌크→이후 증분). 캐시 비활성은 cachedir=None."""
    if cachedir is None:
        panel, fetch = {}, _fetch_one(market)
        for tk in tickers:
            d = fetch(tk, start)
            if d is not None and len(d) >= CFG["min_history"]:
                panel[tk] = d
        return panel
    return CACHE.update_panel(market, tickers, start, _fetch_one(market),
                              cachedir=cachedir, min_history=CFG["min_history"])


def load_pit_fundamentals(market, tickers):
    return PIT.build_pit_fundamentals_kr(tickers) if market == "KR" else PIT.build_pit_fundamentals_us(tickers)


def load_pit_supply(market, tickers):
    return PIT.build_pit_supply_kr(tickers) if market == "KR" else {}


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


def _zc(s):                       # z-score 후 결측은 중립(0)
    return _z(s).fillna(0.0)


def _fwd(d, i, h):
    return d["Close"].iloc[i + h] / d["Close"].iloc[i] - 1 if i + h < len(d) else np.nan


def _has(f, col):
    return col in f and pd.to_numeric(f[col], errors="coerce").notna().any()


def cross_section(panel, date):
    """특정 날짜의 5팩터 z + forward return. attach_pit 컬럼이 있으면 가치·퀄리티·수급도 계산."""
    rows, fwd = {}, {}
    cols = ["ret_12_1", "rs_63", "hi_252", "ma_align", "turnover", "close", "atr14",
            "per", "pbr", "ev_ebitda", "roe", "op_margin", "debt_ratio", "earn_stability",
            "net5", "net20", "consec"]
    for tk, d in panel.items():
        if date not in d.index:
            continue
        r = d.loc[date]
        if pd.isna(r.get("ret_12_1")) or pd.isna(r.get("ma_align")):
            continue
        rows[tk] = {c: r.get(c) for c in cols if c != "close"}
        rows[tk]["close"] = r["Close"]
        i = d.index.get_loc(date)
        fwd[tk] = {f"fwd{h}": _fwd(d, i, h) for h in CFG["horizons"]}
        fwd[tk]["ret_step"] = _fwd(d, i, CFG["bt_step"])
    if len(rows) < 5:
        return pd.DataFrame()
    f = pd.DataFrame(rows).T.join(pd.DataFrame(fwd).T)

    f["mom_z"] = (_zc(f["ret_12_1"]) + _zc(f["rs_63"]) + _zc(f["hi_252"])) / 3.0
    f["trend_z"] = _zc(f["ma_align"])
    # 가치(싼 게 좋음 → 역수), 퀄리티, 수급 — 데이터 있을 때만 유효, 없으면 0
    inv = lambda col: _zc(1.0 / pd.to_numeric(f[col], errors="coerce").where(pd.to_numeric(f[col], errors="coerce") > 0)) if _has(f, col) else pd.Series(0.0, index=f.index)
    zz = lambda col, sign=1: sign * _zc(f[col]) if _has(f, col) else pd.Series(0.0, index=f.index)
    f["value_z"] = (inv("per") + inv("pbr") + inv("ev_ebitda")) / 3.0
    f["quality_z"] = (zz("roe") + zz("op_margin") + zz("debt_ratio", -1) + zz("earn_stability")) / 4.0
    f["supply_z"] = (zz("net5") + zz("net20") + zz("consec")) / 3.0
    return f


# ============================ 백테스트 (5팩터, PIT) ============================
def _sampled(panel):
    all_dates = sorted({dt for d in panel.values() for dt in d.index})
    hz = max(CFG["horizons"])
    if len(all_dates) < CFG["min_history"] + hz:
        return []
    window = all_dates[-(CFG["bt_lookback"] + hz):-hz]
    return [cross_section(panel, dt) for dt in window[::CFG["bt_step"]]]


def _composite(f, W):
    return sum(W[k] * f[ZCOL[k]] for k in FACTORS)


def _evaluate(frames, W):
    hit = {h: {dc: [0, 0] for dc in range(CFG["n_deciles"])} for h in CFG["horizons"]}
    series = []
    for f in frames:
        dec = (_composite(f, W).rank(pct=True) * CFG["n_deciles"]).clip(upper=CFG["n_deciles"] - 1e-9).astype(int)
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


def _derive_weights(ic, present):
    """IC로 5팩터 가중치. 가격 예산은 모멘텀/추세 IC비율로 분할, 재무·수급은 IC>0일 때만 예산."""
    def pos(f):
        v = ic.get(f, {}).get("ic")
        return max(v, 0.0) if v is not None else 0.0
    im, it = pos("momentum"), pos("tech")
    pm, pt = (0.6, 0.4) if im + it <= 0 else (im / (im + it), it / (im + it))
    budget = {"price": GROUP_BUDGET["price"]}
    for g in ["value", "quality", "supply"]:
        budget[g] = GROUP_BUDGET[g] if (present.get(g) and pos(g) > 0) else 0.0
    tot = sum(budget.values()) or 1.0
    budget = {k: v / tot for k, v in budget.items()}
    return {"momentum": round(budget["price"] * pm, 4), "tech": round(budget["price"] * pt, 4),
            "value": round(budget["value"], 4), "quality": round(budget["quality"], 4),
            "supply": round(budget["supply"], 4)}


def _configs(W, present):
    """PBO용 후보 가중치 격자(가용 팩터 조합)."""
    cfgs = [W]
    cfgs.append({"momentum": 0.6, "tech": 0.4, "value": 0, "quality": 0, "supply": 0})      # 가격만
    eq_f = [g for g in ["value", "quality", "supply"] if present.get(g)]
    if eq_f:
        w = {k: 0.0 for k in FACTORS}; each = 0.5 / len(eq_f)
        w["momentum"], w["tech"] = 0.3, 0.2
        for g in eq_f:
            w[g] = each
        cfgs.append(w)                                                                       # 가격50 재무50
    cfgs.append({"momentum": 0.35, "tech": 0.15, "value": 0.2, "quality": 0.2, "supply": 0.1})
    cfgs.append({"momentum": 0.2, "tech": 0.1, "value": 0.25, "quality": 0.25, "supply": 0.2})
    cfgs.append({"momentum": 0.5, "tech": 0.5, "value": 0, "quality": 0, "supply": 0})
    # 결측 팩터 제거 후 재정규화
    out = []
    for c in cfgs:
        c = {k: (c.get(k, 0.0) if (k in ["momentum", "tech"] or present.get(k)) else 0.0) for k in FACTORS}
        s = sum(c.values()) or 1.0
        out.append({k: v / s for k, v in c.items()})
    return out


def backtest_engine(panel):
    frames = [f for f in _sampled(panel) if not f.empty]
    empty = dict(decile={}, weights={k: 0 for k in FACTORS}, pbo=None, dsr=None, ic={},
                 top_hit={f"d{h}": None for h in CFG["horizons"]})
    if len(frames) < 6:
        empty["weights"] = _derive_weights({}, {})
        return empty

    present = {g: any(_has(f, {"value": "per", "quality": "roe", "supply": "net20"}[g]) for f in frames)
               for g in ["value", "quality", "supply"]}
    ic = {}
    for k in FACTORS:
        ic[k] = V.information_coefficient([(f[ZCOL[k]], f["fwd20"]) for f in frames]) \
            if (k in ["momentum", "tech"] or present.get(k)) else {"ic": None, "t": None, "n": 0}

    W = _derive_weights(ic, present)
    cfgs = _configs(W, present)
    cols, sr = {}, []
    for j, c in enumerate(cfgs):
        _, s = _evaluate(frames, c)
        cols[f"c{j}"] = s.reset_index(drop=True)
        sd = s.std(ddof=1)
        sr.append(float(s.mean() / sd) if sd and sd == sd else 0.0)
    pbo = V.pbo_cscv(pd.DataFrame(cols), n_splits=8)

    decile, chosen = _evaluate(frames, W)
    dsr = V.deflated_sharpe_ratio(chosen.dropna(), sr)
    top = CFG["n_deciles"] - 1
    return dict(decile=decile, weights=W, pbo=pbo, dsr=(round(dsr, 3) if dsr == dsr else None),
                ic=ic, top_hit={f"d{h}": decile[top].get(f"d{h}") for h in CFG["horizons"]})


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
def build_market(market, panel, regime, pit_fund=None, pit_supply=None, top_n=40):
    PIT.attach_pit(panel, pit_fund, pit_supply)          # 재무·수급을 일별 PIT 컬럼으로 부착
    for tk in list(panel):
        panel[tk] = add_indicators(panel[tk])

    bt = backtest_engine(panel)
    W = bt["weights"]
    ruleset_ok = (bt["pbo"] is not None and bt["pbo"] <= CFG["gate_pbo"]
                  and bt["dsr"] is not None and bt["dsr"] >= CFG["gate_dsr"])

    date = sorted({dt for d in panel.values() for dt in d.index})[-1]
    f = cross_section(panel, date)
    if f.empty:
        return [], dict(hit={}, pbo=bt["pbo"], dsr=bt["dsr"], ic=bt["ic"], weights=W), 0

    f["score"] = (_composite(f, W).rank(pct=True) * 100).round().astype(int)
    f["pdecile"] = (_composite(f, W).rank(pct=True) * CFG["n_deciles"]).clip(upper=CFG["n_deciles"] - 1e-9).astype(int)

    def disp(col):
        s = pd.to_numeric(f[col], errors="coerce")
        return s.rank(pct=True) if s.abs().sum() > 0 else pd.Series(0.5, index=f.index)
    dsp = {k: disp(ZCOL[k]) for k in FACTORS}

    if f["turnover"].notna().any():
        f = f[pd.to_numeric(f["turnover"], errors="coerce") >= pd.to_numeric(f["turnover"], errors="coerce").quantile(1 - CFG["liquidity_keep"])]

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
            "factors": {k: round(float(dsp[k][tk]), 3) for k in FACTORS},
            "prob_up": {k: round(v, 3) for k, v in prob_up.items()},
            "signal": signal, **px, "verified": bool(verified), "note": "",
        })
    stats = dict(hit=bt["top_hit"], pbo=bt["pbo"], dsr=bt["dsr"], ic=bt["ic"], weights=W)
    return recs, stats, len(f)


def build_payload(markets_data: dict) -> dict:
    recs, regimes, usize, meta = [], {}, {}, {}
    for mk, v in markets_data.items():
        panel, regime = v[0], v[1]
        pit_fund = v[2] if len(v) > 2 else None
        pit_supply = v[3] if len(v) > 3 else None
        r, stats, n = build_market(mk, panel, regime, pit_fund, pit_supply)
        recs += r; regimes[mk.lower()] = regime; usize[mk.lower()] = n; meta[mk] = stats

    def avg(key):
        vals = [meta[m]["hit"].get(key) for m in meta if meta[m]["hit"].get(key) is not None]
        return round(sum(vals) / len(vals), 3) if vals else None

    def avgm(key):
        vals = [meta[m][key] for m in meta if meta[m].get(key) is not None]
        return round(sum(vals) / len(vals), 3) if vals else None

    return {
        "as_of": datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d"),
        "ruleset": "balanced_v3",
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
        md[mk] = (panel, load_regime(mk), load_pit_fundamentals(mk, tickers), load_pit_supply(mk, tickers))
    payload = build_payload(md)
    with open(out, "w", encoding="utf-8") as fp:
        json.dump(payload, fp, ensure_ascii=False, indent=2)
    return payload
