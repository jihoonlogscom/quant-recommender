"""
Point-in-Time (PIT) 정합 + 실데이터 커넥터 — Phase 3

- 미국 재무: SEC EDGAR companyfacts (무키, User-Agent 필요). 연간(10-K/FY) 기준으로 PIT 구성.
- 한국 재무: DART OpenAPI (DART_API_KEY 필요, OpenDartReader 사용). 없으면 빈 결과(중립).
- 한국 수급: pykrx 일별 투자자별 순매수(외국인·기관).
- attach_pit(순수): 재무는 공시일 ffill, 수급은 일별 롤링으로 가격 패널에 부착 → look-ahead 없음.

네트워크가 필요한 build_* 는 모두 방어적으로 try/except 하며, 실패 시 빈 결과를 반환해
파이프라인이 해당 팩터를 자동으로 중립·예산0 처리하도록 한다.
"""
from __future__ import annotations
import os, time, json
import numpy as np
import pandas as pd

FUND_COLS = ["eps_ttm", "bps", "ebitda_ps", "roe", "op_margin", "debt_ratio", "earn_stability"]
SUPPLY_COLS = ["foreign_net", "inst_net"]
_CACHE = os.getenv("PIT_CACHE", ".cache/pit")


# ============================ attach (순수, 테스트 가능) ============================
def attach_pit(panel, fundamentals=None, supply=None):
    """가격 패널 각 종목에 PIT 재무·수급 컬럼을 일별로 부착한다(look-ahead 없음)."""
    for tk, d in panel.items():
        if fundamentals and tk in fundamentals and not fundamentals[tk].empty:
            fp = fundamentals[tk].sort_index()
            a = fp.reindex(d.index.union(fp.index)).ffill().reindex(d.index)  # 공시일 ffill = PIT
            close = d["Close"]
            eps, bps, eb = a.get("eps_ttm"), a.get("bps"), a.get("ebitda_ps")
            d["per"] = close / eps.where(eps > 0) if eps is not None else np.nan
            d["pbr"] = close / bps.where(bps > 0) if bps is not None else np.nan
            d["ev_ebitda"] = close / eb.where(eb > 0) if eb is not None else np.nan
            for c in ["roe", "op_margin", "debt_ratio", "earn_stability"]:
                d[c] = a[c] if c in a else np.nan
        if supply and tk in supply and not supply[tk].empty:
            sp = supply[tk].reindex(d.index).fillna(0.0)
            net = sp.get("foreign_net", 0.0) + sp.get("inst_net", 0.0)
            d["net5"] = net.rolling(5).sum()
            d["net20"] = net.rolling(20).sum()
            sign = np.sign(net)
            grp = (sign != sign.shift()).cumsum()
            d["consec"] = (sign.groupby(grp).cumcount() + 1) * sign
    return panel


# ============================ SEC EDGAR (US 재무, 무키) ============================
_SEC_UA = os.getenv("SEC_UA", "quant-recommender research (contact: set SEC_UA env)")


def _sec_session():
    import requests
    s = requests.Session()
    s.headers.update({"User-Agent": _SEC_UA, "Accept-Encoding": "gzip, deflate"})
    return s


def _sec_ticker_cik(session) -> dict:
    r = session.get("https://www.sec.gov/files/company_tickers.json", timeout=30)
    r.raise_for_status()
    out = {}
    for row in r.json().values():
        out[str(row["ticker"]).upper()] = int(row["cik_str"])
    return out


def _companyfacts(session, cik: int) -> dict | None:
    path = os.path.join(_CACHE, f"cf_{cik:010d}.json")
    # 주 단위 캐시(재무는 분기 갱신이라 잦은 재요청 불필요)
    if os.path.exists(path) and (time.time() - os.path.getmtime(path) < 7 * 86400):
        try:
            return json.load(open(path, encoding="utf-8"))
        except Exception:
            pass
    try:
        r = session.get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json", timeout=30)
        if r.status_code != 200:
            return None
        cf = r.json()
        os.makedirs(_CACHE, exist_ok=True)
        json.dump(cf, open(path, "w", encoding="utf-8"))
        time.sleep(0.12)                                  # SEC 예의상 rate limit
        return cf
    except Exception:
        return None


def _facts(cf, names, units, annual=True) -> pd.Series:
    """companyfacts에서 개념 시계열을 filed일 기준 Series로. annual=True면 10-K/FY만."""
    for tax in ("us-gaap", "dei"):
        facts = cf.get("facts", {}).get(tax, {})
        for nm in names:
            node = facts.get(nm)
            if not node:
                continue
            for unit, arr in node.get("units", {}).items():
                if units and unit not in units:
                    continue
                recs = []
                for x in arr:
                    if not x.get("filed") or x.get("val") is None:
                        continue
                    if annual and not (x.get("fp") == "FY" or str(x.get("form", "")).startswith("10-K")):
                        continue
                    recs.append((pd.to_datetime(x["filed"]), float(x["val"])))
                if recs:
                    s = pd.Series(dict(recs)).sort_index()
                    return s[~s.index.duplicated(keep="last")]
    return pd.Series(dtype=float)


def _fundamentals_from_cf(cf) -> pd.DataFrame:
    ni = _facts(cf, ["NetIncomeLoss", "ProfitLoss"], ["USD"])
    rev = _facts(cf, ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet"], ["USD"])
    opi = _facts(cf, ["OperatingIncomeLoss"], ["USD"])
    da = _facts(cf, ["DepreciationDepletionAndAmortization", "DepreciationAmortizationAndAccretionNet"], ["USD"])
    eq = _facts(cf, ["StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"], ["USD"], annual=False)
    liab = _facts(cf, ["Liabilities"], ["USD"], annual=False)
    sh = _facts(cf, ["WeightedAverageNumberOfDilutedSharesOutstanding", "CommonStockSharesOutstanding",
                     "EntityCommonStockSharesOutstanding"], ["shares"])
    eps = _facts(cf, ["EarningsPerShareDiluted", "EarningsPerShareBasic"], ["USD/shares"])

    idx = pd.Index(sorted(set().union(*[s.index for s in [ni, rev, opi, da, eq, liab, sh, eps] if len(s)])))
    if len(idx) == 0:
        return pd.DataFrame(columns=FUND_COLS)
    A = lambda s: s.reindex(idx).ffill()
    ni, rev, opi, da, eq, liab, sh, eps = map(A, [ni, rev, opi, da, eq, liab, sh, eps])

    out = pd.DataFrame(index=idx)
    out["eps_ttm"] = eps if eps.notna().any() else (ni / sh)
    out["bps"] = eq / sh
    out["ebitda_ps"] = (opi + da.fillna(0)) / sh
    out["roe"] = ni / eq.where(eq != 0)
    out["op_margin"] = opi / rev.where(rev != 0)
    out["debt_ratio"] = liab / eq.where(eq != 0)
    stab = eps if eps.notna().any() else ni
    roll = stab.rolling(4, min_periods=2)
    out["earn_stability"] = 1.0 / (1.0 + (roll.std() / roll.mean().abs()).replace([np.inf, -np.inf], np.nan))
    return out.reindex(columns=FUND_COLS)


def build_pit_fundamentals_us(tickers: list[str]) -> dict[str, pd.DataFrame]:
    try:
        s = _sec_session()
        cik = _sec_ticker_cik(s)
    except Exception:
        return {}
    out = {}
    for tk in tickers:
        c = cik.get(str(tk).upper())
        if not c:
            continue
        cf = _companyfacts(s, c)
        if not cf:
            continue
        df = _fundamentals_from_cf(cf)
        if not df.empty and df.notna().any().any():
            out[tk] = df
    return out


# ============================ DART (KR 재무, 키 필요) ============================
def build_pit_fundamentals_kr(tickers: list[str]) -> dict[str, pd.DataFrame]:
    """OpenDartReader로 최근 연간 재무를 받아 PIT 구성. 키/라이브러리 없으면 빈 결과."""
    key = os.getenv("DART_API_KEY")
    if not key:
        return {}
    try:
        import OpenDartReader
    except Exception:
        return {}
    dart = OpenDartReader(key)
    out = {}
    years = [pd.Timestamp.today().year - i for i in range(1, 5)]
    for tk in tickers:
        rows = {}
        for y in years:
            try:
                fs = dart.finstate(tk, y)                 # 연간 주요계정
                if fs is None or len(fs) == 0:
                    continue
                acc = {r["account_nm"]: r for _, r in fs.iterrows()}
                def val(name):
                    r = acc.get(name)
                    if not r:
                        return np.nan
                    v = str(r.get("thstrm_amount", "")).replace(",", "")
                    return float(v) if v and v.lstrip("-").isdigit() else np.nan
                ni = val("당기순이익"); eq = val("자본총계"); rev = val("매출액")
                opi = val("영업이익"); liab = val("부채총계")
                filed = pd.to_datetime(f"{y+1}-03-31")   # 사업보고서 통상 익년 3월 공시(근사)
                rows[filed] = dict(eps_ttm=np.nan, bps=np.nan, ebitda_ps=np.nan,
                                   roe=(ni / eq if eq else np.nan),
                                   op_margin=(opi / rev if rev else np.nan),
                                   debt_ratio=(liab / eq if eq else np.nan),
                                   earn_stability=np.nan)
            except Exception:
                continue
        if rows:
            out[tk] = pd.DataFrame(rows).T.reindex(columns=FUND_COLS)
    return out


# ============================ pykrx (KR 수급) ============================
def build_pit_supply_kr(tickers: list[str], lookback_days: int = 120) -> dict[str, pd.DataFrame]:
    """pykrx 일별 투자자별 순매수(외국인·기관). 최근 lookback_days만 수집(net5/20/consec에 충분)."""
    try:
        from pykrx import stock
    except Exception:
        return {}
    end = pd.Timestamp.today()
    start = end - pd.Timedelta(days=lookback_days)
    fr, to = start.strftime("%Y%m%d"), end.strftime("%Y%m%d")
    out = {}
    for tk in tickers:
        try:
            df = stock.get_market_trading_value_by_date(fr, to, tk)   # 순매수 기준
            if df is None or df.empty:
                continue
            foreign = next((c for c in df.columns if "외국인" in c), None)
            inst = next((c for c in df.columns if "기관" in c), None)
            if not foreign and not inst:
                continue
            s = pd.DataFrame(index=pd.to_datetime(df.index))
            s["foreign_net"] = df[foreign].values if foreign else 0.0
            s["inst_net"] = df[inst].values if inst else 0.0
            out[tk] = s
        except Exception:
            continue
    return out
