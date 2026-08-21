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


def _sec_get(session, url, tries=3):
    """(response|None, last_status). SEC 일시적 403/429 대비 재시도."""
    import time
    last = None
    for i in range(tries):
        try:
            r = session.get(url, timeout=30)
            if r.status_code == 200:
                return r, 200
            last = r.status_code
        except Exception:
            last = None
        time.sleep(0.6 * (i + 1))
    return None, last


def _sec_ticker_cik(session) -> dict:
    r, status = _sec_get(session, "https://www.sec.gov/files/company_tickers.json")
    if r is None:
        raise RuntimeError(f"company_tickers.json 응답코드 {status}")
    out = {}
    for row in r.json().values():
        out[str(row["ticker"]).upper()] = int(row["cik_str"])
    return out


def _companyfacts(session, cik: int):
    """(cf|None, status). 성공분만 주 단위 캐시."""
    path = os.path.join(_CACHE, f"cf_{cik:010d}.json")
    if os.path.exists(path) and (time.time() - os.path.getmtime(path) < 7 * 86400):
        try:
            return json.load(open(path, encoding="utf-8")), 200
        except Exception:
            pass
    r, status = _sec_get(session, f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json")
    if r is None:
        return None, status
    try:
        cf = r.json()
    except Exception:
        return None, status
    os.makedirs(_CACHE, exist_ok=True)
    try:
        json.dump(cf, open(path, "w", encoding="utf-8"))
    except Exception:
        pass
    time.sleep(0.12)                                          # SEC 예의상 rate limit
    return cf, 200


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
    except Exception as e:
        print(f"[US] SEC 접근 실패: {e} — SEC_UA에 연락 이메일 포함 필요(예: 'app you@mail.com'). "
              f"GitHub Actions IP가 막히면 로컬 실행 권장", flush=True)
        return {}
    if not cik:
        print("[US] SEC 티커→CIK 맵이 비어있음", flush=True)
        return {}
    out, n_ok, tried, first_status = {}, 0, 0, None
    for tk in tickers:
        c = cik.get(str(tk).upper()) or cik.get(str(tk).upper().replace("-", ""))
        if not c:
            continue
        tried += 1
        cf, status = _companyfacts(s, c)
        if not cf:
            if first_status is None:
                first_status = status
            continue
        try:
            df = _fundamentals_from_cf(cf)
        except Exception:
            continue
        if not df.empty and df.notna().any().any():
            out[tk] = df; n_ok += 1
    if n_ok == 0:
        print(f"[US] SEC companyfacts 0건 (시도 {tried}, 첫 응답코드 {first_status}) — "
              f"403이면 UA/IP 차단, 200이면 파싱 문제", flush=True)
    return out


# ============================ DART (KR 재무, 키 필요) ============================
def build_pit_fundamentals_kr(tickers: list[str]) -> dict[str, pd.DataFrame]:
    """OpenDartReader로 최근 연간 재무를 받아 PIT 구성. 키/라이브러리 없으면 빈 결과.
    반환: {ticker: DataFrame(index=filed(근사), columns=FUND_COLS)}."""
    import io, contextlib
    key = os.getenv("DART_API_KEY")
    if not key:
        return {}
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            import OpenDartReader
            dart = OpenDartReader(key)                     # corp_code 매핑 1회 다운로드
    except Exception:
        return {}

    def pick(df, needle):
        rows = df
        if "fs_div" in df.columns:                        # 연결(CFS) 우선
            cfs = df[df["fs_div"] == "CFS"]
            rows = cfs if len(cfs) else df
        m = rows[rows["account_nm"].astype(str).str.contains(needle, na=False)]
        if len(m) == 0:
            return np.nan
        v = str(m.iloc[0].get("thstrm_amount", "")).replace(",", "").strip()
        try:
            return float(v)
        except Exception:
            return np.nan

    out = {}
    years = [pd.Timestamp.today().year - i for i in range(1, 5)]
    for tk in tickers:
        rows = {}
        for y in years:
            try:
                with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                    fs = dart.finstate(tk, y)             # 사업보고서(연간) 주요계정
            except Exception:
                continue
            if fs is None or len(fs) == 0:
                continue
            ni, eq = pick(fs, "당기순이익"), pick(fs, "자본총계")
            rev, opi, liab = pick(fs, "매출"), pick(fs, "영업이익"), pick(fs, "부채총계")
            eps = pick(fs, "주당순이익")
            rows[pd.to_datetime(f"{y+1}-03-31")] = dict(   # 사업보고서 통상 익년 3월 공시(근사)
                eps_ttm=eps, bps=np.nan, ebitda_ps=np.nan,
                roe=(ni / eq if eq else np.nan),
                op_margin=(opi / rev if rev else np.nan),
                debt_ratio=(liab / eq if eq else np.nan),
                earn_stability=np.nan)
        if rows:
            df = pd.DataFrame(rows).T.reindex(columns=FUND_COLS)
            if df.notna().any().any():
                out[tk] = df
    return out


# ============================ pykrx (KR 수급, 무로그인) ============================
def build_pit_supply_kr(tickers: list[str]) -> pd.DataFrame:
    """외국인·기관 순매수 스냅샷(최근 5일·20일). 로그인 불필요한 순매수상위 API 사용.

    반환: DataFrame(index=ticker, columns=[net5, net20]) — 순매수거래대금 합(외국인+기관).
    종목별 일별 API는 최근 KRX에서 로그인을 요구하므로 시장 전체 순매수 집계를 쓴다.
    과거 시계열이 없어 백테스트엔 넣지 않고 '현재 틸트'로만 사용한다(파이프라인 참조).
    """
    import io, contextlib
    want = set(map(str, tickers))
    acc = {"net5": {}, "net20": {}}
    buf = io.StringIO()
    api_err = None
    try:
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            from pykrx import stock                            # 임포트 시 로그인 안내 출력까지 억제
            try:
                to = stock.get_nearest_business_day_in_a_week()   # 가장 최근 거래일(오늘 미확정 회피)
            except Exception:
                to = pd.Timestamp.today().strftime("%Y%m%d")
            to_ts = pd.Timestamp(to)
            windows = {"net5": (to_ts - pd.Timedelta(days=10)).strftime("%Y%m%d"),
                       "net20": (to_ts - pd.Timedelta(days=32)).strftime("%Y%m%d")}
            for col, fr in windows.items():
                for mk in ("KOSPI", "KOSDAQ"):
                    for inv in ("외국인", "기관합계"):
                        try:
                            df = stock.get_market_net_purchases_of_equities(fr, to, mk, inv)
                        except Exception as e:
                            api_err = repr(e); continue
                        if df is None or df.empty:
                            continue
                        vcol = next((c for c in df.columns if "순매수거래대금" in c), None)
                        if not vcol:
                            continue
                        for tk, v in df[vcol].items():
                            tk = str(tk)
                            if tk in want:
                                acc[col][tk] = acc[col].get(tk, 0.0) + float(v)
    except Exception as e:
        api_err = repr(e)

    idx = sorted(set(acc["net5"]) | set(acc["net20"]))
    if not idx:
        print(f"[KR] 수급 0 — pykrx 순매수 API 빈 결과{(' · '+api_err) if api_err else ''}. "
              f"KRX 엔드포인트 변경 가능성(수급은 틸트라 KR은 모멘텀+추세+재무로 계속 동작)", flush=True)
        return pd.DataFrame(columns=["net5", "net20"])
    return pd.DataFrame({"net5": pd.Series(acc["net5"]), "net20": pd.Series(acc["net20"])}).reindex(idx).fillna(0.0)
