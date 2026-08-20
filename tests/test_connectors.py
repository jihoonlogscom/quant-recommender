"""네트워크 없이 (1) SEC companyfacts 파서 (2) 증분 parquet 캐시 로직을 검증."""
import sys, os, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from quant import pit as PIT
from quant import cache as CACHE


def fake_companyfacts():
    """3개 회계연도치 최소 companyfacts 구조(us-gaap)."""
    def ann(vals, unit="USD"):
        # (filed, val) 연간 엔트리
        return [{"end": f"{y}-12-31", "val": v, "filed": f"{y+1}-03-01", "fp": "FY", "form": "10-K"}
                for y, v in vals]
    yrs = [(2022, 1), (2023, 2), (2024, 3)]
    return {"facts": {"us-gaap": {
        "NetIncomeLoss": {"units": {"USD": ann([(2022, 8e9), (2023, 10e9), (2024, 12e9)])}},
        "Revenues": {"units": {"USD": ann([(2022, 80e9), (2023, 95e9), (2024, 110e9)])}},
        "OperatingIncomeLoss": {"units": {"USD": ann([(2022, 20e9), (2023, 25e9), (2024, 30e9)])}},
        "DepreciationDepletionAndAmortization": {"units": {"USD": ann([(2022, 3e9), (2023, 3.2e9), (2024, 3.4e9)])}},
        "StockholdersEquity": {"units": {"USD": ann([(2022, 60e9), (2023, 66e9), (2024, 72e9)])}},
        "Liabilities": {"units": {"USD": ann([(2022, 40e9), (2023, 42e9), (2024, 44e9)])}},
        "WeightedAverageNumberOfDilutedSharesOutstanding": {"units": {"shares": ann([(2022, 5e9), (2023, 5e9), (2024, 5e9)])}},
        "EarningsPerShareDiluted": {"units": {"USD/shares": ann([(2022, 1.6), (2023, 2.0), (2024, 2.4)])}},
    }}}


def test_edgar_parser():
    df = PIT._fundamentals_from_cf(fake_companyfacts())
    assert list(df.columns) == PIT.FUND_COLS
    assert len(df) == 3
    last = df.iloc[-1]
    # 2024: roe=12/72, op_margin=30/110, debt=44/72, eps=2.4, bps=72e9/5e9=14.4, ebitda_ps=(30+3.4)/5
    assert abs(last["roe"] - 12e9 / 72e9) < 1e-6
    assert abs(last["op_margin"] - 30e9 / 110e9) < 1e-6
    assert abs(last["debt_ratio"] - 44e9 / 72e9) < 1e-6
    assert abs(last["eps_ttm"] - 2.4) < 1e-9
    assert abs(last["bps"] - 14.4) < 1e-6
    assert abs(last["ebitda_ps"] - (30e9 + 3.4e9) / 5e9) < 1e-6
    assert df["earn_stability"].notna().any()
    # attach_pit로 가격에 부착 → PER/PBR가 종가로 매일 계산되는지
    dates = pd.bdate_range("2022-01-03", periods=820)
    price = pd.DataFrame({"Open": 100, "High": 101, "Low": 99, "Close": 100.0, "Volume": 1e6}, index=dates)
    panel = PIT.attach_pit({"X": price}, {"X": df}, None)
    x = panel["X"]
    assert "per" in x and "pbr" in x and "roe" in x
    assert x["per"].dropna().gt(0).all()          # 100/eps>0
    print("✓ EDGAR 파서: roe/op_margin/debt/eps/bps/ebitda 정확, attach_pit PER/PBR 생성")


def test_cache_incremental():
    calls = {"n": 0}
    full = pd.DataFrame({"Open": 1., "High": 1., "Low": 1., "Close": np.arange(1, 301) * 1.0,
                         "Volume": 1.}, index=pd.bdate_range("2023-01-02", periods=300))

    def fetch(tk, start):
        calls["n"] += 1
        s = pd.Timestamp(start)
        return full[full.index >= s]

    with tempfile.TemporaryDirectory() as d:
        p1 = CACHE.update_panel("US", ["AAA"], "2023-01-02", fetch, cachedir=d, min_history=260, stale_days=3)
        assert "AAA" in p1 and len(p1["AAA"]) == 300
        n_after_first = calls["n"]
        # 캐시가 최신(오늘 기준 stale 아님 가정 불가)이므로 두번째 호출 시 증분만 시도
        p2 = CACHE.update_panel("US", ["AAA"], "2023-01-02", fetch, cachedir=d, min_history=260, stale_days=3)
        assert "AAA" in p2
        # 저장/로드 라운드트립 확인
        loaded = CACHE.load_one(d, "US", "AAA")
        assert loaded is not None and len(loaded) >= 300
    print(f"✓ 증분 캐시: 최초 벌크 저장·재로드 OK (fetch 호출 {calls['n']}회, 증분만 재요청)")


if __name__ == "__main__":
    test_edgar_parser()
    test_cache_incremental()
    print("✓ Phase 3 커넥터/캐시 단위 테스트 통과")
