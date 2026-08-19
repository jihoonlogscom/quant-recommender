"""네트워크 없이 Phase 2 파이프라인 전체를 합성 데이터로 검증."""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from quant import pipeline as P


def synth_panel(n, n_days=820, seed=0, prefix="T"):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-03", periods=n_days)
    panel = {}
    for k in range(n):
        drift = rng.normal(0.0004, 0.0005); vol = rng.uniform(0.012, 0.03)
        rets = rng.normal(drift, vol, n_days)
        # 모멘텀에 약한 지속성 부여(팩터 IC가 0이 아니도록)
        rets[60:] += 0.15 * pd.Series(rets).rolling(40).mean().shift(1).fillna(0).values[60:]
        close = 100 * np.exp(np.cumsum(rets))
        high = close * (1 + np.abs(rng.normal(0, 0.006, n_days)))
        low = close * (1 - np.abs(rng.normal(0, 0.006, n_days)))
        openp = close * (1 + rng.normal(0, 0.004, n_days))
        volume = rng.integers(1e5, 5e6, n_days) * (1 + k / n)
        panel[f"{prefix}{k:03d}"] = pd.DataFrame(
            {"Open": openp, "High": high, "Low": low, "Close": close, "Volume": volume}, index=dates)
    return panel


def synth_fund(tickers, seed=5):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "per": rng.uniform(5, 40, len(tickers)), "pbr": rng.uniform(0.5, 6, len(tickers)),
        "ev_ebitda": rng.uniform(4, 25, len(tickers)), "roe": rng.uniform(-0.1, 0.35, len(tickers)),
        "op_margin": rng.uniform(-0.05, 0.4, len(tickers)), "debt_ratio": rng.uniform(20, 250, len(tickers)),
        "earn_stability": rng.uniform(0.1, 1.0, len(tickers)),
    }, index=list(tickers))


def synth_supply(tickers, seed=7):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "net5": rng.normal(0, 1e9, len(tickers)), "net20": rng.normal(0, 3e9, len(tickers)),
        "consec": rng.integers(-8, 9, len(tickers)),
    }, index=list(tickers))


def validate(p):
    assert set(p) >= {"as_of","ruleset","market_regime","universe_size","backtest","recommendations","factor_ic","weights"}
    assert p["recommendations"]
    for r in p["recommendations"]:
        assert set(r["factors"]) == {"momentum","value","quality","supply","tech"}
        assert set(r["prob_up"]) == {"d5","d20","d60"}
        assert r["signal"] in {"buy","watch","sell"}
        assert 0 <= r["score"] <= 100
        for v in list(r["factors"].values()) + list(r["prob_up"].values()):
            assert 0 <= v <= 1
    b = p["backtest"]; assert set(b) >= {"hit_d5","hit_d20","hit_d60","deflated_sharpe","pbo"}
    return True


if __name__ == "__main__":
    krp = synth_panel(70, seed=1, prefix="KR"); usp = synth_panel(70, seed=2, prefix="US")
    md = {
        "KR": (krp, dict(state="위험선호", note="합성", led="on"), synth_fund(krp), synth_supply(krp)),
        "US": (usp, dict(state="중립", note="합성", led="neutral"), synth_fund(usp), None),  # US 수급 없음
    }
    p = P.build_payload(md); validate(p)
    sig = {}
    for r in p["recommendations"]:
        sig[r["signal"]] = sig.get(r["signal"], 0) + 1
    print("✓ Phase 2 스키마 검증 통과")
    print(f"  추천 {len(p['recommendations'])}종목  신호분포 {sig}")
    print(f"  backtest: {p['backtest']}")
    print(f"  KR IC: momentum={p['factor_ic']['KR']['momentum']}  tech={p['factor_ic']['KR']['tech']}")
    print(f"  KR weights: {p['weights']['KR']}")
    print(f"  US weights: {p['weights']['US']}  (US는 supply 데이터 없음→0 확인)")
    top = max(p["recommendations"], key=lambda r: r["score"])
    print(f"  최고점수: {top['ticker']} score={top['score']} factors={top['factors']}")
    print(f"    prob={top['prob_up']} signal={top['signal']} verified={top['verified']}")
    kr_val = [r['factors']['value'] for r in p['recommendations'] if r['market']=='KR']
    us_sup = [r['factors']['supply'] for r in p['recommendations'] if r['market']=='US']
    print(f"  KR value 팩터 분산(재무 반영 확인): {round(float(np.std(kr_val)),3)}  (0이 아니어야)")
    print(f"  US supply 팩터(데이터 없음→0.5 중립): set={sorted(set(round(x,2) for x in us_sup))}")
    with open(os.path.join(os.path.dirname(__file__), "sample_output.json"), "w", encoding="utf-8") as f:
        json.dump(p, f, ensure_ascii=False, indent=2)
    print("✓ sample_output.json 생성")
