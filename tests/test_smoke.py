"""네트워크 없이 파이프라인 엔드투엔드 검증.
- PIT 재무는 백테스트/IC/가중치에 정식 편입(합성 filed 재무 주입).
- 수급은 현재 스냅샷 '틸트'로만 반영(백테스트 제외) — 무료 KR 데이터 현실 반영.
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from quant import pipeline as P


def synth(n, n_days=820, seed=0, prefix="T"):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-03", periods=n_days)
    panel, fund = {}, {}
    supply = {}   # 스냅샷: ticker -> (net5, net20)
    for k in range(n):
        drift = rng.normal(0.0004, 0.0005); vol = rng.uniform(0.012, 0.03)
        rets = rng.normal(drift, vol, n_days)
        rets[60:] += 0.15 * pd.Series(rets).rolling(40).mean().shift(1).fillna(0).values[60:]
        close = 100 * np.exp(np.cumsum(rets))
        high = close*(1+np.abs(rng.normal(0,0.006,n_days))); low = close*(1-np.abs(rng.normal(0,0.006,n_days)))
        openp = close*(1+rng.normal(0,0.004,n_days)); volume = rng.integers(1e5,5e6,n_days)*(1+k/n)
        tk = f"{prefix}{k:03d}"
        panel[tk] = pd.DataFrame({"Open":openp,"High":high,"Low":low,"Close":close,"Volume":volume}, index=dates)
        fdates = dates[::63]
        fund[tk] = pd.DataFrame({
            "eps_ttm": rng.uniform(2,12,len(fdates)), "bps": rng.uniform(20,120,len(fdates)),
            "ebitda_ps": rng.uniform(3,18,len(fdates)), "roe": rng.uniform(-0.1,0.35,len(fdates)),
            "op_margin": rng.uniform(-0.05,0.4,len(fdates)), "debt_ratio": rng.uniform(20,250,len(fdates)),
            "earn_stability": rng.uniform(0.1,1.0,len(fdates)),
        }, index=fdates)
        supply[tk] = (rng.normal(0,1e9), rng.normal(0,3e9))
    sup_df = pd.DataFrame(supply, index=["net5","net20"]).T
    return panel, fund, sup_df


def validate(p):
    assert set(p) >= {"as_of","ruleset","market_regime","universe_size","backtest","recommendations","factor_ic","weights"}
    assert p["recommendations"]
    for r in p["recommendations"]:
        assert set(r["factors"]) == {"momentum","value","quality","supply","tech"}
        assert set(r["prob_up"]) == {"d5","d20","d60"}
        assert r["signal"] in {"buy","watch","sell"}
        for v in list(r["factors"].values())+list(r["prob_up"].values()):
            assert 0 <= v <= 1
    return True


if __name__ == "__main__":
    krp, krf, krs = synth(70, seed=1, prefix="KR")
    usp, usf, _   = synth(70, seed=2, prefix="US")
    md = {
        "KR": (krp, dict(state="위험선호",note="합성",led="on"), krf, krs),   # 재무(PIT)+수급(스냅샷 틸트)
        "US": (usp, dict(state="중립",note="합성",led="neutral"), usf, None), # 재무만
    }
    p = P.build_payload(md); validate(p)
    sig = {}
    for r in p["recommendations"]:
        sig[r["signal"]] = sig.get(r["signal"],0)+1
    print("✓ 스키마 검증 통과")
    print(f"  추천 {len(p['recommendations'])}종목  신호분포 {sig}")
    print(f"  backtest: {p['backtest']}")
    kic = p["factor_ic"]["KR"]
    print("  KR 팩터 IC(백테스트):", {k: kic[k]['ic'] for k in ['momentum','value','quality','tech']})
    print(f"  KR weights(현재 점수용, 수급 틸트 포함): {p['weights']['KR']}")
    print(f"  US weights: {p['weights']['US']}")
    assert kic["value"]["ic"] is not None and kic["quality"]["ic"] is not None, "PIT 재무 IC 미계산"
    assert p["weights"]["KR"]["supply"] > 0, "KR 수급 틸트 미적용"
    assert p["weights"]["US"]["supply"] == 0, "US 수급 0 아님"
    kr_sup = [r['factors']['supply'] for r in p['recommendations'] if r['market']=='KR']
    assert float(np.std(kr_sup)) > 0, "KR 수급 팩터가 스냅샷을 반영하지 않음"
    print("  ✓ PIT 재무=백테스트 편입 / 수급=현재 스냅샷 틸트로 분리 반영 확인")
    with open(os.path.join(os.path.dirname(__file__),"sample_output.json"),"w",encoding="utf-8") as f:
        json.dump(p,f,ensure_ascii=False,indent=2)
    print("✓ sample_output.json 생성")
