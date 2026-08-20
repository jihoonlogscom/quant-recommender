"""네트워크 없이 Phase 2.5 파이프라인(PIT 재무·수급 편입)을 합성 데이터로 검증."""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from quant import pipeline as P


def synth(n, n_days=820, seed=0, prefix="T"):
    """가격 패널 + PIT 재무 + PIT 수급(수급은 미래수익에 예측력 갖도록 생성)."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-03", periods=n_days)
    panel, fund, supply = {}, {}, {}
    for k in range(n):
        drift = rng.normal(0.0004, 0.0005); vol = rng.uniform(0.012, 0.03)
        rets = rng.normal(drift, vol, n_days)
        rets[60:] += 0.15 * pd.Series(rets).rolling(40).mean().shift(1).fillna(0).values[60:]  # 모멘텀 지속성
        close = 100 * np.exp(np.cumsum(rets))
        high = close*(1+np.abs(rng.normal(0,0.006,n_days))); low = close*(1-np.abs(rng.normal(0,0.006,n_days)))
        openp = close*(1+rng.normal(0,0.004,n_days)); volume = rng.integers(1e5,5e6,n_days)*(1+k/n)
        tk = f"{prefix}{k:03d}"
        panel[tk] = pd.DataFrame({"Open":openp,"High":high,"Low":low,"Close":close,"Volume":volume}, index=dates)

        # PIT 재무: 분기 공시일마다 값 갱신
        fdates = dates[::63]
        fund[tk] = pd.DataFrame({
            "eps_ttm": rng.uniform(2,12,len(fdates)), "bps": rng.uniform(20,120,len(fdates)),
            "ebitda_ps": rng.uniform(3,18,len(fdates)), "roe": rng.uniform(-0.1,0.35,len(fdates)),
            "op_margin": rng.uniform(-0.05,0.4,len(fdates)), "debt_ratio": rng.uniform(20,250,len(fdates)),
            "earn_stability": rng.uniform(0.1,1.0,len(fdates)),
        }, index=fdates)

        # PIT 수급: 향후 20일 수익 방향에 상관되게(예측력 부여) + 노이즈
        fwd20 = pd.Series(close, index=dates).pct_change(20).shift(-20).fillna(0).values
        base = np.sign(fwd20) * rng.uniform(0.3,1.0,n_days) + rng.normal(0,0.6,n_days)
        supply[tk] = pd.DataFrame({"foreign_net": base*7e8, "inst_net": base*3e8}, index=dates)
    return panel, fund, supply


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
        "KR": (krp, dict(state="위험선호",note="합성",led="on"), krf, krs),   # 재무+수급
        "US": (usp, dict(state="중립",note="합성",led="neutral"), usf, {}),   # 재무만(수급 없음)
    }
    p = P.build_payload(md); validate(p)
    sig = {}
    for r in p["recommendations"]:
        sig[r["signal"]] = sig.get(r["signal"],0)+1
    print("✓ Phase 2.5 스키마 검증 통과")
    print(f"  추천 {len(p['recommendations'])}종목  신호분포 {sig}")
    print(f"  backtest: {p['backtest']}")
    kic = p["factor_ic"]["KR"]
    print("  KR 팩터 IC:")
    for k in ["momentum","value","quality","supply","tech"]:
        print(f"    {k:9s} ic={kic[k]['ic']}  t={kic[k]['t']}  n={kic[k]['n']}")
    print(f"  KR weights: {p['weights']['KR']}")
    print(f"  US weights: {p['weights']['US']}  (수급 데이터 없음→supply 0 확인)")
    assert kic["value"]["ic"] is not None and kic["quality"]["ic"] is not None, "재무 IC 미계산"
    assert kic["supply"]["ic"] is not None, "수급 IC 미계산"
    assert p["weights"]["US"]["supply"] == 0, "US supply 예산 0 아님"
    assert p["weights"]["KR"]["supply"] > 0, "예측력 있는 KR supply가 가중치 0 (PIT 편입 실패)"
    print("  ✓ 가치·퀄리티·수급이 PIT로 IC 산출 & 예측력 있는 수급이 가중치 획득")
    with open(os.path.join(os.path.dirname(__file__),"sample_output.json"),"w",encoding="utf-8") as f:
        json.dump(p,f,ensure_ascii=False,indent=2)
    print("✓ sample_output.json 생성")
