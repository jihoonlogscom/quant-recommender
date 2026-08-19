"""
검증 레이어 (과최적화·데이터누수 방어)

- information_coefficient : 팩터의 예측력(Spearman IC, alphalens 아이디어를 직접 구현)
- probabilistic_sharpe_ratio / deflated_sharpe_ratio : Bailey & López de Prado
- pbo_cscv : Combinatorially-Symmetric Cross-Validation 기반 과최적화 확률(PBO)
- derive_weights : IC에 비례한 비음(non-negative) 팩터 가중치

의존성은 numpy/pandas만. Φ는 math.erf, Φ⁻¹는 Acklam 근사로 직접 구현.
"""
from __future__ import annotations
import math
from itertools import combinations
import numpy as np
import pandas as pd

GAMMA = 0.5772156649015329  # Euler–Mascheroni


def _phi(x: float) -> float:                      # 표준정규 CDF
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _phi_inv(p: float) -> float:                  # 표준정규 역CDF (Acklam)
    if p <= 0.0:
        return -math.inf
    if p >= 1.0:
        return math.inf
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


# ---------------------------------------------------------------- IC
def information_coefficient(pairs: list[tuple[pd.Series, pd.Series]]) -> dict:
    """pairs = [(factor_values, forward_returns), ...] (as-of 날짜별 횡단면).
    각 날짜의 Spearman 상관을 IC로 보고, 평균 IC와 t-stat(연속성 지표)을 낸다."""
    ics = []
    for fac, fwd in pairs:
        df = pd.concat([fac, fwd], axis=1).dropna()
        if len(df) >= 5:
            ic = df.iloc[:, 0].corr(df.iloc[:, 1], method="spearman")
            if pd.notna(ic):
                ics.append(ic)
    if len(ics) < 3:
        return dict(ic=None, t=None, n=len(ics))
    arr = np.array(ics)
    mean, sd = arr.mean(), arr.std(ddof=1)
    t = mean / sd * math.sqrt(len(arr)) if sd > 0 else 0.0
    return dict(ic=round(float(mean), 4), t=round(float(t), 2), n=len(arr))


# ---------------------------------------------------- PSR / Deflated SR
def probabilistic_sharpe_ratio(returns, sr_benchmark: float = 0.0) -> float:
    """관측 Sharpe가 기준 Sharpe보다 클 확률(왜도·첨도 보정). 값 ∈ [0,1]."""
    r = pd.Series(returns).dropna().to_numpy()
    T = len(r)
    if T < 10 or r.std(ddof=1) == 0:
        return float("nan")
    sr = r.mean() / r.std(ddof=1)
    g3 = pd.Series(r).skew()
    g4 = pd.Series(r).kurt() + 3.0                    # pandas kurt=초과첨도 → 피어슨 첨도
    denom = math.sqrt(max(1e-12, 1 - g3 * sr + ((g4 - 1) / 4.0) * sr * sr))
    return _phi((sr - sr_benchmark) * math.sqrt(T - 1) / denom)


def deflated_sharpe_ratio(returns, sr_trials: list[float]) -> float:
    """여러 시도(trial)의 Sharpe 분산을 반영해 기대 최대 Sharpe(SR0) 대비 유의성 평가.
    sr_trials = 시도된 전략들의 (기간당) Sharpe 목록. 값 ∈ [0,1]."""
    N = len(sr_trials)
    if N < 2:
        return probabilistic_sharpe_ratio(returns, 0.0)
    v = np.var(np.array(sr_trials), ddof=1)
    if v <= 0:
        return probabilistic_sharpe_ratio(returns, 0.0)
    sr0 = math.sqrt(v) * ((1 - GAMMA) * _phi_inv(1 - 1.0 / N) + GAMMA * _phi_inv(1 - 1.0 / (N * math.e)))
    return probabilistic_sharpe_ratio(returns, sr0)


# ---------------------------------------------------------------- PBO
def pbo_cscv(config_returns: pd.DataFrame, n_splits: int = 8) -> float:
    """Combinatorially-Symmetric CV로 과최적화 확률(PBO)을 추정.
    config_returns: index=기간, columns=후보 전략(config), 값=기간 수익률.
    IS에서 최고였던 전략이 OOS에서 중앙값 아래로 떨어지는 비율."""
    M = config_returns.dropna()
    T, N = M.shape
    if N < 2 or T < n_splits * 2:
        return float("nan")
    if n_splits % 2:
        n_splits -= 1
    blocks = np.array_split(np.arange(T), n_splits)
    logits = []
    for is_idx in combinations(range(n_splits), n_splits // 2):
        is_rows = np.concatenate([blocks[i] for i in is_idx])
        oos_rows = np.concatenate([blocks[i] for i in range(n_splits) if i not in is_idx])
        IS, OOS = M.iloc[is_rows], M.iloc[oos_rows]
        sr_is = IS.mean() / IS.std(ddof=1).replace(0, np.nan)
        sr_oos = OOS.mean() / OOS.std(ddof=1).replace(0, np.nan)
        if sr_is.isna().all() or sr_oos.isna().all():
            continue
        best = sr_is.idxmax()
        rank = sr_oos.rank().loc[best]               # 1..N
        omega = rank / (N + 1)
        omega = min(max(omega, 1e-6), 1 - 1e-6)
        logits.append(math.log(omega / (1 - omega)))
    if not logits:
        return float("nan")
    return round(float(np.mean(np.array(logits) < 0)), 3)


# ------------------------------------------------------------ weights
def derive_weights(ic_by_factor: dict, defaults: dict) -> dict:
    """양(+)의 IC에 비례한 비음 가중치. IC 없는 팩터는 defaults 사용, 합=1 정규화."""
    raw = {}
    for f, dft in defaults.items():
        ic = ic_by_factor.get(f, {}).get("ic")
        raw[f] = max(ic, 0.0) if ic is not None else dft
    s = sum(raw.values())
    if s <= 0:
        return dict(defaults)
    return {f: round(v / s, 4) for f, v in raw.items()}
