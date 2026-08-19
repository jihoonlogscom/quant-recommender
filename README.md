# 데일리 퀀트 추천 (KR + US)

매일 한국·미국 주식을 **모멘텀·가치·퀄리티·수급·기술** 멀티팩터로 스캔해, 종합점수·보유기간별 상승확률·매수/관망/매도 신호·진입가/목표가를 산출하고 정적 대시보드로 보여준다. **전 구간 무료 스택.**

> 리서치 보조 도구이며 투자 자문이 아니다. 확률은 과거 히트레이트로, 미래 수익을 보장하지 않는다.

## 구조

```
run.py                  # 실행 엔트리 (→ docs/latest.json)
quant/pipeline.py       # 멀티팩터·IC가중치·백테스트·검증·조립
quant/validate.py       # IC / PSR·Deflated Sharpe / PBO(CSCV) / 가중치
quant/fundamentals.py   # 가치·퀄리티 원재료 (DART / SEC EDGAR)
quant/supply.py         # 수급 원재료 (korea-market-data, 한국 특화)
tests/test_smoke.py     # 네트워크 없이 합성 데이터로 로직 검증
docs/                   # GitHub Pages 배포 단위 (index.html + latest.json + history/)
.github/workflows/daily.yml   # 평일 장마감 후 cron 실행 + 커밋
```

## 실행

```bash
pip install -r requirements.txt
python run.py --markets KR US --outdir docs         # 실데이터 (네트워크 필요)
python -m http.server -d docs 8000                  # 대시보드 확인
python tests/test_smoke.py                          # 합성 데이터로 파이프라인 점검(무네트워크)
```

한국 재무는 `DART_API_KEY` 환경변수가 있으면 사용한다(없으면 재무 팩터는 중립 처리).

## 룰 개요 (Phase 2)

**팩터 (5종)**
- 모멘텀: 12-1개월 수익률 · 3개월 상대강도 · 52주 고가 근접도
- 추세(기술): 20/50/200 이동평균 정렬
- 가치: PER·PBR·EV/EBITDA (역수 정규화 · DART/EDGAR)
- 퀄리티: ROE · 영업이익률 · 부채비율 · 이익 안정성
- 수급(한국 특화): 외국인·기관 순매수 추세 (korea-market-data)

**점수·가중치** — 팩터별 횡단면 z-score. "균형" 성향의 그룹 예산(가격 0.40 / 가치 0.20 / 퀄리티 0.20 / 수급 0.20)으로 배분하고, 가격 예산은 팩터 IC 비율로 모멘텀/추세에 분할한다. 데이터 없는 그룹(예: 미국 수급)은 예산 0 후 재정규화. 종합점수는 백분위 0~100.

**확률·검증 (정직성 원칙)** — 확률(prob_up)·PBO·Deflated Sharpe·IC는 **가격 기반 엔진**(모멘텀+추세)으로만 산출한다. 과거 재무/수급 시계열이 없어 오늘 스냅샷을 과거에 적용하면 look-ahead이므로, 재무·수급 팩터는 **현재 랭킹/점수와 표시**에만 반영한다.
- **prob_up:** 점수 decile별 5·20·60일 forward 상승확률(과거 히트레이트, look-ahead 없음)
- **IC:** 팩터-forward수익 Spearman 상관의 평균·t값(예측력)
- **PBO:** Combinatorially-Symmetric CV로 과최적화 확률 추정
- **Deflated Sharpe:** 시도 수를 반영한 유의성(Bailey & López de Prado)
- **verified 배지:** 룰셋이 관문(PBO ≤ 0.5, DSR ≥ 0.5) 통과 **그리고** 해당 종목 decile의 20일 히트레이트 ≥ 0.55

**레짐 게이트** — 지수 200MA·변동성으로 위험선호/중립/위험회피 판정, 위험회피 시 매수 억제.
**가격** — ATR 기반 진입밴드·손절·bear/base/bull 목표. **유동성 필터** — 20일 평균 거래대금 하위 컷.

파라미터는 `quant/pipeline.py`의 `CFG`·`GROUP_BUDGET`에서 조정한다.

## 자동화 (무료)

1. GitHub에 올리고 Settings → Pages → Source를 `main`/`docs`로 지정.
2. `.github/workflows/daily.yml`이 평일 21:30 UTC에 `run.py`를 돌려 `docs/latest.json`을 갱신·커밋(Actions 권한 = Read and write).

연산·저장·호스팅 모두 GitHub 무료 범위.

## 로드맵 (→ Phase 2.5+)

- **Point-in-time 재무:** SEC EDGAR 공시일 정렬로 재무 팩터를 백테스트에 포함(현재는 현재 스냅샷만, 확률 백테스트에서 제외).
- **수급 실연결:** korea-market-data CSV/JSON을 `quant/supply.py`에 연결(현재 골격).
- **엔진 강화(선택):** alphalens-reloaded(IC 리포트), vectorbt(전종목 벡터 백테스트), quantstats(성과 티어시트), purged-cross-validation(CPCV/PBO 정밀).
- **유니버스 확장:** S&P500 → 러셀급, 코스피200/코스닥150 → 전종목(증분 수집·캐시).
- **데이터 견고화:** FinanceDataReader+DART OpenAPI 정본, 커뮤니티 데이터셋은 보조.

## 데이터 계약 (latest.json)

```jsonc
{
  "as_of","ruleset",
  "market_regime": { "kr": {"state","note","led"}, "us": {...} },
  "universe_size": { "kr", "us" },
  "backtest": { "hit_d5","hit_d20","hit_d60","deflated_sharpe","pbo" },
  "factor_ic": { "KR": {"momentum":{"ic","t","n"},"tech":{...}}, "US": {...} },
  "weights":   { "KR": {"momentum","tech","value","quality","supply"}, "US": {...} },
  "recommendations": [{
    "ticker","name","market","sector","score",
    "factors": {"momentum","value","quality","supply","tech"},  // 0~1
    "prob_up": {"d5","d20","d60"},                               // 0~1
    "signal": "buy|watch|sell",
    "entry": {"low","high"}, "targets": {"bear","base","bull"}, "stop",
    "verified": true
  }]
}
```
