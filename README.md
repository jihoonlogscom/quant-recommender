# 데일리 퀀트 추천 (KR + US)

매일 한국·미국 주식을 **모멘텀·가치·퀄리티·수급·기술** 멀티팩터로 스캔해, 종합점수·보유기간별(5·20·60일) 상승확률·매수/관망/매도 신호·진입가/목표가를 산출하고 정적 대시보드로 보여준다. **전 구간 무료 스택.**

> 리서치 보조 도구이며 투자 자문이 아니다. 확률은 과거 히트레이트로, 미래 수익을 보장하지 않는다.

## 구조

```
run.py                  # 실행 엔트리 (→ docs/latest.json), --limit/--cachedir
quant/pipeline.py       # 멀티팩터·IC가중치·백테스트·검증·조립
quant/validate.py       # IC / PSR·Deflated Sharpe / PBO(CSCV)
quant/pit.py            # Point-in-Time 정합 + 실데이터 커넥터(EDGAR/DART/pykrx)
quant/cache.py          # 일봉 증분 parquet 캐시
tests/test_smoke.py         # 파이프라인 엔드투엔드(합성, 무네트워크)
tests/test_connectors.py    # EDGAR 파서 + 캐시 단위 테스트(합성, 무네트워크)
docs/                   # GitHub Pages 배포 단위 (index.html + latest.json + history/)
.github/workflows/daily.yml   # 평일 장마감 후 cron 실행 + 커밋
```

## 데이터 소스 (모두 무료)

| 팩터/용도 | 소스 | 키 |
|---|---|---|
| KR/US 일봉·유니버스 | FinanceDataReader, yfinance | 무 |
| 미국 재무(가치·퀄리티) | SEC EDGAR companyfacts | 무(User-Agent 필요) |
| 한국 재무(가치·퀄리티) | DART OpenAPI (OpenDartReader) | `DART_API_KEY`(선택) |
| 한국 수급(외국인·기관) | pykrx | 무 |
| 매크로/레짐 | 지수(코스피·S&P) MA·변동성 | 무 |

환경변수: `SEC_UA`(EDGAR 요구, 연락처 포함 문자열 권장) · `DART_API_KEY`(한국 재무, 없으면 KR은 모멘텀+추세+수급으로 동작) · `US_UNIVERSE`(기본 S&P500, `NASDAQ`/`NYSE`로 확장).

## 실행

```bash
pip install -r requirements.txt
export SEC_UA="your-app your-email@example.com"
python run.py --markets KR US --outdir docs --limit 400   # 첫 실행은 상한 권장
python -m http.server -d docs 8000                        # 대시보드 확인
python tests/test_smoke.py && python tests/test_connectors.py   # 무네트워크 검증
```

- 설치명 주의: import는 `FinanceDataReader`, PyPI 설치명은 `finance-datareader`. pandas는 `<3` 고정.
- `--limit 0`(또는 삭제) = 전종목. 최초 1회는 전 종목 일봉을 받느라 오래 걸리지만, 이후엔 **증분 캐시**로 마지막 날짜 이후만 받아 빨라진다.

## 룰 개요 (Phase 3)

**팩터 (5종)** — 모멘텀(12-1개월·상대강도·52주 고가) · 추세(20/50/200 MA 정렬) · 가치(PER·PBR·EV/EBITDA 역수) · 퀄리티(ROE·마진·부채비율·이익안정성) · 수급(외국인·기관 순매수, 한국 특화).

**Point-in-Time 정합** — 재무는 공시일(EDGAR filed / DART 접수일) 기준 일별 ffill, 수급은 일별 순매수의 trailing 롤링. 그날 알 수 있던 값만 과거에 적용하므로 **look-ahead 없이** 5개 팩터 전부가 백테스트·IC·가중치에 참여한다.

**점수·가중치(IC 게이팅)** — 팩터별 횡단면 z-score. "균형" 그룹 예산(가격 0.40 / 가치·퀄리티·수급 각 0.20)에서 출발하되, 재무·수급은 **과거 IC가 양(+)일 때만** 예산을 받고 예측력이 없으면(IC ≤ 0 또는 데이터 없음) 자동으로 0이 되어 재정규화된다. 가격 예산은 모멘텀/추세 IC 비율로 분할. 종합점수는 백분위 0~100.

**확률·검증** — decile별 5·20·60일 forward 상승확률(히트레이트) · 팩터 IC(Spearman) · PBO(CSCV 과최적화 확률) · Deflated/Probabilistic Sharpe(시도 수 반영). **verified 배지** = 룰셋 관문(PBO ≤ 0.5, DSR ≥ 0.5) 통과 **그리고** 종목 decile의 20일 히트레이트 ≥ 0.55.

**레짐 게이트** — 지수 200MA·변동성으로 위험선호/중립/위험회피, 위험회피 시 매수 억제. **가격** — ATR 진입밴드·손절·bear/base/bull. **유동성 필터** — 20일 평균 거래대금 하위 컷.

파라미터는 `quant/pipeline.py`의 `CFG`·`GROUP_BUDGET`에서 조정.

## 자동화 (무료)

1. GitHub에 올리고 Settings → Pages → Source를 `main`/`docs`로.
2. Settings → Actions → Workflow permissions = **Read and write**.
3. (선택) Settings → Secrets → `DART_API_KEY` 등록.
4. `.github/workflows/daily.yml`이 평일 21:30 UTC에 `run.py`를 돌려 `docs/latest.json` 갱신·커밋. 일봉 캐시는 `actions/cache`로 러닝 간 지속.

## 검증 상태

- 연산·파서·캐시 로직은 합성 데이터로 단위 검증 완료(`tests/`). 실데이터 fetch(EDGAR/pykrx/FDR/yfinance)는 네트워크가 필요하므로 사용자 환경/Actions에서 처음 실행된다.
- 스모크: 예측력 주입한 수급이 IC로 가중치를 얻고, 무작위 가치·퀄리티는 IC≤0으로 예산0 처리되는 IC 게이팅을 확인.
- EDGAR 파서: 합성 companyfacts에서 ROE·영업이익률·부채비율·EPS·BPS·EBITDA/주 정확 계산 확인.

## 로드맵 (선택 고도화)

- 분기 TTM 재무(현재 연간 기준) · EV 정밀화(순부채·주식수) · alphalens/vectorbt/quantstats/purgedcv 편입 · 유니버스 전종목 상시화 · korea-market-data 병행(수급 폴백).

## 데이터 계약 (latest.json)

```jsonc
{
  "as_of","ruleset",
  "market_regime": { "kr": {"state","note","led"}, "us": {...} },
  "universe_size": { "kr","us" },
  "backtest": { "hit_d5","hit_d20","hit_d60","deflated_sharpe","pbo" },
  "factor_ic": { "KR": {"momentum":{"ic","t","n"}, "value":{...}, "quality":{...}, "supply":{...}, "tech":{...}}, "US": {...} },
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
