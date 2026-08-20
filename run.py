#!/usr/bin/env python3
"""데일리 퀀트 추천 실행기.

    python run.py                      # KR+US, latest.json 생성
    python run.py --markets KR         # 한국만
    python run.py --start 2021-01-01 --outdir docs

docs/latest.json 를 대시보드가 읽는다. 날짜별 아카이브도 docs/history/ 에 남긴다.
"""
import argparse, json, os, shutil, sys
from datetime import datetime, timezone
from quant import pipeline as P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--markets", nargs="+", default=["KR", "US"])
    ap.add_argument("--start", default="2022-01-01")
    ap.add_argument("--outdir", default="docs")
    ap.add_argument("--cachedir", default=".cache/prices", help="일봉 증분 캐시 위치")
    ap.add_argument("--limit", type=int, default=0,
                    help="시장별 스캔 종목 상한(0=전체). 첫 실행/Actions 시간 관리용.")
    args = ap.parse_args()

    os.makedirs(os.path.join(args.outdir, "history"), exist_ok=True)
    md = {}
    for mk in args.markets:
        print(f"[{mk}] 유니버스 로딩…", flush=True)
        tickers = P.load_universe(mk)
        if args.limit:
            tickers = tickers[:args.limit]
        print(f"[{mk}] {len(tickers)}종목 가격 수집(증분 캐시)…", flush=True)
        panel = P.load_prices(mk, tickers, args.start, cachedir=args.cachedir)
        print(f"[{mk}] 유효 {len(panel)}종목 · 재무/수급/레짐…", flush=True)
        md[mk] = (panel, P.load_regime(mk),
                  P.load_pit_fundamentals(mk, tickers), P.load_pit_supply(mk, tickers))

    payload = P.build_payload(md)
    latest = os.path.join(args.outdir, "latest.json")
    with open(latest, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    shutil.copy(latest, os.path.join(args.outdir, "history", f"{stamp}.json"))

    print(f"✓ {latest}  ({len(payload['recommendations'])}종목, 기준일 {payload['as_of']})")
    print(f"  backtest={payload['backtest']}")


if __name__ == "__main__":
    sys.exit(main())
