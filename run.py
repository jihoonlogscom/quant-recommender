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
    args = ap.parse_args()

    os.makedirs(os.path.join(args.outdir, "history"), exist_ok=True)
    md = {}
    for mk in args.markets:
        print(f"[{mk}] 유니버스 로딩…", flush=True)
        tickers = P.load_universe(mk)
        print(f"[{mk}] {len(tickers)}종목 가격 수집…", flush=True)
        panel = P.load_prices(mk, tickers, args.start)
        print(f"[{mk}] 유효 {len(panel)}종목 · 레짐 판정…", flush=True)
        regime = P.load_regime(mk)
        md[mk] = (panel, regime)

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
