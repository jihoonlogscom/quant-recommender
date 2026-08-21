/* 공통 유틸 — 모든 도구 페이지 공유 */
const QW = (() => {
  let _data = null;

  async function load() {
    if (_data) return _data;
    try {
      const r = await fetch("latest.json", { cache: "no-store" });
      if (r.ok) _data = await r.json();
    } catch (e) { /* noop */ }
    if (!_data) _data = { as_of: null, recommendations: [], weights: {}, factor_ic: {},
                          backtest: {}, backtest_by_market: {}, market_regime: {}, universe_size: {} };
    (_data.recommendations || []).forEach(r => { if (r.note) r.note = String(r.note).replace(/<[^>]*>/g, ""); });
    return _data;
  }

  const esc = s => String(s ?? "").replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  const pct = v => (v == null ? "—" : Math.round(v * 100) + "%");
  const pct1 = v => (v == null ? "—" : (v * 100).toFixed(1) + "%");
  const money = (v, mk) => (!v ? "—" : mk === "US"
    ? "$" + Number(v).toLocaleString(undefined, { maximumFractionDigits: 2 })
    : "₩" + Number(v).toLocaleString());
  const SIGLAB = { buy: "매수", watch: "관망", sell: "매도" };
  const FACTORS = ["momentum", "value", "quality", "supply", "tech"];
  const FLABEL = { momentum: "모멘텀", value: "가치", quality: "퀄리티", supply: "수급", tech: "기술" };
  const FCOLOR = { momentum: "var(--buy)", value: "var(--cyan)", quality: "var(--kr)", supply: "var(--watch)", tech: "var(--us)" };

  function header(data, elId) {
    const el = document.getElementById(elId || "asof");
    if (el && data.as_of) el.textContent = data.as_of;
  }

  /* 로컬 저장 (워치리스트 / 보유) */
  const store = {
    get(key, dflt) {
      try { const v = localStorage.getItem("qw:" + key); return v ? JSON.parse(v) : dflt; }
      catch (e) { return dflt; }
    },
    set(key, val) {
      try { localStorage.setItem("qw:" + key, JSON.stringify(val)); return true; }
      catch (e) { return false; }
    }
  };

  function rowName(r) {
    return `<span class="badge-mk ${r.market}">${r.market}</span> <b>${esc(r.name)}</b> `
         + `<span class="tk">${esc(r.ticker)}</span>${r.verified ? ' <span class="vf">검증</span>' : ""}`;
  }

  function segment(el, onPick) {
    if (!el) return;
    el.addEventListener("click", e => {
      const b = e.target.closest("button"); if (!b) return;
      [...el.children].forEach(x => x.setAttribute("aria-pressed", x === b));
      onPick(b.dataset);
    });
  }

  /* 간단 툴팁 (근거 등) */
  function tooltip() {
    let tip = document.getElementById("qtip");
    if (!tip) {
      tip = document.createElement("div"); tip.id = "qtip";
      tip.style.cssText = "position:fixed;z-index:60;max-width:340px;background:#0f1420;color:var(--ink);"
        + "border:1px solid var(--line-2);border-radius:10px;padding:9px 12px;font-size:12.5px;line-height:1.5;"
        + "box-shadow:0 12px 30px -10px rgba(0,0,0,.6);opacity:0;transition:opacity .12s;pointer-events:none";
      document.body.appendChild(tip);
    }
    const move = e => {
      const pad = 14, w = tip.offsetWidth, h = tip.offsetHeight;
      let x = e.clientX + pad, y = e.clientY + pad;
      if (x + w > innerWidth - 8) x = e.clientX - w - pad;
      if (y + h > innerHeight - 8) y = e.clientY - h - pad;
      tip.style.left = Math.max(8, x) + "px"; tip.style.top = Math.max(8, y) + "px";
    };
    return {
      bind(el, text) {
        if (!text) return;
        el.addEventListener("mouseenter", e => { tip.textContent = text; tip.style.opacity = "1"; move(e); });
        el.addEventListener("mousemove", move);
        el.addEventListener("mouseleave", () => { tip.style.opacity = "0"; });
      }
    };
  }

  return { load, esc, pct, pct1, money, SIGLAB, FACTORS, FLABEL, FCOLOR, header, store, rowName, segment, tooltip };
})();
