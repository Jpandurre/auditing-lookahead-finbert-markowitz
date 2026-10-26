"""
R2 (dictamen Revista Ingeniantes): auditoría cronológica de titulares y
análisis controlado con universo, señal y reglas operativas constantes.

CONVENCIÓN TEMPORAL DEL MOTOR
  Los pesos fijados en el rebalanceo t (primer día hábil del mes) se aplican al
  retorno de t = cierre_t / cierre_{t-1} − 1. La cartera se forma, por tanto, al
  CIERRE de la sesión previa p (16:00 ET). Precios: retornos con índice < t.

ZONA HORARIA DE time_published (Alpha Vantage)
  No documentada por la API; se verificó empíricamente que corresponde a hora del
  Este de EE. UU. (p. ej., JPM 2T25 publicado por MarketWatch a las 06:45 del
  15-jul-2025; cobertura de resultados de NVIDIA del 27-ago-2025 a las 18:37,
  posterior a su publicación a las 16:20 ET). 352 marcas son exactamente 00:00:00:
  fechas sin hora (p. ej., el comunicado de resultados de NVIDIA del 27-ago).

REGLAS DE SELECCIÓN DE TITULARES PARA EL REBALANCEO t
  calendario : [t − 7 d, t)                       (especificación original)
  auditada   : [c − 7 d, c), c = p 16:00 ET, excluyendo marcas 00:00:00 del día p
               (hora desconocida -> no se puede acreditar precedencia)
  fuga       : titulares del mes de tenencia [t, t_siguiente)  (fuga deliberada)

Uso (desde la raíz del repositorio):
    python scripts/auditoria_y_ablacion.py            # todo excepto placebo
    python scripts/auditoria_y_ablacion.py placebo 500 1   # R réplicas, semilla
    python scripts/auditoria_y_ablacion.py grid            # exploración α × lookback
Entradas: data/returns_daily.csv, data/news_sentiment.csv
Salidas: results/r2_*.csv
Entorno de referencia: Python 3.x, numpy 2.2, pandas 2.3, scipy 1.15.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import logging
logging.disable(logging.CRITICAL)

import numpy as np
import pandas as pd
from scipy.stats import norm

from investment_platform.portfolio_optimization.allocators import MarkowitzAllocator, SentimentAllocator
from investment_platform.portfolio_optimization.backtest import BacktestEngine
from investment_platform.portfolio_optimization.inference import (
    net_daily_returns, stationary_bootstrap_indices, bootstrap_metrics,
    percentile_ci, bootstrap_pvalue, metrics_from_returns)

DATA = ROOT / "data"
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)
EVAL_START = pd.Timestamp("2025-06-01")
WIN, LB, ALPHA, COST, CAP0, RF = 90, 7, 0.5, 12.0, 10_000.0, 0.04
SEED, B, L = 42, 10_000, 10

ret_full = pd.read_csv(DATA / "returns_daily.csv", parse_dates=["date"]).set_index("date")
news_all = pd.read_csv(DATA / "news_sentiment.csv", parse_dates=["date"])
TK = list(ret_full.columns)
DAYS = list(ret_full.index)
EV = ret_full.index[ret_full.index >= EVAL_START]
RET_EV = ret_full.loc[EV]
REB = BacktestEngine.generate_rebalance_dates(EV)
REB_T1 = [DAYS[DAYS.index(t) + 1] for t in REB]
ENG = BacktestEngine(initial_capital=CAP0, transaction_cost_bps=COST)
EW = {t: {k: 1 / len(TK) for k in TK} for t in REB}


def prev(t):
    """Sesión bursátil inmediatamente anterior a t (p en el artículo)."""
    return DAYS[DAYS.index(t) - 1]


def titulares(t, regla, news=news_all, dates=REB):
    """Titulares admitidos en el rebalanceo t según la regla temporal.

    Reglas: 'calendario' = [t − 7 d, t); 'auditada_ET' = [c − 7 d, c) con c = p 16:00 ET;
    'auditada' = igual que 'auditada_ET' excluyendo marcas 00:00:00 del día p;
    'fuga' = titulares del mes de tenencia [t, t_siguiente) (fuga deliberada, solo como contraste).
    """
    p = prev(t)
    c = p + pd.Timedelta(hours=16)
    if regla == "calendario":
        lo, hi = t - pd.Timedelta(days=LB), t
    elif regla in ("auditada", "auditada_ET"):
        lo, hi = c - pd.Timedelta(days=LB), c
    elif regla == "fuga":
        k = dates.index(t)
        lo, hi = t, dates[k + 1] if k + 1 < len(dates) else EV[-1] + pd.Timedelta(days=1)
    n = news[(news["date"] >= lo) & (news["date"] < hi)]
    if regla == "auditada":
        n = n[n["date"] != p]          # marca 00:00:00 del día p: hora desconocida
    return n


def schedules(regla, cap=1.0, dates=REB, news=news_all, rng=None):
    """Calendarios de pesos {fecha: {ticker: peso}} de Markowitz y Sentimiento IA.

    Args:
        regla: regla de selección de titulares (ver `titulares`).
        cap: tope de peso por activo (1.0 = sin tope).
        dates: fechas de rebalanceo (REB = primer día hábil de cada mes).
        news: corpus de noticias a usar.
        rng: si se pasa, permuta las puntuaciones entre emisoras (placebo).

    Returns:
        (calendario_markowitz, calendario_sentimiento).
    """
    mw, sa = MarkowitzAllocator(max_weight=cap), SentimentAllocator(alpha=ALPHA, max_weight=cap)
    a, b = {}, {}
    for t in dates:
        w = ret_full[ret_full.index < t].tail(WIN)
        a[t] = mw.optimize(w)
        sc = titulares(t, regla, news, dates).groupby("ticker")["sentiment_score"].mean().to_dict()
        if rng is not None and len(sc) >= 2:
            k = list(sc)
            sc = dict(zip(k, rng.permutation([sc[x] for x in k])))
        b[t] = sa.optimize(w, sc)
    return a, b


def run(s):
    """Backtest encadenado con costos sobre la ventana de evaluación (197 sesiones)."""
    return ENG.run_strategy_with_schedule("x", s, RET_EV)


def activas(a, b, dates):
    """Número de rebalanceos en que la señal modifica los pesos de Markowitz (distancia L1 > 0)."""
    return sum(1 for t in dates if sum(abs(b[t].get(k, 0) - a[t].get(k, 0)) for k in TK) > 1e-6)


def jobson_korkie_memmel(ra, rb, rf=RF):
    """Test JK con corrección de Memmel (2003) sobre retornos diarios en exceso."""
    ea, eb = ra - rf / 252, rb - rf / 252
    sa, sb = ea.mean() / ea.std(ddof=1), eb.mean() / eb.std(ddof=1)
    rho = np.corrcoef(ea, eb)[0, 1]
    T = len(ea)
    theta = (2 * (1 - rho) + 0.5 * (sa ** 2 + sb ** 2 - 2 * sa * sb * rho ** 2)) / T
    z = (sa - sb) / np.sqrt(theta)
    return z, 2 * (1 - norm.cdf(abs(z))), rho


if len(sys.argv) > 1 and sys.argv[1] == "grid":
    # Exploración α × lookback sobre la MISMA muestra (no es validación independiente)
    mw = MarkowitzAllocator()
    base = {t: mw.optimize(ret_full[ret_full.index < t].tail(WIN)) for t in REB}
    r_mw = run(base)
    rows = []
    for lb in (3, 7, 14, 21, 30):
        LB = lb
        sc = {t: titulares(t, "auditada").groupby("ticker")["sentiment_score"].mean().to_dict() for t in REB}
        for al in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
            sa = SentimentAllocator(alpha=al)
            r = run({t: sa.optimize(ret_full[ret_full.index < t].tail(WIN), sc[t]) for t in REB})
            rows.append(dict(alpha=al, lookback=lb, ret_mw=r_mw.total_return, ret_ia=r.total_return,
                             delta_pp=100 * (r.total_return - r_mw.total_return),
                             sharpe_mw=r_mw.sharpe_ratio, sharpe_ia=r.sharpe_ratio))
    g = pd.DataFrame(rows)
    g.to_csv(OUT / "r2_sensibilidad_auditada.csv", index=False)
    print(g.pivot(index="alpha", columns="lookback", values="delta_pp").round(2).to_string())
    print(g.groupby("lookback")["delta_pp"].agg(["mean", "min", "max", lambda x: (x > 0).mean()]).round(3))
    print("positivos totales:", (g.delta_pp > 0).sum(), "de", len(g), "| máx:", g.loc[g.delta_pp.idxmax()].to_dict())
    sys.exit()

if len(sys.argv) > 1 and sys.argv[1] == "placebo":
    R, seed = int(sys.argv[2]), int(sys.argv[3])
    rng = np.random.default_rng(seed)
    a, _ = schedules("auditada")
    base = run(a).total_return
    d = [run(schedules("auditada", rng=rng)[1]).total_return - base for _ in range(R)]
    pd.DataFrame({"diferencial_total": d}).to_csv(OUT / f"r2_placebo_seed{seed}.csv", index=False)
    print(f"seed {seed}: {R} réplicas, media {np.mean(d):+.4f}")
    sys.exit()

pd.set_option("display.width", 260, "display.max_columns", 40)

# ── 1. Auditoría cronológica por rebalanceo ────────────────────────────────
aud = []
for t in REB:
    p = prev(t)
    c = p + pd.Timedelta(hours=16)
    w = titulares(t, "calendario")
    wa = titulares(t, "auditada")
    aud.append(dict(
        rebalanceo=t.date(), cierre_ejecucion=c, ventana_calendario=len(w),
        posteriores_al_cierre=int((w["date"] >= c).sum()),
        sin_hora_dia_ejecucion=int((w["date"] == p).sum()),
        ultima_marca_calendario=w["date"].max() if len(w) else None,
        ventana_auditada=len(wa), emisoras_auditada=wa["ticker"].nunique(),
        primera_marca_auditada=wa["date"].min() if len(wa) else None,
        ultima_marca_auditada=wa["date"].max() if len(wa) else None,
        precedencia_ok=bool((wa["date"] < c).all()),
    ))
aud = pd.DataFrame(aud)
print(aud.to_string())

# ── 2. Resultados principales (regla auditada) y ablación ──────────────────
mw0, ia_aud = schedules("auditada")
_, ia_cal = schedules("calendario")
_, ia_fuga = schedules("fuga")
mw40, ia40 = schedules("auditada", cap=0.40)
mwk1, iak1 = schedules("auditada", dates=REB[:1])
dedup = news_all.drop_duplicates(subset=["ticker", "headline_id"])
_, ia_dedup = schedules("auditada", news=dedup)

V = [("A", "Markowitz walk-forward", mw0, REB),
     ("B", "Sentimiento IA — regla auditada", ia_aud, REB),
     ("EW", "Igual ponderación", EW, REB),
     ("B-cal", "Sentimiento IA — ventana de calendario [t−7 d, t)", ia_cal, REB),
     ("C", "Sentimiento IA — fuga deliberada (mes de tenencia)", ia_fuga, REB),
     ("B-dedup", "Sentimiento IA — auditada, titulares sindicados deduplicados", ia_dedup, REB),
     ("A40", "Markowitz, tope 40 %", mw40, REB),
     ("B40", "Sentimiento IA auditada, tope 40 %", ia40, REB),
     ("A-k1", "Markowitz, partición única (k = 1)", mwk1, REB[:1]),
     ("B-k1", "Sentimiento IA, partición única (k = 1)", iak1, REB[:1])]
rows, daily = [], {}
for v, desc, s, dts in V:
    r = run(s)
    daily[v] = net_daily_returns(r.portfolio_values, CAP0)
    rows.append(dict(variante=v, descripcion=desc, capital_final=r.final_value,
                     ret_total=r.total_return, ret_anual=r.annual_return,
                     vol=r.annual_volatility, sharpe=r.sharpe_ratio, sortino=r.sortino_ratio,
                     cvar95=r.cvar_95, mdd=r.max_drawdown, costos=r.transaction_cost_paid,
                     senal_activa=(f"{activas(mw40 if v=='B40' else mw0, s, dts)}/{len(dts)}"
                                   if v.startswith(("B", "C")) else "")))
abl = pd.DataFrame(rows)
abl.to_csv(OUT / "r2_ablacion_metricas.csv", index=False)
print(abl.round(4).to_string())

# Pesos por rebalanceo (trazabilidad)
pw = []
for t in REB:
    for lab, s in [("Markowitz", mw0), ("Sentimiento IA", ia_aud)]:
        pw.append(dict(rebalanceo=t.date(), estrategia=lab, **{k: s[t].get(k, 0) for k in TK}))
pd.DataFrame(pw).to_csv(OUT / "r2_pesos_por_rebalanceo.csv", index=False)
print(pd.DataFrame(pw).round(3).to_string())

# ── 3. Inferencia ──────────────────────────────────────────────────────────
z, pjk, rho = jobson_korkie_memmel(daily["B"].to_numpy(), daily["A"].to_numpy())
z2, pjk2, _ = jobson_korkie_memmel(daily["EW"].to_numpy(), daily["B"].to_numpy())
print(f"JK-Memmel IA vs MW: z={z:.3f} p={pjk:.3f} rho={rho:.3f} | EW vs IA: z={z2:.3f} p={pjk2:.3f}")
cols = ["A", "B", "EW", "B-cal", "C", "A40", "B40"]
mat = pd.DataFrame({c: daily[c] for c in cols}).to_numpy()
om = {c: metrics_from_returns(daily[c].to_numpy()) for c in cols}
bt = []
for Lb in (5, 10, 20):
    idx = stationary_bootstrap_indices(mat.shape[0], B=B, L=Lb, rng=np.random.default_rng(SEED))
    bm = bootstrap_metrics(mat, idx)
    for a, b_, lab in [("B", "A", "IA auditada − Markowitz"),
                       ("EW", "B", "Igual ponderación − IA auditada"),
                       ("EW", "A", "Igual ponderación − Markowitz"),
                       ("B-cal", "B", "Calendario − auditada (efecto de la fuga horaria)"),
                       ("C", "B", "Fuga deliberada − auditada"),
                       ("B40", "A40", "IA − Markowitz (tope 40 %)"),
                       ("A40", "A", "Tope 40 % − sin tope (Markowitz)")]:
        if Lb != 10 and lab != "IA auditada − Markowitz":
            continue
        ia_, ib_ = cols.index(a), cols.index(b_)
        for m, key in [("Δ retorno anualizado", "annual_return"), ("Δ Sharpe", "sharpe_ratio")]:
            d = bm[key][:, ia_] - bm[key][:, ib_]
            lo, hi = percentile_ci(d)
            bt.append(dict(L=Lb, contraste=lab, cantidad=m, observado=om[a][key] - om[b_][key],
                           ic95_inf=lo, ic95_sup=hi, p_bootstrap=bootstrap_pvalue(d)))
bt = pd.DataFrame(bt)
bt.to_csv(OUT / "r2_bootstrap_contrastes.csv", index=False)
print(bt.round(4).to_string())

# ── 4. Sensibilidad a la convención temporal ───────────────────────────────
conv = []
for lab, regla, dts in [("Calendario [t−7 d, t)", "calendario", REB),
                        ("Corte al cierre previo (16:00 ET)", "auditada_ET", REB),
                        ("Auditada (corte + sin marcas sin hora)", "auditada", REB),
                        ("Ejecución al cierre de t (auditada)", "auditada", REB_T1)]:
    a, b = schedules(regla, dates=dts)
    ra, rb = run(a), run(b)
    da, db = net_daily_returns(ra.portfolio_values, CAP0), net_daily_returns(rb.portfolio_values, CAP0)
    idx = stationary_bootstrap_indices(len(da), B=B, L=L, rng=np.random.default_rng(SEED))
    bm = bootstrap_metrics(np.column_stack([db, da]), idx)
    d = bm["annual_return"][:, 0] - bm["annual_return"][:, 1]
    lo, hi = percentile_ci(d)
    conv.append(dict(convencion=lab, mw_total=ra.total_return, ia_total=rb.total_return,
                     delta_total_pp=100 * (rb.total_return - ra.total_return),
                     delta_anual_pp=100 * (rb.annual_return - ra.annual_return),
                     mw_sharpe=ra.sharpe_ratio, ia_sharpe=rb.sharpe_ratio,
                     ic_inf=lo, ic_sup=hi, p=bootstrap_pvalue(d), activa=f"{activas(a, b, dts)}/10"))
conv = pd.DataFrame(conv)
conv.to_csv(OUT / "r2_convenciones_temporales.csv", index=False)
print(conv.round(4).to_string())

# ── 5. CVaR / MDD coincidentes ─────────────────────────────────────────────
ra, rb = daily["A"], daily["B"]
same = (ra - rb).abs() < 1e-12
ta, tb = ra[ra <= ra.quantile(.05)].index, rb[rb <= rb.quantile(.05)].index
pa, pb = (1 + ra).cumprod(), (1 + rb).cumprod()
da_, db_ = pa / pa.cummax() - 1, pb / pb.cummax() - 1
cv = dict(dias_identicos=int(same.sum()), dias=len(ra), cola_A=len(ta), cola_compartida=len(set(ta) & set(tb)),
          cola_en_dias_identicos=int(same.loc[list(ta)].sum()),
          cvar_A=ra[ra <= ra.quantile(.05)].mean(), cvar_B=rb[rb <= rb.quantile(.05)].mean(),
          mdd_A=da_.min(), mdd_B=db_.min(), pico=pa.loc[:da_.idxmin()].idxmax().date(), valle=da_.idxmin().date())
pd.DataFrame([cv]).to_csv(OUT / "r2_cvar_mdd.csv", index=False)
print(cv)

# ── 6. Corpus: cobertura, duplicados ───────────────────────────────────────
cal = pd.date_range(EVAL_START, EV[-1], freq="D")
have = set(news_all["date"].dt.normalize())
emp = pd.Series([d not in have for d in cal], index=cal)
mid = (news_all["date"] == news_all["date"].dt.normalize()).sum()
print(f"días calendario {len(cal)}, sin noticias {emp.sum()} ({emp.mean():.1%}); marcas 00:00:00 {mid}; "
      f"duplicados (ticker,url) {news_all.duplicated(['ticker','url']).sum()}; "
      f"titulares sindicados repetidos (ticker,titular) {news_all.duplicated(['ticker','headline_id']).sum()}")
print("noticias por mes y emisora:")
print(news_all.groupby([news_all["date"].dt.to_period("M"), "ticker"]).size().unstack(fill_value=0).to_string())
aud.to_csv(OUT / "r2_auditoria_temporal.csv", index=False)
