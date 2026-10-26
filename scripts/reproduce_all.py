"""
Reproduce todas las cifras del artículo y las verifica contra los valores publicados.

Uso (desde la raíz del repositorio):
    python scripts/reproduce_all.py                    # calcula y verifica
    python scripts/reproduce_all.py --solo-verificar   # solo verifica results/
Tiempo aproximado: 3–5 minutos (el placebo de 800 réplicas domina).
"""
import subprocess, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
py = [sys.executable, str(ROOT / "scripts" / "auditoria_y_ablacion.py")]
# --solo-verificar: compara results/ existentes sin volver a calcular
RUNS = [] if "--solo-verificar" in sys.argv else ([], ["grid"], ["placebo", "400", "1"], ["placebo", "400", "2"])
for args in RUNS:
    print(">>", " ".join(["auditoria_y_ablacion.py"] + args), flush=True)
    subprocess.run(py + args, cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
if RUNS:
    subprocess.run([sys.executable, str(ROOT / "scripts" / "make_figura1.py")], cwd=ROOT, check=True,
                   stdout=subprocess.DEVNULL)

fails = []
def check(name, got, exp, tol=5e-4):
    """Compara un valor reproducido con el del artículo y registra la diferencia si excede `tol`."""
    ok = abs(got - exp) <= tol
    print(f"  {'OK ' if ok else 'FAIL'} {name}: {got:.4f} (artículo: {exp})")
    if not ok:
        fails.append(name)

abl = pd.read_csv(RES / "r2_ablacion_metricas.csv").set_index("variante")
print("Tabla 3 y Tabla 5 (retorno total y Sharpe)")
for v, r, s in [("A", 0.1999, 1.1827), ("B", 0.1915, 1.0970), ("EW", 0.2182, 1.6494), ("B-cal", 0.2218, 1.3149),
                ("A-k1", -0.1368, -0.9310), ("A40", 0.2435, 1.6992), ("B40", 0.2201, 1.4870)]:
    check(f"{v} retorno total", abl.loc[v, "ret_total"], r)
    check(f"{v} Sharpe", abl.loc[v, "sharpe"], s)

aud = pd.read_csv(RES / "r2_auditoria_temporal.csv")
print("Tabla 4 (auditoría cronológica)")
check("titulares ventana calendario", aud["ventana_calendario"].sum(), 1958, 0)
check("posteriores al cierre", aud["posteriores_al_cierre"].sum(), 1203, 0)
check("titulares regla auditada", aud["ventana_auditada"].sum(), 897, 0)
check("precedencia acreditada en todos los rebalanceos", float(aud["precedencia_ok"].all()), 1.0, 0)

bt = pd.read_csv(RES / "r2_bootstrap_contrastes.csv")
b = bt[(bt.L == 10) & (bt.contraste == "IA auditada − Markowitz") & (bt.cantidad == "Δ retorno anualizado")].iloc[0]
print("Inferencia (sección 4.1)")
check("Δ anual IA − Markowitz", b.observado, -0.0113)
check("IC 95 % inferior", b.ic95_inf, -0.0738)
check("IC 95 % superior", b.ic95_sup, 0.0463)
check("p bootstrap", b.p_bootstrap, 0.700, 1e-3)

g = pd.read_csv(RES / "r2_sensibilidad_auditada.csv")
print("Sensibilidad (sección 4.4)")
check("configuraciones con diferencial positivo", float((g.delta_pp > 0).sum()), 9, 0)

pl = pd.concat([pd.read_csv(f) for f in sorted(RES.glob("r2_placebo_seed*.csv"))])["diferencial_total"]
print("Placebo (sección 4.1)")
check("réplicas", float(len(pl)), 800, 0)
check("percentil 2,5 %", pl.quantile(0.025), -0.0301)
check("percentil 97,5 %", pl.quantile(0.975), 0.0281)

print("Consistencia: la librería (build_walk_forward_schedules) reproduce al script")
sys.path.insert(0, str(ROOT))
import logging
logging.disable(logging.CRITICAL)
from investment_platform.portfolio_optimization import BacktestEngine, build_walk_forward_schedules
ret = pd.read_csv(ROOT / "data" / "returns_daily.csv", parse_dates=["date"]).set_index("date")
news = pd.read_csv(ROOT / "data" / "news_sentiment.csv", parse_dates=["date"])
eng = BacktestEngine(initial_capital=10_000, transaction_cost_bps=12)
ev = ret[ret.index >= "2025-06-01"]
for regla, var in [("auditada", "B"), ("calendario", "B-cal")]:
    s = build_walk_forward_schedules(ret, news, "2025-06-01", 90, 7, 0.5, regla_titulares=regla)
    check(f"librería regla={regla}", eng.run_strategy_with_schedule("x", s["Sentimiento IA"], ev).total_return,
          abl.loc[var, "ret_total"], 1e-9)

print("\nTODO COINCIDE" if not fails else f"\nDIFERENCIAS: {fails}")
sys.exit(1 if fails else 0)
