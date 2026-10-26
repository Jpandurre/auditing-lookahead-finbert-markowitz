"""Figura 1: protocolo walk-forward con la convención temporal auditada.
Uso (desde la raíz del repositorio): python scripts/make_figura1.py  ->  results/figura1.png"""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.dates as mdates
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import pandas as pd
idx = pd.read_csv("data/returns_daily.csv", parse_dates=["date"])["date"]
idx = pd.DatetimeIndex(idx)
EVAL_START = pd.Timestamp("2025-06-01")
ev = idx[idx >= EVAL_START]
origins = [ev[ev.to_period("M") == m][0] for m in ev.to_period("M").unique()]
assert len(ev) == 197 and len(origins) == 10, (len(ev), len(origins))
prev = lambda t: idx[idx < t][-1]
def est_start(t):
    """Primera fecha de la ventana de 90 retornos que termina en p."""
    past = idx[idx < t]; return past[-90]
C_EST, C_SENT, C_APP, C_WARM = "#9ecae9", "#e07b39", "#4f4f4f", "#ececec"
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"], "font.size": 7, "axes.edgecolor": "#999999", "axes.linewidth": 0.5})
fig, ax = plt.subplots(figsize=(3.5, 2.45), dpi=300)
ax.axvspan(idx[0], EVAL_START, facecolor=C_WARM, zorder=0)
H = 0.32
for i, t in enumerate(origins):
    y = 10 - i
    c = prev(t) + pd.Timedelta(hours=16)
    t_next = prev(origins[i + 1]) + pd.Timedelta(hours=16) if i + 1 < len(origins) else ev[-1]
    s = est_start(t)
    ax.barh(y, (prev(t) - s).days + 0.6, left=s, height=2 * H, color=C_EST, edgecolor="white", linewidth=0.3, zorder=2)
    ax.barh(y - H / 1.5, 7, left=c - pd.Timedelta(days=7), height=H * 0.8, color=C_SENT, edgecolor="none", zorder=3)
    ax.barh(y, (t_next - c) / pd.Timedelta(days=1), left=c, height=2 * H, color=C_APP, edgecolor="white",
            linewidth=0.3, hatch="///", zorder=2, alpha=0.92)
    ax.plot([c, c], [y - H - 0.05, y + H + 0.05], color="#222222", linewidth=0.7, zorder=4)
ax.text(idx[0] + (EVAL_START - idx[0]) / 2, 10.95, "calentamiento", ha="center", va="bottom", fontsize=6, color="#777777", style="italic")
ax.set_ylim(0.35, 11.6)
ax.set_yticks([10 - i for i in range(10)]); ax.set_yticklabels([f"t{i+1}" for i in range(10)], fontsize=6)
ax.set_ylabel("Rebalanceo", fontsize=7)
ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[2, 5, 8, 11]))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b-%y"))
ax.set_xlim(idx[0] - pd.Timedelta(days=5), ev[-1] + pd.Timedelta(days=5))
ax.tick_params(axis="x", labelsize=6)
for sp in ("top", "right"): ax.spines[sp].set_visible(False)
ax.grid(axis="x", color="#e8e8e8", linewidth=0.4, zorder=1); ax.set_axisbelow(True)
ax.legend(handles=[
    Patch(facecolor=C_EST, label="Markowitz (90 retornos)"),
    Patch(facecolor=C_SENT, label="Titulares [c − 7 d, c)"),
    Line2D([0], [0], color="#222222", lw=0.8, label="c: cierre de p, 16:00 ET"),
    Patch(facecolor=C_APP, hatch="///", label="Tenencia"),
    Patch(facecolor=C_WARM, label="Calentamiento"),
], loc="lower left", fontsize=5.4, frameon=True, framealpha=0.95, edgecolor="#cccccc",
   handlelength=1.2, borderpad=0.4, labelspacing=0.3)
fig.tight_layout(pad=0.3)
import os; os.makedirs("results", exist_ok=True)
fig.savefig("results/figura1.png", dpi=300, bbox_inches="tight")
print("ok")
