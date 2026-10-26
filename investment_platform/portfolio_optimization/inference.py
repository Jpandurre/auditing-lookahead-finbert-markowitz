"""
Inferencia estadística para métricas de backtest.

Implementa el bootstrap estacionario de Politis & Romano (1994) sobre retornos
diarios, con remuestreo PAREADO entre estrategias: todas se remuestrean con los
mismos índices, preservando la correlación contemporánea entre carteras, lo que
hace válidos los intervalos de confianza de DIFERENCIAS entre estrategias.

Uso típico:
    net = pd.DataFrame({...})                     # (n_días, k_estrategias)
    idx = stationary_bootstrap_indices(len(net), B=10_000, L=10, rng=rng)
    bm  = bootstrap_metrics(net.to_numpy(), idx)  # dict de (B, k) arrays
    lo, hi = percentile_ci(bm["annual_return"][:, 0])
"""

from typing import Dict, Tuple

import numpy as np
import pandas as pd

from investment_platform.config import RISK_FREE_RATE, TRADING_DAYS_PER_YEAR

TRADING_DAYS = TRADING_DAYS_PER_YEAR


def net_daily_returns(portfolio_values: pd.Series, initial_capital: float) -> pd.Series:
    """
    Retornos diarios NETOS derivados de la trayectoria de capital.

    A diferencia del retorno bruto ponderado, esta serie incorpora la merma por
    costos de transacción en los días de rebalanceo — es la serie correcta a
    remuestrear cuando el backtest cobra turnover.
    """
    r = portfolio_values.pct_change()
    r.iloc[0] = portfolio_values.iloc[0] / initial_capital - 1.0
    return r


def metrics_from_returns(
    r: np.ndarray, risk_free_rate: float = RISK_FREE_RATE, trading_days: int = TRADING_DAYS
) -> Dict[str, float]:
    """Métricas anualizadas a partir de una serie de retornos diarios."""
    n = len(r)
    total = float(np.prod(1.0 + r) - 1.0)
    annual = float((1.0 + total) ** (trading_days / n) - 1.0) if n > 1 else 0.0
    vol = float(np.std(r, ddof=1) * np.sqrt(trading_days)) if n > 1 else 0.0
    sharpe = (annual - risk_free_rate) / vol if vol > 0 else 0.0
    return {"total_return": total, "annual_return": annual,
            "annual_volatility": vol, "sharpe_ratio": sharpe}


def stationary_bootstrap_indices(
    n: int, B: int, L: float, rng: np.random.Generator
) -> np.ndarray:
    """
    Matriz (B, n) de índices del bootstrap estacionario (Politis & Romano, 1994).

    Bloques de longitud geométrica con media L y envolvimiento circular:
    preserva autocorrelación y agrupamiento de volatilidad que un bootstrap
    i.i.d. destruiría.
    """
    p = 1.0 / L
    starts = rng.integers(0, n, size=(B, n))
    cont = rng.random((B, n)) >= p          # True -> continuar el bloque
    idx = np.empty((B, n), dtype=np.int64)
    idx[:, 0] = starts[:, 0]
    for t in range(1, n):
        idx[:, t] = np.where(cont[:, t], (idx[:, t - 1] + 1) % n, starts[:, t])
    return idx


def bootstrap_metrics(
    returns_mat: np.ndarray,
    idx: np.ndarray,
    risk_free_rate: float = RISK_FREE_RATE,
    trading_days: int = TRADING_DAYS,
) -> Dict[str, np.ndarray]:
    """
    Métricas bootstrap (B, k) bajo remuestreo pareado.

    Args:
        returns_mat: (n, k) retornos diarios, una columna por estrategia.
        idx: (B, n) índices — los MISMOS para todas las columnas.
    """
    B, n = idx.shape
    boot = returns_mat[idx]                                  # (B, n, k)
    total = np.prod(1.0 + boot, axis=1) - 1.0                # (B, k)
    annual = (1.0 + total) ** (trading_days / n) - 1.0
    vol = boot.std(axis=1, ddof=1) * np.sqrt(trading_days)
    sharpe = np.where(vol > 0, (annual - risk_free_rate) / vol, 0.0)
    return {"total_return": total, "annual_return": annual,
            "annual_volatility": vol, "sharpe_ratio": sharpe}


def percentile_ci(x: np.ndarray, level: float = 0.95) -> Tuple[float, float]:
    """Intervalo percentil bilateral."""
    a = (1.0 - level) / 2.0
    return float(np.quantile(x, a)), float(np.quantile(x, 1.0 - a))


def bootstrap_pvalue(diff: np.ndarray) -> float:
    """p-valor bootstrap bilateral para H0: diferencia = 0."""
    p_le = float(np.mean(diff <= 0.0))
    p_ge = float(np.mean(diff >= 0.0))
    return float(min(1.0, 2.0 * min(p_le, p_ge)))
