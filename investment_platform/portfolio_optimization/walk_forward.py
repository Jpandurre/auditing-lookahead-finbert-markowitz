"""
Motor de backtesting walk-forward (validación de origen móvil a nivel de portafolio).

En cada fecha de rebalanceo mensual, los pesos se recalculan usando solo
información anterior a la ejecución:
  - Markowitz: ventana móvil de trailing_days días de retornos.
  - Sentimiento IA: Markowitz + FinBERT sobre noticias de los últimos
    sentiment_lookback días antes del rebalanceo.
  - Equal Weight: 1/N constante.

La función `build_walk_forward_schedules()` devuelve tres schedules de pesos
(uno por estrategia) listos para pasarle a BacktestEngine.run_strategy_with_schedule().

Convención temporal: los pesos del rebalanceo t se aplican al retorno de t
(cierre de la sesión previa p -> cierre de t), así que la cartera se forma al
cierre de p. Con `regla_titulares="calendario"` (especificación original del
artículo) se usan noticias de [t - lookback, t), lo que admite titulares
publicados después del cierre de p. Con `regla_titulares="auditada"` se usan
noticias de [c - lookback, c), con c = p a las 16:00 ET, y se excluyen las
marcas sin hora del día p. La regla auditada es la especificación principal.
"""

import logging
from typing import Dict, List

import pandas as pd

from investment_platform.config import RISK_FREE_RATE

from .allocators import (
    MarkowitzAllocator,
    SentimentAllocator,
    prepare_sentiment_scores,
)
from .backtest import BacktestEngine


logger = logging.getLogger("WalkForward")


def build_walk_forward_schedules(
    returns_matrix: pd.DataFrame,
    news_df: pd.DataFrame,
    eval_start: str,
    markowitz_window: int = 90,
    sentiment_lookback: int = 7,
    alpha: float = 0.5,
    risk_free_rate: float = RISK_FREE_RATE,
    rebalance_frequency: str = "monthly",
    max_weight: float = 1.0,
    sentiment_fallback_days: int = None,
    regla_titulares: str = "calendario",
) -> Dict[str, Dict[pd.Timestamp, Dict[str, float]]]:
    """
    Genera tres schedules de pesos sin look-ahead bias.

    Para cada fecha de rebalanceo mensual t (≥ eval_start):
      - Markowitz: estima sobre returns[t-markowitz_window : t] (solo pasado).
      - Sentimiento IA: mismo Markowitz + scores FinBERT de noticias en
        [t-sentiment_lookback, t) (estrictamente previas a t).
      - Equal Weight: 1/N fijo.

    Args:
        returns_matrix: DataFrame de retornos diarios (índice=fechas, cols=tickers).
                        Puede incluir un período de warmup previo a eval_start.
        news_df: DataFrame de noticias con columnas ticker, date, sentiment_score.
        eval_start: Primera fecha de evaluación 'YYYY-MM-DD'. El primer rebalanceo
                    se agenda a/después de esta fecha.
        markowitz_window: Días de trading de lookback para estimar covarianza.
        sentiment_lookback: Días calendario de noticias previas al rebalanceo.
        alpha: Intensidad del ajuste por sentimiento [0, 1].
        risk_free_rate: Tasa libre de riesgo anual para Markowitz.
        rebalance_frequency: Frecuencia de rebalanceo: "monthly" | "quarterly" | "weekly".
        max_weight: Cota superior de peso por activo (1.0 = sin restricción).
            Valores < 1 evitan soluciones de esquina que dejan inerte la señal.
        sentiment_fallback_days: Si la ventana de sentiment_lookback días no
            contiene noticias, reintentar con esta ventana más amplia (siempre
            estrictamente previa al corte -> sin fuga). None desactiva el fallback.
        regla_titulares: "calendario" (ventana [t - lookback, t)) o "auditada"
            (ventana [c - lookback, c) con c = cierre de la sesión previa, 16:00 ET,
            excluyendo marcas sin hora de esa sesión). Ver docstring del módulo.

    Returns:
        Dict con claves "Markowitz", "Sentimiento IA", "Equal Weight", cada una
        mapeando {pd.Timestamp -> {ticker: peso}}.
    """
    eval_start_ts = pd.Timestamp(eval_start)
    eval_index = returns_matrix.index[returns_matrix.index >= eval_start_ts]

    if eval_index.empty:
        raise ValueError(f"Sin datos de retornos desde {eval_start}.")

    rebalance_dates = BacktestEngine.generate_rebalance_dates(eval_index, frequency=rebalance_frequency)
    tickers = list(returns_matrix.columns)
    n = len(tickers)

    markowitz_alloc = MarkowitzAllocator(
        risk_free_rate=risk_free_rate, max_weight=max_weight
    )
    sentiment_alloc = SentimentAllocator(
        alpha=alpha, risk_free_rate=risk_free_rate, max_weight=max_weight
    )

    schedules: Dict[str, Dict[pd.Timestamp, Dict[str, float]]] = {
        "Markowitz": {},
        "Sentimiento IA": {},
        "Equal Weight": {},
    }

    logger.info(
        f"Walk-forward: {len(rebalance_dates)} rebalanceos ({rebalance_frequency}) desde {eval_start} "
        f"| ventana Markowitz={markowitz_window}d | lookback sentimiento={sentiment_lookback}d"
    )

    for t in rebalance_dates:
        # ── Ventana de retornos para Markowitz ──────────────────────────
        window_ret = returns_matrix[returns_matrix.index < t].tail(markowitz_window)

        if len(window_ret) < 10:
            logger.warning(
                f"  {t.date()}: solo {len(window_ret)} días de retornos disponibles "
                f"(necesita ≥10). Usando equal weight como fallback."
            )
            ew = {tk: round(1.0 / n, 6) for tk in tickers}
            schedules["Markowitz"][t] = ew
            schedules["Sentimiento IA"][t] = ew
            schedules["Equal Weight"][t] = ew
            continue

        # ── Markowitz ───────────────────────────────────────────────────
        mw_weights = markowitz_alloc.optimize(window_ret)
        schedules["Markowitz"][t] = mw_weights
        logger.info(f"  {t.date()} Markowitz: {mw_weights}")

        # ── Equal Weight ────────────────────────────────────────────────
        ew_weights = {tk: round(1.0 / n, 6) for tk in tickers}
        schedules["Equal Weight"][t] = ew_weights

        # ── Sentimiento IA ──────────────────────────────────────────────
        sent_scores: Dict[str, float] = {}
        if regla_titulares == "auditada":
            p = returns_matrix.index[returns_matrix.index < t][-1]   # sesión de ejecución
            corte, sin_hora = p + pd.Timedelta(hours=16), p
        elif regla_titulares == "calendario":
            corte, sin_hora = t, None
        else:
            raise ValueError(f"regla_titulares desconocida: {regla_titulares}")
        if news_df is not None and not news_df.empty and "sentiment_score" in news_df.columns:
            sent_scores = prepare_sentiment_scores(
                news_df,
                lookback_days=sentiment_lookback,
                as_of_date=corte,
                fallback_lookback_days=sentiment_fallback_days,
                excluir_sin_hora_de=sin_hora,
            )

        sent_weights = sentiment_alloc.optimize(window_ret, sent_scores)
        schedules["Sentimiento IA"][t] = sent_weights
        logger.info(f"  {t.date()} Sentimiento IA: {sent_weights}")

    logger.info(
        f"Schedules walk-forward construidos: "
        f"Markowitz={len(schedules['Markowitz'])} puntos | "
        f"Sentimiento IA={len(schedules['Sentimiento IA'])} puntos"
    )
    return schedules
