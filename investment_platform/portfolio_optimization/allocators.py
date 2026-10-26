"""
Estrategias de asignación de portafolios.

Implementa dos enfoques:
    1. MarkowitzAllocator  — Media-Varianza clásica (Benchmark)
    2. SentimentAllocator  — Markowitz + ajuste por sentimiento FinBERT (IA)
"""

import logging
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from investment_platform.config import RISK_FREE_RATE, TRADING_DAYS_PER_YEAR


logger = logging.getLogger("PortfolioAllocators")


def _apply_max_weight_cap(weights: np.ndarray, cap: float) -> np.ndarray:
    """
    Proyecta un vector de pesos (suma 1) al simplex con cota superior `cap`.

    Redistribuye iterativamente el exceso de los activos que superan el cap
    entre los no saturados, proporcionalmente a su peso. Garantiza suma = 1
    y w_i ≤ cap (si cap·n ≥ 1; en caso contrario devuelve equal weight).
    """
    n = len(weights)
    if n == 0 or cap >= 1.0:
        return weights
    if cap * n < 1.0 - 1e-9:
        # Infeasible: ni repartiendo al máximo se llega a suma 1
        return np.full(n, 1.0 / n)

    w = np.clip(weights, 0.0, None)
    s = w.sum()
    w = w / s if s > 0 else np.full(n, 1.0 / n)

    for _ in range(100):
        over = w > cap + 1e-12
        if not over.any():
            break
        excess = float((w[over] - cap).sum())
        w[over] = cap
        under = ~over
        under_sum = float(w[under].sum())
        if under_sum > 1e-12:
            w[under] += excess * w[under] / under_sum
        else:
            w[under] += excess / max(int(under.sum()), 1)
    return w


class MarkowitzAllocator:
    """
    Optimización de portafolios por Media-Varianza (Markowitz).
    
    Maximiza el Ratio de Sharpe sujeto a:
        - Pesos entre 0 y 1 (long only)
        - Suma de pesos = 1
    
    Ejemplo:
        >>> allocator = MarkowitzAllocator(risk_free_rate=0.04)
        >>> weights = allocator.optimize(returns_df)
        >>> print(weights)  # {'AAPL': 0.45, 'MSFT': 0.35, 'SPY': 0.20}
    """

    def __init__(
        self,
        risk_free_rate: float = RISK_FREE_RATE,
        trading_days: int = TRADING_DAYS_PER_YEAR,
        max_weight: float = 1.0,
    ):
        """
        Args:
            risk_free_rate: Tasa libre de riesgo anual. Default: config.RISK_FREE_RATE
                (4 %, T-bill 13 semanas; ver justificación en config.py).
            trading_days: Días de trading por año para anualización.
            max_weight: Cota superior de peso por activo [1/n, 1]. Con 1.0
                (default) se recupera el comportamiento clásico sin restricción
                de concentración. Valores < 1 evitan soluciones de esquina
                (100% en un solo activo) frecuentes en universos pequeños y
                correlacionados.
        """
        self.risk_free_rate = risk_free_rate
        self.trading_days = trading_days
        self.max_weight = max_weight

    def optimize(self, returns_df: pd.DataFrame) -> Dict[str, float]:
        """
        Calcula pesos óptimos que maximizan el Sharpe Ratio.
        
        Args:
            returns_df: DataFrame con retornos diarios.
                Columnas = tickers, filas = fechas, valores = retornos.
                
        Returns:
            Dict {ticker: peso} con pesos óptimos (suman 1.0).
        """
        tickers = list(returns_df.columns)
        n_assets = len(tickers)

        if n_assets == 0:
            logger.warning("Sin activos para optimizar.")
            return {}

        if n_assets == 1:
            logger.info(f"  - Un solo activo ({tickers[0]}), asignando 100%.")
            return {tickers[0]: 1.0}

        # Estadísticas anualizadas
        mean_returns = returns_df.mean() * self.trading_days
        cov_matrix = returns_df.cov() * self.trading_days

        # Función objetivo: negativo del Sharpe Ratio (minimizamos)
        def neg_sharpe(weights):
            """Objetivo de SLSQP: Sharpe anualizado con signo negativo (se minimiza)."""
            port_return = np.dot(weights, mean_returns)
            port_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            if port_volatility == 0:
                return 0
            return -(port_return - self.risk_free_rate) / port_volatility

        # Restricciones y límites (cap efectivo ≥ 1/n para que sea factible)
        cap = max(self.max_weight, 1.0 / n_assets)
        constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
        bounds = tuple((0.0, cap) for _ in range(n_assets))
        initial_weights = np.array([1.0 / n_assets] * n_assets)

        # Optimización
        result = minimize(
            neg_sharpe,
            initial_weights,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 1000, "ftol": 1e-12},
        )

        if not result.success:
            logger.warning(f"Optimización no convergió: {result.message}")
            # Fallback: pesos iguales
            weights = {t: 1.0 / n_assets for t in tickers}
        else:
            # Limpiar pesos muy pequeños (< 0.1%)
            raw_weights = result.x
            raw_weights[raw_weights < 0.001] = 0
            raw_weights = raw_weights / raw_weights.sum()  # Renormalizar
            # La renormalización puede exceder el cap -> re-proyectar
            raw_weights = _apply_max_weight_cap(raw_weights, cap)
            weights = {t: round(float(w), 6) for t, w in zip(tickers, raw_weights)}

        # Métricas del portafolio óptimo
        w_arr = np.array(list(weights.values()))
        port_return = float(np.dot(w_arr, mean_returns))
        port_vol = float(np.sqrt(np.dot(w_arr.T, np.dot(cov_matrix, w_arr))))
        sharpe = (port_return - self.risk_free_rate) / port_vol if port_vol > 0 else 0

        logger.info(f"  Markowitz — Sharpe: {sharpe:.4f}")
        logger.info(f"     Retorno anual: {port_return:.2%} | Volatilidad: {port_vol:.2%}")
        logger.info(f"     Pesos: {weights}")

        return weights

    def get_portfolio_stats(
        self, weights: Dict[str, float], returns_df: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Calcula estadísticas del portafolio con los pesos dados.
        
        Returns:
            Dict con: annual_return, annual_volatility, sharpe_ratio.
        """
        tickers = list(weights.keys())
        w = np.array([weights[t] for t in tickers])
        ret = returns_df[tickers]

        mean_ret = ret.mean() * self.trading_days
        cov = ret.cov() * self.trading_days

        port_return = float(np.dot(w, mean_ret))
        port_vol = float(np.sqrt(np.dot(w.T, np.dot(cov, w))))
        sharpe = (port_return - self.risk_free_rate) / port_vol if port_vol > 0 else 0

        return {
            "annual_return": port_return,
            "annual_volatility": port_vol,
            "sharpe_ratio": sharpe,
        }


class SentimentAllocator:
    """
    Optimización de portafolios con ajuste por sentimiento (IA Heurística).

    Flujo:
        1. Calcula pesos base usando Markowitz
        2. Normaliza los scores de sentimiento a [-1, +1] relativos entre tickers
        3. Ajusta pesos: w_adj = w_base * (1 + α * score_norm)
        4. Renormaliza pesos a suma = 1.0

    Ejemplo:
        >>> allocator = SentimentAllocator(alpha=0.5)
        >>> weights = allocator.optimize(returns_df, sentiment_scores)
        >>> print(weights)  # {'AAPL': 0.52, 'MSFT': 0.28, 'SPY': 0.20}
    """

    def __init__(
        self,
        alpha: float = 0.5,
        risk_free_rate: float = RISK_FREE_RATE,
        max_weight: float = 1.0,
    ):
        """
        Args:
            alpha: Intensidad del ajuste por sentimiento [0, 1].
                   0 = ignora sentimiento (puro Markowitz).
                   1 = máxima influencia del sentimiento.
            risk_free_rate: Tasa libre de riesgo anual.
            max_weight: Cota superior de peso por activo (ver MarkowitzAllocator).
                Se aplica tanto a los pesos base como a los ajustados.
        """
        self.alpha = alpha
        self.max_weight = max_weight
        self.markowitz = MarkowitzAllocator(
            risk_free_rate=risk_free_rate, max_weight=max_weight
        )

    def optimize(
        self,
        returns_df: pd.DataFrame,
        sentiment_scores: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Calcula pesos ajustados por sentimiento.
        
        Args:
            returns_df: DataFrame de retornos diarios (columnas = tickers).
            sentiment_scores: Dict {ticker: avg_sentiment_score} con scores
                             en rango [-1, +1].
        
        Returns:
            Dict {ticker: peso_ajustado} (suman 1.0).
        """
        tickers = list(returns_df.columns)
        n_assets = len(tickers)

        if n_assets == 0:
            return {}

        # Paso 1: Pesos base Markowitz
        logger.info("  Calculando pesos base (Markowitz)...")
        base_weights = self.markowitz.optimize(returns_df)

        if n_assets == 1:
            return base_weights

        # Paso 1.5: Normalización relativa de scores entre tickers con cobertura real.
        # El promedio de cientos de artículos FinBERT converge a ~0.05 para todos los
        # tickers, haciendo el ajuste trivial. Mapear a [-1,+1] según posición relativa
        # amplifica el diferencial efectivo sin cambiar la dirección de la señal.
        # Tickers sin noticias (score por defecto 0.0) quedan excluidos de la normalización
        # y mantienen sus pesos Markowitz sin ajuste.
        covered = [t for t in tickers if t in sentiment_scores]
        uncovered = [t for t in tickers if t not in sentiment_scores]

        relative_scores: Dict[str, float] = {}
        if len(covered) >= 2:
            raw_vals = [sentiment_scores[t] for t in covered]
            s_min, s_max = min(raw_vals), max(raw_vals)
            score_range = s_max - s_min
            if score_range > 1e-4:
                relative_scores = {
                    t: float(2.0 * (sentiment_scores[t] - s_min) / score_range - 1.0)
                    for t in covered
                }
                logger.info(
                    f"  Scores normalizados: rango bruto [{s_min:.3f}, {s_max:.3f}]"
                    f" -> relativo [-1, +1]  ({len(covered)} con noticias,"
                    f" {len(uncovered)} sin cobertura)"
                )
            else:
                relative_scores = {t: 0.0 for t in covered}
                logger.info("  Scores demasiado similares entre tickers -- sin ajuste relativo")
        elif len(covered) == 1:
            relative_scores = {covered[0]: 0.0}
        # Tickers sin cobertura: score relativo 0 -> sin modificacion de peso
        for t in uncovered:
            relative_scores[t] = 0.0

        # Paso 2: Ajuste por sentimiento — w_adj = w_base * (1 + α * score_norm)
        logger.info(f"  Ajustando por sentimiento (alpha={self.alpha})...")
        adjusted = {}

        for ticker in tickers:
            w_base = base_weights.get(ticker, 1.0 / n_assets)
            score = relative_scores.get(ticker, 0.0)
            w_adjusted = max(0.0, w_base * (1.0 + self.alpha * score))
            adjusted[ticker] = w_adjusted

            logger.info(
                f"    {ticker}: base={w_base:.4f} x"
                f" (1 + {self.alpha}x{score:.3f})"
                f" = {w_adjusted:.4f}"
            )

        # Paso 3: Renormalizar a suma = 1.0 (y re-proyectar al cap si aplica)
        total = sum(adjusted.values())
        if total > 0:
            w_arr = np.array([adjusted[t] / total for t in tickers])
            cap = max(self.max_weight, 1.0 / n_assets)
            w_arr = _apply_max_weight_cap(w_arr, cap)
            adjusted = {t: round(float(w), 6) for t, w in zip(tickers, w_arr)}
        else:
            # Fallback: pesos iguales
            adjusted = {t: 1.0 / n_assets for t in tickers}

        logger.info(f"  Sentiment — Pesos finales: {adjusted}")

        return adjusted


def prepare_returns_matrix(
    price_df: pd.DataFrame,
    tickers: Optional[list] = None,
) -> pd.DataFrame:
    """
    Prepara la matriz de retornos diarios desde el DataFrame de precios.
    
    Convierte formato long (date, ticker, close, daily_return) a formato wide
    (columna por ticker) requerido por los allocators.
    
    Args:
        price_df: DataFrame en formato long con 'date', 'ticker', 'daily_return'.
        tickers: Lista de tickers a incluir (None = todos).
        
    Returns:
        pd.DataFrame con fechas como índice y tickers como columnas.
    """
    df = price_df.copy()
    df["date"] = pd.to_datetime(df["date"])

    if tickers:
        df = df[df["ticker"].isin(tickers)]

    # Pivotear: filas=fechas, columnas=tickers, valores=retornos diarios
    returns = df.pivot_table(
        index="date",
        columns="ticker",
        values="daily_return",
    )

    # Eliminar first row (NaN por primer día)
    returns = returns.dropna()

    return returns


def prepare_sentiment_scores(
    news_df: pd.DataFrame,
    lookback_days: int = 30,
    as_of_date: Optional[pd.Timestamp] = None,
    fallback_lookback_days: Optional[int] = None,
    excluir_sin_hora_de: Optional[pd.Timestamp] = None,
) -> Dict[str, float]:
    """
    Calcula el sentiment_score promedio reciente por ticker.

    Args:
        news_df: DataFrame de noticias con 'ticker', 'date', 'sentiment_score'.
        lookback_days: Número de días recientes a considerar.
        as_of_date: Si se provee, solo usa noticias ESTRICTAMENTE ANTERIORES a esta
                    fecha (elimina look-ahead bias en walk-forward). La ventana es
                    [as_of_date - lookback_days, as_of_date). Si None, usa
                    comportamiento original (ventana relativa al máximo del corpus).
        fallback_lookback_days: Solo con as_of_date. Si la ventana primaria no
                    contiene ninguna noticia (huecos del corpus), se reintenta con
                    esta ventana más amplia — SIEMPRE estrictamente anterior a
                    as_of_date, por lo que no introduce fuga de información. None
                    (default) desactiva el fallback (comportamiento previo:
                    sin noticias -> pesos Markowitz puros).
        excluir_sin_hora_de: Fecha (sin hora) cuyas marcas exactamente a las 00:00:00
                    se descartan por tener hora desconocida. La regla auditada pasa
                    aquí la sesión de ejecución p, porque esas noticias pudieron
                    publicarse después del cierre. None no excluye nada.

    Returns:
        Dict {ticker: avg_sentiment_score}. Tickers sin noticias en la ventana
        se omiten del dict -> SentimentAllocator mantiene sus pesos Markowitz puros.
    """
    df = news_df.copy()
    df["date"] = pd.to_datetime(df["date"])

    if as_of_date is not None:
        as_of = pd.Timestamp(as_of_date)
        cutoff = as_of - pd.Timedelta(days=lookback_days)
        recent = df[(df["date"] >= cutoff) & (df["date"] < as_of)]
        if excluir_sin_hora_de is not None:
            recent = recent[recent["date"] != pd.Timestamp(excluir_sin_hora_de).normalize()]
        label = f"ventana [{cutoff} -> {as_of})"

        if recent.empty and fallback_lookback_days is not None \
                and fallback_lookback_days > lookback_days:
            cutoff_fb = as_of - pd.Timedelta(days=fallback_lookback_days)
            recent = df[(df["date"] >= cutoff_fb) & (df["date"] < as_of)]
            label = (f"ventana fallback [{cutoff_fb.date()} -> {as_of.date()}) "
                     f"(primaria de {lookback_days}d vacía)")
    else:
        cutoff = df["date"].max() - pd.Timedelta(days=lookback_days)
        recent = df[df["date"] >= cutoff]
        label = f"últimos {lookback_days} días"

    if recent.empty:
        if as_of_date is None:
            recent = df
        # Con as_of_date: devolver dict vacío (sin noticias -> pesos Markowitz puros)

    scores = recent.groupby("ticker")["sentiment_score"].mean().to_dict()

    logger.info(f"  Sentiment scores ({label}): {scores}")
    return scores
