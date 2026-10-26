"""
Motor de backtesting para comparar estrategias de portafolio.

Simula una inversión con rebalanceo periódico y calcula métricas
de rendimiento para comparación cuantitativa entre estrategias.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from investment_platform.config import RISK_FREE_RATE, TRADING_DAYS_PER_YEAR


logger = logging.getLogger("BacktestEngine")


@dataclass
class StrategyResult:
    """Resultado de una estrategia de backtesting."""
    name: str
    weights: Dict[str, float]
    portfolio_values: pd.Series = field(default_factory=pd.Series)
    daily_returns: pd.Series = field(default_factory=pd.Series)

    # Métricas calculadas
    total_return: float = 0.0
    annual_return: float = 0.0
    annual_volatility: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    cvar_95: float = 0.0
    max_drawdown: float = 0.0
    final_value: float = 0.0

    # Costo de transacción descontado al establecer posiciones (USD).
    # None -> backtest sin costos (compatibilidad con llamadores antiguos).
    transaction_cost_paid: Optional[float] = None


class BacktestEngine:
    """
    Motor de backtesting para simular y comparar estrategias de portafolio.
    
    Simula inversión con capital inicial, aplica asignaciones de pesos
    fijos y calcula métricas de rendimiento histórico.
    
    Ejemplo:
        >>> engine = BacktestEngine(initial_capital=10000)
        >>> result = engine.run_strategy("Markowitz", weights, returns_df)
        >>> print(f"Return: {result.total_return:.2%}")
    """

    def __init__(
        self,
        initial_capital: float = 10_000.0,
        risk_free_rate: float = RISK_FREE_RATE,
        trading_days: int = TRADING_DAYS_PER_YEAR,
        transaction_cost_bps: float = 10.0,
    ):
        """
        Args:
            initial_capital: Capital inicial de inversión (USD).
            risk_free_rate: Tasa libre de riesgo anual. Default: config.RISK_FREE_RATE
                (4 %, T-bill 13 semanas; ver justificación en config.py).
            trading_days: Días de trading por año.
            transaction_cost_bps: Costo de transacción en basis points
                (1 bp = 0.01%). Aplicado al establecer las posiciones
                iniciales sobre el turnover (suma de pesos = 1). Default 10
                (0.10%) refleja una operativa retail estándar. Usar 0 para
                desactivar costos.
        """
        self.initial_capital = initial_capital
        self.risk_free_rate = risk_free_rate
        self.trading_days = trading_days
        self.transaction_cost_bps = transaction_cost_bps

    def run_strategy(
        self,
        name: str,
        weights: Dict[str, float],
        returns_df: pd.DataFrame,
    ) -> StrategyResult:
        """
        Ejecuta backtesting de una estrategia con pesos fijos.
        
        Args:
            name: Nombre de la estrategia.
            weights: Dict {ticker: peso}.
            returns_df: DataFrame de retornos diarios (columnas = tickers).
            
        Returns:
            StrategyResult con métricas y series temporales.
        """
        tickers = list(weights.keys())
        available = [t for t in tickers if t in returns_df.columns]

        if not available:
            logger.error(f"Ningún ticker disponible para {name}")
            return StrategyResult(name=name, weights=weights)

        # Renormalizar pesos con tickers disponibles
        w = np.array([weights.get(t, 0) for t in available])
        if w.sum() > 0:
            w = w / w.sum()

        ret = returns_df[available].copy()

        # Retorno diario del portafolio (promedio ponderado)
        port_daily_returns = ret.dot(w)

        # Costo de transacción al establecer posiciones (turnover día 1 = suma(weights) = 1.0)
        cost_rate = self.transaction_cost_bps / 10_000.0
        turnover = float(w.sum())
        cost_paid = self.initial_capital * turnover * cost_rate
        effective_capital = self.initial_capital - cost_paid

        # Valor del portafolio (crecimiento acumulativo sobre capital efectivo post-costo)
        cumulative = (1 + port_daily_returns).cumprod()
        portfolio_values = cumulative * effective_capital

        # Métricas
        result = StrategyResult(
            name=name,
            weights=dict(zip(available, [round(float(x), 6) for x in w])),
            portfolio_values=portfolio_values,
            daily_returns=port_daily_returns,
            transaction_cost_paid=round(cost_paid, 2),
        )

        # Valor final (ya descuenta costo de transacción inicial)
        result.final_value = (
            float(portfolio_values.iloc[-1]) if len(portfolio_values) > 0
            else effective_capital
        )

        # Retorno total NETO (desde el capital inicial del inversor, post-costo)
        result.total_return = (
            float(result.final_value / self.initial_capital - 1)
            if self.initial_capital > 0 else 0.0
        )

        # Retorno anualizado (derivado del retorno neto)
        n_days = len(port_daily_returns)
        if n_days > 1:
            result.annual_return = float(
                (1 + result.total_return) ** (self.trading_days / n_days) - 1
            )
        else:
            result.annual_return = 0.0

        # Volatilidad anualizada (independiente del costo)
        result.annual_volatility = float(
            port_daily_returns.std() * np.sqrt(self.trading_days)
        )

        # Sharpe Ratio
        if result.annual_volatility > 0:
            result.sharpe_ratio = float(
                (result.annual_return - self.risk_free_rate) / result.annual_volatility
            )
        else:
            result.sharpe_ratio = 0.0

        # Sortino Ratio (solo volatilidad de retornos negativos)
        neg_rets = port_daily_returns[port_daily_returns < 0]
        if len(neg_rets) > 1:
            downside_std = float(neg_rets.std() * np.sqrt(self.trading_days))
            result.sortino_ratio = float(
                (result.annual_return - self.risk_free_rate) / downside_std
            ) if downside_std > 0 else 0.0
        else:
            result.sortino_ratio = 0.0

        # CVaR 95% diario — pérdida esperada en el peor 5% de jornadas
        threshold = port_daily_returns.quantile(0.05)
        tail = port_daily_returns[port_daily_returns <= threshold]
        result.cvar_95 = float(tail.mean()) if len(tail) > 0 else 0.0

        # Max Drawdown (sobre la trayectoria del portafolio post-establecimiento)
        running_max = cumulative.cummax()
        drawdowns = cumulative / running_max - 1
        result.max_drawdown = float(drawdowns.min())

        return result

    @staticmethod
    def generate_rebalance_dates(
        index: pd.DatetimeIndex, frequency: str = "monthly"
    ) -> List[pd.Timestamp]:
        """
        Retorna el primer día de trading de cada mes presente en `index`.

        Args:
            index: DatetimeIndex del DataFrame de retornos.
            frequency: Solo "monthly" por ahora (extensible a "weekly"/"biweekly").

        Returns:
            Lista de Timestamps, uno por mes, en orden ascendente.
        """
        if frequency != "monthly":
            raise ValueError(f"Frecuencia no soportada: {frequency}. Usar 'monthly'.")

        rebalance_dates = []
        seen_months: set = set()
        for date in sorted(index):
            key = (date.year, date.month)
            if key not in seen_months:
                seen_months.add(key)
                rebalance_dates.append(date)
        return rebalance_dates

    def run_strategy_with_schedule(
        self,
        name: str,
        weight_schedule: Dict[pd.Timestamp, Dict[str, float]],
        returns_df: pd.DataFrame,
    ) -> StrategyResult:
        """
        Ejecuta backtest con rebalanceo dinámico y costos de transacción por turnover.

        Los pesos cambian en cada fecha del schedule (calculados SOLO con datos
        pasados en el motor walk-forward). En cada transición de bloque se cobra el
        turnover: `Σ|w_nuevo − w_viejo| × valor_portafolio × cost_rate`.

        Args:
            name: Nombre de la estrategia.
            weight_schedule: Dict {fecha_rebalanceo: {ticker: peso}}.
            returns_df: DataFrame de retornos diarios (eval window = índice completo).

        Returns:
            StrategyResult con métricas y series temporales.
        """
        if not weight_schedule:
            logger.error(f"Schedule vacío para {name}")
            return StrategyResult(name=name, weights={})

        sorted_dates = sorted(weight_schedule.keys())

        all_tickers: set = set()
        for w in weight_schedule.values():
            all_tickers.update(w.keys())
        available = [t for t in sorted(all_tickers) if t in returns_df.columns]

        if not available:
            logger.error(f"Ningún ticker disponible para {name}")
            return StrategyResult(name=name, weights={})

        def _normalize(w_dict: Dict[str, float]) -> np.ndarray:
            """Vector de pesos ordenado como `available`, reescalado a suma 1 (1/N si todos son 0)."""
            w = np.array([w_dict.get(t, 0.0) for t in available])
            s = w.sum()
            return w / s if s > 0 else np.full(len(available), 1.0 / len(available))

        cost_rate = self.transaction_cost_bps / 10_000.0
        schedule_idx = 0
        prev_w_arr = _normalize(weight_schedule[sorted_dates[0]])

        # Costo inicial (compra desde cash -> turnover = 1.0 normalizado)
        initial_cost = self.initial_capital * 1.0 * cost_rate
        portfolio_value = self.initial_capital - initial_cost
        total_tx_costs = initial_cost

        portfolio_values_list: List[float] = []
        daily_returns_list: List[float] = []

        # Pre-convertir retornos a NumPy: el bucle opera exclusivamente sobre arrays,
        # sin lookups Pandas en cada iteración.
        returns_matrix: np.ndarray = returns_df[available].to_numpy()   # shape (T, N)
        dates_list = list(returns_df.index)

        for i, date in enumerate(dates_list):
            # Avanzar al bloque de pesos vigente para esta fecha
            while (
                schedule_idx < len(sorted_dates) - 1
                and date >= sorted_dates[schedule_idx + 1]
            ):
                schedule_idx += 1

            current_w_arr = _normalize(weight_schedule[sorted_dates[schedule_idx]])

            # Si hay cambio de bloque (rebalanceo), cobrar turnover
            if not np.allclose(current_w_arr, prev_w_arr, atol=1e-6):
                turnover = float(np.sum(np.abs(current_w_arr - prev_w_arr)))
                rebalance_cost = portfolio_value * turnover * cost_rate
                portfolio_value -= rebalance_cost
                total_tx_costs += rebalance_cost

            # Retorno diario — producto punto sobre fila NumPy, sin .loc
            daily_ret = float(returns_matrix[i] @ current_w_arr)
            portfolio_value *= (1 + daily_ret)
            portfolio_values_list.append(portfolio_value)
            daily_returns_list.append(daily_ret)
            prev_w_arr = current_w_arr

        portfolio_values = pd.Series(portfolio_values_list, index=returns_df.index)
        port_daily_returns = pd.Series(daily_returns_list, index=returns_df.index)

        final_w_arr = _normalize(weight_schedule[sorted_dates[-1]])
        result = StrategyResult(
            name=name,
            weights=dict(zip(available, [round(float(x), 6) for x in final_w_arr])),
            portfolio_values=portfolio_values,
            daily_returns=port_daily_returns,
            transaction_cost_paid=round(total_tx_costs, 2),
        )
        result.final_value = float(portfolio_values.iloc[-1]) if len(portfolio_values) > 0 else 0.0
        result.total_return = (
            float(result.final_value / self.initial_capital - 1)
            if self.initial_capital > 0 else 0.0
        )
        n_days = len(port_daily_returns)
        result.annual_return = (
            float((1 + result.total_return) ** (self.trading_days / n_days) - 1)
            if n_days > 1 else 0.0
        )
        result.annual_volatility = float(
            port_daily_returns.std() * np.sqrt(self.trading_days)
        )
        if result.annual_volatility > 0:
            result.sharpe_ratio = float(
                (result.annual_return - self.risk_free_rate) / result.annual_volatility
            )

        # Sortino Ratio
        neg_rets = port_daily_returns[port_daily_returns < 0]
        if len(neg_rets) > 1:
            downside_std = float(neg_rets.std() * np.sqrt(self.trading_days))
            result.sortino_ratio = float(
                (result.annual_return - self.risk_free_rate) / downside_std
            ) if downside_std > 0 else 0.0

        # CVaR 95% diario
        threshold = port_daily_returns.quantile(0.05)
        tail = port_daily_returns[port_daily_returns <= threshold]
        result.cvar_95 = float(tail.mean()) if len(tail) > 0 else 0.0

        running_max = portfolio_values.cummax()
        result.max_drawdown = float((portfolio_values / running_max - 1).min())

        logger.info(
            f"  {name} (walk-forward, {len(sorted_dates)} rebalanceos): "
            f"retorno={result.total_return:.2%}  sharpe={result.sharpe_ratio:.4f}  "
            f"tx_costs=${total_tx_costs:.2f}"
        )
        return result

    def compare_strategies(
        self,
        strategies: List[StrategyResult],
    ) -> pd.DataFrame:
        """
        Genera tabla comparativa de métricas entre estrategias.
        
        Args:
            strategies: Lista de StrategyResult.
            
        Returns:
            DataFrame con métricas comparativas.
        """
        any_cost = any(
            (s.transaction_cost_paid or 0) > 0 for s in strategies
        )
        rows = []
        for s in strategies:
            row = {
                "Estrategia": s.name,
                "Capital Inicial": f"${self.initial_capital:,.2f}",
                "Capital Final": f"${s.final_value:,.2f}",
                "Retorno Total": f"{s.total_return:.2%}",
                "Retorno Anual": f"{s.annual_return:.2%}",
                "Volatilidad Anual": f"{s.annual_volatility:.2%}",
                "Sharpe Ratio": f"{s.sharpe_ratio:.4f}",
                "Sortino Ratio": f"{s.sortino_ratio:.4f}",
                "CVaR 95% (diario)": f"{s.cvar_95:.4%}",
                "Max Drawdown": f"{s.max_drawdown:.2%}",
            }
            if any_cost:
                cost = s.transaction_cost_paid if s.transaction_cost_paid is not None else 0.0
                row["Costo TX"] = f"${cost:,.2f}"
            rows.append(row)

        return pd.DataFrame(rows)

    def print_comparison(self, strategies: List[StrategyResult]):
        """Imprime comparación formateada de estrategias."""
        print("\n" + "=" * 80)
        print("COMPARACIÓN DE ESTRATEGIAS DE PORTAFOLIO")
        if self.transaction_cost_bps > 0:
            print(f"   (costos de transacción aplicados: {self.transaction_cost_bps:.1f} bps)")
        print("=" * 80)

        comparison = self.compare_strategies(strategies)
        print(comparison.to_string(index=False))

        print("\n" + "─" * 80)
        print("DETALLE DE ASIGNACIÓN DE PESOS")
        print("─" * 80)

        for s in strategies:
            print(f"\n  {s.name}:")
            for ticker, weight in sorted(s.weights.items()):
                bar = "█" * int(weight * 40)
                print(f"    {ticker:6s} {weight:7.2%}  {bar}")

        # ¿Quién ganó?
        if len(strategies) >= 2:
            print("\n" + "─" * 80)
            print("RESULTADO")
            print("─" * 80)

            best = max(strategies, key=lambda s: s.total_return)
            worst = min(strategies, key=lambda s: s.total_return)

            diff = best.total_return - worst.total_return
            diff_value = best.final_value - worst.final_value

            print(
                f"  {best.name} superó a {worst.name} por {diff:.2%} "
                f"(${diff_value:,.2f} USD)"
            )

            if best.sharpe_ratio > worst.sharpe_ratio:
                print(
                    f"  Mejor ratio riesgo/retorno: {best.name} "
                    f"(Sharpe: {best.sharpe_ratio:.4f} vs {worst.sharpe_ratio:.4f})"
                )

            # Diferencial específico Sentimiento IA vs Markowitz (para el paper)
            s_ai = next((s for s in strategies if s.name == "Sentimiento IA"), None)
            s_mw = next((s for s in strategies if s.name == "Markowitz"), None)
            if s_ai and s_mw:
                delta_ret = s_ai.total_return - s_mw.total_return
                delta_sharpe = s_ai.sharpe_ratio - s_mw.sharpe_ratio
                delta_val = s_ai.final_value - s_mw.final_value
                sign = "+" if delta_ret >= 0 else ""
                print(
                    f"\n  DIFERENCIAL PAPER (Sentimiento IA − Markowitz):"
                    f"\n     Retorno:  {sign}{delta_ret:.2%} ({sign}${delta_val:,.2f})"
                    f"\n     Sharpe:   {'+' if delta_sharpe>=0 else ''}{delta_sharpe:.4f}"
                )

        print("=" * 80)

    def get_daily_comparison(
        self, strategies: List[StrategyResult]
    ) -> pd.DataFrame:
        """
        Retorna DataFrame con valores diarios de cada estrategia.
        
        Útil para gráficos de evolución del portafolio.
        """
        frames = {}
        for s in strategies:
            frames[s.name] = s.portfolio_values
        
        return pd.DataFrame(frames)
