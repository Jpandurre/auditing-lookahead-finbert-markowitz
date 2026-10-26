"""Motor de optimización: asignadores (Markowitz, Sentimiento IA), backtest walk-forward e inferencia."""
from .allocators import MarkowitzAllocator, SentimentAllocator
from .backtest import BacktestEngine
from .walk_forward import build_walk_forward_schedules

__all__ = ["MarkowitzAllocator", "SentimentAllocator", "BacktestEngine", "build_walk_forward_schedules"]
