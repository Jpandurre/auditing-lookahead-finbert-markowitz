"""
Configuración centralizada del pipeline ETL.
Todas las constantes y parámetros del sistema se definen aquí.
"""

import os
from datetime import datetime, timedelta
from pathlib import Path


# ──────────────────────────────────────────────
# Rutas del proyecto
# ──────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
LOG_DIR = BASE_DIR / "logs"

# Crear directorios si no existen
OUTPUT_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)


# ──────────────────────────────────────────────
# Tickers por defecto
# ──────────────────────────────────────────────
DEFAULT_TICKERS = ["AAPL", "MSFT", "SPY", "GOOGL", "AMZN"]


# ──────────────────────────────────────────────
# Rango de fechas por defecto (último año)
# ──────────────────────────────────────────────
DEFAULT_END_DATE = datetime.now().strftime("%Y-%m-%d")
DEFAULT_START_DATE = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")


# ──────────────────────────────────────────────
# Parámetros de transformación
# ──────────────────────────────────────────────
MOVING_AVERAGE_WINDOWS = [20, 50]
VOLATILITY_WINDOW = 20


# ──────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────
LOG_FORMAT = "%(asctime)s | %(name)-25s | %(levelname)-8s | %(message)s"
LOG_LEVEL = "INFO"
LOG_FILE = LOG_DIR / "etl_pipeline.log"


# ──────────────────────────────────────────────
# Configuración de APIs
# ──────────────────────────────────────────────
NEWS_API_KEY = os.environ.get("NEWS_API_KEY", "")
ALPHA_VANTAGE_KEY = os.environ.get("ALPHA_VANTAGE_KEY", "")
FINNHUB_API_KEY = os.environ.get("FINNHUB_API_KEY", "")


# ──────────────────────────────────────────────
# FinBERT / Análisis de Sentimiento
# ──────────────────────────────────────────────
FINBERT_MODEL_NAME = "ProsusAI/finbert"
SENTIMENT_BATCH_SIZE = 16
USE_SENTIMENT_ANALYSIS = os.environ.get("USE_SENTIMENT", "true").lower() in ("true", "1", "yes")


# ──────────────────────────────────────────────
# Tasa libre de riesgo (Sharpe / Sortino / objetivo SLSQP)
# ──────────────────────────────────────────────
# Referencia: rendimiento del T-bill de EE.UU. a 13 semanas
# (Yahoo Finance ^IRX; equivalente a la serie DTB3 de FRED).
#
# Promedios observados del período de evaluación del paper
# (1-jun-2025 -> 16-mar-2026, n = 198 ruedas):
#     media 3,85 %   |   mín 3,51 %   |   máx 4,26 %
#
# Se adopta 4,00 % como valor único de referencia para todo el período:
#   - Redondeo conservador: sobreestima rf en 15 bps, lo que REDUCE todos
#     los Sharpe/Sortino por igual (≈ -0,011 con volatilidad de 13 %).
#   - Al ser un desplazamiento constante del numerador, no altera el orden
#     relativo entre estrategias ni el signo de los diferenciales.
#   - Una tasa constante evita introducir una segunda serie temporal en el
#     backtest, manteniendo la comparación entre estrategias limpia.
RISK_FREE_RATE = float(os.environ.get("RISK_FREE_RATE", "0.04"))

# Días de negociación por año usados para anualizar retorno y volatilidad.
TRADING_DAYS_PER_YEAR = int(os.environ.get("TRADING_DAYS_PER_YEAR", "252"))


# ──────────────────────────────────────────────
# Base de datos PostgreSQL
# ──────────────────────────────────────────────
DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://localhost:5432/investment_platform",
)
USE_DATABASE = os.environ.get("USE_DATABASE", "true").lower() in ("true", "1", "yes")
DB_BATCH_SIZE = int(os.environ.get("DB_BATCH_SIZE", "1000"))


# ──────────────────────────────────────────────
# Archivos de salida
# ──────────────────────────────────────────────
PRICE_OUTPUT_FILE = OUTPUT_DIR / "historical_prices.csv"
NEWS_OUTPUT_FILE = OUTPUT_DIR / "financial_news.csv"
TRANSFORMED_OUTPUT_FILE = OUTPUT_DIR / "transformed_data.csv"
ALIGNED_OUTPUT_FILE = OUTPUT_DIR / "aligned_data.csv"
