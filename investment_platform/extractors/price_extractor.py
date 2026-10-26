"""
Extractor de datos históricos de precios usando yfinance.
Descarga datos OHLCV para una lista de tickers en un rango de fechas.
"""

import logging
from typing import List

import pandas as pd
import yfinance as yf

from .base import BaseExtractor


class PriceExtractor(BaseExtractor):
    """
    Extrae datos históricos de precios del mercado usando yfinance.
    
    Retorna un DataFrame con formato long (un registro por ticker/fecha)
    con columnas: date, ticker, open, high, low, close, volume, adj_close.
    
    Ejemplo:
        >>> extractor = PriceExtractor(["AAPL", "MSFT"], "2024-01-01", "2025-01-01")
        >>> df = extractor.extract()
        >>> print(df.head())
    """

    def __init__(self, tickers: List[str], start_date: str, end_date: str):
        """Inicializa el extractor de precios diarios.

        Args:
            tickers: símbolos a descargar (p. ej. ['AAPL', 'MSFT']).
            start_date: fecha inicial 'YYYY-MM-DD' (incluida).
            end_date: fecha final 'YYYY-MM-DD' (excluida, convención de yfinance).
        """
        super().__init__(tickers, start_date, end_date)
        self.logger = logging.getLogger("PriceExtractor")

    def extract(self) -> pd.DataFrame:
        """
        Descarga datos históricos de precios para todos los tickers configurados.
        
        Returns:
            pd.DataFrame con columnas:
                - date (datetime): Fecha de la cotización
                - ticker (str): Símbolo bursátil
                - open (float): Precio de apertura
                - high (float): Precio máximo del día
                - low (float): Precio mínimo del día
                - close (float): Precio de cierre
                - volume (int): Volumen de operaciones
                - adj_close (float): Precio de cierre ajustado
        """
        self.logger.info(
            f"Descargando precios para {len(self.tickers)} tickers "
            f"[{self.start_date} -> {self.end_date}]"
        )

        all_data = []

        for ticker in self.tickers:
            try:
                self.logger.info(f"  - Descargando {ticker}...")
                stock = yf.Ticker(ticker)
                hist = stock.history(start=self.start_date, end=self.end_date)

                if hist.empty:
                    self.logger.warning(f"  Sin datos para {ticker}")
                    continue

                # Normalizar columnas
                df = hist.reset_index()
                df.columns = [c.lower().replace(" ", "_") for c in df.columns]

                # Renombrar columnas para consistencia
                column_mapping = {
                    "date": "date",
                    "open": "open",
                    "high": "high",
                    "low": "low",
                    "close": "close",
                    "volume": "volume",
                }

                # Manejar adj_close (puede venir como 'adj close' o no existir)
                if "adj_close" in df.columns:
                    column_mapping["adj_close"] = "adj_close"
                elif "adj close" in df.columns:
                    df = df.rename(columns={"adj close": "adj_close"})
                    column_mapping["adj_close"] = "adj_close"
                else:
                    # Si no existe, usar close como adj_close
                    df["adj_close"] = df["close"]

                df["ticker"] = ticker

                # Seleccionar solo columnas relevantes
                relevant_cols = ["date", "ticker", "open", "high", "low", "close", "volume", "adj_close"]
                available_cols = [c for c in relevant_cols if c in df.columns]
                df = df[available_cols]

                # Asegurar tipo datetime
                df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)

                all_data.append(df)
                self.logger.info(f"  {ticker}: {len(df)} registros")

            except Exception as e:
                self.logger.error(f"  Error descargando {ticker}: {e}")
                continue

        if not all_data:
            self.logger.error("No se pudieron descargar datos de ningún ticker.")
            self._data = pd.DataFrame()
            return self._data

        self._data = pd.concat(all_data, ignore_index=True)
        self._data = self._data.sort_values(["ticker", "date"]).reset_index(drop=True)

        self.logger.info(
            f"Extracción completa: {len(self._data)} registros totales "
            f"para {self._data['ticker'].nunique()} tickers."
        )

        self.validate(self._data)
        return self._data

    def validate(self, df: pd.DataFrame) -> bool:
        """
        Valida los datos de precios extraídos.
        Verifica columnas esperadas y rangos de valores.
        """
        if not super().validate(df):
            return False

        expected_cols = {"date", "ticker", "open", "high", "low", "close", "volume"}
        missing = expected_cols - set(df.columns)
        if missing:
            self.logger.warning(f"Columnas faltantes: {missing}")
            return False

        # Verificar que no hay precios negativos
        price_cols = ["open", "high", "low", "close"]
        for col in price_cols:
            if col in df.columns and (df[col] < 0).any():
                self.logger.warning(f"Precios negativos encontrados en '{col}'")
                return False

        return True
