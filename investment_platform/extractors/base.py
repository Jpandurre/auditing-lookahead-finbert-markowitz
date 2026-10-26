"""
Clase base abstracta para todos los extractores de datos.
Garantiza una interfaz uniforme en el pipeline ETL.
"""

from abc import ABC, abstractmethod
from typing import List, Optional
import logging

import pandas as pd


class BaseExtractor(ABC):
    """
    Extractor base que define la interfaz para todos los módulos de extracción.
    
    Todos los extractores deben implementar:
        - extract(): Ejecuta la extracción y retorna un DataFrame.
        - validate(): Valida los datos extraídos.
    """

    def __init__(self, tickers: List[str], start_date: str, end_date: str):
        """
        Args:
            tickers: Lista de símbolos bursátiles (ej. ['AAPL', 'MSFT']).
            start_date: Fecha de inicio en formato 'YYYY-MM-DD'.
            end_date: Fecha de fin en formato 'YYYY-MM-DD'.
        """
        self.tickers = [t.upper() for t in tickers]
        self.start_date = start_date
        self.end_date = end_date
        self.logger = logging.getLogger(self.__class__.__name__)
        self._data: Optional[pd.DataFrame] = None

    @abstractmethod
    def extract(self) -> pd.DataFrame:
        """
        Ejecuta la extracción de datos.
        
        Returns:
            pd.DataFrame con los datos extraídos.
        """
        pass

    def validate(self, df: pd.DataFrame) -> bool:
        """
        Validación básica de los datos extraídos.
        Puede ser sobreescrita por subclases para validaciones específicas.
        
        Args:
            df: DataFrame a validar.
            
        Returns:
            True si los datos son válidos.
        """
        if df is None or df.empty:
            self.logger.warning("DataFrame vacío o nulo.")
            return False

        self.logger.info(f"Validación OK — {len(df)} registros, {len(df.columns)} columnas.")
        return True

    @property
    def data(self) -> Optional[pd.DataFrame]:
        """Acceso al último DataFrame extraído."""
        return self._data

    def __repr__(self) -> str:
        """Representación legible: clase, tickers y rango de fechas."""
        return (
            f"{self.__class__.__name__}("
            f"tickers={self.tickers}, "
            f"start='{self.start_date}', "
            f"end='{self.end_date}')"
        )
