"""
Extractor de noticias financieras.
Soporta Alpha Vantage (fuente primaria, histórico real), Finnhub y datos mock.
"""

import logging
import os
import random
import time
from datetime import datetime, timedelta
from typing import List, Optional, Tuple

import pandas as pd
import requests

from .base import BaseExtractor
from .. import config

try:
    import finnhub
    FINNHUB_AVAILABLE = True
except ImportError:
    FINNHUB_AVAILABLE = False


class NewsExtractor(BaseExtractor):
    """
    Extrae titulares de noticias financieras para los tickers configurados.

    MODOS (en orden de prioridad):
        1. Alpha Vantage API — fuente primaria. Respeta time_from/time_to,
           cobertura histórica desde 2022. Requiere ALPHA_VANTAGE_KEY.
        2. Finnhub API — fallback. Solo retorna ~7 días recientes (free-tier);
           documentado como limitación del diseño experimental original.
        3. Mock — datos sintéticos para pruebas del pipeline.

    FinBERT es el motor de sentimiento en todos los casos; esta clase
    solo provee titulares + fechas + fuente. El análisis de sentimiento
    se realiza posteriormente por SentimentAnalyzer.

    Retorna un DataFrame con columnas:
        date, ticker, headline, summary, source, url, sentiment_hint

    Ejemplo:
        >>> extractor = NewsExtractor(["AAPL", "MSFT"], "2025-06-01", "2026-03-16")
        >>> df = extractor.extract()
        >>> print(df.head())
    """

    # Plantillas de titulares por categoría para datos mock
    HEADLINE_TEMPLATES = {
        "earnings": [
            "{ticker} Beats Q{quarter} Earnings Estimates with Record Revenue",
            "{ticker} Reports Mixed Results: Revenue Up but Margins Decline",
            "Wall Street Cuts Expectations for {ticker} Ahead of Quarterly Report",
            "{ticker} Announces ${amount}B Stock Buyback After Strong Earnings",
        ],
        "market": [
            "{ticker} Leads Tech Sector Gains in Bullish Trading Session",
            "Analysts Raise {ticker} Price Target to ${price} Per Share",
            "{ticker} Falls {pct}% on Antitrust Regulation Concerns",
            "Institutional Funds Increase {ticker} Holdings by {pct}%",
        ],
        "product": [
            "{ticker} Launches New Product Line with Integrated AI Technology",
            "{ticker} Announces Strategic Partnership to Expand Cloud Services",
            "{ticker} CEO Presents 5-Year Strategic Vision",
            "{ticker} Invests ${amount}B in Next-Generation Data Center",
        ],
        "macro": [
            "Interest Rate Impact: How It Affects {ticker} and the Tech Sector",
            "{ticker} Among Top Beneficiaries of New Fiscal Stimulus",
            "Persistent Inflation Pressures {ticker} and Competitor Valuations",
            "Geopolitical Tensions Generate Volatility in {ticker}",
        ],
    }

    SOURCES = [
        "Bloomberg", "Reuters", "CNBC", "MarketWatch",
        "Financial Times", "Yahoo Finance", "The Wall Street Journal",
        "Seeking Alpha", "Barron's", "Investing.com",
    ]

    # Alpha Vantage: tiempo mínimo entre llamadas para free-tier (~5 req/min)
    _AV_SLEEP_BETWEEN_TICKERS = 12  # segundos

    def __init__(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str,
        api_key: Optional[str] = None,
        alpha_vantage_key: Optional[str] = None,
        news_per_ticker_per_week: int = 3,
    ):
        """
        Args:
            tickers: Lista de símbolos bursátiles.
            start_date: Fecha inicio 'YYYY-MM-DD'.
            end_date: Fecha fin 'YYYY-MM-DD'.
            api_key: Finnhub API key (fallback si no hay Alpha Vantage key).
            alpha_vantage_key: Alpha Vantage API key (fuente primaria histórica).
            news_per_ticker_per_week: Noticias por ticker por semana en modo mock.
        """
        super().__init__(tickers, start_date, end_date)
        self.alpha_vantage_key = alpha_vantage_key or config.ALPHA_VANTAGE_KEY
        self.api_key = api_key or config.FINNHUB_API_KEY
        self.news_per_ticker_per_week = news_per_ticker_per_week
        self.logger = logging.getLogger("NewsExtractor")

    def extract(self) -> pd.DataFrame:
        """
        Extrae noticias financieras.
        Prioridad: Alpha Vantage (histórico) -> Finnhub (snapshot) -> Mock.

        Returns:
            pd.DataFrame con columnas:
                - date (datetime): Fecha de publicación
                - ticker (str): Símbolo bursátil relacionado
                - headline (str): Titular de la noticia
                - summary (str): Resumen de la noticia
                - source (str): Fuente de la noticia
                - url (str): URL de la noticia
                - sentiment_hint (str): Indicador de sentimiento (vacío, llenado por FinBERT)
        """
        if self.alpha_vantage_key:
            self.logger.info("Modo ALPHA VANTAGE activado — extrayendo noticias históricas reales")
            try:
                self._data = self._fetch_from_alpha_vantage()
                if not self._data.empty:
                    self.validate(self._data)
                    return self._data
                self.logger.warning("Alpha Vantage no retornó datos. Intentando Finnhub.")
            except Exception as e:
                self.logger.error(f"Error con Alpha Vantage: {e}. Intentando Finnhub.")

        if self.api_key and FINNHUB_AVAILABLE:
            self.logger.info("Modo FINNHUB activado (fallback) — solo últimos ~7 días")
            try:
                self._data = self._fetch_from_finnhub()
                if not self._data.empty:
                    self.validate(self._data)
                    return self._data
                self.logger.warning("Finnhub no retornó datos. Usando mock.")
            except Exception as e:
                self.logger.error(f"Error con Finnhub: {e}. Usando mock.")
        elif self.api_key and not FINNHUB_AVAILABLE:
            self.logger.warning(
                "finnhub-python no instalado. "
                "Ejecuta: pip install finnhub-python. Usando mock."
            )

        self.logger.info("Modo MOCK activado — generando noticias sintéticas")
        self._data = self._fetch_mock_news()
        self.validate(self._data)
        return self._data

    # ──────────────────────────────────────────────
    # Alpha Vantage — fuente primaria histórica
    # ──────────────────────────────────────────────

    @staticmethod
    def _chunk_date_range(
        start: datetime, end: datetime, months: int = 3
    ) -> List[Tuple[datetime, datetime]]:
        """Parte [start, end] en sub-rangos de `months` meses contiguos."""
        chunks = []
        chunk_start = start
        while chunk_start < end:
            # avanzar N meses desde chunk_start
            month = chunk_start.month - 1 + months
            year = chunk_start.year + month // 12
            month = month % 12 + 1
            chunk_end = min(datetime(year, month, 1) - timedelta(days=1), end)
            chunks.append((chunk_start, chunk_end))
            chunk_start = chunk_end + timedelta(days=1)
        return chunks

    def _fetch_av_chunk(
        self, ticker: str, chunk_start: datetime, chunk_end: datetime
    ) -> List[dict]:
        """Ejecuta UNA llamada AV NEWS_SENTIMENT para (ticker, chunk).

        Returns:
            Lista de dicts con keys: date, ticker, headline, summary, source, url, sentiment_hint.
            Lista vacía si AV reporta rate-limit (la llamada debe reintentarse otro día).

        Raises:
            RuntimeError: si AV devuelve un mensaje de rate limit diario — señal para
                          que el bucle externo aborte la extracción del día.
        """
        base_url = "https://www.alphavantage.co/query"
        params = {
            "function": "NEWS_SENTIMENT",
            "tickers": ticker,
            "time_from": chunk_start.strftime("%Y%m%dT%H%M"),
            "time_to": chunk_end.strftime("%Y%m%dT2359"),
            "limit": 1000,
            "sort": "EARLIEST",
            "apikey": self.alpha_vantage_key,
        }

        response = requests.get(base_url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()

        if "Information" in data or "Note" in data:
            msg = data.get("Information") or data.get("Note", "")
            raise RuntimeError(f"AV rate-limit/daily-cap: {msg[:200]}")

        articles = []
        for article in data.get("feed", []):
            ticker_sentiments = article.get("ticker_sentiment", [])
            if not any(ts.get("ticker", "").upper() == ticker.upper() for ts in ticker_sentiments):
                continue

            raw_time = article.get("time_published", "")
            try:
                pub_date = datetime.strptime(raw_time, "%Y%m%dT%H%M%S")
            except ValueError:
                continue

            headline = (article.get("title") or "").strip()
            if not headline:
                continue

            articles.append({
                "date": pub_date,
                "ticker": ticker,
                "headline": headline,
                "summary": (article.get("summary") or "").strip(),
                "source": (article.get("source") or "Unknown"),
                "url": article.get("url", ""),
                "sentiment_hint": "",
            })
        return articles

    def _fetch_from_alpha_vantage(self) -> pd.DataFrame:
        """
        Extrae noticias históricas usando Alpha Vantage NEWS_SENTIMENT endpoint.

        Fragmenta el rango [start_date, end_date] en trimestres para garantizar
        cobertura uniforme sin chocar con el tope de 1000 artículos/llamada.
        Free-tier: ~5 req/min (sleep 12s entre llamadas), 25 req/día.
        Con chunking trimestral y 10 tickers -> ~40 llamadas (~2 días en free-tier).

        Persistencia incremental: tras cada ticker completado, hace append a
        `output/news_corpus_av_partial.csv`. Al reiniciar, saltea tickers ya
        presentes en ese CSV — permite reanudar extracción multi-día.

        Returns:
            pd.DataFrame con noticias históricamente distribuidas.
        """
        partial_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "output", "news_corpus_av_partial.csv"
        )

        # Cargar progreso previo si existe
        done_tickers: set = set()
        existing_rows: List[dict] = []
        if os.path.exists(partial_path):
            try:
                prev = pd.read_csv(partial_path, parse_dates=["date"])
                existing_rows = prev.to_dict("records")
                done_tickers = set(prev["ticker"].unique())
                self.logger.info(
                    f"  Retomando extracción parcial: {len(existing_rows)} noticias ya descargadas "
                    f"para tickers {sorted(done_tickers)}"
                )
            except Exception as e:
                self.logger.warning(f"  No se pudo leer el CSV parcial: {e}. Empezando de cero.")

        start_dt = datetime.strptime(self.start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(self.end_date, "%Y-%m-%d")
        chunks = self._chunk_date_range(start_dt, end_dt, months=3)

        all_news: List[dict] = list(existing_rows)
        call_count = 0
        daily_cap_hit = False

        for ticker in self.tickers:
            if ticker in done_tickers:
                self.logger.info(f"  {ticker}: ya descargado, saltando.")
                continue

            ticker_news: List[dict] = []
            ticker_ok = True

            for chunk_start, chunk_end in chunks:
                cs = chunk_start.strftime("%Y-%m-%d")
                ce = chunk_end.strftime("%Y-%m-%d")
                self.logger.info(f"  - AV: {ticker} | chunk {cs} -> {ce}")

                try:
                    chunk_articles = self._fetch_av_chunk(ticker, chunk_start, chunk_end)
                    ticker_news.extend(chunk_articles)
                    call_count += 1
                    self.logger.info(
                        f"     {len(chunk_articles)} noticias (acum. ticker: {len(ticker_news)})"
                    )
                except RuntimeError as e:
                    # Cap diario alcanzado — abortar y guardar lo que hay
                    remaining_tickers = [t for t in self.tickers if t not in done_tickers]
                    self.logger.error(
                        f"  Cap diario AV alcanzado tras {call_count} llamadas. "
                        f"Tickers pendientes: {remaining_tickers}. "
                        f"Reinicia mañana — el CSV parcial retomará desde donde paró. "
                        f"Error: {e}"
                    )
                    daily_cap_hit = True
                    ticker_ok = False
                    break
                except requests.RequestException as e:
                    self.logger.error(f"     Error HTTP: {e}")
                except Exception as e:
                    self.logger.error(f"     Error inesperado: {e}")

                # Rate limit entre TODAS las llamadas
                self.logger.debug(f"     Esperando {self._AV_SLEEP_BETWEEN_TICKERS}s...")
                time.sleep(self._AV_SLEEP_BETWEEN_TICKERS)

            if ticker_ok and ticker_news:
                # Dedup por (ticker, url)
                seen_urls: set = set()
                deduped = []
                for item in ticker_news:
                    key = (item["ticker"], item["url"])
                    if key not in seen_urls:
                        seen_urls.add(key)
                        deduped.append(item)
                all_news.extend(deduped)
                done_tickers.add(ticker)
                self.logger.info(f"  {ticker}: {len(deduped)} noticias únicas guardadas.")

                # Persistencia incremental
                try:
                    os.makedirs(os.path.dirname(partial_path), exist_ok=True)
                    df_partial = pd.DataFrame(all_news)
                    df_partial["date"] = pd.to_datetime(df_partial["date"])
                    df_partial.to_csv(partial_path, index=False)
                    self.logger.info(f"     CSV parcial actualizado ({len(df_partial)} filas).")
                except Exception as e:
                    self.logger.warning(f"     No se pudo guardar CSV parcial: {e}")
            elif ticker_ok and not ticker_news:
                self.logger.warning(f"  {ticker}: 0 noticias en todo el período.")
                done_tickers.add(ticker)

            if daily_cap_hit:
                break

        df = pd.DataFrame(all_news)
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"])
            df = df.drop_duplicates(subset=["ticker", "url"]).sort_values(
                ["ticker", "date"]
            ).reset_index(drop=True)

        status = "INCOMPLETA (cap diario)" if daily_cap_hit else "completa"
        self.logger.info(
            f"Extracción Alpha Vantage {status}: {len(df)} noticias reales "
            f"con fechas históricas ({self.start_date} -> {self.end_date}). "
            f"Llamadas realizadas: {call_count}."
        )
        return df

    # ──────────────────────────────────────────────
    # Finnhub — fallback (snapshot ~7 días)
    # ──────────────────────────────────────────────

    def _fetch_from_finnhub(self) -> pd.DataFrame:
        """
        Extrae noticias usando Finnhub API company_news endpoint.

        LIMITACIÓN: el free-tier solo retorna ~7 días recientes sin importar
        el rango from_date/to_date. Se mantiene como fallback si Alpha Vantage
        no está disponible.

        Returns:
            pd.DataFrame con noticias (fechas limitadas a ~7 días recientes).
        """
        client = finnhub.Client(api_key=self.api_key)
        all_news = []

        for i, ticker in enumerate(self.tickers):
            self.logger.info(f"  - Finnhub: descargando noticias para {ticker}...")

            try:
                news = client.company_news(
                    symbol=ticker,
                    _from=self.start_date,
                    to=self.end_date,
                )

                if not news:
                    self.logger.warning(f"  Sin noticias para {ticker}")
                    continue

                for article in news:
                    pub_date = datetime.fromtimestamp(article.get("datetime", 0))
                    headline = article.get("headline", "").strip()
                    if not headline:
                        continue

                    all_news.append({
                        "date": pub_date,
                        "ticker": ticker,
                        "headline": headline,
                        "summary": article.get("summary", "").strip(),
                        "source": article.get("source", "Unknown"),
                        "url": article.get("url", ""),
                        "sentiment_hint": "",
                    })

                self.logger.info(
                    f"  {ticker}: {sum(1 for n in all_news if n['ticker'] == ticker)} noticias"
                )

            except Exception as e:
                self.logger.error(f"  Error descargando {ticker}: {e}")
                continue

            if i < len(self.tickers) - 1:
                time.sleep(1)

        df = pd.DataFrame(all_news)
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

        self.logger.info(f"Extracción Finnhub completa: {len(df)} noticias.")
        return df

    # ──────────────────────────────────────────────
    # Mock — datos sintéticos para pruebas
    # ──────────────────────────────────────────────

    def _fetch_mock_news(self) -> pd.DataFrame:
        """Genera noticias mock realistas para pruebas del pipeline."""
        start = datetime.strptime(self.start_date, "%Y-%m-%d")
        end = datetime.strptime(self.end_date, "%Y-%m-%d")
        total_weeks = max(1, (end - start).days // 7)

        all_news = []

        for ticker in self.tickers:
            self.logger.info(f"  - Generando noticias mock para {ticker}...")

            for week in range(total_weeks):
                n_news = random.randint(
                    max(1, self.news_per_ticker_per_week - 1),
                    self.news_per_ticker_per_week + 1,
                )

                for _ in range(n_news):
                    day_offset = week * 7 + random.randint(0, 6)
                    news_date = start + timedelta(days=day_offset)

                    if news_date > end:
                        continue
                    if news_date.weekday() >= 5:
                        continue

                    category = random.choice(list(self.HEADLINE_TEMPLATES.keys()))
                    template = random.choice(self.HEADLINE_TEMPLATES[category])

                    headline = template.format(
                        ticker=ticker,
                        quarter=random.randint(1, 4),
                        amount=random.choice([1, 2, 5, 10, 15, 20]),
                        price=random.randint(100, 500),
                        pct=round(random.uniform(1.5, 8.0), 1),
                    )

                    sentiment_map = {
                        "earnings": random.choice(["positive", "positive", "neutral"]),
                        "market": random.choice(["positive", "negative", "neutral"]),
                        "product": random.choice(["positive", "positive", "positive"]),
                        "macro": random.choice(["negative", "neutral", "neutral"]),
                    }

                    all_news.append({
                        "date": news_date,
                        "ticker": ticker,
                        "headline": headline,
                        "summary": "",
                        "source": random.choice(self.SOURCES),
                        "url": f"https://example.com/news/{ticker.lower()}/{news_date.strftime('%Y%m%d')}/{random.randint(1000,9999)}",
                        "sentiment_hint": sentiment_map[category],
                    })

            self.logger.info(
                f"  {ticker}: {sum(1 for n in all_news if n['ticker'] == ticker)} noticias"
            )

        df = pd.DataFrame(all_news)
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

        self.logger.info(f"Extracción mock completa: {len(df)} noticias generadas.")
        return df
