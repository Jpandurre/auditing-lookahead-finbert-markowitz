"""
Analizador de sentimiento financiero usando FinBERT.
Modelo preentrenado: ProsusAI/finbert (optimizado para textos financieros).
"""

import logging
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd

from .. import config

# Lazy import de transformers y torch
_model = None
_tokenizer = None
_pipeline_available = False

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    _pipeline_available = True
except ImportError:
    _pipeline_available = False

logger = logging.getLogger("SentimentAnalyzer")


class SentimentAnalyzer:
    """
    Analiza sentimiento de textos financieros usando FinBERT (ProsusAI/finbert).
    
    Características:
        - Score continuo: P(positive) - P(negative) -> rango [-1, +1]
        - Labels: Positive, Negative, Neutral
        - Procesamiento por lotes (batches) para eficiencia
        - Lazy loading del modelo (solo al primer uso)
        - Fallback graceful si torch/transformers no están instalados
    
    Ejemplo:
        >>> analyzer = SentimentAnalyzer()
        >>> labels, scores = analyzer.analyze_batch(["Apple beats earnings", "Stock crashes"])
        >>> print(labels)  # ['Positive', 'Negative']
        >>> print(scores)  # [0.85, -0.72]
    """

    # Mapeo de labels del modelo FinBERT
    LABEL_MAP = {0: "Positive", 1: "Negative", 2: "Neutral"}

    def __init__(
        self,
        model_name: Optional[str] = None,
        batch_size: Optional[int] = None,
        device: Optional[str] = None,
    ):
        """
        Args:
            model_name: Nombre del modelo HuggingFace. Default: config.FINBERT_MODEL_NAME.
            batch_size: Tamaño de lote para inferencia. Default: config.SENTIMENT_BATCH_SIZE.
            device: 'cuda', 'cpu', o None (auto-detect).
        """
        self.model_name = model_name or config.FINBERT_MODEL_NAME
        self.batch_size = batch_size or config.SENTIMENT_BATCH_SIZE
        self.logger = logging.getLogger("SentimentAnalyzer")

        # Auto-detect device
        if device:
            self.device = device
        elif _pipeline_available and torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        self._model = None
        self._tokenizer = None
        self._loaded = False

    @property
    def is_available(self) -> bool:
        """Verifica si transformers y torch están disponibles."""
        return _pipeline_available

    def _load_model(self):
        """Carga el modelo y tokenizer de FinBERT (lazy loading)."""
        if self._loaded:
            return

        if not _pipeline_available:
            self.logger.warning(
                "transformers/torch no instalados. "
                "Ejecuta: pip install transformers torch"
            )
            return

        self.logger.info(f"Cargando modelo FinBERT ({self.model_name})...")
        self.logger.info(f"   Dispositivo: {self.device}")

        try:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self._model.to(self.device)
            self._model.eval()
            self._loaded = True
            self.logger.info("Modelo FinBERT cargado exitosamente.")
        except Exception as e:
            self.logger.error(f"Error cargando FinBERT: {e}")
            self._loaded = False

    def analyze_text(self, text: str) -> Tuple[str, float]:
        """
        Analiza el sentimiento de un texto individual.
        
        Args:
            text: Texto financiero a analizar.
            
        Returns:
            Tuple (sentiment_label, sentiment_score):
                - label: 'Positive', 'Negative', 'Neutral'
                - score: Valor continuo [-1, +1]
        """
        labels, scores = self.analyze_batch([text])
        return labels[0], scores[0]

    def analyze_batch(self, texts: List[str]) -> Tuple[List[str], List[float]]:
        """
        Analiza sentimiento de un lote de textos en batch.
        
        Args:
            texts: Lista de textos financieros.
            
        Returns:
            Tuple (labels, scores):
                - labels: Lista de 'Positive'/'Negative'/'Neutral'
                - scores: Lista de floats [-1, +1]
        """
        if not texts:
            return [], []

        # Cargar modelo si no está cargado
        self._load_model()

        if not self._loaded:
            self.logger.warning("Modelo no disponible. Retornando 'Neutral' por defecto.")
            return ["Neutral"] * len(texts), [0.0] * len(texts)

        all_labels = []
        all_scores = []
        total_batches = (len(texts) + self.batch_size - 1) // self.batch_size

        self.logger.info(
            f"Analizando sentimiento: {len(texts)} textos en {total_batches} lotes..."
        )

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i: i + self.batch_size]
            batch_num = i // self.batch_size + 1

            try:
                # Limpiar textos vacíos
                clean_batch = [t[:512] if t and t.strip() else "neutral" for t in batch]

                # Tokenizar
                inputs = self._tokenizer(
                    clean_batch,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                ).to(self.device)

                # Inferencia sin gradientes
                with torch.no_grad():
                    outputs = self._model(**inputs)
                    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

                # Extraer labels y scores
                probs_np = probs.cpu().numpy()

                for prob in probs_np:
                    # prob[0]=positive, prob[1]=negative, prob[2]=neutral
                    label_idx = np.argmax(prob)
                    label = self.LABEL_MAP[label_idx]

                    # Score continuo: P(positive) - P(negative) -> [-1, +1]
                    score = float(prob[0] - prob[1])

                    all_labels.append(label)
                    all_scores.append(round(score, 4))

                if batch_num % 5 == 0 or batch_num == total_batches:
                    self.logger.info(
                        f"  - Lote {batch_num}/{total_batches} completado"
                    )

            except Exception as e:
                self.logger.error(f"  Error en lote {batch_num}: {e}")
                all_labels.extend(["Neutral"] * len(batch))
                all_scores.extend([0.0] * len(batch))

        self.logger.info(
            f"  Análisis completo: "
            f"{sum(1 for l in all_labels if l == 'Positive')} positivas, "
            f"{sum(1 for l in all_labels if l == 'Negative')} negativas, "
            f"{sum(1 for l in all_labels if l == 'Neutral')} neutrales"
        )

        return all_labels, all_scores

    def analyze_dataframe(
        self,
        df: pd.DataFrame,
        text_column: str = "headline",
        fallback_column: str = "summary",
    ) -> pd.DataFrame:
        """
        Aplica análisis de sentimiento a un DataFrame de noticias.
        
        Usa la columna `text_column` como entrada principal.
        Si está vacía, usa `fallback_column` (summary).
        
        Agrega columnas: 'sentiment_label' y 'sentiment_score'.
        
        Args:
            df: DataFrame con noticias.
            text_column: Columna principal de texto.
            fallback_column: Columna de fallback si el texto está vacío.
            
        Returns:
            DataFrame con columnas de sentimiento agregadas.
        """
        if df.empty:
            df["sentiment_label"] = pd.Series(dtype="str")
            df["sentiment_score"] = pd.Series(dtype="float")
            return df

        df = df.copy()

        # Construir textos: headline + summary si disponible
        texts = []
        for _, row in df.iterrows():
            text = str(row.get(text_column, "") or "")
            if fallback_column in df.columns:
                summary = str(row.get(fallback_column, "") or "")
                if summary and len(text) < 50:
                    # Agregar summary si el headline es muy corto
                    text = f"{text}. {summary}" if text else summary
            texts.append(text.strip() or "neutral")

        labels, scores = self.analyze_batch(texts)

        df["sentiment_label"] = labels
        df["sentiment_score"] = scores

        return df
