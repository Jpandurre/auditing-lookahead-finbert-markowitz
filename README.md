# Auditing Look-Ahead Bias in NLP-Enhanced Portfolio Optimization

Código, datos derivados y archivos de auditoría del artículo:

> J. P. Pandur-Reyes, E. G. Martínez-Rodríguez y W. A. Mata-López, «Auditing Look-Ahead Bias in NLP-Enhanced Portfolio
> Optimization: A Walk-Forward FinBERT–Markowitz Framework», *Revista Ingeniantes*, Edición Especial CIREDII 2026
> (en revisión).

Universidad de Colima, Facultad de Ingeniería Mecánica y Eléctrica.

## Qué contiene

Un backtesting *walk-forward* (validación cruzada temporal de origen móvil a nivel de portafolio) que compara tres
estrategias sobre seis acciones estadounidenses (AAPL, MSFT, GOOGL, NVDA, JPM, BAC), del 2 de junio de 2025 al 13 de
marzo de 2026 (197 sesiones, 10 rebalanceos mensuales):

| Estrategia | Descripción |
|---|---|
| Markowitz | Máximo Sharpe con SLSQP, ventana de 90 retornos, solo largos |
| Sentimiento IA | Markowitz ajustado por FinBERT: `ŵᵢ = wᵢᴹᵛ · (1 + α · sᵢ*)`, α = 0,5 |
| Igual Ponderación | 1/6 por activo |

El aporte principal es la **auditoría cronológica**: para cada rebalanceo se verifica que cada titular usado se publicó
antes del instante de ejecución (cierre de la sesión previa, 16:00 ET).

## Reproducir el artículo

```bash
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python scripts/reproduce_all.py
```

`reproduce_all.py` regenera todas las cifras en `results/` y las compara con los valores publicados; termina con
`TODO COINCIDE` si la reproducción es exacta (también verifica que la librería y el script den lo mismo). Tarda 3–5 minutos. Verificado con Python 3.10, NumPy 2.2, pandas 2.3,
SciPy 1.15 y Matplotlib 3.10.

| Elemento del artículo | Script | Archivo en `results/` |
|---|---|---|
| Tabla 3 y Tabla 5 | `auditoria_y_ablacion.py` | `r2_ablacion_metricas.csv` |
| Tabla 4 (auditoría) | `auditoria_y_ablacion.py` | `r2_auditoria_temporal.csv` |
| Inferencia (JKM, bootstrap) | `auditoria_y_ablacion.py` | `r2_bootstrap_contrastes.csv` (JKM en la salida de consola) |
| Convenciones temporales (4.2) | `auditoria_y_ablacion.py` | `r2_convenciones_temporales.csv` |
| CVaR y máxima caída | `auditoria_y_ablacion.py` | `r2_cvar_mdd.csv` |
| Pesos por rebalanceo | `auditoria_y_ablacion.py` | `r2_pesos_por_rebalanceo.csv` |
| Sensibilidad α × ventana (4.4) | `auditoria_y_ablacion.py grid` | `r2_sensibilidad_auditada.csv` |
| Placebo (800 réplicas) | `auditoria_y_ablacion.py placebo 400 1` y `... 400 2` | `r2_placebo_seed1.csv`, `r2_placebo_seed2.csv` |
| Figura 1 | `make_figura1.py` | `figura1.png` |

## Convención temporal

- Los pesos del rebalanceo *t* (primer día hábil del mes) se aplican al retorno de *t* (cierre de *p* → cierre de *t*),
  donde *p* es la sesión previa: la cartera se forma al **cierre ajustado de *p***.
- Markowitz usa los 90 retornos con fecha anterior a *t*.
- **Regla auditada de titulares:** marcas en `[c − 7 días, c)`, con `c = p 16:00 ET`, excluyendo marcas `00:00:00` del
  día *p* (fecha sin hora). La ventana original `[t − 7 días, t)` se conserva como contraste (`regla="calendario"`).
- `time_published` de Alpha Vantage no declara zona horaria; se verificó que corresponde a hora del Este de EE. UU.
- Costos: 12 pb × Σ|Δw| en cada rebalanceo (y 12 pb en la compra inicial). rf = 4 % anual. Anualización con 252 días.

## Datos (`data/`)

| Archivo | Contenido |
|---|---|
| `returns_daily.csv` | Retornos diarios simples de cierres ajustados (yfinance), 16-ene-2025 a 13-mar-2026 |
| `news_sentiment.csv` | 17.659 noticias de Alpha Vantage NEWS_SENTIMENT (extracción: mayo de 2026): marca horaria, emisora, fuente, URL, identificador del titular y puntuación FinBERT |

La puntuación es `P(positivo) − P(negativo)` de **ProsusAI/finbert** (sin ajuste fino) aplicada al titular.
**No se redistribuyen los titulares ni los resúmenes** por los términos de uso del proveedor. Cada noticia conserva su
URL y su marca horaria para que la auditoría se pueda verificar contra la fuente. `headline_id` es un hash (SHA-1, 12
caracteres) del titular, que permite reproducir la deduplicación de titulares sindicados.

Los extractores originales están en `investment_platform/extractors/` y `investment_platform/ia_models/`. Volver a
extraer requiere una clave propia de Alpha Vantage (`ALPHA_VANTAGE_KEY`), y el resultado puede diferir porque la API no
publica versiones de su archivo histórico. Para el pipeline completo: `pip install -r requirements-pipeline.txt`.

## Documentación

La arquitectura, la convención temporal, el modelo, el diccionario de datos y la descripción de cada archivo de
resultados están en [`docs/DOCUMENTACION.md`](docs/DOCUMENTACION.md). Todas las funciones tienen docstrings.

Para usar la regla auditada desde la librería:

```python
from investment_platform.portfolio_optimization import build_walk_forward_schedules
sched = build_walk_forward_schedules(returns, news, "2025-06-01", 90, 7, 0.5, regla_titulares="auditada")
```

## Estructura

```
data/                      datos derivados usados en el artículo
docs/                      documentación técnica
investment_platform/       motor: asignadores, backtest walk-forward, inferencia, extractores
scripts/                   reproducción de tablas, auditoría y figura
results/                   salidas de referencia (se regeneran con reproduce_all.py)
```

## Licencia

Código bajo licencia MIT (ver `LICENSE`). Los datos derivados se ofrecen solo para fines de verificación académica;
los derechos sobre las noticias pertenecen a sus editores.
