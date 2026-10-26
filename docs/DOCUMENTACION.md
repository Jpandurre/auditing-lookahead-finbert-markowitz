# Documentación técnica

Este documento describe la arquitectura del código, el flujo de datos, la convención temporal y el formato de cada
archivo. Para reproducir el artículo basta con el `README.md`; aquí está el detalle para revisarlo o extenderlo.

## 1. Arquitectura

```
investment_platform/
├── config.py                      constantes: tasa libre de riesgo (4 %), 252 días/año, modelo FinBERT
├── portfolio_optimization/        núcleo usado en el artículo
│   ├── allocators.py              MarkowitzAllocator, SentimentAllocator, prepare_sentiment_scores
│   ├── walk_forward.py            build_walk_forward_schedules (calendario de pesos por rebalanceo)
│   ├── backtest.py                BacktestEngine (trayectoria encadenada con costos, métricas)
│   └── inference.py               bootstrap estacionario pareado, IC y p-valores
├── extractors/                    obtención de datos (no se ejecuta al reproducir)
│   ├── price_extractor.py         precios diarios con yfinance
│   └── news_extractor.py          noticias de Alpha Vantage NEWS_SENTIMENT
└── ia_models/
    └── sentiment_analyzer.py      FinBERT (ProsusAI/finbert): P(positivo) − P(negativo)
scripts/
├── auditoria_y_ablacion.py        todas las tablas, la auditoría, la sensibilidad y el placebo
├── make_figura1.py                Figura 1
└── reproduce_all.py               ejecuta todo y verifica contra el artículo
```

`config.py` conserva algunas constantes de la plataforma completa (base de datos, tickers por defecto) que no
intervienen en el artículo.

## 2. Flujo de datos

```
yfinance ──> precios ajustados ──> retornos diarios ──────────────┐
                                    (data/returns_daily.csv)      │
                                                                  ▼
Alpha Vantage ──> noticias ──> FinBERT ──> puntuación por titular ──> walk-forward ──> backtest ──> métricas
                               (data/news_sentiment.csv)          ▲      (10 rebalanceos)   (costos)    e inferencia
                                                                  │
                                         regla temporal de titulares (calendario | auditada)
```

Los dos primeros pasos (extracción y FinBERT) se ejecutaron en mayo de 2026. Sus resultados están en `data/`, así que
la reproducción parte de ahí y no necesita claves de API, GPU ni red.

## 3. Convención temporal

| Elemento | Regla |
|---|---|
| Rebalanceo *t* | Primer día hábil de cada mes con datos (10 orígenes, del 2-jun-2025 al 2-mar-2026) |
| Sesión de ejecución *p* | Sesión inmediatamente anterior a *t* |
| Formación de la cartera | Cierre ajustado de *p* (16:00 ET); los pesos se aplican al retorno de *t* |
| Ventana de Markowitz | Los 90 retornos con fecha anterior a *t* (el último es el de *p*) |
| Titulares, regla **auditada** | Marca en `[c − 7 días, c)`, con `c = p 16:00 ET`; se excluyen marcas `00:00:00` del día *p* |
| Titulares, regla **calendario** | Marca en `[t − 7 días, t)`; admite titulares de la tarde de *p* y del fin de semana (fuga) |
| Zona horaria de las marcas | Hora del Este de EE. UU. (no la declara la API; verificada con eventos de hora conocida) |

En la librería, la regla se elige con `build_walk_forward_schedules(..., regla_titulares="auditada")`. El valor por
defecto, `"calendario"`, se mantiene para reproducir la especificación original. Si usas el código para otro estudio,
usa `"auditada"`.

Fuentes de fuga y control de cada una:

| Etapa | Control |
|---|---|
| Estimación con precios | Ventana estrictamente anterior a *t* |
| Selección de titulares | Regla auditada (corte al cierre de ejecución) |
| Modelo de lenguaje | FinBERT preentrenado, sin ajuste fino sobre el período |
| Hiperparámetros | α = 0,5 y ventana de 7 días fijados a priori; la sensibilidad es solo exploratoria |

## 4. Modelo

**Markowitz** (`MarkowitzAllocator.optimize`). Maximiza `(μᵀw − rf) / √(wᵀΣw)` con μ y Σ muestrales anualizados
(×252), sujeto a `Σw = 1` y `0 ≤ wᵢ ≤ cap` (por defecto `cap = 1`, sin tope), con SLSQP de SciPy (`ftol = 1e-12`,
`maxiter = 1000`), partiendo de 1/N. Los pesos menores de 0,1 % se anulan y se renormaliza. Si no converge, usa 1/N.

**Sentimiento IA** (`SentimentAllocator.optimize`). Parte de los pesos de Markowitz y aplica
`ŵᵢ = wᵢᴹᵛ · (1 + α · sᵢ*)`, donde `sᵢ*` es la puntuación media de la emisora en la ventana, reescalada min–max a
`[−1, +1]` entre las emisoras con titulares. Las emisoras sin titulares quedan con `sᵢ* = 0`. Con una sola emisora
cubierta no hay ajuste. Después se renormaliza a suma 1. Si una emisora tiene el 100 % del peso, el ajuste no puede
cambiar la cartera (solución de esquina).

**Igual Ponderación.** 1/6 por activo en todo el período.

**Backtest** (`BacktestEngine.run_strategy_with_schedule`). Una sola trayectoria de capital con capital inicial de
$10.000:
- Compra inicial: costo de `12 pb × 1`.
- En cada rebalanceo: costo de `12 pb × Σ|w_nuevo − w_anterior|` sobre el valor del portafolio.
- Dentro de cada bloque los pesos objetivo son constantes.
- Métricas:
  - retorno anualizado `(1 + R)^(252/n) − 1`;
  - volatilidad `σ_diaria · √252`;
  - Sharpe `(r_anual − rf) / vol`;
  - Sortino, igual que Sharpe pero con la desviación de los retornos negativos;
  - CVaR 95 %, media del 5 % de peores días;
  - máxima caída sobre la trayectoria.

**Inferencia** (`inference.py` y `scripts/auditoria_y_ablacion.py`):
- *Jobson–Korkie con corrección de Memmel* sobre los retornos diarios netos en exceso.
- *Bootstrap estacionario pareado* (Politis–Romano): bloques de longitud geométrica con media `L = 10` (5 y 20 como
  sensibilidad), `B = 10.000`, semilla 42. Todas las estrategias se remuestrean con los mismos índices. IC percentil y
  `p = 2 · mín{P(Δ* ≤ 0), P(Δ* ≥ 0)}`.
- *Placebo:* en cada rebalanceo se permutan las puntuaciones entre las emisoras con titulares (800 réplicas: semillas
  1 y 2, 400 cada una).

## 5. Diccionario de datos

`data/returns_daily.csv`: una fila por sesión (16-ene-2025 a 13-mar-2026) y una columna por emisora, con el retorno
simple diario del cierre ajustado.

`data/news_sentiment.csv`:

| Columna | Descripción |
|---|---|
| `date` | Marca de publicación (`time_published`), hora del Este; `00:00:00` = fecha sin hora |
| `ticker` | Emisora a la que la API vincula la noticia (una URL puede aparecer en varias emisoras) |
| `source` | Medio de publicación |
| `url` | Enlace a la noticia original, para verificar la marca horaria |
| `headline_id` | SHA-1 (12 caracteres) del titular; permite detectar titulares sindicados repetidos |
| `sentiment_label` | Clase más probable de FinBERT (Positive, Negative, Neutral) |
| `sentiment_score` | `P(positivo) − P(negativo)` ∈ [−1, +1] |

No se incluyen titulares ni resúmenes por los términos de uso del proveedor.

## 6. Archivos de resultados (`results/`)

| Archivo | Contenido |
|---|---|
| `r2_ablacion_metricas.csv` | Métricas por variante: `A` Markowitz, `B` Sentimiento IA auditada, `EW` Igual Ponderación, `B-cal` ventana de calendario, `C` fuga deliberada, `B-dedup` sin titulares sindicados, `A40`/`B40` tope de 40 %, `A-k1`/`B-k1` partición única |
| `r2_auditoria_temporal.csv` | Por rebalanceo: titulares de cada regla, cuántos son posteriores al cierre, primera y última marca admitida, y `precedencia_ok` |
| `r2_pesos_por_rebalanceo.csv` | Pesos de Markowitz y Sentimiento IA en cada origen |
| `r2_bootstrap_contrastes.csv` | Diferencias observadas, IC 95 % y p bootstrap de cada contraste (`L` = bloque medio) |
| `r2_convenciones_temporales.csv` | Diferencial IA − Markowitz con cada convención temporal |
| `r2_cvar_mdd.csv` | Por qué coinciden el CVaR y la máxima caída de Markowitz y Sentimiento IA |
| `r2_sensibilidad_auditada.csv` | Rejilla exploratoria α × ventana con la regla auditada |
| `r2_placebo_seed*.csv` | Diferencial de cada réplica placebo frente a Markowitz |
| `figura1.png` | Figura 1 del artículo |

## 7. Cómo extender

- **Otra estrategia:** escribe un asignador con un método `optimize(returns_df, ...) -> {ticker: peso}`, genera su
  calendario `{fecha: pesos}` y pásalo a `BacktestEngine.run_strategy_with_schedule`.
- **Otro universo o período:** reemplaza los archivos de `data/` con el mismo formato y ajusta `EVAL_START` en
  `scripts/auditoria_y_ablacion.py`. Para descargar datos nuevos usa los extractores (`requirements-pipeline.txt` y
  `ALPHA_VANTAGE_KEY`).
- **Otra regla temporal:** agrégala en `titulares()` (script) o en `build_walk_forward_schedules` (librería).
  Comprueba siempre que `precedencia_ok` sea verdadero en la auditoría.

## 8. Limitaciones conocidas

- La API no publica la zona horaria ni versiones de su archivo histórico; una nueva extracción puede dar otro corpus.
- El límite de 1.000 artículos por consulta, en orden ascendente, deja lagunas de 45 y 52 días.
- Con un universo de seis activos, Markowitz cae a menudo en soluciones de esquina, que anulan la señal en 7 de 10
  rebalanceos.
- 197 sesiones de un solo régimen de mercado limitan la potencia de los contrastes.
