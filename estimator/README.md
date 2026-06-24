# Estimator — Servicio IA de estimación de software

Servicio IA en FastAPI que estima proyectos de software a partir de un formulario tipado. Es la pieza Python del programa **Master en AI Engineering**: un endpoint pensado para ser consumido por un backend de negocio (Rails, Streamlit u otro), no por un usuario final.

A partir de la **Sesión 04** el contrato es deliberadamente estrecho:
- entrada tipada (`description` + tres enums),
- salida en texto libre,
- prompt fuera del código en templates Jinja2 versionados (`app/foundation/prompts/<use_case>/<version>/`).

La inteligencia adicional (output estructurado, guardrails, cache semántico) se construye encima de esta base en directo.

## Cómo levantar

### Con Docker (recomendado)

```bash
cd estimator
cp .env.example .env  # añade al menos OPENAI_API_KEY o ANTHROPIC_API_KEY
docker compose up --build
```

El servicio queda en `http://localhost:8000` (Swagger en `/docs`, health en `/health`). Redis arranca como servicio vecino para el cache exact-match del wrapper.

### Sin Docker

```bash
cd estimator
uv sync
uv run uvicorn app.main:app --reload
```

### Probar el endpoint

```bash
curl -X POST http://localhost:8000/api/v1/estimate \
  -H "Content-Type: application/json" \
  -d '{
    "description": "A small B2B SaaS to manage employee equipment loans across teams. Role-based access, audit trail, weekly digest.",
    "project_type": "web_saas",
    "detail_level": "medium",
    "output_format": "phases_table"
  }'
```

Respuesta:

```json
{
  "text": "| phase | duration_weeks | cost_eur | confidence_pct | …",
  "prompt_version": "v1"
}
```

### Cliente Streamlit

El cliente Streamlit es un formulario que construye el JSON y muestra el `text` recibido. Corre fuera de Docker y consume la API por HTTP:

```bash
cd estimator
uv run streamlit run streamlit_app.py
# Abrir http://localhost:8501
```

La URL del servicio se lee de `ESTIMATOR_API_BASE_URL` (default `http://localhost:8000`).

## Cómo testar

```bash
cd estimator
uv run pytest
```

La batería corre en milisegundos sin tocar APIs externas. Cubre cuatro categorías:

- `tests/test_schemas.py` — validaciones del `EstimationRequest` (longitudes, enums, campos obligatorios).
- `tests/test_prompts.py` — render del template `v1`: `description` aparece dentro de `<project_description>`, los bloques condicionales por `output_format` y `detail_level` solo se incluyen cuando aplica, y `StrictUndefined` falla early ante variables faltantes.
- `tests/test_estimate_endpoint.py` — endpoint con el wrapper LLM mockeado vía `app.dependency_overrides`: comprueba el contrato 200/422, que `system_prompt` y `user_message` viajan separados, y que la respuesta lleva `prompt_version="v1"`.
- `tests/test_llm_wrapper.py` y `tests/test_cache.py` — wrapper y cache de la Sesión 03, intactos.

## Estructura del proyecto

```
estimator/
├── app/
│   ├── main.py                        # FastAPI app, CORS, lifespan, /health
│   ├── config.py                      # Settings (Pydantic Settings, .env)
│   ├── dependencies.py                # Singletons cacheados: cache + LLMWrapper
│   ├── routers/
│   │   └── estimations.py             # POST /api/v1/estimate
│   ├── schemas/
│   │   └── estimation.py              # EstimationRequest, EstimationResponse, enums
│   ├── prompts/
│   │   ├── loader.py                  # Environment Jinja2 + render_estimation_prompt
│   │   └── estimation/
│   │       └── v1/
│   │           ├── system.j2          # rol + reglas + bloques condicionales + include
│   │           ├── user.j2            # bloque <project_description>
│   │           └── examples.j2        # few-shot examples
│   └── services/
│       ├── llm_wrapper.py             # LiteLLM Router con fallback y cost tracking
│       └── cache.py                   # Redis exact-match cache
├── tests/
│   ├── test_schemas.py
│   ├── test_prompts.py
│   ├── test_estimate_endpoint.py
│   ├── test_llm_wrapper.py
│   └── test_cache.py
├── streamlit_app.py                   # Formulario que consume /api/v1/estimate
├── Dockerfile                         # Multi-stage con uv
├── docker-compose.yml                 # Servicio IA + Redis
└── pyproject.toml
```

### Versionado de prompts

La estructura `app/foundation/prompts/<use_case>/<version>/` no es opcional: `v1/` ya existe desde el primer día porque versionar un prompt es la forma más barata de habilitar A/B testing y rollback en producción. Cuando una iteración del prompt se cocina, se crea `v2/` al lado y `render_estimation_prompt(request, version="v2")` lo recoge sin tocar router ni schemas.

Lo que vive **fuera** del template (en código): el contrato (`EstimationRequest`), el switch de versión y el wrapper. Todo lo demás (rol del modelo, reglas, ejemplos, formatos de salida, niveles de detalle) vive dentro del `.j2`. Si para cambiar el comportamiento del modelo hay que tocar Python, la separación está rota.

## Variables de entorno

| Variable | Default | Notas |
|---|---|---|
| `OPENAI_API_KEY` | — | Requerido al menos uno de los dos |
| `ANTHROPIC_API_KEY` | — | Requerido al menos uno de los dos |
| `PRIMARY_MODEL` | `gpt-4o-mini` | Modelo principal del Router |
| `FALLBACK_MODEL` | `claude-haiku-4-5-20251001` | Se usa si el primario falla |
| `REDIS_URL` | `redis://localhost:6379` | Cache exact-match |
| `CACHE_TTL` | `86400` | Segundos |
| `APP_ENV` | `development` | Controla el renderer de structlog |
| `ESTIMATOR_API_BASE_URL` | `http://localhost:8000` | Lo lee el cliente Streamlit |

`get_settings()` es un singleton cacheado con `lru_cache`: cualquier cambio en `.env` requiere reiniciar uvicorn (no basta con `--reload`). **Excepción: los modelos LLM** — ver la sección siguiente.

## Configuración de modelos en runtime

Los knobs de modelo (`PRIMARY_MODEL`, `FALLBACK_MODEL`, `CRITIC_MODEL`, `METADATA_EXTRACTOR_MODEL`, `COMPRESSION_MODEL`, `PROPOSITIONAL_CHUNKER_MODEL`, `CONTEXTUAL_CHUNKER_MODEL`) se pueden **sobreescribir en caliente** sin tocar `.env` ni recrear contenedores — pensado para cambiar de modelo en mitad de un directo (la pestaña *Ajustes* del cliente Rails usa este endpoint).

```
GET /api/v1/config/models
  → {"models": {KEY: {"effective", "default", "overridden"}},
     "available_models": [...], "embedding_model": "..."}

PUT /api/v1/config/models
  Body: {"models": {"PRIMARY_MODEL": "gpt-4o", "CRITIC_MODEL": null}}   # null = reset
  → mismo shape que el GET (snapshot fresco)
  422 key desconocida / modelo fuera de catálogo · 400 modelo sin API key · 503 Redis caído
```

Cómo funciona (`app/foundation/llm/runtime_config.py`):

- Los overrides viven en un hash de Redis (`estimator:runtime_config`): **sobreviven a `--reload` y reinicios**, y todos los workers los ven al instante. `.env` sigue siendo la capa de defaults.
- El wrapper y el servicio resuelven el modelo **por llamada** (properties), así que el cambio aplica en la siguiente petición. El catálogo (`AVAILABLE_MODELS`) se filtra por las API keys configuradas.
- Con un override de primario activo no hay fallback automático de provider (misma semántica que `model_override`: llamada directa, sin Router).
- Las caches se particionan por modelo (la exacta ya lo hacía; la semántica incluye el modelo en su bucket desde este cambio), así que cambiar de modelo nunca sirve respuestas generadas por otro.
- `EMBEDDING_MODEL` queda fuera a propósito: cambiarlo invalidaría todos los vectores almacenados.

```bash
http PUT :8000/api/v1/config/models models:='{"PRIMARY_MODEL": "gpt-4o"}'
http PUT :8000/api/v1/config/models models:='{"PRIMARY_MODEL": null}'     # volver al .env
```

---

## Sesión 5 — Memoria conversacional y adjuntos

A partir de la Sesión 05 el estimator deja de ser puramente transaccional y soporta **sesiones conversacionales**: el cliente puede refinar el alcance del proyecto a lo largo de varios turnos, subir documentos (PDF/Word) y el sistema recuerda el proyecto en curso entre llamadas. El endpoint `POST /api/v1/estimate` original se mantiene intacto para compatibilidad y para la demo transaccional.

### Endpoints nuevos

```
POST /sessions                              → 201 {"session_id": "<uuid>"}
GET  /sessions/{session_id}                 → 200 {session_id, message_count, max_turns, metadata}
POST /sessions/{session_id}/estimate        → 200 EstimationResponse
   (multipart/form-data: transcript, project_type, detail_level, output_format, attachments[])
```

Ejemplo end-to-end con httpie:

```bash
http POST :8000/sessions
# {"session_id": "abc-123"}

http -f POST :8000/sessions/abc-123/estimate \
  transcript="Queremos estimar un CRM llamado Nimbus en React + Postgres para el equipo de ventas." \
  project_type=web_saas detail_level=medium output_format=phases_table \
  attachments@spec.pdf

http GET :8000/sessions/abc-123
# Inspecciona el ProjectMetadata acumulado y el tamaño del historial.
```

Y un segundo turno reutilizando el mismo `session_id` sin repetir el contexto:

```bash
http -f POST :8000/sessions/abc-123/estimate \
  transcript="Añade un módulo de facturación con Stripe." \
  project_type=web_saas detail_level=medium output_format=phases_table
```

La respuesta del segundo turno integra Nimbus + React + Postgres + facturación porque el `<project_metadata>` se inyecta en el system prompt y el historial reciente viaja en el array `messages`.

### Decisiones de diseño

1. **Camino B para los adjuntos.** Extraemos el texto del PDF/Word **dentro del servicio IA** con `pypdf` y `python-docx`, lo recortamos a `MAX_ATTACHMENT_CHARS` y lo concatenamos al transcript con fences explícitos (`--- attachment: spec.pdf ---`). La alternativa (Camino A: subir el binario a la Files API de OpenAI o Anthropic) habría sido más corta de implementar pero acopla el wrapper a un proveedor multimodal concreto. Camino B mantiene `complete_structured_chat` agnóstico de proveedor (texto en, texto fuera vía LiteLLM Router + Instructor) y prepara el terreno para el chunking real de RAG en el módulo 3. La extracción es robusta a páginas corruptas (fallos por página se loguean y se ignoran) y a archivos vacíos.

2. **`project_metadata` con extractor LLM, no heurística.** Tras cada respuesta del estimador, una **segunda llamada** al LLM (modelo barato configurable vía `METADATA_EXTRACTOR_MODEL`, por defecto `gpt-4o-mini`) lee el último turno y devuelve un `ProjectMetadata` parcial vía Instructor. Lo fusionamos con el previo: campos escalares sobrescriben si vienen no-nulos, la lista de tecnologías se une case-insensitively. Se eligió el extractor LLM frente a una heurística regex porque el coste de una llamada con prompt corto es marginal y la robustez frente a paráfrasis del usuario es mucho mejor — y porque el curso enseña precisamente cómo construir estos pasos con LLMs. Si la llamada falla, se loguea y se conserva la metadata previa: la conversación no se cae por una extracción rota.

3. **Memoria en proceso, no Redis ni Postgres.** El `SessionStore` es un `dict` en memoria del worker FastAPI. La volatilidad (estado perdido al reiniciar el contenedor) es **intencional** para esta fase y está documentada en el docstring del store. La persistencia entre reinicios entra en el directo cuando hablemos de compresión de memoria con anclas.

4. **Cachés desactivadas en el path conversacional.** Cada turno depende del historial + metadata + adjuntos: dos transcripciones idénticas en sesiones distintas **no** son la misma llamada. El método nuevo `EstimationService.estimate_conversational` por tanto no consulta ni el cache exact-match ni el semántico, y `EstimationResponse.cached` siempre es `false` en este path. El endpoint transaccional original `POST /api/v1/estimate` sigue usando las dos cachés sin cambios.

5. **Ventana deslizante con `MAX_CONVERSATION_TURNS=6` por defecto.** El system prompt se regenera fresco cada turno desde el `ProjectMetadata` actual, así que no consume slot. Lo que llega al LLM en el turno N es: `[system_v2] + últimos N pares (user, assistant) + nuevo user`. Cuando el historial supera el tope, los pares más antiguos se descartan en bloque para preservar la alternancia de roles. El siguiente paso (resumen acumulativo + anclas) lo construimos en el directo.

### Variables de entorno nuevas

| Variable | Default | Notas |
|---|---|---|
| `MAX_CONVERSATION_TURNS` | `6` | Pares user+assistant que mantiene la ventana. |
| `MAX_ATTACHMENT_CHARS` | `60000` | Corte por archivo extraído. Trunca, no rechaza. |
| `METADATA_EXTRACTOR_MODEL` | `gpt-4o-mini` | Modelo de la segunda llamada por turno. |

### Tests del Paso 7

```bash
uv run pytest tests/test_sessions_metadata.py tests/test_sessions_attachments.py tests/test_sessions_window.py -v
```

Los tres tests son de integración con `TestClient`, un `FakeLLMWrapper` que captura cada llamada y devuelve resultados scripted, y un `SessionStore` aislado por test (sin singleton). Cubren los tres criterios del enunciado: dos turnos acumulan metadata, el contenido de un PDF llega al `messages` del LLM, y enviar más turnos que `MAX_CONVERSATION_TURNS` nunca infla el array de mensajes más allá del límite.

### Cliente Rails

El cliente Rails (`estimator-web/`) se adaptó al flujo conversacional con un nuevo controller `ChatSessionsController` (rutas `/chat_sessions`, root re-apuntado aquí), un panel lateral con el `ProjectMetadata` actual, multipart vía `faraday-multipart` y un botón "Nueva conversación" que destruye el mirror local y arranca una sesión limpia. El endpoint transaccional `EstimationsController` se mantiene operativo para la demo histórica.

## Sesión 7 — Pipeline de embeddings

Primer paso hacia la búsqueda semántica: convertir presupuestos históricos (JSON) en vectores. El módulo nuevo vive en `app/generation/rag/` y expone un único endpoint. En la Sesión 07 no se persistía nada — los vectores se generaban en memoria y se devolvían por HTTP; **desde la Sesión 08 el endpoint persiste en pgvector** (ver la sección de la Sesión 8 más abajo).

Piezas:

- `chunker.py` (`JSONStructuralChunker`) — chunking **estructural**: un componente del presupuesto = un chunk. A cada chunk se le antepone una cabecera de contexto del presupuesto padre (proyecto, sector, tecnología) para que no pierda la pista de a quién pertenece. Cuenta tokens con `tiktoken`.
- `embedder.py` (`OpenAIEmbedder`) — invoca `text-embedding-3-small` (1536 dims) en **batches** de 100, con reintento exponencial (1s/2s/4s) ante `RateLimitError` y logging por batch.
- `router.py` — orquesta `chunk → embed → stats`.

### Endpoint nuevo

> **Contrato actualizado en la Sesión 08.** El contrato original de la S07
> (`{"budgets": [...]}` → chunks+vectores por HTTP, sin persistencia) fue
> reemplazado por el contrato persistente de un documento por petición que se
> documenta en la sección de la Sesión 8. Las piezas de esta sección (chunker,
> embedder) siguen siendo las mismas; lo que cambió es qué se hace con los
> vectores.

Con el sample completo: 17 presupuestos → 60 chunks (`text-embedding-3-small`, 1536 dims).

### Script CLI `compare.py`

Sanity check de los embeddings: embebe dos textos y devuelve su similitud coseno (calculada a mano, sin numpy). Reutiliza `OpenAIEmbedder`.

```bash
# Fuera del contenedor (desde estimator/, con el .env cargado):
uv run python scripts/compare.py \
  --text-a "OAuth 2.0 authentication backend for fintech" \
  --text-b "JWT-based authorization service for banking app"

# Dentro del contenedor (scripts/ está bind-montado en docker-compose.yml):
docker compose exec estimator python scripts/compare.py \
  --text-a "..." --text-b "..."
```

Los resultados de las tres parejas de validación del enunciado están en [`app/generation/rag/SANITY_CHECK.md`](app/generation/rag/SANITY_CHECK.md).

### Comparativa de estrategias de chunking (sesión en vivo)

Ocho estrategias de chunking tras una interfaz común (`app/generation/rag/chunking/base.py::Chunker`): `structural`, `fixed_size`, `recursive`, `sentence_window`, `semantic`, `propositional`, `contextual_retrieval`, `hierarchical`. Viven en `app/generation/rag/chunking/strategies/` (el estructural en `structural.py`).

```
POST /embeddings/compare
  Input:  {"budgets": [...], "queries": [...], "strategies": [...], "top_k": 3}
  Output: {"stats_per_strategy": {...}, "queries_per_strategy": {...}}
```

CLI del comparador (la herramienta de las demos), que carga `data/budgets_sample.json` + `data/test_queries.json`:

```bash
# Estadísticos + coste de todas las estrategias
uv run python scripts/compare_chunkers.py --strategies all --queries all --show-stats --show-cost

# Top-k de una consulta para dos estrategias
uv run python scripts/compare_chunkers.py --strategies sentence-window,structural \
  --queries "OAuth authentication for fintech mobile app" --show-top-k 3

# Comparar dimensiones del modelo (1536 vs 768 / Matryoshka)
uv run python scripts/compare_chunkers.py --models small-1536,small-768

# Generar el reporte de respaldo
uv run python scripts/compare_chunkers.py --strategies all --queries all \
  --show-stats --show-cost --output app/generation/rag/COMPARISON_REPORT.md
```

Las estrategias `semantic`, `propositional` y `contextual_retrieval` llaman a APIs externas durante la ingesta (necesitan `OPENAI_API_KEY` / `ANTHROPIC_API_KEY`) y reportan su coste en `chunking_done`. `sentence_window` usa NLTK (`punkt`/`punkt_tab`, descarga perezosa). El endpoint de comparación no persiste nada — la persistencia vectorial vive en `/embeddings/ingest` desde la **Sesión 08**.

### Dependencias y scope

- Dependencias del pre-ejercicio: `tiktoken>=0.7.0` (`openai` ya estaba desde Sesión 01).
- Dependencias de la sesión en vivo: `langchain-text-splitters`, `langchain-experimental`, `langchain-openai`, `nltk` (`anthropic` ya estaba). No se añade numpy/scikit-learn ni `sentence-transformers`; la coseno y los percentiles son stdlib.
- **Late chunking** se trata como concepto en el directo (no hay código ejecutable: requiere modelos con token-level embeddings que no son el del proyecto).
- **Fuera de scope** → **Sesión 08**: persistencia vectorial (pgvector), búsqueda semántica / retrieval real y métricas formales de retrieval (recall@k, NDCG).
- El guion del directo está en `guides/session-7-live-guide.md` (git-ignored, material de instructor).

## Sesión 8 — Persistencia vectorial y búsqueda semántica

El pipeline de la S07 deja de devolver vectores por HTTP y los persiste en Postgres + pgvector (`pgvector/pgvector:pg16`, ya presente en compose). Schema gestionado con Alembic (`alembic/versions/0002_session8_pgvector.py`: extensión `vector` + tablas `documents` y `chunks`). Código nuevo: `app/generation/rag/store/` (modelos ORM + repositorio async con `asyncpg`), `app/generation/rag/ingest_service.py` (orquestación chunk → embed → persist) y `app/generation/rag/retriever.py` (búsqueda). El stack async convive con el sync de la S06: una sola `DATABASE_URL`, el engine async deriva el driver.

### Endpoints

```
POST /embeddings/ingest   (refactorizado: ahora persiste)
  Input:  {"source_path": "...", "document_type": "historical_budget", "content": <Budget>}
  Output: {"document_id": 1, "chunks_created": 4, "embedding_dimension": 1536, "ingestion_time_ms": 2431}
  200 OK · 409 {"detail": "Document already ingested", "document_id": N} · 422 · 500
  Todo en UNA transacción: si el embedder falla, rollback — sin documents huérfanos.

POST /search
  Input:  {"query": "REST API with OAuth authentication", "k": 5}
  Output: {"query", "k", "search_time_ms", "results": [{chunk_id, document_id, chunk_type, content, distance, metadata}]}
  k-NN por distancia coseno (operador <=>) vía SQL. Sin índice vectorial: sequential scan.
```

### Script `query_examples.py`

Ingesta el corpus completo (idempotente: los 409 se saltan) y lanza 5 queries que ejercitan ángulos distintos (match directo, reformulación semántica, dominio ajeno, ambigua, muy específica). Su salida real contra el corpus está en [`output_examples.txt`](output_examples.txt).

```bash
docker compose up -d
docker compose run --rm estimator python scripts/query_examples.py
```

No hay tests de integración con BD viva (no existen fixtures de Postgres en la suite); la evidencia end-to-end es este script. Los tests HTTP usan fakes vía `dependency_overrides`.

### Decisiones de schema

- **Dos tablas y no una.** Un presupuesto produce N chunks: es un uno-a-muchos real. Una tabla única duplicaría la metadata del documento en cada fila y perdería integridad referencial. Con `ON DELETE CASCADE`, borrar un presupuesto elimina sus chunks automáticamente; `documents` posee la procedencia (`source_path`, `ingested_at`), `chunks` posee los vectores.
- **`metadata` como JSONB y no columnas tipadas.** Lo estable (tipo de documento, tipo de chunk, fechas) va en columnas tipadas; lo que el chunker puede enriquecer (sector, tecnologías, horas) va a JSONB. El índice GIN permite consultar por claves arbitrarias sin una migración por cada clave nueva. Una columna se promociona a tipada solo cuando se convierte en filtro caliente.
- **`cosine_distance` y no L2 ni inner product.** Los embeddings de OpenAI vienen normalizados, así que el ranking sería equivalente; usamos coseno por convención de la literatura RAG y, sobre todo, para quedar alineados con la operator class `vector_cosine_ops` del índice HNSW que se añade en el directo. Si la query usa un operador y el índice está construido con otra operator class, Postgres ignora el índice **en silencio** y cae a sequential scan.
- **Sin índice vectorial todavía (deliberado).** Con 17 presupuestos / 60 chunks el sequential scan responde en pocos cientos de ms y es el baseline contra el que el directo mide el impacto del HNSW. Añadirlo ahora ocultaría justamente lo que queremos observar.
- **`embedding` nullable.** Permite insertar el chunk y rellenar el vector después (ingesta asíncrona, sesiones posteriores). En esta sesión chunk+embedding se escriben atómicamente.
- **`vector(1536)` hardcodeado.** Es la dimensionalidad de `text-embedding-3-small`; cambiarla implica re-embedear todo el corpus, no es configuración dinámica.

**Fuera de scope (se construye en el directo):** índices vectoriales (HNSW/IVFFlat), filtros por metadata en SQL, búsqueda híbrida (full-text + vector) y tuning de Postgres.

## Live Session 08 — Indexación vectorial y operación

Material de la sesión en vivo que cierra el Módulo 3: cómo se **indexa** (HNSW), **optimiza** (halfvec) y **opera** (monitorización + mantenimiento) la base de datos vectorial construida en el previo. Foco exclusivo en la capa de datos — el retrieval en estimación llega en la Sesión 09.

### Scripts Python (`scripts/*_s08.py`)

Todos se ejecutan con `docker compose run --rm estimator python scripts/<script>` (o `docker compose exec estimator python scripts/<script>` con el stack levantado). Reutilizan la configuración, la sesión async y el embedder del proyecto; `s08_common.py` es el módulo compartido (no es un script).

| Script | Qué hace |
|---|---|
| `measure_baseline_s08.py` | Latencia SQL de las 5 queries del benchmark (warm-up + 2 mediciones, media y desviación). Ejecutar antes y después de crear el índice. Imprime al final el literal pgvector de la primera query para los demos en psql. |
| `sweep_ef_search_s08.py` | Barre `hnsw.ef_search` en [10..200], mide latencia y recall contra la verdad de fondo (seq scan forzado) e imprime la tabla con la recomendación ★. |
| `compare_indexes_s08.py` | Las 5 queries contra el índice `vector` y el `halfvec` (forzados por expresión, sin dropear nada): top-5, overlap y latencias lado a lado. |
| `report_index_sizes_s08.py` | Estado de los índices de `chunks`: tipo (btree/gin/hnsw), tamaño, `idx_scan`, último uso. Ejecutar antes/después de cada decisión. |
| `insert_synthetic_chunks_s08.py` | Inserta chunks sintéticos con embeddings **reales** (`count` posicional, default 100). Con `30000` engorda el corpus en el pre-flight para que el baseline sin índice sea medible. Limpieza: `DELETE FROM documents WHERE document_type = 'synthetic_test';` |

### Snippets SQL (`scripts/sql_s08/`)

Se ejecutan en psql, en este orden durante el directo. psql vive en el contenedor de Postgres (que no monta `scripts/`), así que: redirigir el archivo o pegar bloques.

```bash
# Archivo completo:
docker compose exec -T estimator-postgres psql -U estimator -d estimator \
  < estimator/scripts/sql_s08/01_create_hnsw.sql
# Interactivo (pegar bloques):
docker compose exec estimator-postgres psql -U estimator -d estimator
```

| Orden | Snippet | Bloque del directo |
|---|---|---|
| 1 | `01_create_hnsw.sql` | Construcción del índice HNSW (`vector_cosine_ops`, m=16, ef_construction=128) |
| 2 | `02_test_antipatron.sql` | El antipatrón silencioso: `<=>` vs `<->` con `EXPLAIN ANALYZE` |
| 3 | `03_create_halfvec.sql` | Índice halfvec paralelo sobre `(embedding::halfvec(1536))` |
| 4 | `04_monitoring_queries.sql` | Monitorización con `pg_stat_user_indexes` |
| 5 | `05_maintenance_cycle.sql` | ANALYZE → VACUUM → REINDEX CONCURRENTLY |

### Operational queries

La query canónica de monitorización — para tenerla a mano siempre:

```sql
SELECT indexrelname, idx_scan, last_idx_scan,
       pg_size_pretty(pg_relation_size(indexrelid)) AS size
FROM pg_stat_user_indexes
WHERE relname = 'chunks'
ORDER BY idx_scan DESC;
```

Si un índice vectorial tiene `idx_scan = 0` después de servir queries semánticas: casi seguro el operador de la query no coincide con la operator class del índice (p. ej. `<->` contra `vector_cosine_ops`). Verificación de operator classes y estadísticas de tabla: en `scripts/sql_s08/04_monitoring_queries.sql`.

El tuning de Postgres para builds de índices vive en `docker-compose.yml` (servicio `estimator-postgres`): `shm_size`, `shared_buffers`, `maintenance_work_mem`, `max_parallel_maintenance_workers`. Valores conservadores de desarrollo; en producción escalan con la RAM.

### Entregable post-directo (a Lia)

1. Repositorio actualizado: índice halfvec activo, flags de tuning en compose, queries de monitorización en el README.
2. Documento corto con los números observados en **vuestro** barrido de `ef_search` (tabla del script) y la decisión razonada del valor adoptado: qué recall ganáis y qué latencia pagáis frente a las alternativas.

## Sesión 9 — Pre-work: diagnóstico arquitectónico

Antes del directo, el ejercicio documenta **por qué** el flujo S08 no basta para estimar desde una transcripción de reunión. El entregable vive en la raíz del repo (`arquitectura-actual.md`), no en el servicio: es un diagnóstico con cuatro piezas.

1. **Diagrama del estado al cierre de S08** — tres capas (Rails → FastAPI → pgvector) con el borde sombreado donde el flujo muere: *lista de chunks + distancias*, sin flecha transcripción → estimación.
2. **Trace reproducible** — `examples/trace_s09.py` embebe la transcripción completa con `text-embedding-3-small` (mismo modelo que la ingesta) y hace `POST /search` con k=5. Es un **cliente**, no añade comportamiento al servicio: no hay reformulación, filtros ni generación todavía.
3. **Cinco fallos anclados al trace** — embedding de texto largo y ambiguo, ausencia de query understanding, sin filtros por metadata, sin ensamblado de contexto, sin generación fundamentada.
4. **Diagrama de evolución** — las cajas nuevas que el directo implementará: Query Understanding → Metadata-filtered Retriever → Context Assembler → Generation.

Transcripciones de ejemplo en `examples/transcripts/` (`01_clear.txt`, `02_ambiguous.txt` — la del trace —, `03_hard.txt`) y plantilla del entregable en `examples/transcripts/TEMPLATE.md`.

```bash
# Con el stack levantado y el corpus S08 ingerido (query_examples.py):
export OPENAI_API_KEY=sk-...
uv run examples/trace_s09.py examples/transcripts/02_ambiguous.txt
```

## Live Session 09 — Estimación RAG fundamentada

El directo cierra el bucle que faltaba: **transcripción → estimación con citaciones**, fundamentada en presupuestos históricos. Es un pipeline **nuevo e independiente** del transaccional/conversacional (S4–5): produce un `Estimate` (engineer-days, módulos→tareas, citaciones) en lugar de `EstimationResult` (EUR/semanas/fases). Toda la generación pasa por `LLMWrapper` (Instructor), no por la API cruda del proveedor.

### Pipeline E2E — `estimate_from_transcript()`

Orquestador en `app/generation/rag/estimator.py`. Etapas:

1. **Idempotencia** — `idempotency_key` opcional; hit en Redis (o `dict` in-process si no hay Redis) devuelve el `Estimate` cacheado sin re-ejecutar el pipeline.
2. **Reformulación** — `reformulate_query()` destila la transcripción en una `EstimationQuery` estructurada (`gpt-5-mini`); `compose_search_text()` arma un texto técnico corto en inglés. Fallback degradado a rewrite de una frase si la extracción estructurada falla.
3. **Embed + retrieval** — embed del `search_text` + `search_chunks()` con filtros estructurales (`client_sector`, `year` desde JSONB; `chunk_type`) y umbral de distancia coseno. Si nada cruza el umbral → `low_confidence` y `Estimate` con `confidence='insufficient'` (soft-fail, sin generación).
4. **Ensamblado** — cada chunk se envuelve en `<source id=… sector=… project_year=… chunk_type=… distance=…>`; `truncate_to_token_budget()` descarta chunks enteros desde la cola hasta `MAX_CONTEXT_TOKENS`.
5. **Generación** — `generate_estimate()` con grounding estricto (`gpt-5`, `reasoning_effort=high`, `GENERATION_MAX_TOKENS=64000`).
6. **Guards post-generación** — `validate_citations()` detecta `source_id` fabricados (1 reintento correctivo; si persisten → `confidence='low'`); `check_coherence()` exige que `insufficient` no lleve cifras (1 reintento; si persiste → `MalformedEstimateError` → 502).

Módulos nuevos: `query_reformulator.py`, `context_assembler.py`, `prompt_builder.py`, `validation.py`, `idempotency.py`, `observability.py` (`log_stage()` + correlación por `X-Request-ID`), `errors.py`.

### Endpoints nuevos

```
POST /v1/estimate/from-transcript
  Header: X-API-Key: $ESTIMATE_API_KEY
  Body:   {"transcript": "...", "idempotency_key": "<uuid>?"}
  → Estimate (idempotente) · 10/min · 502 si el pipeline falla

POST /v1/retrieval/search
  Header: X-API-Key: $RETRIEVAL_API_KEY
  Body:   {"query_text": "...", "top_k": 10, "distance_threshold": 0.6,
           "sectors": [...], "project_year_min": ..., "chunk_types": [...]}
  → RetrievalResult · 120/min · 200 con low_confidence=true si nada cruza el umbral
```

El `/v1/retrieval/search` autenticado **supersede** el `POST /search` de S08 (sin auth), que se mantiene solo para compatibilidad (Chunking Lab / demos S08).

Ejemplo con httpie:

```bash
http POST :8000/v1/estimate/from-transcript \
  X-API-Key:demo-estimate-key \
  transcript="We need a B2B portal for equipment loans with OAuth and audit trail."

http POST :8000/v1/retrieval/search \
  X-API-Key:demo-retrieval-key \
  query_text="OAuth authentication backend for fintech" top_k:=5 distance_threshold:=0.6
```

### Seguridad y rate limiting

- **Dos API keys independientes** (`RETRIEVAL_API_KEY`, `ESTIMATE_API_KEY`): comparación en tiempo constante (`secrets.compare_digest`). Sin clave configurada → 401 a todo el router.
- **Rate limiting por clave** (slowapi): el bucket es el `X-API-Key`, no la IP. Límites por ruta; 429 con `Retry-After`.
- **Correlación** — middleware `request_id_middleware` asigna `X-Request-ID` por request y lo refleja en la respuesta.

### Variables de entorno nuevas

| Variable | Default | Notas |
|---|---|---|
| `RETRIEVAL_API_KEY` | — | Protege `/v1/retrieval/*`; vacío = router deshabilitado |
| `ESTIMATE_API_KEY` | — | Protege `/v1/estimate/*`; debe coincidir con `estimator-web/.env` |
| `REFORMULATION_MODEL` | `gpt-5-mini` | Destila la transcripción |
| `GENERATION_MODEL` | `gpt-5` | Genera el `Estimate` fundamentado |
| `GENERATION_REASONING_EFFORT` | `high` | Esfuerzo de razonamiento del generador |
| `GENERATION_MAX_TOKENS` | `64000` | Techo reasoning+output (los tokens de razonamiento cuentan) |
| `RETRIEVAL_TOP_K` | `10` | Chunks devueltos por retrieval |
| `RETRIEVAL_DISTANCE_THRESHOLD` | `0.6` | Umbral coseno (menor = más cercano) |
| `MAX_CONTEXT_TOKENS` | `16384` | Presupuesto del bloque `<source>` (tiktoken `cl100k_base`) |
| `IDEMPOTENCY_TTL` | `86400` | TTL de la caché de idempotencia (segundos) |

### Tests del directo

```bash
uv run pytest tests/api/test_security.py tests/api/test_rate_limiting.py \
  tests/api/test_idempotency.py \
  tests/generation/rag/test_estimator.py tests/generation/rag/test_context_assembler.py \
  tests/generation/rag/test_query_reformulator.py tests/generation/rag/test_retriever.py \
  tests/generation/rag/test_validation.py -v
```

Integración con `TestClient`, LLM y embedder mockeados, y store de idempotencia aislado por test.

### Cierre post-directo (Sesión 9 completa)

Material que completa el módulo tras el directo:

**Endpoints por etapas** (`POST /v1/estimate/stages/*`) — el mismo pipeline expuesto stage a stage para el RAG Wizard. Contrato stateless: el llamador persiste la salida de cada etapa y la pasa a la siguiente. Reutilizan las mismas funciones puras que el orquestador.

```
POST /v1/estimate/stages/reformulate  → ReformulationResult   (30/min)
POST /v1/estimate/stages/retrieve     → RetrievalResult       (60/min)
POST /v1/estimate/stages/assemble     → AssembleResult        (60/min)
POST /v1/estimate/stages/generate     → GenerateResult        (15/min)
```

El stage `generate` **no** replica el bucle correctivo del orquestador: devuelve `fabricated_source_ids` y `coherent` para que la UI los muestre como momento didáctico.

**Task corpus task-granular** — el corpus S08 es grueso (un chunk por componente de presupuesto), pero la estimación S9 es módulo→tarea. `scripts/build_task_corpus.py` sintetiza proyectos históricos deterministas (seed), cada uno descompuesto en módulos y tareas con horas; cada tarea es un `BudgetComponent` etiquetado con su `module`, de modo que el chunker estructural emite un chunk por tarea (`chunk_type='historical_task'`, `document_type='historical_task_breakdown'`).

```bash
# Solo generar data/task_corpus.json (revisar antes de ingerir)
docker compose run --rm estimator python scripts/build_task_corpus.py --generate-only

# Generar + ingerir en pgvector
docker compose run --rm estimator python scripts/build_task_corpus.py --ingest

# Borrar el corpus sintético
# DELETE FROM documents WHERE document_type = 'historical_task_breakdown';
```

**Cliente Rails — RAG Wizard** (`estimator-web/`): `Rag::EstimationRunsController` recorre el pipeline en 6 pasos vía `EstimatorAi::RagEstimateClient` (`/v1/estimate/stages/*`). Tabla `estimation_runs` persiste el estado del wizard; paso final de verificación humana del desglose (editor Stimulus de módulos/tareas). Cuarta tarjeta en el dashboard (`home#index`). Las claves API viajan en `X-API-Key` desde `config/initializers/estimator_ai.rb`.

```bash
uv run pytest tests/api/test_estimate_stages.py tests/test_task_corpus_generator.py \
  tests/generation/rag/test_structural_chunker.py -v
```

## Sesión 10 — Pre-work: full-text search en PostgreSQL

Primer paso hacia el retrieval híbrido (vector + léxico) de la Sesión 10: preparar la **rama léxica** en la capa de datos. El corpus de muestra está en inglés; la búsqueda full-text se indexa con la regconfig `'english'`, que debe coincidir con la del lado de consulta (`plainto_tsquery('english', …)`) o Postgres ignora el índice GIN en silencio.

### Migración Alembic `0003_session10_fts`

Nueva revisión encadenada a `0002_session8_pgvector` (`alembic/versions/0003_session10_fts.py`):

```sql
ALTER TABLE chunks
ADD COLUMN content_tsv tsvector
GENERATED ALWAYS AS (to_tsvector('english', content)) STORED;

CREATE INDEX ix_chunks_content_tsv ON chunks USING gin (content_tsv);
```

Piezas:

- **`content_tsv`** — columna **generada y almacenada** (`GENERATED ALWAYS AS … STORED`). Postgres recalcula `to_tsvector('english', content)` en cada insert/update de `content` y persiste el resultado. Sin triggers ni lógica de aplicación: el tsvector no puede desincronizarse del texto.
- **`ix_chunks_content_tsv`** — índice **GIN** sobre esa columna, necesario para que los filtros `@@` y el ranking `ts_rank_cd` sean eficientes.

El `downgrade()` elimina el índice y la columna. Al arrancar con Docker, Alembic aplica `0001` + `0002` + `0003` automáticamente sobre el corpus ya ingerido: los chunks existentes reciben `content_tsv` sin re-ingestar.

### Modelo ORM actualizado

`app/generation/rag/store/models.py` refleja el esquema para que el repositorio pueda referenciar la columna en consultas SQL:

- Atributo `content_tsv` (`TSVECTOR` + `Computed("to_tsvector('english', content)", persisted=True)`): solo lectura desde Python, nunca se escribe en ingesta.
- Índice GIN `ix_chunks_content_tsv` declarado en `__table_args__` de `ChunkRow`.

### Verificación manual

```bash
docker compose up -d
docker compose exec estimator-postgres psql -U estimator -d estimator -c "\d chunks"
# Debe listar content_tsv (generated) e ix_chunks_content_tsv (gin)

docker compose exec estimator-postgres psql -U estimator -d estimator -c "SELECT id, content_tsv FROM chunks LIMIT 3;"
```

### comparativa (retrieval híbrido + reranking)

Material que completa el módulo sobre la base FTS del pre-work. El retrieval del pipeline RAG deja de ser solo k-NN denso: una única entrada `retrieve()` (`app/generation/rag/retrieval/pipeline.py`) compone **cuatro configuraciones** detrás de dos conmutadores (`search_mode`, `rerank`) que el llamador resuelve por precedencia **param de request → override runtime → default `.env`**.

| Config | `search_mode` | `rerank` | Comportamiento |
|--------|---------------|----------|----------------|
| **A** (baseline S9) | `vector` | `false` | k-NN denso, top-k |
| **B** | `hybrid` | `false` | ramas densa + léxica fusionadas con RRF |
| **C** | `vector` | `true` | recall denso amplio → cross-encoder → top-n |
| **D** | `hybrid` | `true` | recall híbrido amplio → cross-encoder → top-n |

**Rama léxica y fusión RRF.** `ChunkStore.search_lexical()` (`store/repository.py`) implementa la búsqueda por palabras clave sobre `content_tsv`: normaliza la query con `plainto_tsquery('english', …)` y **convierte el `&` (AND) en `|` (OR)** para que un chunk haga match por **cualquier** término relevante (`plainto_tsquery` con AND no devuelve nada ante descripciones largas: ningún chunk contiene *todos* los términos); `@@` filtra y `ts_rank_cd` rankea (los chunks que comparten más términos suben), con los **mismos filtros estructurales** que la rama densa (extraídos a `_structural_filters`). `retrieval/fusion.py` fusiona ambas ramas en un único ranking con **Reciprocal Rank Fusion**: una función pura `score = Σ 1/(k + rank)` que solo confía en el orden de cada rama (la densa puntúa por distancia coseno, la léxica por `ts_rank_cd`: escalas incomparables). La constante de suavizado es **`RRF_K=60`** (Cormack et al.). El modo de búsqueda se pasa como parámetro del endpoint (`RetrievalRequest.search_mode`).

**Integración del reranker.** `retrieval/reranker.py` conecta un cross-encoder multilingüe (`cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`, ES+EN, CPU) siguiendo el patrón **recall-then-rerank**: recuperación amplia (`RETRIEVAL_RECALL_TOP_K=50`) → reordenación fina → mejores `RERANK_TOP_N=5`. El modelo carga **perezosamente** en el primer rerank (nunca en import/arranque), guardado por lock. El reranking se activa/desactiva **sin tocar código**: por request (`rerank`), en caliente (`PUT /api/v1/config/retrieval`) o por `.env` (`RERANKER_ENABLED`). `scripts/verify_reranker.py` fuerza la carga para warmup.

**Conmutadores en caliente.** `RuntimeRetrievalConfig` (`app/foundation/llm/runtime_retrieval_config.py`, hash Redis `estimator:runtime_retrieval`, separado del de modelos) expone `search_mode` y `rerank`; lecturas degradan al default si Redis cae, escrituras re-lanzan (503). El pipeline RAG S9 (`estimator.py`) hereda estos toggles sin cambiar su contrato.

```
GET  /api/v1/config/retrieval  → snapshot {search_mode, rerank} (effective/default/overridden) + reranker_model
PUT  /api/v1/config/retrieval  → override parcial (null = reset al default); 422 si search_mode inválido, 503 si Redis cae
POST /v1/retrieval/search      → acepta search_mode: "vector"|"hybrid"|null y rerank: bool|null (null = runtime/settings)
```

**Golden set y medición.** `evals/golden_retrieval.json` recoge **5 consultas representativas** del dominio (descripciones de proyectos a estimar: banca/pagos, telemedicina, e-commerce, IoT industrial, logística), cada una anotada a mano con los `relevant_budget_ids` realmente relevantes del corpus. `scripts/eval_retrieval_s10.py` ejecuta las **cuatro configuraciones de forma reproducible** (las fija explícitamente, sin leer el runtime) y reporta **precisión sobre los 5 primeros (precision@5)** y **latencia** de la consulta (embeddings excluidos de los tiempos; reranker precalentado; umbral permisivo para medir ranking, no el soft-fail), recogiéndolo en una tabla comparativa.

```bash
# test manual
docker exec estimator-postgres psql -U estimator -d estimator -t -c "SELECT count(*) FROM chunks WHERE content_tsv @@ replace(plainto_tsquery('english','We are scoping a secure payments platform for a retail bank real-time payment authorization and capture streaming fraud detection')::text, ' & ', ' | ')::tsquery;"

# verify_reranker.py solo necesita la dependencia sentence-transformers (no toca BD ni API).
docker compose run --rm estimator python scripts/verify_reranker.py      # warmup del cross-encoder

# eval_retrieval_s10.py habla directamente con Postgres (no con la API): necesita el corpus
# base ya ingerido (lo ingiere scripts/query_examples.py) y OPENAI_API_KEY para las queries.
docker compose run --rm estimator python scripts/eval_retrieval_s10.py   # precision@5 + latencia (4 configs)
```

> Nota: el reranking añade la dependencia `sentence-transformers` (arrastra torch; pesos en primer uso). Tras un `git pull` ejecuta `uv sync` o recrea el contenedor antes de usar las rutas con `rerank`.

**Resultados observados.** Se midió la misma batería en dos momentos. Las cifras varían con el corpus y el hardware.

**Prueba 1 — corpus base (60 chunks `budget_component` de `budgets_sample.json`), golden set original ** (sólo `BUD-*` anotados - guardado en `Golden_retrieval_BUD.json`):

| Config | Búsqueda | Reranking | precision@5 | latencia/query |
|--------|----------|-----------|-------------|----------------|
| A | vector | no | 0.92 | ~12 ms |
| B | híbrida | no | 0.92 | ~6 ms |
| C | vector | sí | 0.76 | ~2.8 s |
| D | híbrida | sí | 0.76 | ~3.2 s |

**Prueba 2 — corpus completo (282 chunks: 60 `budget_component` + 222 `historical_task` de `task_corpus.json`), golden set reanotado** (añadidos los `TASK-*` relevantes por sector):

| Config | Búsqueda | Reranking | precision@5 | latencia/query |
|--------|----------|-----------|-------------|----------------|
| A | vector | no | 1.00 | ~15 ms |
| B | híbrida | no | 1.00 | ~17 ms |
| C | vector | sí | 0.92 | ~1.5 s |
| D | híbrida | sí | 0.92 | ~3.5 s |

> El golden set de la prueba 2 está anotado por dominio: cada proyecto `TASK-*` es una plataforma sectorial completa, así que se marca relevante cuando su `client_sector` coincide con la consulta (finance↔banca, healthcare↔telemedicina, ecommerce↔e-commerce, industrial↔IoT). La consulta de logística no tiene sector equivalente en el corpus de tareas, por lo que sólo conserva sus `BUD-*`.

> ⚠️ Importante: al cargar los documentos `TASK-*` SIN reanotar el golden set, la precisión@5 cayó aparentemente a ~0.64 (A/B) / ~0.48 (C/D). No era una regresión del sistema, sino del *measurement*: los `TASK-*` recuperados eran relevantes por contenido pero contaban como fallos porque no figuraban en `relevant_budget_ids`. La prueba 2 corrige la anotación y restablece la lectura correcta. Es el recordatorio clásico: un golden set hay que mantenerlo al día con el corpus.

Lecturas de esta caracterización (no son bugs, son el motivo de medir):

- **El reranking empeora la precisión en ambas pruebas** (0.76 vs 0.92 en la 1; 0.92 vs 1.00 en la 2) y añade ~1.5–3.5 s por consulta (cross-encoder en CPU). Cuando el bi-encoder + RRF ya separan limpiamente los dominios, el reranker genérico multilingüe reordena introduciendo ruido (mezcla componentes/`TASK-*` de otros sectores arriba). Justifica el default `RERANKER_ENABLED=false` y que el toggle sea por request/runtime: se activa solo donde aporta.
- **La rama léxica no cambia la precisión@5 frente a la vectorial** (A ≡ B en aciertos en ambas pruebas), pero aporta cobertura por término exacto: en el corpus completo la rama contribuye 50 hits por query (`lexical_hits=50`), mientras que con el `plainto_tsquery` AND original contribuía 0 ante descripciones largas. Su valor real se nota en queries con vocabulario técnico raro que el embedding no captura.
- **El único punto que baja con reranking en la prueba 2 es `gq-05` (logística) a 0.60**: al no haber `TASK-*` de logística, el reranker promueve tareas de otros sectores por solapamiento de vocabulario (almacén, rutas, capacidad). Es una ambigüedad real del corpus, útil como caso de discusión.
- Las latencias **excluyen el embedding de la query** (round-trip a OpenAI, ajeno a la config) y usan un umbral de distancia permisivo: se mide el *ranking*, no el soft-fail.

```bash
uv run pytest tests/generation/rag/test_rrf_fusion.py tests/generation/rag/test_reranker.py \
  tests/generation/rag/test_hybrid_retrieve.py tests/test_runtime_retrieval_config.py \
  tests/test_retrieval_config_endpoint.py -v
```

---

> Este proyecto forma parte del **Master en AI Engineering** y es la base sobre la que se construye en directo el resto de la Sesión 04 (output estructurado, guardrails, cache semántico) y de la Sesión 05 (compresión avanzada de memoria con anclas, tier dinámico, patrón Actor-Critic-Boss).
