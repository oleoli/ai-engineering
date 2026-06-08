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

Primer paso hacia la búsqueda semántica: convertir presupuestos históricos (JSON) en vectores. El módulo nuevo vive en `app/generation/rag/` y expone un único endpoint. **No se persiste nada todavía** — los vectores se generan en memoria y se devuelven por HTTP; la persistencia en pgvector entra en la Sesión 08.

Piezas:

- `chunker.py` (`JSONStructuralChunker`) — chunking **estructural**: un componente del presupuesto = un chunk. A cada chunk se le antepone una cabecera de contexto del presupuesto padre (proyecto, sector, tecnología) para que no pierda la pista de a quién pertenece. Cuenta tokens con `tiktoken`.
- `embedder.py` (`OpenAIEmbedder`) — invoca `text-embedding-3-small` (1536 dims) en **batches** de 100, con reintento exponencial (1s/2s/4s) ante `RateLimitError` y logging por batch.
- `router.py` — orquesta `chunk → embed → stats`.

### Endpoint nuevo

```
POST /embeddings/ingest
  Input  (IngestRequest):  {"budgets": [ <Budget>, ... ]}
  Output (IngestResponse): {"chunks": [ <EmbeddedChunk>, ... ], "stats": {...}}
  200 OK · 422 validación Pydantic · 500 error de la API de embeddings (mensaje genérico, detalle en logs)
```

Aparece en Swagger (`http://localhost:8000/docs`) y se puede invocar desde ahí con el sample de datos.

Desde línea de comandos, alimentando los 15 presupuestos de ejemplo (`data/budgets_sample.json` es un array; el endpoint espera `{"budgets": [...]}`):

```bash
# httpie (envuelve el array en el campo "budgets")
http POST :8000/embeddings/ingest budgets:=@data/budgets_sample.json

# curl equivalente
curl -s -X POST http://localhost:8000/embeddings/ingest \
  -H 'Content-Type: application/json' \
  -d "{\"budgets\": $(cat data/budgets_sample.json)}" | python -m json.tool | head -40
```

Con el sample: 15 presupuestos → 52 chunks → ~4.1k tokens → coste estimado ~$0.00008.

### Script CLI `compare.py`

Sanity check de búsqueda semántica: invoca `POST /embeddings/search` con cinco queries representativas (match directo, reformulación semántica, dominio distinto, consulta ambigua y consulta muy específica) e imprime el top-5 de cada una con `chunk_id`, `distance`, `chunk_type` y un preview del `content`.

Requiere el servicio levantado y el corpus ya ingestado vía `POST /embeddings/ingest`.

```bash
# Fuera del contenedor (desde estimator/):
uv run python scripts/compare.py

# Dentro del contenedor (scripts/ está bind-montado en docker-compose.yml):
docker compose exec estimator python scripts/compare.py

# Opcional: cambiar base URL o número de resultados
uv run python scripts/compare.py --base-url http://localhost:8000 --k 5
```

La base URL se lee de `ESTIMATOR_API_BASE_URL` (default `http://localhost:8000`).

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

Las estrategias `semantic`, `propositional` y `contextual_retrieval` llaman a APIs externas durante la ingesta (necesitan `OPENAI_API_KEY` / `ANTHROPIC_API_KEY`) y reportan su coste en `chunking_done`. `sentence_window` usa NLTK (`punkt`/`punkt_tab`, descarga perezosa). Nada se persiste todavía — la persistencia vectorial con pgvector entra en la **Sesión 08**.

### Dependencias y scope

- Dependencias del pre-ejercicio: `tiktoken>=0.7.0` (`openai` ya estaba desde Sesión 01).
- Dependencias de la sesión en vivo: `langchain-text-splitters`, `langchain-experimental`, `langchain-openai`, `nltk` (`anthropic` ya estaba). No se añade numpy/scikit-learn ni `sentence-transformers`; la coseno y los percentiles son stdlib.
- **Late chunking** se trata como concepto en el directo (no hay código ejecutable: requiere modelos con token-level embeddings que no son el del proyecto).
- **Fuera de scope** → **Sesión 08**: persistencia vectorial (pgvector), búsqueda semántica / retrieval real y métricas formales de retrieval (recall@k, NDCG).
- El guion del directo está en `guides/session-7-live-guide.md` (git-ignored, material de instructor).

## Sesión 8 (pre-ejercicio) — Persistencia pgvector y búsqueda semántica

Primer paso hacia el RAG persistente de la Sesión 08 completa. Reutiliza el chunker estructural y el `OpenAIEmbedder` de la Sesión 07, pero **cambia el contrato de ingesta**: `POST /embeddings/ingest` deja de devolver vectores en memoria y pasa a escribirlos en Postgres (`documents` + `chunks`). Se añade `POST /embeddings/search` para recuperar los top-k chunks por distancia coseno. `POST /embeddings/compare` sigue siendo in-memory (sin tocar la BD).

### Pasos implementados

1. **Migración Alembic `0002_session8_pre`** — activa la extensión `vector` y crea las tablas `documents` y `chunks` (columna `embedding vector(1536)`, índice GIN sobre `metadata`). Cadena: `0001_session6_initial` → `0002_session8_pre`.

2. **`alembic/env.py` + `alembic.ini`** — la URL sale de la variable de entorno `DATABASE_URL` (fallback a `Settings`). Antes de ejecutar migraciones se registra `pgvector.sqlalchemy.Vector` en el dialecto para que `alembic check` y el autogenerate reconozcan columnas `vector` sin generar diffs inconsistentes.

3. **Capa de persistencia dual** (`app/foundation/persistence/database.py`):
   - **Sync + `psycopg`** — Sesión 6 (`BackgroundTasks` de ingesta batch, Alembic).
   - **Async + `asyncpg`** — Sesión 8 (`get_async_session`, derivado automáticamente de `DATABASE_URL` sustituyendo `+psycopg` por `+asyncpg`).

4. **Modelos ORM** (`app/generation/rag/store/models.py`) — `DocumentRow` y `ChunkRow` mapean las tablas nuevas. El paquete `app/generation/rag/store/` expone también `search_similar_chunks`.

5. **Endpoints** (`app/api/embeddings.py`):
   - `POST /embeddings/ingest` — chunk → embed (en threadpool) → persistencia en una transacción. Idempotencia por `source_path`: reingestar el mismo path devuelve **409**.
   - `POST /embeddings/search` — embed de la query + ranking por `cosine_distance` sobre chunks persistidos.

### Decisiones de diseño

1. **Normalización 1:N (`documents` → `chunks`).** La relación presupuesto–fragmento es uno-a-muchos: un documento origen genera varios chunks. Desnormalizar todo en una sola tabla violaría la **3FN** (atributos del documento repetidos en cada fila de chunk), introduciría **anomalías de actualización** y dificultaría garantizar **integridad referencial**. El modelo relacional estándar es entidad padre + entidad hija con **clave foránea** (`chunks.document_id → documents.id`) y **`ON DELETE CASCADE`**, de modo que la eliminación del padre propaga de forma consistente a los hijos sin huérfanos.

2. **Esquema híbrido: columnas tipadas + JSONB.** Los atributos con cardinalidad y tipo conocidos (`document_type`, `chunk_type`, timestamps) van en columnas SQL con restricciones `NOT NULL` donde aplica: el motor puede indexarlos, validarlos y documentarlos en el catálogo. Los atributos extensibles —los que el pipeline de chunking puede ir añadiendo— se delegan a **`metadata JSONB`**, patrón habitual cuando el esquema evoluciona más rápido que las migraciones. El **índice GIN** sobre JSONB habilita predicados por clave (`@>`, `?`, `->>`) sin alterar el DDL cada vez que aparece un nuevo campo de metadata.

3. **Dimensión fija del tipo `vector(1536)`.** En pgvector la dimensionalidad es parte de la definición de columna, no un valor runtime. Fijarla en **1536** alinea el esquema con el modelo de embeddings del proyecto (`text-embedding-3-small`). Cambiar esa dimensión posteriormente exige **migración destructiva** y **re-indexación completa** del corpus; por eso se trata como invariante de diseño, no como configuración dinámica.

4. **`embedding` nullable (columna opcional en fase de ingesta).** A nivel de restricciones, `NULL` distingue «chunk persistido sin vector aún» de «chunk listo para búsqueda». Permite un patrón con **pipelines por fases** (persistir texto → calcular embedding → actualizar fila) frente al commit atómico que usa este ejercicio. El endpoint actual escribe texto y vector en la misma transacción; la nulabilidad reserva el esquema para **ingesta desacoplada** en fases posteriores sin otra migración.

### Infraestructura

Levantar Postgres (pgvector) y el servicio:

```bash
# Desde estimator/ (aislado) o desde ai-engineering/ (monorepo)
docker compose up -d estimator-postgres estimator
```

Al arrancar, el contenedor `estimator` ejecuta `alembic upgrade head` antes de uvicorn. Postgres del estimator escucha en el host en el puerto **5433** (`estimator-postgres`, usuario `estimator`); no confundir con el `postgres` de Rails (puerto 5432, usuario `postgres`).

Comprobar migraciones y esquema:

```bash
docker compose exec estimator alembic current
# Debe mostrar 0002_session8_pre (head)

docker compose exec estimator alembic check
# Sin drift entre modelos y BD

docker compose exec estimator-postgres psql -U estimator -d estimator -c "\dt"
docker compose exec estimator-postgres psql -U estimator -d estimator \
  -c "SELECT extname FROM pg_extension WHERE extname = 'vector';"
```

Si el contenedor falla con `Can't locate revision identified by '0002_session8_pgvector'`, el volumen de Postgres conserva una revisión de otra rama. Opciones: borrar el volumen (`docker volume rm …_estimator_postgres_data`) o alinear con `alembic stamp`.

### Probar los endpoints

Necesitas `OPENAI_API_KEY` en `.env` (el embedder llama a `text-embedding-3-small`).

**1. Ingestar un presupuesto** — el body lleva un único `Budget` (no un array). Construye un JSON con `source_path`, `document_type` y `content` (un elemento de `data/budgets_sample.json`):

```bash
# Generar el payload (Linux/macOS o PowerShell con Python en PATH)
python -c "
import json
budget = json.load(open('data/budgets_sample.json'))[0]
payload = {
    'source_path': 'data/budgets_sample.json#BUD-2024-001',
    'document_type': 'historical_budget',
    'content': budget,
}
json.dump(payload, open('ingest_payload.json', 'w'))
"

# Enviar
curl -s -X POST http://localhost:8000/embeddings/ingest \
  -H 'Content-Type: application/json' \
  -d @ingest_payload.json | python -m json.tool

# Equivalente con httpie
http POST :8000/embeddings/ingest < ingest_payload.json
```

Respuesta esperada (200):

```json
{
  "document_id": 1,
  "chunks_created": 4,
  "embedding_dimension": 1536,
  "ingestion_time_ms": 850
}
```

Repetir la misma petición devuelve **409** con `document_id` del registro existente.

**2. Búsqueda semántica** — solo devuelve chunks ya persistidos por `/ingest`:

```bash
http POST :8000/embeddings/search \
  query="OAuth 2.0 authentication backend for fintech mobile banking" \
  k:=3
```

Respuesta esperada (200): `results[]` con `chunk_id`, `content`, `distance` (coseno; menor = más similar) y `metadata`.

**3. Verificar filas en Postgres**:

```bash
docker compose exec estimator-postgres psql -U estimator -d estimator \
  -c "SELECT id, source_path FROM documents;"
docker compose exec estimator-postgres psql -U estimator -d estimator \
  -c "SELECT id, document_id, chunk_type, left(content, 60) AS preview FROM chunks LIMIT 5;"
```

**4. Chunking Lab sin persistir** — `/embeddings/compare` sigue igual que en Sesión 07 (estrategias en memoria, útil para el cliente Rails).

### Tests automatizados

Todavía no hay tests de integración con Postgres real para esta fase. La batería existente sigue pasando sin tocar la API de embeddings:

```bash
uv run pytest
```

La validación de la sesión 8 pre es manual: migración aplicada, ingest 200 → search con resultados coherentes, re-ingest 409, y `alembic check` limpio.

### Dependencias y scope

- Nuevas / relevantes: `asyncpg`, `pgvector` (además de `psycopg` y `sqlalchemy` de Sesión 6).
- **Dentro de scope (pre)**: tablas pgvector, ingest persistente, search standalone.
- **Fuera de scope (Sesión 08 completa)**: índice HNSW, integración del retriever en `estimate()`, métricas formales de retrieval (recall@k, NDCG), cliente Rails para search.

---

> Este proyecto forma parte del **Master en AI Engineering** y es la base sobre la que se construye en directo el resto de la Sesión 04 (output estructurado, guardrails, cache semántico) y de la Sesión 05 (compresión avanzada de memoria con anclas, tier dinámico, patrón Actor-Critic-Boss).

