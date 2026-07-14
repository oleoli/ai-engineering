# Sesión 13 — Grafo explícito de estimación con LangGraph

Este documento resume lo implementado en el servicio IA (`estimator/`), cómo lanzarlo y los problemas encontrados durante el desarrollo y las pruebas.

## Objetivo

Sustituir el bucle agentico de la S12 por un **grafo explícito** con LangGraph. Desde fuera, el contrato no cambia: entra una transcripción y sale una estimación estructurada con su estado de salida. El backend de negocio no sabe qué hay debajo; el grafo vive **dentro del servicio IA**.

Se reutiliza la lógica RAG de las sesiones 9–12 (`reformulate_query`, `generate_structure`, `retrieve`, `generate_estimate`, `verify_citations`, etc.) envuelta en nodos del grafo.

## Qué se ha implementado

### Dependencias nuevas

Instaladas con `uv`:

- `langgraph`
- `langgraph-checkpoint-postgres`
- `logfire[asyncpg,fastapi,httpx]`

### Estructura del grafo

Ubicación: `app/domain/graph/` (conductor de dominio, respeta `ARCHITECTURE.md`).

| Fichero | Responsabilidad |
|---------|-----------------|
| `state.py` | Estado tipado `EstimationState` con reducers acumuladores (`budget_matches`, `errors`) |
| `nodes.py` | Cinco nodos async, cada uno con `logfire.span("node: …")` |
| `build.py` | Cableado secuencial + arista condicional al final |
| `schemas.py` | `GraphEstimateResponse { estimate, status }` |

### Flujo de nodos (secuencial)

```
START → extract_requirements → classify_components → search_budgets
      → generate_estimate → validate_and_consolidate → END
```

| Nodo | Qué hace | Lógica reutilizada |
|------|----------|-------------------|
| `extract_requirements` | Transcripción → lista de requisitos + brief estructurado | `reformulate_query` |
| `classify_components` | Agrupa requisitos en componentes con categoría | `generate_structure` → cada `WorkModule` es un `Component` |
| `search_budgets` | Por cada componente, recupera presupuestos de referencia (secuencial) | `embedder` + `retrieve(collection=BUDGET, chunk_types=["historical_task"])` |
| `generate_estimate` | Consolida presupuestos en una estimación | `build_context_block` + `generate_estimate` |
| `validate_and_consolidate` | Revisa citas y coherencia; fija el `status` | `verify_citations_for_chunks` + `check_coherence` |

### Nivel 2 — Persistencia y observabilidad

- **Checkpointer:** `AsyncPostgresSaver` sobre el Postgres del proyecto (`estimator-postgres`, pgvector). Crea sus propias tablas en el primer `setup()` y convive con las de embeddings.
- **`thread_id`:** el identificador de la estimación (`idempotency_key` del payload o un `uuid4` nuevo).
- **Observabilidad:** **Logfire** con un span por nodo. Variables opcionales en `.env`: `LOGFIRE_TOKEN`, `LOGFIRE_SERVICE_NAME` (default `estimator`). Sin token, los spans se imprimen en la **consola** durante la ejecución (no se persisten en un fichero ni se envían al cloud).

### Nivel 3 — Primera arista condicional

Tras `validate_and_consolidate`, una arista condicional enruta a `END` con:

- `status = "validated"` si pasó la validación
- `status = "needs_review"` si hay citas colgantes, incoherencia u otros errores

### Endpoint HTTP

Nuevo endpoint (el original **no se ha tocado**):

```
POST /v1/estimate/graph/from-transcript
```

- **Auth:** `X-API-Key: $ESTIMATE_API_KEY`
- **Rate limit:** 10/min
- **Request:** mismo `EstimateRequest` que S9 (`transcript` + `idempotency_key` opcional)
- **Response:** `{ "estimate": Estimate, "status": "validated" | "needs_review" }`

El endpoint clásico `POST /v1/estimate/from-transcript` sigue devolviendo solo `Estimate` (con `confidence`, no `status`).

### Script CLI

`scripts/run_graph_s13.py` — ejecuta el grafo sobre un fichero de transcripción (por defecto `exercises/session-12/sample_transcript_complex.txt`) y escribe el JSON de salida.

### Tests

`tests/domain/graph/test_graph_wiring.py` — tests network-free con `MemorySaver` que verifican el cableado y las dos ramas de `status`.

---

## Cómo lanzarlo

### Requisitos previos

1. Stack levantado: Postgres (`estimator-postgres`), Redis y el servicio `estimator`.
2. `.env` con al menos `OPENAI_API_KEY` (y `ESTIMATE_API_KEY` para el endpoint HTTP).
3. Corpus histórico ingerido (`scripts/build_task_corpus.py --ingest`) para que `search_budgets` encuentre analogías.
4. (Opcional) `LOGFIRE_TOKEN` para exportar trazas a Logfire Cloud. El token correcto está en `estimator/.logfire/logfire_credentials.json` tras `logfire projects use <nombre>`.

### Opción A — Dentro del contenedor (recomendado)

Desde la raíz del monorepo o desde `estimator/`:

```bash
docker compose up -d --build
# Si acabas de añadir el volumen exercises/, recrea el contenedor:
docker compose up -d --force-recreate estimator

docker compose exec estimator python scripts/run_graph_s13.py \
  exercises/session-12/sample_transcript_complex.txt
```

Guardar la salida en un fichero:

```bash
docker compose exec estimator python scripts/run_graph_s13.py \
  exercises/session-12/sample_transcript_complex.txt \
  --out exercises/session-13/estimacio.json
```

### Opción B — Fuera del contenedor (host con `uv`)

También es posible, pero el `.env` debe apuntar a **localhost**, no a hostnames de Docker:

```env
DATABASE_URL=postgresql+psycopg://estimator:estimator@localhost:5433/estimator
REDIS_URL=redis://localhost:6379
```

```bash
cd estimator
uv sync
uv run python scripts/run_graph_s13.py \
  exercises/session-12/sample_transcript_complex.txt
```

Postgres y Redis deben estar levantados (`docker compose up -d estimator-postgres redis` como mínimo).

### Endpoint HTTP

Con el servicio en marcha (`http://localhost:8000`):

```bash
http POST :8000/v1/estimate/graph/from-transcript \
  X-API-Key:demo-estimate-key \
  transcript:="$(cat exercises/session-12/sample_transcript_complex.txt)"
```

O con curl:

```bash
curl -X POST http://localhost:8000/v1/estimate/graph/from-transcript \
  -H "X-API-Key: demo-estimate-key" \
  -H "Content-Type: application/json" \
  -d "{\"transcript\": \"$(cat exercises/session-12/sample_transcript_complex.txt | jq -Rs .)\"}"
```

### Tests unitarios

```bash
cd estimator
uv run pytest tests/domain/graph/test_graph_wiring.py -v
```

---

## Problemas encontrados

### 1. `exercises/` no montado en el contenedor

**Síntoma:** al ejecutar dentro del contenedor:

```
FileNotFoundError: ... '/app/exercises/session-12/sample_transcript_complex.txt'
```

**Causa:** `docker-compose.yml` montaba `app/`, `scripts/`, `data/`, etc., pero **no** `exercises/`. El fichero existía en el host pero no era visible dentro del contenedor.

**Solución:** añadir el bind-mount `./exercises:/app/exercises` y recrear el contenedor:

```bash
docker compose up -d --force-recreate estimator
```

### 2. Ejecución fuera vs dentro de Docker

**Síntoma:** errores de resolución DNS (`getaddrinfo failed`) al conectar a `redis:6379` o `postgres:5432` desde el host.

**Causa:** el `.env.example` usa hostnames de la red interna de Compose (`postgres`, `redis`), que solo resuelven **dentro** de los contenedores.

**Solución:** fuera de Docker, usar `localhost:5433` para Postgres y `localhost:6379` para Redis. Dentro del contenedor, Compose sobrescribe las URLs automáticamente.

### 3. Extras de Logfire no instalados de primeras

**Síntoma:** al importar `app.main` (p. ej. en tests):

```
RuntimeError: logfire.instrument_fastapi() requires opentelemetry-instrumentation-fastapi
```

**Causa:** `logfire` base no incluye los instrumentadores de FastAPI, asyncpg y httpx.

**Solución:** instalar con extras:

```bash
uv add "logfire[fastapi,asyncpg,httpx]"
```

### 4. Contrato de respuesta: `status` vs `confidence`

El esquema RAG existente (`Estimate`) usa el campo **`confidence`** (`high` | `medium` | `low` | `insufficient`). El grafo S13 expone además un **`status`** de workflow (`validated` | `needs_review`) en un wrapper `GraphEstimateResponse`. Son conceptos distintos: `status` indica si la validación del grafo pasó; `confidence` sigue siendo la confianza del modelo sobre la estimación.

### 5. Sin token Logfire: ¿dónde queda la traza?

**Síntoma:** tras ejecutar `run_graph_s13.py` solo aparece el JSON de la estimación; no hay fichero de traza guardado.

**Causa:** con `send_to_logfire="if-token-present"` y sin `LOGFIRE_TOKEN`, Logfire **no envía nada al cloud** ni escribe un fichero local. Los spans solo se ven en **stdout/stderr** mientras corre el proceso. El checkpointer Postgres guarda el **estado del grafo** (para reanudar), no los spans de observabilidad.

**Solución:**

- **Consola (sin cuenta):** capturar la salida al ejecutar:

  ```powershell
  docker compose exec estimator python scripts/run_graph_s13.py `
    exercises/session-12/sample_transcript_complex.txt 2>&1 `
    | Select-String "graph_run|node:"
  ```

- **Logfire Cloud (recomendado):** configurar `LOGFIRE_TOKEN` en `.env` y ver la traza en el dashboard del proyecto.

### 6. `LOGFIRE_TOKEN` en `.env` pero error 401 (Invalid token)

**Síntoma:**

```
Logfire API returned status code 401. Detail: Invalid token
Failed to export span batch code: 401, reason: Unauthorized
```

**Causa:** el token del `.env` **no coincide** con el write token del proyecto Logfire. Suele pasar si:

- se copia un token viejo, incomento o de otro proyecto;
- se confunde el token de autenticación CLI (`logfire auth`) con el **write token** del proyecto;
- el proyecto está en la región **EU** (`logfire-eu.pydantic.dev`) pero el token no es el generado para ese proyecto.

El contenedor **no** lee `~/.logfire/` ni `estimator/.logfire/`; solo carga variables desde `.env` vía `env_file`.

**Solución:**

1. En el host, dentro de `estimator/`:

   ```bash
   uv run logfire auth          # si hace falta (región EU: logfire --region eu auth)
   uv run logfire projects use estimator-s13
   ```

2. Copiar el `"token"` de `estimator/.logfire/logfire_credentials.json` al `.env`:

   ```env
   LOGFIRE_TOKEN=pylf_v1_eu_...
   LOGFIRE_SERVICE_NAME=estimator
   ```

   Alternativa: crear un **New write token** en Settings del proyecto en https://logfire-eu.pydantic.dev.

3. Recrear el contenedor para que cargue el `.env` nuevo:

   ```bash
   docker compose up -d --force-recreate estimator
   ```

4. Verificar que el token llega al contenedor:

   ```bash
   docker compose exec estimator printenv LOGFIRE_TOKEN
   ```

### 7. ¿Hay que reconstruir la imagen para volver a ejecutar el script?

**Síntoma:** duda sobre si hace falta `docker compose up --build` cada vez que se relanza `run_graph_s13.py`.

**Causa:** confusión entre cambios de código/dependencias y simple re-ejecución.

**Solución:**

| Acción | ¿Rebuild (`--build`)? | ¿Recrear (`--force-recreate`)? |
|--------|----------------------|--------------------------------|
| Volver a ejecutar el script | No | No |
| Cambios en `app/`, `scripts/`, `exercises/` (bind-mount) | No | No |
| Cambios en `.env` (`LOGFIRE_TOKEN`, claves API…) | No | **Sí** |
| Nuevas dependencias en `pyproject.toml` / `uv.lock` | **Sí** | Opcional |

Comando habitual:

```bash
docker compose exec estimator python scripts/run_graph_s13.py \
  exercises/session-12/sample_transcript_complex.txt
```

### 8. `search_budgets` devuelve 0 resultados dentro del contenedor

**Síntoma:** la traza muestra los 5 nodos, pero la estimación sale con `confidence: "insufficient"` y logs como `rag_retrieve_done … results=0`.

**Causa:** el corpus histórico (`historical_task`) no está ingerido en el Postgres del contenedor (`estimator-postgres`). Es independiente del token Logfire: el grafo funciona, pero no hay presupuestos de referencia.

**Solución:** ingerir el corpus dentro del mismo entorno Docker:

```bash
docker compose exec estimator python scripts/build_task_corpus.py --ingest
```

Comprobar que hay chunks antes de relanzar el grafo.

---

## Ejemplo de traza Logfire

Espacio reservado para documentar un ejemplo real de la traza S13 (captura, enlace al proyecto Logfire o extracto de consola).

<!-- TODO: pegar aquí el ejemplo del entregable -->

**Qué debería verse** en una ejecución correcta:

```
graph_run
  node: extract_requirements
    POST api.openai.com/v1/chat/completions
  node: classify_components
    POST api.openai.com/v1/chat/completions
  node: search_budgets
    POST api.openai.com/v1/embeddings
    BEGIN; … (consultas Postgres)
  node: generate_estimate
    POST api.openai.com/v1/chat/completions
  node: validate_and_consolidate
```

**Enlace al proyecto (EU):**

```
https://logfire-eu.pydantic.dev/ole-oli/estimator-s13
```

**Captura / extracto:**

_(añadir aquí screenshot o pegado de la traza `graph_run` con los 5 spans hijos)_
![Traza Logfire S13](traza%20logfire.png)


---

## Fuera de alcance (pendiente para el directo)

- Paralelización de `search_budgets` por componente (Send API de LangGraph)
- Reintentos con backoff, nodo de fallback, timeouts, circuit breakers
- Intervención humana (HITL) con `interrupt()` en la validación
- Optimización a partir de la lectura fina de trazas Logfire

---

## Referencias rápidas

| Recurso | Ruta |
|---------|------|
| Grafo | `app/domain/graph/` |
| Router HTTP | `app/api/routers/estimate_graph.py` |
| Startup (checkpointer + Logfire) | `app/main.py` |
| DSN LangGraph | `app/foundation/persistence/database.py` → `langgraph_conn_string()` |
| Script CLI | `scripts/run_graph_s13.py` |
| Transcripción de prueba | `exercises/session-12/sample_transcript_complex.txt` |
| Ejemplo de salida (estimación) | `exercises/session-13/estimacio.json` |
| Credenciales Logfire (local) | `estimator/.logfire/logfire_credentials.json` |
| Traza S12 (referencia, no S13) | `exercises/session-12/example_trace_complex.txt` |
