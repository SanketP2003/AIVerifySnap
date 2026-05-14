<!-- Copilot instructions for AI coding agents working in this repo -->
# Copilot instructions — AIVerifySnap repository

Purpose
- Provide concise, actionable guidance so an AI assistant can be productive immediately.

Big picture
- This repository contains three cooperating services: a Spring Boot Java backend ([backend/pom.xml](backend/pom.xml)), a Next.js frontend ([frontend/package.json](frontend/package.json)), and a Python ML microservice ([ml_service/main.py](ml_service/main.py)).
- The backend exposes REST endpoints (see [backend/src/main/java/com/backend/aiverifysnap/controller](backend/src/main/java/com/backend/aiverifysnap/controller)) and persists detection records to PostgreSQL via JPA.
- The backend calls the ML microservice at the configured `ml.service.url` (default in [backend/src/main/resources/application-dev.yml](backend/src/main/resources/application-dev.yml)) to perform image analysis.

Key developer workflows (commands to run)
- Backend (Maven wrapper):
  - Build: `./mvnw package` (Linux/macOS) or `mvnw.cmd package` (Windows)
  - Run locally: `java -jar target/*.jar` or use the Dockerfile ([backend/Dockerfile](backend/Dockerfile)) which sets `PORT=8080`.
- Frontend (Next.js):
  - Dev: `npm run dev` (serves on :3000)
  - Build: `npm run build` and `npm run start` for production
- ML service (FastAPI):
  - Install: `pip install -r ml_service/requirements.txt`
  - Run locally: `python -m uvicorn main:app --reload --port 8000` (Dockerfile exposes 8000)

Project-specific conventions & patterns
- Package layout: follow `controller`, `service`, `repository`, `model`, `dto` packages under `com.backend.aiverifysnap`. Use existing classes as examples (e.g. [DetectionController](backend/src/main/java/com/backend/aiverifysnap/controller/DetectionController.java)).
- ML integration pattern: backend uses a `WebClient` (configured in `AiVerifySnapApplication`) and `DetectionService` builds a multipart request using a `ByteArrayResource` to avoid corrupted uploads — preserve this approach when modifying ML call logic ([backend/src/main/java/com/backend/aiverifysnap/service/DetectionService.java]).
- Configuration: environment-specific settings live in `application-dev.yml` and `application-prod.yml`; `application.yml` sets the active profile. Read `ml.service.url` and `server.port` from these files.
- Datastore: PostgreSQL configured in `application-dev.yml` (JDBC URL, Hikari pool). JPA entities are used for persistence (`model` package).
- Java version: project targets Java 21 (see `pom.xml`). Lombok is used for models; ensure annotation-processing remains enabled in IDEs/builds.

ML service contract (what the backend expects)
- Endpoint: POST `/detect` accepting multipart `file` (image). See [ml_service/main.py](ml_service/main.py).
- Response shape (ML -> backend): JSON with keys at least: `verdict`, `confidence`, `filename`, `raw_output`, `ela`, `processing_time_ms`. The backend maps these to the frontend-friendly shape in `DetectionService.transformMlResponse`.

Errors, fallbacks, and logging
- `DetectionService` treats non-2xx as structured errors and returns a map with `error` and `details` fields rather than throwing. When adding features, follow the same error-map pattern so callers (controllers/frontend) can handle errors uniformly.
- If `/detect` returns 404, the service retries with a trailing `/detect/` — keep that fallback when changing ML call behavior.

Where to look (quick links)
- Backend entry: [backend/src/main/java/com/backend/aiverifysnap/AiVerifySnapApplication.java](backend/src/main/java/com/backend/aiverifysnap/AiVerifySnapApplication.java)
- Detection flow: [backend/src/main/java/com/backend/aiverifysnap/service/DetectionService.java](backend/src/main/java/com/backend/aiverifysnap/service/DetectionService.java)
- Controllers: [backend/src/main/java/com/backend/aiverifysnap/controller](backend/src/main/java/com/backend/aiverifysnap/controller)
- Backend config: [backend/src/main/resources/application-dev.yml](backend/src/main/resources/application-dev.yml)
- ML service: [ml_service/main.py](ml_service/main.py) and [ml_service/requirements.txt](ml_service/requirements.txt)
- Frontend: [frontend/package.json](frontend/package.json) and [frontend/README.md](frontend/README.md)

Notes for code edits
- Preserve the ML request shape and error-map responses when changing detection logic.
- Avoid switching to `file.getResource()` in `DetectionService` — it causes corrupted uploads; use the `ByteArrayResource` pattern already implemented.
- When changing database schema or JPA entities, update `application-*.yml` profiles or migration logic if present.

If anything here is unclear or you'd like more detailed examples (tests, local docker-compose, or CI conventions), ask and I'll iterate.
