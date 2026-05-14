<div align="center">
  <h1 style="font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; font-size: 3.5rem; font-weight: 800; letter-spacing: -0.05em; color: #E65100; margin-bottom: 20px;">
    AI Verify Snap
  </h1>
  
  <img src="./banner.png" alt="AI Verify Snap Banner" width="800" style="border-radius: 12px; margin-top: 10px;" />
</div>

<br />

## 🌟 Project Summary

**AI Verify Snap** is a state-of-the-art deepfake detection and digital forensics platform designed to verify the authenticity of digital media. As visual misinformation becomes increasingly sophisticated, this platform serves as a reliable tool to detect synthetic modifications, deepfakes, and manipulated imagery.

By analyzing forensic artifacts through **Error Level Analysis (ELA)** and advanced **Machine Learning** models, the platform identifies hidden pixel-level inconsistencies that are invisible to the naked eye. Users can securely upload images to run deep scans, view detailed visual analysis, and generate comprehensive, downloadable forensic reports. 

Built with a focus on speed, accuracy, and a premium user experience, AI Verify Snap empowers individuals and organizations to trust their digital content in an era of AI-generated media.

## 🔧 Quickstart

Prerequisites:
- Java 21 (for backend)
- Python 3.10+ (for ML service)
- Node.js 18+ / npm (for frontend)

Start services from the project root:

1. ML service (FastAPI / Uvicorn):

```powershell
c:/python314/python.exe -m pip install -r ML_Service/requirements.txt
c:/python314/python.exe -m uvicorn ML_Service.main:app --reload --port 8000
```

2. Backend (Spring Boot):

```powershell
cd backend
.\mvnw.cmd spring-boot:run
```

3. Frontend (Next.js dev server):

```powershell
npm --prefix frontend install --legacy-peer-deps
npm --prefix frontend run dev
```

Access the apps:
- Frontend: http://localhost:3000
- Backend API: http://localhost:8080
- ML service: http://127.0.0.1:8000

## 🧩 Configuration
- Copy `.env.example` to `.env` at the repo root and fill secrets. There are also service-scoped examples:
  - `ML_Service/.env.example`
  - `backend/.env.example`
  - `frontend/.env.example`

Important keys:
- `SERP_API_KEY` — SerpAPI Google Lens API key (used by reverse image search)
- `HF_TOKEN` — Optional HuggingFace token to speed model downloads
- Database settings for the backend (see `backend/.env.example`)

## ⚙️ What I changed while running/debugging locally
- Fixed ML service imports and added `ML_Service/__init__.py` so the package imports work under the project root.
- Made reverse-search import robust and installed `google-search-results` if missing.
- Added `.env.example` files and updated `.gitignore` to avoid committing secrets and build artifacts.

## Contributing
- Create issues or PRs on the GitHub repo: https://github.com/SanketP2003/AIVerifySnap
- For local contributions, create a feature branch, make changes, run tests, and open a PR.

## License
This repository does not include an explicit license file. Add a `LICENSE` if you intend to open-source the project.

---
If you want, I can open a PR with these changes, add a short run script, or expand the README with architecture diagrams. Which would you like next?
