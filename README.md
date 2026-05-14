<div align="center">
  
  # 🔍 AI Verify Snap
  
  **AI-Powered Deepfake Detection & Digital Forensics Platform**
  
  [![Java](https://img.shields.io/badge/Java-21-ED8B00?style=flat-square&logo=openjdk)](https://www.java.com)
  [![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python)](https://www.python.org)
  [![Node.js](https://img.shields.io/badge/Node.js-18+-339933?style=flat-square&logo=node.js)](https://nodejs.org)
  [![React](https://img.shields.io/badge/React-18-61DAFB?style=flat-square&logo=react)](https://react.dev)
  [![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)
  [![GitHub](https://img.shields.io/badge/GitHub-Repo-181717?style=flat-square&logo=github)](https://github.com/SanketP2003/AIVerifySnap)
  
</div>

---

## 🎯 Overview

**AI Verify Snap** is a state-of-the-art deepfake detection and digital forensics platform designed to verify the authenticity of digital media. As visual misinformation becomes increasingly sophisticated, this platform serves as your reliable defense against synthetic modifications, deepfakes, and manipulated imagery.

### Key Capabilities
- 🤖 **Advanced ML Detection** — Dual-stream ResNet18 + ELA CNN fusion for accurate deepfake identification
- 🔬 **Forensic Analysis** — Error Level Analysis (ELA) to reveal hidden pixel-level inconsistencies
- 📊 **Interactive Heatmaps** — Visualize tampering artifacts with detailed heatmap overlays
- 🔍 **Reverse Image Search** — Google Lens integration for cross-platform verification
- 📄 **Detailed Reports** — Generate downloadable forensic reports with full analysis
- ⚡ **Real-time Processing** — Fast detection on GPU-accelerated infrastructure

## 🏗️ Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Backend** | Spring Boot 3.2 + Java 21 | REST API, Database, Business Logic |
| **Frontend** | Next.js 14 + React 18 + TypeScript | Web UI, Real-time Detection |
| **ML Service** | FastAPI + PyTorch | Deepfake Detection & ELA Analysis |
| **Database** | PostgreSQL | Detection History & User Management |
| **GPU** | CUDA/PyTorch | Accelerated ML Inference |

## 📋 Features

- ✅ **Upload & Analyze** — Drag-and-drop image upload with instant analysis
- ✅ **Dual Detection Models** — Custom AIVerifySnap model + HuggingFace SigLIP fallback
- ✅ **Error Level Analysis (ELA)** — Forensic heatmap visualization
- ✅ **Reverse Image Search** — Find similar images across the web via Google Lens
- ✅ **User Authentication** — Secure sign-up and detection history tracking
- ✅ **Download Reports** — Export forensic findings as PDF/JSON
- ✅ **Responsive UI** — Works on desktop, tablet, mobile
- ✅ **Real-time Dashboard** — Admin stats and monitoring

## 🚀 Quick Start

### Prerequisites
- **Java 21** — Backend runtime
- **Python 3.10+** — ML service runtime
- **Node.js 18+** — Frontend tooling
- **PostgreSQL 13+** — Database
- **4GB+ RAM, GPU recommended** — For ML inference

### Installation & Setup

#### 1️⃣ Clone Repository
```bash
git clone https://github.com/SanketP2003/AIVerifySnap.git
cd AIVerifySnap
```

#### 2️⃣ Environment Configuration
Copy env templates and fill in your secrets:
```bash
cp .env.example .env
cp ML_Service/.env.example ML_Service/.env
cp backend/.env.example backend/.env
cp frontend/.env.example frontend/.env
```

**Key Environment Variables:**
```bash
# ML Service
SERP_API_KEY=<your-serpapi-key>        # For reverse image search
HF_TOKEN=<your-huggingface-token>      # Optional, speeds model downloads

# Backend
SPRING_DATASOURCE_URL=jdbc:postgresql://localhost:5432/aiverifysnap
SPRING_DATASOURCE_USERNAME=postgres
SPRING_DATASOURCE_PASSWORD=<your-db-password>
ML_SERVICE_URL=http://localhost:8000

# Frontend
NEXT_PUBLIC_API_URL=http://localhost:8080
NEXT_PUBLIC_ML_URL=http://localhost:8000
```

#### 3️⃣ Start Services

**Terminal 1 — ML Service:**
```powershell
cd ML_Service
python -m pip install -r requirements.txt
python -m uvicorn main:app --reload --port 8000
```

**Terminal 2 — Backend:**
```powershell
cd backend
./mvnw.cmd spring-boot:run
# or build then run:
./mvnw.cmd package -DskipTests
java -jar target/backend-0.0.1-SNAPSHOT.jar
```

**Terminal 3 — Frontend:**
```powershell
cd frontend
npm install --legacy-peer-deps
npm run dev
```

#### 4️⃣ Access Application
- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8080
- **ML Service**: http://127.0.0.1:8000
- **API Docs**: http://localhost:8080/swagger-ui.html

## 📁 Project Structure

```
AIVerifySnap/
├── backend/                  # Spring Boot REST API
│   ├── src/main/java/...    # Java services, controllers, models
│   ├── pom.xml              # Maven dependencies
│   └── Dockerfile           # Docker image for backend
├── frontend/                 # Next.js React web app
│   ├── src/app/             # Pages and routes
│   ├── src/components/      # React components
│   ├── package.json         # Node dependencies
│   └── Dockerfile           # Docker image for frontend
├── ML_Service/              # FastAPI microservice
│   ├── main.py              # FastAPI app & endpoints
│   ├── model.py             # ML model architectures
│   ├── utils.py             # ELA & preprocessing
│   ├── requirements.txt     # Python dependencies
│   └── Dockerfile           # Docker image for ML service
└── README.md                # This file
```

## 🔌 API Endpoints

### Detection API
```
POST /api/detect
- Upload image and get deepfake detection
- Returns: verdict, confidence, ELA heatmap, forensic scores

POST /api/reverse-search
- Upload image for Google Lens reverse search
- Returns: visual matches, knowledge graph, metadata
```

### User API
```
POST /api/users/signup
GET /api/users/history
DELETE /api/users/{id}
```

See [Swagger UI](http://localhost:8080/swagger-ui.html) for full API documentation.

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| **ML Service crashes on import** | Run `python -m uvicorn ML_Service.main:app` from project root (not from ML_Service dir) |
| **Reverse search returns 500** | Set `SERP_API_KEY` environment variable with valid SerpAPI key |
| **Frontend can't reach backend** | Ensure `NEXT_PUBLIC_API_URL` matches backend port (default: http://localhost:8080) |
| **Database connection fails** | Verify PostgreSQL is running and credentials in `.env` are correct |
| **GPU not detected in ML service** | Check CUDA installation and PyTorch GPU support: `python -c "import torch; print(torch.cuda.is_available())"` |

## 📚 Documentation

- **Backend**: See `backend/README.md` for Spring Boot architecture
- **Frontend**: See `frontend/README.md` for Next.js setup
- **ML Service**: See `ML_Service/` for model training & inference details

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push to branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) file for details.

---

## 👥 Authors & Credits

- **Sanket Patil** — Project Creator  
- Built with ❤️ for digital media verification

## 🙏 Acknowledgments

- HuggingFace for SigLIP model
- SerpAPI for reverse image search integration
- PyTorch & TensorFlow community
- Spring Boot & Next.js communities

---

<div align="center">
  
  **[⬆ back to top](#-ai-verify-snap)**
  
  Made with 🔍 for a safer, more trustworthy internet
  
</div>
