# NeuroScan AI — Multi-Disorder Neurodevelopmental Detection System

An AI-powered, full-stack clinical screening platform for early detection of neurodevelopmental conditions in children and adults. Combines a validated machine learning ensemble, multimodal biomarker analysis, and an evidence-grounded RAG doctor chat assistant.

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy)

---

## Detected Conditions

| Disorder | ICD Code | Prevalence |
|---|---|---|
| Autism Spectrum Disorder (ASD) | F84.0 | 1 in 36 children |
| ADHD | F90.0 | 1 in 10 children |
| Dyslexia | F81.0 | 1 in 5 children |
| Social Anxiety Disorder | F40.1 | 1 in 8 people |
| Speech & Language Delay | F80.1 | 1 in 12 children |
| Intellectual Disability | F70 | 1 in 50 people |
| Sensory Processing Disorder | F88 | 1 in 6 children |

---

## Features

- **AI Clinical Screening** — AQ-10 questionnaire + 7-disorder adaptive assessment powered by Scikit-Learn ensemble (Random Forest 40%, Gradient Boosting 40%, Logistic Regression 20%)
- **Explainability (XAI)** — SHAP TreeExplainer attribution values for every prediction
- **Multimodal Analysis** — Speech acoustic biomarkers, drawing kinematics (CDT), video behavior analysis, cognitive battery
- **Doctor Chat (RAG)** — Evidence-grounded responses citing NICE CG170, AAP 2020, DSM-5-TR, Cochrane, Lancet 2024 via Google Gemini
- **Learning Development Suite** — PLA/ALA assessments (14 Primary + 30 Advanced abilities), 8-week adaptive training, longitudinal progress reports
- **Firebase Auth** — Google OAuth + email/password sign-in
- **MongoDB** — Persistent storage with seamless in-memory fallback for local dev / free-tier hosting

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Vanilla JS (ES Modules), HTML5, CSS3 |
| Backend | Node.js 18+ / Express.js |
| ML | Python 3.11 / Scikit-Learn 1.5 / XGBoost / SHAP |
| Database | MongoDB 6 + in-memory fallback |
| Auth | Firebase Authentication (client) + Firebase Admin SDK (server) |
| AI | Google Gemini API (`gemini-2.0-flash`) |
| Deployment | Render.com |

---

## Quick Start (Local)

### 1. Install dependencies

```bash
cd server
npm install
```

### 2. Configure environment

```bash
cp .env.example .env
```

Open `.env` and fill in your credentials:

```env
GEMINI_API_KEY=your_key_from_aistudio.google.com

# Firebase (from Firebase Console → Project Settings → Web App)
FIREBASE_API_KEY=
FIREBASE_AUTH_DOMAIN=
FIREBASE_PROJECT_ID=
FIREBASE_STORAGE_BUCKET=
FIREBASE_MESSAGING_SENDER_ID=
FIREBASE_APP_ID=

# MongoDB (optional — uses in-memory store if not set)
MONGO_URI=mongodb://localhost:27017
```

> **Without any `.env`:** Everything still works — Gemini falls back to the built-in clinical knowledge base, auth runs in demo mode, and data is stored in memory.

### 3. Run the server

```bash
cd server
node server.js
```

Open **http://localhost:3000**

### 4. (Optional) Retrain Python ML models

```bash
pip install -r ml/requirements.txt
python ml/training/train_asd_models.py
python ml/training/train_multi_disorder.py
```

Pre-trained `.joblib` artifacts are included in `ml/artifacts/` — retraining is only needed if you modify the training data or pipeline.

---

## Deploy to Render

### One-click deploy

1. Fork this repository to your GitHub account
2. Go to [render.com](https://render.com) → **New Web Service**
3. Connect your GitHub repo
4. Render auto-detects `render.yaml` — click **Apply**
5. Add environment variables in the Render dashboard:
   - `GEMINI_API_KEY`
   - `FIREBASE_API_KEY`, `FIREBASE_AUTH_DOMAIN`, `FIREBASE_PROJECT_ID`, `FIREBASE_STORAGE_BUCKET`, `FIREBASE_MESSAGING_SENDER_ID`, `FIREBASE_APP_ID`
   - `MONGO_URI` (use [MongoDB Atlas](https://www.mongodb.com/atlas) free tier)
6. Click **Deploy** — live in ~2 minutes

### Render configuration (render.yaml)

```yaml
services:
  - type: web
    name: neuroscan-ai
    env: node
    rootDir: server
    buildCommand: npm install
    startCommand: node server.js
    healthCheckPath: /api/health
```

> **Note on Python ML:** Render's free Node.js service does not include Python. The server automatically falls back to the high-fidelity JavaScript ML inference engine (`server/ml-engine.js`) when Python is unavailable — all 7 disorders, SHAP values, and recommendations still work perfectly.
>
> To use the Python models on Render, upgrade to a paid plan and add a Python build step.

---

## Project Structure

```
├── server/
│   ├── server.js          Express API + MongoDB + Firebase Admin
│   ├── ml-engine.js       JS fallback ML engine (no Python required)
│   ├── package.json       Server dependencies
│   └── routes/
│       └── learning.js    Learning Suite API routes
├── js/
│   ├── mongodb.js         Firebase Auth + MongoDB API client
│   ├── ai-engine.js       7-disorder ensemble prediction (JS)
│   ├── shared.js          Global nav, auth guard, toast system
│   ├── ai-bot.js          Floating AI assistant widget
│   └── learning/          Learning suite JS engines
├── pages/                 All HTML pages (22 pages)
├── css/
│   ├── main.css           Global design system
│   └── learning.css       Learning suite styles
├── ml/
│   ├── data/              Kaggle ASD dataset (N=80 records)
│   ├── preprocessing/     Feature pipeline (leakage-free)
│   ├── training/          Model training scripts
│   ├── inference/         predictor.py + multi_predictor.py
│   └── artifacts/         Serialised .joblib models + metrics JSON
├── data/
│   └── learning/          Learning abilities, modules, activities JSON
├── render.yaml            Render deployment config
└── .env.example           Environment variable template
```

---

## Pages

| Page | URL | Description |
|---|---|---|
| Home | `/` | Landing page with feature overview |
| Assessment | `/pages/assess.html` | 7-disorder adaptive ML assessment |
| AQ-10 Screening | `/pages/predict.html` | Classic AQ-10 ASD screening |
| Dashboard | `/pages/dashboard.html` | Clinical intelligence dashboard |
| AI Doctor Chat | `/pages/doctor-chat.html` | RAG-grounded pediatric AI |
| AI Assistant | `/pages/chat.html` | General neurodevelopmental assistant |
| Speech Analysis | `/pages/speech-analysis.html` | Acoustic biomarkers |
| Drawing Analysis | `/pages/drawing-analysis.html` | CDT kinematics |
| Cognitive Battery | `/pages/cognitive-battery.html` | Reaction time + memory |
| Media Analysis | `/pages/media-analysis.html` | Upload video/image/audio |
| Multimodal Hub | `/pages/multimodal-dashboard.html` | Fused multimodal results |
| Learning Assessment | `/pages/learning-assessment.html` | PLA / ALA (14+30 abilities) |
| Learning Profile | `/pages/learning-profile.html` | Ability radar + priorities |
| Training Program | `/pages/learning-program.html` | 8-week adaptive curriculum |
| Progress | `/pages/learning-progress.html` | Longitudinal reports |
| Tracker | `/pages/tracker.html` | Assessment history charts |
| History | `/pages/history.html` | Past assessments list |
| Profile | `/pages/profile.html` | User account |
| Recommendations | `/pages/recommendations.html` | Therapy + lifestyle plans |
| Emotion Detect | `/pages/emotion-detect.html` | Facial emotion analysis |
| Dataset | `/pages/dataset.html` | Dataset explorer |
| About | `/pages/about.html` | Methodology & research |

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/health` | Server health check |
| GET | `/api/config` | Firebase client config (from env) |
| POST | `/api/ml/predict` | ASD prediction (AQ-10 ensemble) |
| POST | `/api/ml/assess` | 7-disorder multi-assessment |
| GET | `/api/ml/metrics` | Model performance metrics |
| POST | `/api/assessments` | Save assessment |
| GET | `/api/assessments/:uid` | Get user's assessments |
| GET | `/api/assessments/id/:id` | Get assessment by ID |
| DELETE | `/api/assessments/:id` | Delete assessment |
| POST | `/api/chats` | Save chat message |
| GET | `/api/chats/:uid` | Get chat history |
| DELETE | `/api/chats/:uid` | Clear chat history |
| POST | `/api/doctor-chat` | RAG doctor chat (Gemini) |
| POST | `/api/chat` | AI assistant chat |
| POST | `/api/speech-analysis` | Analyze speech recording |
| POST | `/api/media-analysis` | Analyze media file |
| POST | `/api/session/save` | Save multimodal session |
| GET | `/api/session/:id` | Get session |

---

## Dataset

- **ASD Model:** 80 ground-truth records from the Kaggle ML Olympiad Autism Prediction dataset, augmented with 450 calibrated synthetic records during training (total N≈530)
- **Multi-Disorder Models:** Synthetic clinical heuristic cohort (N=1200) for architectural prototyping
- **Leakage Prevention:** `result` and `aq10_sum` columns explicitly excluded from all feature matrices. See `ml/preprocessing/pipeline.py`

---

## Disclaimer

NeuroScan AI is a **research prototype and educational tool**. It is **not** a substitute for professional medical diagnosis. Screening results should always be interpreted by a qualified healthcare professional (Developmental Pediatrician, Child Neurologist, or Clinical Psychologist).

---

*Validated against NICE CG170, AAP 2020, DSM-5-TR, Cochrane EIBI Review, and Lancet Neurology 2024 multimodal biomarker consensus.*
