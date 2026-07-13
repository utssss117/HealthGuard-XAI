# 🛡️ HealthGuard-XAI

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Streamlit-1.35+-FF4B4B?logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/XAI-SHAP%20%2B%20Counterfactuals-7209B7" />
  <img src="https://img.shields.io/badge/LLM-Groq%20LLaMA--3.1-06D6A0" />
  <img src="https://img.shields.io/badge/Models-5%20Ensemble-4361EE" />
  <img src="https://img.shields.io/badge/License-MIT-green" />
</p>

> **An end-to-end, production-grade Explainable AI system for probabilistic disease risk prediction, counterfactual health guidance, and LLM-powered health coaching — built with Next.js & FastAPI.**

🌍 **Live Web App:** [health-guard-xai-i7ou.vercel.app](https://health-guard-xai-i7ou.vercel.app)
🖥️ **Live Backend API:** [healthguard-xai.onrender.com](https://healthguard-xai.onrender.com)

---

## 📸 Screenshots

| Risk Assessment | SHAP Explainability | Counterfactual Analysis |
|:-:|:-:|:-:|
| Animated risk gauge with clinical reference bands | SHAP waterfall + protective vs. risk factors | "What-if" action plan with before/after comparison |

---

## ✨ Features

### 🧠 Machine Learning Pipeline
- **5-model ensemble**: Logistic Regression, Random Forest, XGBoost, MLP, Stacking Classifier
- **Stratified K-Fold Cross-Validation** (5-fold) for robust performance estimation
- **Probability calibration** with Brier score tracking
- **Best model auto-selection** by ROC-AUC

### 🔬 Explainable AI (XAI)
- **SHAP feature attribution** — per-patient local explanations (supports TreeExplainer, LinearExplainer, KernelExplainer)
- **Counterfactual generation** — gradient-free clinical optimization to find the *minimum* biomarker change needed to reach Low risk
- **Feature importance** waterfall charts and magnitude rankings

### 📋 Recommendation Engine
- **Clinical rule engine** — evidence-based rules for BMI, glucose, blood pressure, insulin, age, comorbidity
- **Risk-weighted scoring** — SHAP-informed prioritization
- **LLM personalization** — Groq API adds conversational context to clinical rules

### 🤖 AI Health Assistant
- **Groq LLaMA-3.1-8B-Instant** — sub-second inference
- **3-layer safety guardrail system**:
  - Emergency detection → immediate escalation
  - Medication boundary → physician redirect
  - Diagnostic boundary → appropriate framing
- **Multi-turn conversation** with patient context injection

### 📊 Model Analytics Dashboard
- Interactive ROC curves, calibration curves, confusion matrices
- Side-by-side model comparison bar chart
- Full metrics table (Accuracy, Precision, Recall, F1, ROC-AUC, Brier)

---

## 📈 Model Performance

| Model | Accuracy | Precision | Recall | F1 | **ROC-AUC** | Brier↓ |
|-------|----------|-----------|--------|----|-------------|--------|
| Logistic Regression | 0.714 | 0.609 | 0.519 | 0.560 | **0.823** | 0.168 |
| Random Forest | 0.760 | 0.681 | 0.593 | 0.634 | 0.815 | 0.165 |
| XGBoost | 0.734 | 0.623 | 0.611 | 0.617 | 0.805 | 0.203 |
| MLP | 0.610 | 0.286 | 0.074 | 0.118 | 0.375 | 0.247 |
| Stacking Ensemble | 0.740 | 0.667 | 0.519 | 0.583 | 0.803 | 0.173 |

> 🏆 **Best model**: Logistic Regression — AUC 0.823 (5-fold CV AUC: 0.833 ± 0.035)

---

## 🏗️ Architecture

```
Patient Biomarker Input  (8 clinical features)
        │
        ▼
Preprocessing Pipeline  →  MedianImputer  →  StandardScaler
        │
        ▼
ML Ensemble  →  LogReg · RandomForest · XGBoost · MLP · Stacking
        │
        ▼
   Risk Score  (0.0 → 1.0 probability)
        │
   ┌────┴────┐────────────────────┐
   ▼         ▼                    ▼
[A] SHAP   [B] Counterfactual  [C] Recommendation Engine
Explainer    Optimizer          Rule-based + LLM (Groq)
        │
        ▼
Safety Guardrails  →  Emergency · Medication · Diagnostic
        │
        ▼
Health Guidance Assistant  (multi-turn, context-aware)
```

---

## 🚀 Quick Start

### Option 1 — Local (Recommended)

```bash
# 1. Clone
git clone https://github.com/yourusername/HealthGuard-XAI.git
cd HealthGuard-XAI

# 2. Create virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set your API key
echo GROQ_API_KEY="your_key_here" > .env

# 5. Train the ML model (first time only)
python -m risk_modeling.train

# 6. Launch the dashboard
streamlit run app.py
```

Visit **http://localhost:8501**

### Option 2 — Docker

```bash
# Build and run with a single command
GROQ_API_KEY=your_key docker compose up

# Or build manually
docker build -t healthguard-xai .
docker run -p 8501:8501 -e GROQ_API_KEY=your_key healthguard-xai
```

---

## 🧪 Running Tests

```bash
# Run all tests
pytest tests/ -v

# With coverage report
pytest tests/ -v --cov=. --cov-report=term-missing

# Run specific test file
pytest tests/test_safety_filter.py -v
```

**Test Coverage:**
- `test_safety_filter.py` — 20 tests covering emergency, medication, diagnostic, safe inputs, priority ordering
- `test_rule_engine.py`   — 15 tests covering BMI, glucose, BP, insulin, age, comorbidity, risk classification
- `test_preprocessing.py` — 12 tests covering scaling, imputation, stratification, session tracking

---

## 🗂️ Project Structure

```
HealthGuard-XAI/
├── app.py                              # 🎯 Main Streamlit dashboard (6-tab premium UI)
├── config.py                           # Global configuration
├── requirements.txt                    # Dependencies
├── Dockerfile                          # Container deployment
├── docker-compose.yml                  # One-command deployment
│
├── risk_modeling/                      # 📊 ML Pipeline
│   ├── data_loader.py                  # CSV ingestion
│   ├── preprocessing.py               # Imputation + scaling
│   ├── models.py                       # 5-model definitions
│   ├── evaluation.py                   # Metrics + Plotly charts
│   └── train.py                        # Training entrypoint
│
├── explainability/                     # 🔬 XAI Modules
│   └── counterfactual_advisor.py       # Min-distance CF optimization
│
├── health_llm_assistant/               # 🤖 LLM Assistant
│   ├── assistant.py                    # Orchestrator
│   ├── safety_filter.py                # 3-layer guardrails
│   ├── prompt_builder.py               # Context injection
│   └── llm_interface.py                # Groq SDK wrapper
│
├── recommendation_engine/              # 📋 Recommendations
│   ├── rule_engine.py                  # Clinical rules (BMI, glucose, BP…)
│   ├── hybrid_recommender.py           # Rule + LLM pipeline
│   ├── risk_weighting.py               # SHAP-weighted scoring
│   └── clinical_thresholds.py          # Evidence-based thresholds
│
├── backend/                            # ⚙️ FastAPI REST API
│   ├── main.py                         # App + lifespan
│   ├── auth.py                         # Clerk JWT auth
│   ├── database.py                     # SQLite user store
│   └── routers/                        # predict · explain · recommend · chat
│
├── utils/
│   └── session_tracker.py              # In-session prediction history
│
├── data/
│   └── diabetes.csv                    # PIMA Diabetes dataset (768 samples)
│
├── outputs/
│   ├── models/best_model.pkl           # Trained model artifact
│   └── metrics/model_comparison.csv    # Training results
│
└── tests/                              # 🧪 Unit test suite
    ├── test_safety_filter.py
    ├── test_rule_engine.py
    └── test_preprocessing.py
```

---

## 🔐 Configuration

| Variable | Required | Description |
|----------|----------|-------------|
| `GROQ_API_KEY` | ✅ Yes | Groq API key for LLM assistant. Get one free at [console.groq.com](https://console.groq.com) |
| `CLERK_SECRET_KEY` | ⚙️ Optional | Only needed if using the FastAPI backend with auth |

Set in `.env` file:
```bash
GROQ_API_KEY="gsk_your_key_here"
```

Or on **Streamlit Cloud** → App Settings → Secrets.

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | Streamlit 1.35 + Custom CSS (Glassmorphism) |
| ML Models | scikit-learn, XGBoost, NumPy, pandas |
| Explainability | SHAP (TreeExplainer, LinearExplainer, KernelExplainer) |
| Counterfactuals | Custom gradient-free optimizer (clinical bounds) |
| LLM | Groq API (LLaMA-3.1-8B-Instant) |
| Visualizations | Plotly 5.x (interactive) + Matplotlib |
| REST API | FastAPI + Pydantic + Uvicorn |
| Auth | Clerk JWT + python-jose |
| Database | SQLite (dev) |
| Testing | pytest + pytest-cov |
| CI/CD | GitHub Actions |
| Deployment | Docker + docker-compose |

---

## 📚 Dataset

**PIMA Indians Diabetes Database**
- **Source**: National Institute of Diabetes and Digestive and Kidney Diseases
- **Samples**: 768 female patients (≥21 years, Pima Indian heritage)
- **Features**: 8 biomarkers — Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age
- **Target**: Binary (0 = No diabetes, 1 = Diabetes) — 34.9% positive rate

---

## ⚕️ Medical Disclaimer

This system is built for **educational and research purposes only**. It does not constitute medical advice, diagnosis, or treatment. All health decisions should be made in consultation with a qualified healthcare professional. The AI predictions are statistical estimates based on population-level biomarker patterns and are **NOT equivalent to clinical diagnosis**.

---

## 📄 License

MIT License — see [LICENSE](LICENSE) file for details.

---

<p align="center">
  Built with ❤️ using Python · scikit-learn · SHAP · Groq · Streamlit
</p>
