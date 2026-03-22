# HealthGuard-XAI ⚕️

An AI-driven Health Dashboard and Risk Diagnostic Engine.

## 🚀 How to Run

To run the full application, you need to start both the **Backend API** and the **Frontend Dashboard**.

### 1. Backend (FastAPI + Python)
The backend handles AI risk modeling, clinical synchronization, and patient chat.

1.  **Activate Virtual Environment** (Recommended):
    ```powershell
    # Windows
    .\.venv\Scripts\activate
    ```
2.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
3.  **Start the Server**:
    ```bash
    python -m uvicorn api.main:app --reload
    ```
    *Access API documentation at: [http://localhost:8000/docs](http://localhost:8000/docs)*

### 2. Frontend (Next.js + React)
The frontend provides the "dashing" UI and Clerk-protected dashboard.

1.  **Install Dependencies**:
    ```bash
    npm install
    ```
2.  **Start the Dev Server**:
    ```bash
    npm run dev
    ```
    *Access the dashboard at: [http://localhost:3000](http://localhost:3000)*

## 🔐 Configuration
Ensure your `.env.local` contains valid **Clerk API Keys**:
- `NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY`
- `CLERK_SECRET_KEY`

## ✨ Features
- **Explainable AI Integration**: Understand the biomarkers driving every risk.
- **Dashing UI**: Modern dark theme with medical-grade visualizations.
- **Clerk Auth**: Secure, branded sign-in and sign-up flows.
- **AI Health Assistant**: Interactive chat powered by Groq Llama-3.1.
