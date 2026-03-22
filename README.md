# HealthGuard-XAI 🛡️

An AI-driven Health Dashboard and Risk Diagnostic Engine, built cleanly in pure Python using Streamlit!

🌍 **Live Application:** [https://healthguard-xai-ht2rajyhy4app2wnym3cxtx.streamlit.app/](https://healthguard-xai-ht2rajyhy4app2wnym3cxtx.streamlit.app/)

## ✨ Features
- **Machine Learning Risk Prediction**: Uses advanced modeling to probabilistically estimate disease risk.
- **Explainable AI (SHAP)**: Understand exactly which biomarkers are driving your specific risk up or down natively mapped.
- **AI Health Assistant**: Interactive chat layer powered by Groq's high-speed inference.
- **Hybrid Recommendation Engine**: Delivers a fully-personalized lifestyle plan merging clinical rules with LLM synthesis.

## 🚀 How to Run Locally

Because the entire architecture was transitioned to a unified Streamlit build, running it is incredibly simple:

1. **Activate Virtual Environment** (Recommended):
    ```powershell
    # Windows
    .\.venv\Scripts\activate
    ```
2. **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
3. **Start the Dashboard**:
    ```bash
    streamlit run app.py
    ```

*Access the dashboard at: [http://localhost:8501](http://localhost:8501)*

## 🔐 Configuration
To enable the **AI Assistant** and **Personalized Recommendations** tabs, you must supply a Groq API Key:

- Create a `.env` file or export it directly into your terminal: `GROQ_API_KEY="gsk_your_key_here"`
- On Streamlit Community Cloud, provide this inside the **App Settings > Secrets** panel!
