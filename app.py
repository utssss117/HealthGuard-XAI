"""
app.py
──────
HealthGuard-XAI — Recruiter-Grade Streamlit Dashboard

6-tab premium dark UI featuring:
  Tab 1 · Overview          — Animated hero + system metrics + architecture
  Tab 2 · Risk Assessment   — Biomarker form + animated gauge + risk badge
  Tab 3 · Explainability    — SHAP waterfall + SHAP bar chart + top factors
  Tab 4 · Counterfactuals   — "What-If" action plan with interactive sliders
  Tab 5 · Model Analytics   — ROC, calibration, confusion matrix, model table
  Tab 6 · AI Assistant      — Multi-turn LLM health chat with safety guardrails
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os, json
from datetime import datetime

# ── Page Config (MUST be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="HealthGuard-XAI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Local Imports ─────────────────────────────────────────────────────────────
try:
    from backend.dependencies import load_model_and_scaler, get_model, get_scaler, biomarkers_to_df
    from health_llm_assistant.assistant import ask as health_llm_ask
    from recommendation_engine.hybrid_recommender import generate_recommendations
    from explainability.counterfactual_advisor import generate_counterfactual, CLINICAL_BOUNDS
    from utils.session_tracker import SessionTracker
    from backend.database import (
        init_db,
        register_user,
        authenticate_user,
        save_prediction,
        get_prediction_history,
        get_unique_patients,
        save_chat_message,
        get_chat_history,
        clear_chat_history,
    )
except ImportError as e:
    st.error(f"Import Error: {e}. Ensure you are running from the project root.")
    st.stop()

# ── Global CSS — Premium Dark Glassmorphism Theme ────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Reset & Base ── */
*, *::before, *::after { box-sizing: border-box; }
html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, sans-serif;
}
.stApp {
    background: linear-gradient(135deg, #0a0a1a 0%, #0d1b2a 40%, #0f2641 100%);
    min-height: 100vh;
}

/* ── Hide default Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.5rem !important; max-width: 1300px; }

/* ── Hero Banner ── */
.hero-banner {
    background: linear-gradient(135deg, rgba(67,97,238,0.15) 0%, rgba(114,9,183,0.12) 50%, rgba(247,37,133,0.08) 100%);
    border: 1px solid rgba(67,97,238,0.3);
    border-radius: 20px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    backdrop-filter: blur(20px);
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: '';
    position: absolute;
    top: -50%; left: -50%;
    width: 200%; height: 200%;
    background: radial-gradient(circle at 30% 50%, rgba(67,97,238,0.08) 0%, transparent 60%),
                radial-gradient(circle at 70% 50%, rgba(114,9,183,0.06) 0%, transparent 60%);
    pointer-events: none;
}
.hero-title {
    font-size: 2.8rem;
    font-weight: 800;
    background: linear-gradient(135deg, #4CC9F0, #4361EE, #7209B7);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 0.5rem 0;
    line-height: 1.2;
}
.hero-subtitle {
    font-size: 1.1rem;
    color: rgba(255,255,255,0.65);
    margin: 0;
    font-weight: 400;
}
.hero-badges { margin-top: 1.2rem; display: flex; gap: 0.6rem; flex-wrap: wrap; }
.badge {
    background: rgba(67,97,238,0.2);
    border: 1px solid rgba(67,97,238,0.4);
    color: #4CC9F0;
    padding: 0.3rem 0.8rem;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.5px;
}

/* ── Metric Cards ── */
.metric-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; margin: 1.5rem 0; }
.metric-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 1.4rem 1.2rem;
    text-align: center;
    transition: all 0.3s ease;
    backdrop-filter: blur(10px);
    position: relative;
    overflow: hidden;
}
.metric-card:hover {
    border-color: rgba(67,97,238,0.5);
    background: rgba(67,97,238,0.08);
    transform: translateY(-3px);
    box-shadow: 0 8px 30px rgba(67,97,238,0.15);
}
.metric-card::after {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    border-radius: 16px 16px 0 0;
}
.mc-blue::after   { background: linear-gradient(90deg, #4361EE, #4CC9F0); }
.mc-purple::after { background: linear-gradient(90deg, #7209B7, #F72585); }
.mc-green::after  { background: linear-gradient(90deg, #06D6A0, #4CC9F0); }
.mc-orange::after { background: linear-gradient(90deg, #F77F00, #F72585); }
.metric-value {
    font-size: 2rem;
    font-weight: 800;
    color: #fff;
    line-height: 1;
    margin-bottom: 0.4rem;
}
.metric-label {
    font-size: 0.78rem;
    color: rgba(255,255,255,0.5);
    text-transform: uppercase;
    letter-spacing: 1px;
    font-weight: 600;
}
.metric-icon { font-size: 1.8rem; margin-bottom: 0.5rem; }

/* ── Section Headers ── */
.section-header {
    display: flex;
    align-items: center;
    gap: 0.8rem;
    margin: 2rem 0 1.2rem 0;
    padding-bottom: 0.8rem;
    border-bottom: 1px solid rgba(255,255,255,0.08);
}
.section-title {
    font-size: 1.4rem;
    font-weight: 700;
    color: #fff;
    margin: 0;
}
.section-icon { font-size: 1.5rem; }

/* ── Glass Cards ── */
.glass-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    backdrop-filter: blur(10px);
    transition: border-color 0.3s ease;
}
.glass-card:hover { border-color: rgba(67,97,238,0.3); }

/* ── Risk Indicator Badge ── */
.risk-badge {
    display: inline-block;
    padding: 0.6rem 1.8rem;
    border-radius: 50px;
    font-size: 1.1rem;
    font-weight: 700;
    letter-spacing: 1px;
    text-transform: uppercase;
    margin-top: 0.5rem;
}
.risk-low    { background: rgba(6,214,160,0.15); border: 2px solid #06D6A0; color: #06D6A0; }
.risk-medium { background: rgba(255,165,0,0.15); border: 2px solid #FFA500; color: #FFA500; }
.risk-high   { background: rgba(247,37,133,0.15); border: 2px solid #F72585; color: #F72585; }

/* ── Counterfactual Change Cards ── */
.cf-card {
    background: rgba(6,214,160,0.08);
    border: 1px solid rgba(6,214,160,0.3);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    margin: 0.4rem 0;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.cf-feature { font-weight: 600; color: #fff; font-size: 0.95rem; }
.cf-change  { font-weight: 700; color: #06D6A0; font-size: 0.95rem; }
.cf-values  { color: rgba(255,255,255,0.55); font-size: 0.85rem; font-family: 'JetBrains Mono'; }

/* ── Chat UI ── */
.chat-container {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 1.2rem;
    max-height: 420px;
    overflow-y: auto;
    margin-bottom: 1rem;
    scrollbar-width: thin;
    scrollbar-color: rgba(67,97,238,0.4) transparent;
}
.chat-msg-user {
    background: linear-gradient(135deg, rgba(67,97,238,0.3), rgba(114,9,183,0.25));
    border-radius: 16px 16px 4px 16px;
    padding: 0.8rem 1.1rem;
    margin: 0.5rem 0 0.5rem 3rem;
    color: #fff;
    font-size: 0.95rem;
    border: 1px solid rgba(67,97,238,0.2);
}
.chat-msg-assistant {
    background: rgba(255,255,255,0.05);
    border-radius: 16px 16px 16px 4px;
    padding: 0.8rem 1.1rem;
    margin: 0.5rem 3rem 0.5rem 0;
    color: rgba(255,255,255,0.9);
    font-size: 0.95rem;
    border: 1px solid rgba(255,255,255,0.08);
}
.chat-role {
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 0.3rem;
    color: rgba(255,255,255,0.4);
}

/* ── Info Callouts ── */
.info-callout {
    background: rgba(76,201,240,0.08);
    border-left: 4px solid #4CC9F0;
    border-radius: 0 12px 12px 0;
    padding: 1rem 1.2rem;
    margin: 1rem 0;
    color: rgba(255,255,255,0.8);
    font-size: 0.9rem;
}
.warning-callout {
    background: rgba(255,165,0,0.08);
    border-left: 4px solid #FFA500;
    border-radius: 0 12px 12px 0;
    padding: 1rem 1.2rem;
    margin: 1rem 0;
    color: rgba(255,255,255,0.8);
    font-size: 0.9rem;
}

/* ── Tab Styling ── */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.04);
    border-radius: 12px;
    padding: 0.3rem;
    gap: 0.2rem;
    border: 1px solid rgba(255,255,255,0.06);
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    color: rgba(255,255,255,0.55);
    font-weight: 600;
    font-size: 0.85rem;
    padding: 0.6rem 1rem;
    transition: all 0.2s ease;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, rgba(67,97,238,0.4), rgba(114,9,183,0.3)) !important;
    color: white !important;
    border: 1px solid rgba(67,97,238,0.4) !important;
}
.stTabs [data-baseweb="tab-panel"] {
    padding-top: 1.5rem;
}

/* ── Streamlit Widget Overrides ── */
.stNumberInput input, .stTextInput input, .stSelectbox select {
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    color: white !important;
    border-radius: 8px !important;
}
.stSlider [data-testid="stSlider"] .st-ae { background: rgba(67,97,238,0.4) !important; }
.stButton > button {
    background: linear-gradient(135deg, #4361EE, #7209B7) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    padding: 0.6rem 1.5rem !important;
    transition: all 0.25s ease !important;
    letter-spacing: 0.3px;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(67,97,238,0.4) !important;
}
.stDataFrame { border-radius: 12px; overflow: hidden; }
label, .stMarkdown p { color: rgba(255,255,255,0.8) !important; }

/* ── SHAP colors ── */
.shap-pos { color: #F72585; font-weight: 700; }
.shap-neg { color: #06D6A0; font-weight: 700; }

/* ── Architecture diagram card ── */
.arch-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px;
    padding: 1.5rem;
    font-family: 'JetBrains Mono', monospace;
    color: rgba(255,255,255,0.6);
    font-size: 0.82rem;
    line-height: 1.8;
}
.arch-hl { color: #4CC9F0; font-weight: 700; }
</style>
""", unsafe_allow_html=True)


# ── System Initialization ─────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading ML model and scaler…")
def init_system():
    load_model_and_scaler()
    return get_model(), get_scaler()

@st.cache_data(show_spinner=False)
def load_model_comparison():
    path = os.path.join("outputs", "metrics", "model_comparison.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

try:
    model, scaler = init_system()
except Exception as e:
    st.error(f"**System Initialization Error:** {e}")
    st.markdown("Run `python -m risk_modeling.train` from the project root first.")
    st.stop()


# ── Session State & Database Initialization ────────────────────────────────────

# Ensure the database is initialized
try:
    init_db()
except Exception as db_err:
    st.error(f"**Database Initialization Error:** {db_err}")

def _init_session():
    defaults = {
        "logged_in_user":   None,
        "selected_patient_email": None,
        "patient_data":     None,
        "risk_prob":        None,
        "risk_level":       None,
        "messages":         [],
        "tracker":          SessionTracker(),
        "cf_result":        None,
        "shap_values":      None,
        "shap_features":    None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_session()


def load_user_history():
    """Load user's saved predictions and chat history into the Streamlit session state."""
    user = st.session_state.logged_in_user
    email = st.session_state.selected_patient_email
    if user and email:
        st.session_state.tracker.clear()
        records = get_prediction_history(user['id'], email)
        for r in records:
            biomarkers = {
                "Pregnancies": r["pregnancies"],
                "Glucose": r["glucose"],
                "BloodPressure": r["blood_pressure"],
                "SkinThickness": r["skin_thickness"],
                "Insulin": r["insulin"],
                "BMI": r["bmi"],
                "DiabetesPedigreeFunction": r["diabetes_pedigree"],
                "Age": r["age"],
            }
            st.session_state.tracker.add_prediction(
                biomarkers=biomarkers,
                risk_probability=r["risk_probability"],
                risk_level=r["risk_level"],
                label=r["label"],
            )
        # Set the latest prediction as the active patient data
        latest = st.session_state.tracker.latest()
        if latest:
            st.session_state.patient_data = latest["_biomarkers"]
            st.session_state.risk_prob = latest["Risk (%)"] / 100.0
            st.session_state.risk_level = latest["Risk Level"]
        else:
            st.session_state.patient_data = None
            st.session_state.risk_prob = None
            st.session_state.risk_level = None
        # Load chat history
        chat_msgs = get_chat_history(user['id'], email)
        st.session_state.messages = [{"role": msg["role"], "content": msg["content"]} for msg in chat_msgs]


# ── Helpers ───────────────────────────────────────────────────────────────────

FEATURE_ORDER = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age",
]

FEATURE_LABELS = {
    "Pregnancies":              "Pregnancies",
    "Glucose":                  "Fasting Glucose (mg/dL)",
    "BloodPressure":            "Blood Pressure (mmHg)",
    "SkinThickness":            "Skin Thickness (mm)",
    "Insulin":                  "Serum Insulin (μU/mL)",
    "BMI":                      "BMI (kg/m²)",
    "DiabetesPedigreeFunction": "Diabetes Pedigree Function",
    "Age":                      "Age (years)",
}

CLINICAL_REFS = {
    "Glucose":       ("Normal: < 100 mg/dL", "Pre-diabetic: 100–125", "Diabetic: ≥ 126"),
    "BloodPressure": ("Normal: < 120 mmHg", "Elevated: 120–129", "High: ≥ 130"),
    "BMI":           ("Normal: 18.5–24.9", "Overweight: 25–29.9", "Obese: ≥ 30"),
}

def _risk_class(prob):
    if prob <= 0.33:  return "Low",    "risk-low",    "✅"
    if prob <= 0.66:  return "Medium", "risk-medium", "⚠️"
    return "High", "risk-high", "🔴"


def _compute_shap(patient_df):
    import shap
    X_scaled = pd.DataFrame(scaler.transform(patient_df), columns=FEATURE_ORDER)
    model_type = type(model).__name__
    if "RandomForest" in model_type or "XGB" in model_type:
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X_scaled)
        sv = sv[1][0] if isinstance(sv, list) else sv[0]
    elif "Logistic" in model_type or "Linear" in model_type:
        background = np.zeros((1, X_scaled.shape[1]))
        explainer = shap.LinearExplainer(model, pd.DataFrame(background, columns=FEATURE_ORDER))
        sv = explainer.shap_values(X_scaled)
        sv = sv[0] if sv.ndim == 2 else sv
    else:
        background = np.zeros((1, X_scaled.shape[1]))
        explainer = shap.KernelExplainer(model.predict_proba, pd.DataFrame(background, columns=FEATURE_ORDER))
        sv = explainer.shap_values(X_scaled)
        sv = sv[1][0] if isinstance(sv, list) else sv[0]
    return sv


# ── Login/Signup UI & Auth Interceptor ────────────────────────────────────────

def render_login_signup_ui():
    st.markdown("""
    <div style="text-align: center; margin-top: 3rem; margin-bottom: 2rem;">
        <h1 class="hero-title" style="font-size: 3.2rem; background: linear-gradient(135deg, #4CC9F0, #4361EE, #7209B7); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">🛡️ HealthGuard-XAI</h1>
        <p class="hero-subtitle" style="font-size: 1.2rem; color: rgba(255,255,255,0.65);">Secure Patient &amp; Clinician Portal</p>
    </div>
    """, unsafe_allow_html=True)

    _, col, _ = st.columns([1, 1.4, 1])
    with col:
        st.markdown('<div class="glass-card" style="padding: 2.5rem; border-radius: 20px; background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.1); backdrop-filter: blur(20px);">', unsafe_allow_html=True)
        
        tab_choice = st.radio("Access Type", ["Sign In", "Sign Up"], horizontal=True, label_visibility="collapsed")
        
        if tab_choice == "Sign In":
            st.markdown("### 🔑 Account Login")
            email = st.text_input("Email Address", key="login_email")
            password = st.text_input("Password", type="password", key="login_password")
            
            st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
            col_demo1, col_demo2 = st.columns(2)
            with col_demo1:
                demo_patient = st.button("👥 Patient Demo", use_container_width=True)
            with col_demo2:
                demo_md = st.button("🩺 Clinician Demo", use_container_width=True)
                
            if demo_patient:
                user = authenticate_user("patient@healthguard.local", "demo123")
                if not user:
                    register_user("patient@healthguard.local", "demo123", "Jane", "Doe", "patient")
                    user = authenticate_user("patient@healthguard.local", "demo123")
                st.session_state.logged_in_user = user
                st.session_state.selected_patient_email = user['email']
                load_user_history()
                st.rerun()
                
            if demo_md:
                user = authenticate_user("doctor@healthguard.local", "demo123")
                if not user:
                    register_user("doctor@healthguard.local", "demo123", "Dr. Alexander", "Fleming", "physician")
                    user = authenticate_user("doctor@healthguard.local", "demo123")
                st.session_state.logged_in_user = user
                st.session_state.selected_patient_email = "patient@healthguard.local"
                load_user_history()
                st.rerun()
            
            if st.button("🚀 Enter Portal", use_container_width=True):
                if not email or not password:
                    st.error("Please fill in all fields.")
                else:
                    user = authenticate_user(email, password)
                    if user:
                        st.session_state.logged_in_user = user
                        st.session_state.selected_patient_email = email if user['role'] == 'patient' else None
                        if user['role'] == 'physician':
                            patients = get_unique_patients(user['id'])
                            if patients:
                                st.session_state.selected_patient_email = patients[0]
                        load_user_history()
                        st.success("Login successful! Redirecting...")
                        st.rerun()
                    else:
                        st.error("Invalid email or password.")
                        
        else:
            st.markdown("### 📝 Register New Account")
            new_email = st.text_input("Email Address", key="reg_email")
            new_pass = st.text_input("Password", type="password", key="reg_password")
            col_fn, col_ln = st.columns(2)
            with col_fn:
                first_name = st.text_input("First Name", key="reg_fn")
            with col_ln:
                last_name = st.text_input("Last Name", key="reg_ln")
                
            role = st.selectbox(
                "Register Account As:", 
                ["patient", "physician"], 
                format_func=lambda x: "Patient (Self Health Tracking)" if x == "patient" else "Clinician / Physician (Multi-Patient Tracking)"
            )
            
            if st.button("✨ Create Account", use_container_width=True):
                if not new_email or not new_pass or not first_name:
                    st.error("Please fill in all required fields (Email, Password, and First Name).")
                elif len(new_pass) < 6:
                    st.error("Password must be at least 6 characters.")
                else:
                    user = register_user(new_email, new_pass, first_name, last_name, role)
                    if user:
                        st.session_state.logged_in_user = user
                        st.session_state.selected_patient_email = new_email if role == 'patient' else None
                        load_user_history()
                        st.success("Account created successfully! Redirecting...")
                        st.rerun()
                    else:
                        st.error("Email address is already registered.")
                        
        st.markdown('</div>', unsafe_allow_html=True)

if st.session_state.logged_in_user is None:
    render_login_signup_ui()
    st.stop()

# ── Authentication Top Bar & Patient Controls ─────────────────────────────────
user = st.session_state.logged_in_user

col_user, col_logout = st.columns([7, 3])
with col_user:
    role_label = "🩺 Clinician" if user['role'] == 'physician' else "👤 Patient"
    st.markdown(f"""
    <div style="background: rgba(67, 97, 238, 0.15); padding: 0.6rem 1.2rem; border-radius: 12px; border: 1px solid rgba(67, 97, 238, 0.3); margin-bottom: 1rem;">
        <span style="color: #4CC9F0; font-weight: 700;">Logged in as:</span> {user['first_name']} {user['last_name']} ({user['email']}) <span style="background: rgba(255,255,255,0.1); padding: 2px 6px; border-radius: 4px; font-size: 0.8rem; margin-left: 5px;">{role_label}</span>
    </div>
    """, unsafe_allow_html=True)
with col_logout:
    if st.button("🚪 Log Out", use_container_width=True):
        st.session_state.logged_in_user = None
        st.session_state.selected_patient_email = None
        st.session_state.tracker.clear()
        st.session_state.patient_data = None
        st.session_state.risk_prob = None
        st.session_state.risk_level = None
        st.session_state.messages = []
        st.rerun()

# If Clinician, render the Multi-Patient dashboard panel
if user['role'] == 'physician':
    st.markdown('<div class="glass-card" style="padding: 1.5rem; border-radius: 15px; margin-bottom: 1.5rem;">', unsafe_allow_html=True)
    st.markdown("#### 📋 Clinician Dashboard — Patient Records")
    
    patients = get_unique_patients(user['id'])
    
    col_sel, col_add = st.columns([6, 4])
    with col_sel:
        # Create list of patient profile emails
        options = patients.copy()
        if not options:
            options = ["No patient records found"]
            
        selected = st.selectbox(
            "Select Patient Profile Context:",
            options,
            index=0 if patients else None,
            label_visibility="visible"
        )
        
        # If selection changed, reload patient data
        if selected and selected != "No patient records found" and selected != st.session_state.selected_patient_email:
            st.session_state.selected_patient_email = selected
            load_user_history()
            st.rerun()
            
    with col_add:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True) # spacing
        # Button to add/create a new patient profile
        with st.popover("➕ Add New Patient Profile", use_container_width=True):
            new_p_email = st.text_input("Patient Email Address")
            new_p_name = st.text_input("Patient Full Name")
            if st.button("Create Profile Context", use_container_width=True):
                if not new_p_email or not new_p_name:
                    st.error("Please enter email and name.")
                else:
                    st.session_state.selected_patient_email = new_p_email.lower().strip()
                    st.session_state.tracker.clear()
                    st.session_state.patient_data = None
                    st.session_state.risk_prob = None
                    st.session_state.risk_level = None
                    st.session_state.messages = []
                    # Pre-seed a dummy entry or let them perform an assessment
                    st.success(f"Selected new patient: {new_p_name}")
                    st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# HERO BANNER
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="hero-banner">
  <h1 class="hero-title">🛡️ HealthGuard-XAI</h1>
  <p class="hero-subtitle">
    Explainable AI for Probabilistic Disease Risk Prediction &amp; Personalized Health Guidance
  </p>
  <div class="hero-badges">
    <span class="badge">🤖 Groq LLaMA-3.1</span>
    <span class="badge">🧠 SHAP Explainability</span>
    <span class="badge">🔄 Counterfactual AI</span>
    <span class="badge">⚡ 5-Model Ensemble</span>
    <span class="badge">🩺 Clinical Rule Engine</span>
    <span class="badge">🔒 Safety Guardrails</span>
  </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🏠 Overview",
    "🩺 Risk Assessment",
    "🔬 Explainability",
    "🔄 Counterfactuals",
    "📊 Model Analytics",
    "🤖 AI Assistant",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════

with tab1:
    # Metric cards
    tracker = st.session_state.tracker
    n_assessments = tracker.count()
    latest        = tracker.latest()
    delta         = tracker.get_delta()

    st.markdown(f"""
    <div class="metric-grid">
      <div class="metric-card mc-blue">
        <div class="metric-icon">📊</div>
        <div class="metric-value">{n_assessments}</div>
        <div class="metric-label">Assessments This Session</div>
      </div>
      <div class="metric-card mc-purple">
        <div class="metric-icon">🎯</div>
        <div class="metric-value">0.823</div>
        <div class="metric-label">Best Model ROC-AUC</div>
      </div>
      <div class="metric-card mc-green">
        <div class="metric-icon">🧬</div>
        <div class="metric-value">8</div>
        <div class="metric-label">Biomarkers Analyzed</div>
      </div>
      <div class="metric-card mc-orange">
        <div class="metric-icon">🛡️</div>
        <div class="metric-value">3</div>
        <div class="metric-label">Safety Guardrail Layers</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown("""
        <div class="section-header">
          <span class="section-icon">🏗️</span>
          <h2 class="section-title">System Architecture</h2>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="arch-card">
          <span class="arch-hl">Patient Biomarker Input</span> (8 clinical features)<br>
          &nbsp;&nbsp;↓<br>
          <span class="arch-hl">Preprocessing Pipeline</span> → MedianImputer → StandardScaler<br>
          &nbsp;&nbsp;↓<br>
          <span class="arch-hl">ML Ensemble</span> → LogReg · RandomForest · XGBoost · MLP · Stacking<br>
          &nbsp;&nbsp;↓<br>
          ┌──────────────────────────────────────────┐<br>
          │ <span class="arch-hl">Risk Score</span> (0.0 → 1.0 probability)         │<br>
          └──────────────────────────────────────────┘<br>
          &nbsp;&nbsp;↓ (branches into 3 parallel pipelines)<br>
          <span class="arch-hl">[A] SHAP Explainability</span> — TreeExplainer / LinearExplainer<br>
          <span class="arch-hl">[B] Counterfactual Engine</span> — Min-distance clinical optimization<br>
          <span class="arch-hl">[C] Recommendation Engine</span> — Rule-based + LLM (Groq LLaMA-3.1)<br>
          &nbsp;&nbsp;↓<br>
          <span class="arch-hl">Safety Guardrails</span> → Emergency · Medication · Diagnostic filters<br>
          &nbsp;&nbsp;↓<br>
          <span class="arch-hl">Health Guidance Assistant</span> — Context-aware multi-turn LLM chat
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="section-header">
          <span class="section-icon">✨</span>
          <h2 class="section-title">Key Capabilities</h2>
        </div>
        """, unsafe_allow_html=True)

        capabilities = [
            ("🎯", "Probabilistic Risk Prediction", "5-model ensemble with calibrated probabilities"),
            ("🧠", "SHAP Explainability",           "Local & global feature attribution"),
            ("🔄", "Counterfactual Analysis",       "Actionable 'what-if' guidance"),
            ("📋", "Hybrid Recommendations",        "Clinical rules + LLM personalisation"),
            ("🤖", "AI Health Assistant",           "Groq LLaMA-3.1 with safety boundaries"),
            ("⚖️", "Fairness Analysis",             "Age-group bias detection"),
        ]

        for icon, title, desc in capabilities:
            st.markdown(f"""
            <div class="glass-card" style="padding: 1rem 1.2rem; margin: 0.5rem 0;">
              <div style="display:flex;align-items:center;gap:0.7rem;">
                <span style="font-size:1.4rem;">{icon}</span>
                <div>
                  <div style="font-weight:700;color:#fff;font-size:0.95rem;">{title}</div>
                  <div style="color:rgba(255,255,255,0.45);font-size:0.8rem;">{desc}</div>
                </div>
              </div>
            </div>
            """, unsafe_allow_html=True)

    # Risk trend (if assessments exist)
    if n_assessments >= 2:
        st.markdown("""
        <div class="section-header">
          <span class="section-icon">📈</span>
          <h2 class="section-title">Risk Trend — This Session</h2>
        </div>
        """, unsafe_allow_html=True)

        trend = tracker.get_risk_trend()
        level_colors = {"Low": "#06D6A0", "Medium": "#FFA500", "High": "#F72585"}
        bar_colors = [level_colors.get(lv, "#4361EE") for lv in trend["levels"]]

        fig_trend = go.Figure(go.Bar(
            x=trend["labels"], y=trend["risks"],
            marker_color=bar_colors,
            text=[f"{r:.1f}%" for r in trend["risks"]],
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>Risk: %{y:.1f}%<extra></extra>",
        ))
        fig_trend.add_hline(y=33, line_dash="dash", line_color="rgba(6,214,160,0.5)",
                            annotation_text="Low Threshold (33%)")
        fig_trend.add_hline(y=66, line_dash="dash", line_color="rgba(247,37,133,0.5)",
                            annotation_text="High Threshold (66%)")
        fig_trend.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(255,255,255,0.03)",
            yaxis=dict(range=[0, 110], title="Risk (%)"),
            xaxis_title="Assessment",
            height=320,
            showlegend=False,
        )
        st.plotly_chart(fig_trend, use_container_width=True)
    else:
        st.markdown("""
        <div class="info-callout">
          💡 Run at least 2 assessments in the <strong>Risk Assessment</strong> tab to see your risk trend here.
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — RISK ASSESSMENT
# ══════════════════════════════════════════════════════════════════════════════

with tab2:
    st.markdown("""
    <div class="section-header">
      <span class="section-icon">🩺</span>
      <h2 class="section-title">Patient Biomarker Assessment</h2>
    </div>
    """, unsafe_allow_html=True)

    with st.form("biomarker_form"):
        col1, col2 = st.columns(2)

        with col1:
            pregnancies    = st.number_input("Pregnancies",                     min_value=0,   max_value=20,    value=1,     step=1)
            glucose        = st.number_input("Fasting Glucose (mg/dL)",          min_value=70,  max_value=300,   value=110,   step=1)
            blood_pressure = st.number_input("Blood Pressure — Diastolic (mmHg)",min_value=40,  max_value=150,   value=72,    step=1)
            skin_thickness = st.number_input("Skin Thickness (mm)",              min_value=5,   max_value=100,   value=25,    step=1)

        with col2:
            insulin        = st.number_input("Serum Insulin (μU/mL)",            min_value=0,   max_value=900,   value=80,    step=1)
            bmi            = st.number_input("BMI (kg/m²)",                      min_value=10.0, max_value=80.0, value=27.5,  step=0.1, format="%.1f")
            dpf            = st.number_input("Diabetes Pedigree Function",        min_value=0.05, max_value=2.50, value=0.35,  step=0.01, format="%.2f")
            age            = st.number_input("Age (years)",                      min_value=18,  max_value=100,   value=35,    step=1)

        # Clinical reference hints
        st.markdown("""
        <div style="display:flex;gap:1rem;flex-wrap:wrap;margin:0.5rem 0;">
          <span style="color:rgba(255,255,255,0.4);font-size:0.78rem;">📌 Glucose: Normal &lt;100 · Pre-DM 100–125 · DM ≥126</span>
          <span style="color:rgba(255,255,255,0.4);font-size:0.78rem;">📌 BMI: Normal 18.5–24.9 · Overweight 25–29.9 · Obese ≥30</span>
          <span style="color:rgba(255,255,255,0.4);font-size:0.78rem;">📌 BP: Normal &lt;120 · Elevated 120–129 · High ≥130</span>
        </div>
        """, unsafe_allow_html=True)

        submitted = st.form_submit_button("🔍 Predict Diabetes Risk", use_container_width=True)

    if submitted:
        biomarkers = {
            "Pregnancies":              float(pregnancies),
            "Glucose":                  float(glucose),
            "BloodPressure":            float(blood_pressure),
            "SkinThickness":            float(skin_thickness),
            "Insulin":                  float(insulin),
            "BMI":                      float(bmi),
            "DiabetesPedigreeFunction": float(dpf),
            "Age":                      float(age),
        }
        st.session_state.patient_data = biomarkers
        st.session_state.shap_values  = None  # clear old SHAP
        st.session_state.cf_result    = None  # clear old CF

        df = biomarkers_to_df(biomarkers)
        X_scaled = scaler.transform(df)
        prob = float(model.predict_proba(X_scaled)[0][1])
        level, css_class, icon = _risk_class(prob)

        st.session_state.risk_prob  = prob
        st.session_state.risk_level = level
        
        # Save to database if authenticated
        if st.session_state.logged_in_user and st.session_state.selected_patient_email:
            save_prediction(
                user_id=st.session_state.logged_in_user['id'],
                patient_email=st.session_state.selected_patient_email,
                biomarkers=biomarkers,
                risk_probability=prob,
                risk_level=level,
                label=f"Assessment {st.session_state.tracker.count() + 1}"
            )
            load_user_history()
        else:
            st.session_state.tracker.add_prediction(biomarkers, prob, level)

        # Gauge
        gauge_color = "#06D6A0" if prob <= 0.33 else "#FFA500" if prob <= 0.66 else "#F72585"
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=prob * 100,
            number=dict(suffix="%", font=dict(size=48, color="white")),
            delta=dict(
                reference=33,
                valueformat=".1f",
                suffix="%",
                increasing=dict(color="#F72585"),
                decreasing=dict(color="#06D6A0"),
            ),
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": "Diabetes Risk Probability", "font": {"size": 18, "color": "rgba(255,255,255,0.7)"}},
            gauge={
                "axis":  {"range": [0, 100], "tickcolor": "rgba(255,255,255,0.3)"},
                "bar":   {"color": gauge_color, "thickness": 0.25},
                "bgcolor": "rgba(255,255,255,0.05)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0,  33], "color": "rgba(6,214,160,0.12)"},
                    {"range": [33, 66], "color": "rgba(255,165,0,0.12)"},
                    {"range": [66,100], "color": "rgba(247,37,133,0.12)"},
                ],
                "threshold": {
                    "line":  {"color": gauge_color, "width": 4},
                    "thickness": 0.75,
                    "value": prob * 100,
                },
            },
        ))
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white"),
            height=340,
            margin=dict(t=60, b=20, l=30, r=30),
        )

        col_gauge, col_info = st.columns([3, 2])
        with col_gauge:
            st.plotly_chart(fig, use_container_width=True)

        with col_info:
            st.markdown(f"""
            <br>
            <div style="text-align:center;">
              <div style="font-size:3rem;margin-bottom:0.5rem;">{icon}</div>
              <div class="risk-badge {css_class}">{level} Risk</div>
              <br>
              <div style="color:rgba(255,255,255,0.5);font-size:0.85rem;margin-top:1rem;">
                Assessment #{st.session_state.tracker.count()} · {datetime.now().strftime("%H:%M")}
              </div>
            </div>
            """, unsafe_allow_html=True)

            if prob > 0.33:
                st.markdown("""
                <div class="warning-callout" style="margin-top:1rem;">
                  ⚠️ Head to the <strong>Counterfactuals</strong> tab to see what changes could bring your risk down.
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style="background:rgba(6,214,160,0.08);border-left:4px solid #06D6A0;border-radius:0 12px 12px 0;padding:1rem 1.2rem;margin-top:1rem;">
                  ✅ Your current biomarker profile indicates <strong>low risk</strong>. Keep maintaining your healthy lifestyle!
                </div>
                """, unsafe_allow_html=True)

        # Feature summary table
        df_display = pd.DataFrame([
            {"Biomarker": FEATURE_LABELS.get(k, k), "Value": f"{v:.2f}" if isinstance(v, float) else v}
            for k, v in biomarkers.items()
        ])
        st.dataframe(df_display, use_container_width=True, hide_index=True)

    elif st.session_state.patient_data is None:
        st.markdown("""
        <div class="info-callout">
          👆 Fill in your biomarker values above and click <strong>Predict Diabetes Risk</strong> to get started.
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — EXPLAINABILITY
# ══════════════════════════════════════════════════════════════════════════════

with tab3:
    st.markdown("""
    <div class="section-header">
      <span class="section-icon">🔬</span>
      <h2 class="section-title">SHAP Feature Explainability</h2>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.patient_data is None:
        st.markdown("""
        <div class="info-callout">
          🩺 Run a <strong>Risk Assessment</strong> first to generate SHAP explanations.
        </div>
        """, unsafe_allow_html=True)
    else:
        if st.button("🧠 Generate SHAP Explanation", use_container_width=True):
            with st.spinner("Computing SHAP values…"):
                df = biomarkers_to_df(st.session_state.patient_data)
                try:
                    sv = _compute_shap(df)
                    st.session_state.shap_values   = sv.tolist()
                    st.session_state.shap_features = FEATURE_ORDER
                except Exception as e:
                    st.error(f"SHAP computation error: {e}")

        if st.session_state.shap_values is not None:
            sv      = np.array(st.session_state.shap_values)
            feats   = st.session_state.shap_features
            data    = st.session_state.patient_data

            col1, col2 = st.columns([3, 2])

            with col1:
                # Waterfall-style bar chart
                sorted_idx = np.argsort(np.abs(sv))[::-1]
                colors = ["#F72585" if sv[i] > 0 else "#06D6A0" for i in sorted_idx]

                fig_shap = go.Figure(go.Bar(
                    x=[sv[i] for i in sorted_idx],
                    y=[feats[i] for i in sorted_idx],
                    orientation="h",
                    marker_color=colors,
                    text=[f"{sv[i]:+.4f}" for i in sorted_idx],
                    textposition="outside",
                    hovertemplate="<b>%{y}</b><br>SHAP value: %{x:.4f}<extra></extra>",
                ))
                fig_shap.add_vline(x=0, line_color="rgba(255,255,255,0.3)", line_width=1)
                fig_shap.update_layout(
                    title="Feature Contribution to Risk (SHAP Values)",
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(255,255,255,0.03)",
                    xaxis_title="SHAP Value (→ increases risk  |  ← decreases risk)",
                    height=380,
                    margin=dict(l=10, r=60, t=50, b=40),
                )
                st.plotly_chart(fig_shap, use_container_width=True)

            with col2:
                # Top risk vs protective factors
                risk_factors   = [(feats[i], sv[i], data[feats[i]]) for i in sorted_idx if sv[i] > 0]
                protect_factors = [(feats[i], sv[i], data[feats[i]]) for i in sorted_idx if sv[i] < 0]

                st.markdown("#### 🔴 Top Risk-Increasing Factors")
                if risk_factors:
                    for feat, val, actual in risk_factors[:4]:
                        st.markdown(f"""
                        <div class="glass-card" style="padding:0.8rem 1rem;margin:0.3rem 0;border-left:3px solid #F72585;">
                          <div style="font-weight:700;color:#fff;">{FEATURE_LABELS.get(feat,feat)}</div>
                          <div style="display:flex;justify-content:space-between;">
                            <span style="color:rgba(255,255,255,0.5);font-size:0.82rem;">Value: {actual:.1f}</span>
                            <span class="shap-pos">+{val:.4f}</span>
                          </div>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.caption("No risk-increasing factors found.")

                st.markdown("#### 🟢 Protective Factors")
                if protect_factors:
                    for feat, val, actual in protect_factors[:4]:
                        st.markdown(f"""
                        <div class="glass-card" style="padding:0.8rem 1rem;margin:0.3rem 0;border-left:3px solid #06D6A0;">
                          <div style="font-weight:700;color:#fff;">{FEATURE_LABELS.get(feat,feat)}</div>
                          <div style="display:flex;justify-content:space-between;">
                            <span style="color:rgba(255,255,255,0.5);font-size:0.82rem;">Value: {actual:.1f}</span>
                            <span class="shap-neg">{val:.4f}</span>
                          </div>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.caption("No protective factors identified.")

            # SHAP magnitude bar
            st.markdown("#### 📊 Feature Importance Magnitude")
            abs_sv      = np.abs(sv)
            sorted_abs  = np.argsort(abs_sv)

            fig_abs = go.Figure(go.Bar(
                x=[abs_sv[i] for i in sorted_abs],
                y=[feats[i] for i in sorted_abs],
                orientation="h",
                marker=dict(
                    color=[abs_sv[i] for i in sorted_abs],
                    colorscale=[[0,"#1a1a2e"],[0.5,"#4361EE"],[1,"#F72585"]],
                    showscale=False,
                ),
                text=[f"{abs_sv[i]:.4f}" for i in sorted_abs],
                textposition="outside",
            ))
            fig_abs.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(255,255,255,0.03)",
                xaxis_title="|SHAP value| — absolute feature importance",
                height=300,
                margin=dict(l=10, r=60, t=20, b=30),
            )
            st.plotly_chart(fig_abs, use_container_width=True)

        else:
            st.markdown("""
            <div class="info-callout">
              Click the button above to compute SHAP values for your current biomarkers.
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — COUNTERFACTUALS
# ══════════════════════════════════════════════════════════════════════════════

with tab4:
    st.markdown("""
    <div class="section-header">
      <span class="section-icon">🔄</span>
      <h2 class="section-title">Counterfactual "What-If" Analysis</h2>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-callout">
      🔄 Counterfactual explanations answer: <strong>"What is the minimum change needed to your biomarkers to bring your risk below the Low threshold?"</strong>
      This helps translate model output into an actionable lifestyle plan.
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.patient_data is None:
        st.markdown("""
        <div class="warning-callout">⚠️ Please run a Risk Assessment first.</div>
        """, unsafe_allow_html=True)
    else:
        col_btn, col_target = st.columns([3, 2])
        with col_target:
            target_pct = st.slider(
                "Target Risk Threshold (%)",
                min_value=10, max_value=50, value=33, step=1,
                help="The algorithm finds the minimum biomarker changes to bring risk below this threshold.",
            )
        with col_btn:
            if st.button("🔄 Generate Counterfactual Action Plan", use_container_width=True):
                with st.spinner("Running counterfactual optimization…"):
                    cf = generate_counterfactual(
                        model=model,
                        scaler=scaler,
                        original_biomarkers=st.session_state.patient_data,
                        target_risk=target_pct / 100.0,
                    )
                    st.session_state.cf_result = cf

        if st.session_state.cf_result is not None:
            cf = st.session_state.cf_result
            orig_risk = cf["original_risk"]
            cf_risk   = cf["counterfactual_risk"]
            changes   = cf["changes"]

            # Summary metrics
            reduction = orig_risk - cf_risk
            c1, c2, c3 = st.columns(3)
            c1.metric("Original Risk", f"{orig_risk:.1%}", delta=None)
            c2.metric("Counterfactual Risk", f"{cf_risk:.1%}",
                      delta=f"-{reduction:.1%}" if reduction > 0 else "0%",
                      delta_color="inverse")
            c3.metric("Risk Reduction", f"{reduction:.1%}",
                      delta="✅ Target achieved!" if cf["achieved"] else "⚠️ Partial",
                      delta_color="normal" if cf["achieved"] else "off")

            if changes:
                # Before/after gauge comparison
                c_before, c_after = st.columns(2)

                def _mini_gauge(val, label, color):
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=val * 100,
                        number=dict(suffix="%", font=dict(size=32, color="white")),
                        domain={"x": [0, 1], "y": [0, 1]},
                        title={"text": label, "font": {"size": 14, "color": "rgba(255,255,255,0.6)"}},
                        gauge={
                            "axis": {"range": [0, 100], "tickcolor": "rgba(255,255,255,0.3)"},
                            "bar":  {"color": color, "thickness": 0.25},
                            "bgcolor": "rgba(255,255,255,0.05)",
                            "borderwidth": 0,
                            "steps": [
                                {"range": [0,  33], "color": "rgba(6,214,160,0.12)"},
                                {"range": [33, 66], "color": "rgba(255,165,0,0.12)"},
                                {"range": [66,100], "color": "rgba(247,37,133,0.12)"},
                            ],
                        },
                    ))
                    fig.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)", font=dict(color="white"),
                        height=240, margin=dict(t=50, b=10, l=20, r=20),
                    )
                    return fig

                orig_color = "#06D6A0" if orig_risk <= 0.33 else "#FFA500" if orig_risk <= 0.66 else "#F72585"
                cf_color   = "#06D6A0" if cf_risk  <= 0.33 else "#FFA500" if cf_risk  <= 0.66 else "#F72585"

                c_before.plotly_chart(_mini_gauge(orig_risk, "Current Risk", orig_color), use_container_width=True)
                c_after.plotly_chart(_mini_gauge(cf_risk, "After Changes", cf_color), use_container_width=True)

                # Change list
                st.markdown("#### 🎯 Required Biomarker Adjustments")
                for ch in changes:
                    feat_label = FEATURE_LABELS.get(ch["feature"], ch["feature"])
                    st.markdown(f"""
                    <div class="cf-card">
                      <div>
                        <div class="cf-feature">{feat_label}</div>
                        <div class="cf-values">{ch['original_value']} → {ch['new_value']}</div>
                      </div>
                      <div style="text-align:right;">
                        <div class="cf-change">{ch['direction']} {abs(ch['change']):.2f}</div>
                        <div style="color:rgba(255,255,255,0.4);font-size:0.78rem;">change needed</div>
                      </div>
                    </div>
                    """, unsafe_allow_html=True)

                # Horizontal bar chart of changes
                fig_cf = go.Figure(go.Bar(
                    x=[abs(ch["change"]) for ch in changes],
                    y=[FEATURE_LABELS.get(ch["feature"], ch["feature"]) for ch in changes],
                    orientation="h",
                    marker_color=["#4CC9F0"] * len(changes),
                    text=[f"{ch['direction']} {abs(ch['change']):.2f}" for ch in changes],
                    textposition="outside",
                ))
                fig_cf.update_layout(
                    title="Magnitude of Required Changes",
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(255,255,255,0.03)",
                    height=max(250, len(changes) * 55),
                    margin=dict(l=10, r=60, t=40, b=30),
                )
                st.plotly_chart(fig_cf, use_container_width=True)

                if not cf["achieved"]:
                    st.markdown("""
                    <div class="warning-callout">
                      ⚠️ Target risk threshold could not be fully achieved within clinically plausible bounds.
                      Try a higher target threshold or consult a healthcare professional.
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.success("✅ " + cf["message"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — MODEL ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════

with tab5:
    st.markdown("""
    <div class="section-header">
      <span class="section-icon">📊</span>
      <h2 class="section-title">Model Performance Analytics</h2>
    </div>
    """, unsafe_allow_html=True)

    results_df = load_model_comparison()

    if results_df is None:
        st.markdown("""
        <div class="warning-callout">
          ⚠️ Model comparison data not found. Run <code>python -m risk_modeling.train</code> to generate it.
        </div>
        """, unsafe_allow_html=True)
    else:
        from risk_modeling.evaluation import (
            get_plotly_model_comparison,
            get_plotly_roc_curves,
            get_plotly_calibration_curves,
            get_plotly_confusion_matrix,
        )

        # Metrics table with color coding
        st.markdown("#### 📋 Model Comparison Table")
        display_cols = ["Model", "Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC", "Brier Score", "CV_AUC_mean"]
        df_show = results_df[display_cols].copy()
        df_show.columns = ["Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "Brier↓", "CV AUC"]

        # Highlight best model
        best_idx = df_show["ROC-AUC"].idxmax()

        styled = df_show.style \
            .format({c: "{:.4f}" for c in df_show.columns if c != "Model"}) \
            .highlight_max(subset=["Accuracy","Precision","Recall","F1","ROC-AUC","CV AUC"],
                           color="#1a2a4a") \
            .highlight_min(subset=["Brier↓"], color="#1a2a4a") \
            .set_properties(**{"background-color": "transparent", "color": "white"})

        st.dataframe(df_show, use_container_width=True, hide_index=True)

        best_model_name = results_df.loc[best_idx, "Model"]
        best_auc = results_df.loc[best_idx, "ROC-AUC"]
        st.markdown(f"""
        <div style="background:rgba(67,97,238,0.12);border:1px solid rgba(67,97,238,0.3);border-radius:10px;padding:0.8rem 1.2rem;margin:0.5rem 0;">
          🏆 <strong>Best Model:</strong> {best_model_name} &nbsp;|&nbsp; <strong>ROC-AUC:</strong> {best_auc:.4f}
        </div>
        """, unsafe_allow_html=True)

        # Charts
        st.markdown("#### 📈 Interactive Performance Charts")
        chart_tab1, chart_tab2, chart_tab3 = st.tabs(["ROC Curves", "Model Comparison", "Calibration"])

        with chart_tab1:
            st.markdown("""
            <div class="info-callout">
              ROC curves show the trade-off between sensitivity (True Positive Rate) and specificity (False Positive Rate).
              A model with perfect discrimination has AUC = 1.0. The dashed line represents random chance (AUC = 0.5).
            </div>
            """, unsafe_allow_html=True)

            # We need test probabilities — load from disk if available
            roc_img_path = os.path.join("outputs", "plots", "roc_curves.png")
            if os.path.exists(roc_img_path):
                st.image(roc_img_path, use_container_width=True, caption="ROC Curves — All Models (from training run)")
            else:
                st.info("ROC curve plot not found. Run `python -m risk_modeling.train` to generate it.")

        with chart_tab2:
            fig_cmp = get_plotly_model_comparison(results_df)
            st.plotly_chart(fig_cmp, use_container_width=True)

        with chart_tab3:
            cal_img_path = os.path.join("outputs", "plots", "calibration_curves.png")
            if os.path.exists(cal_img_path):
                st.image(cal_img_path, use_container_width=True, caption="Calibration Curves — All Models")
                st.markdown("""
                <div class="info-callout">
                  Well-calibrated models have curves close to the dashed diagonal.
                  Calibration indicates whether predicted probabilities match actual outcome rates.
                </div>
                """, unsafe_allow_html=True)
            else:
                st.info("Calibration curves not found. Run training first.")

        # Individual model confusion matrices
        st.markdown("#### 🔲 Confusion Matrices")
        cm_dir = os.path.join("outputs", "plots")
        cm_files = []
        if os.path.exists(cm_dir):
            cm_files = [f for f in os.listdir(cm_dir) if f.startswith("confusion_matrix_")]

        if cm_files:
            cm_cols = st.columns(min(3, len(cm_files)))
            for i, fname in enumerate(cm_files[:6]):
                model_lbl = fname.replace("confusion_matrix_", "").replace(".png", "").replace("_", " ")
                cm_cols[i % 3].image(
                    os.path.join(cm_dir, fname),
                    caption=model_lbl,
                    use_container_width=True,
                )
        else:
            st.info("Confusion matrix plots not found. Run training first.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — AI ASSISTANT
# ══════════════════════════════════════════════════════════════════════════════

with tab6:
    st.markdown("""
    <div class="section-header">
      <span class="section-icon">🤖</span>
      <h2 class="section-title">HealthGuard AI Assistant</h2>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-callout">
      🤖 Powered by <strong>Groq LLaMA-3.1-8B</strong> with multi-layer safety guardrails.
      This assistant provides <em>educational health information only</em> — not medical advice.
      Emergency, medication, and diagnosis requests are automatically redirected.
    </div>
    """, unsafe_allow_html=True)

    # Display chat history
    chat_html = ""
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            chat_html += f"""
            <div class="chat-msg-user">
              <div class="chat-role">You</div>
              {msg['content']}
            </div>"""
        else:
            # Build interactive collapsible RAG references if present
            rag_html = ""
            if msg.get("rag_sources"):
                rag_html += '<div style="margin-top: 10px; border-top: 1px dashed rgba(255,255,255,0.1); padding-top: 8px;">'
                rag_html += '<span style="font-size: 0.75rem; color: #4CC9F0; font-weight: 600; display: block; margin-bottom: 6px;">📚 Clinical Reference Sources (RAG):</span>'
                for doc in msg["rag_sources"]:
                    rag_html += f"""
                    <details style="margin-bottom: 4px; font-size: 0.8rem; background: rgba(255, 255, 255, 0.03); border-radius: 6px; padding: 6px 10px; border: 1px solid rgba(255,255,255,0.06);">
                      <summary style="cursor: pointer; font-weight: 500; color: rgba(255,255,255,0.85); outline: none;">{doc['title']} (similarity: {doc['score']:.2f})</summary>
                      <div style="margin-top: 6px; color: rgba(255,255,255,0.6); line-height: 1.35;">{doc['content']}</div>
                    </details>
                    """
                rag_html += '</div>'

            chat_html += f"""
            <div class="chat-msg-assistant">
              <div class="chat-role">🛡️ HealthGuard AI</div>
              {msg['content']}
              {rag_html}
            </div>"""

    if chat_html:
        st.markdown(f'<div class="chat-container">{chat_html}</div>', unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="chat-container" style="display:flex;align-items:center;justify-content:center;height:180px;">
          <div style="text-align:center;color:rgba(255,255,255,0.3);">
            <div style="font-size:2.5rem;margin-bottom:0.5rem;">💬</div>
            <div>Ask me anything about your health profile, lifestyle improvements, or risk factors.</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    # Context status
    if st.session_state.patient_data:
        prob = st.session_state.risk_prob or 0
        level = st.session_state.risk_level or "Unknown"
        st.markdown(f"""
        <div style="background:rgba(67,97,238,0.1);border:1px solid rgba(67,97,238,0.2);border-radius:8px;padding:0.6rem 1rem;margin-bottom:0.8rem;display:flex;gap:1rem;align-items:center;">
          <span style="color:#4CC9F0;font-size:0.85rem;font-weight:600;">🔗 Patient context loaded</span>
          <span style="color:rgba(255,255,255,0.4);font-size:0.82rem;">Risk: {prob:.1%} ({level})</span>
        </div>
        """, unsafe_allow_html=True)

    # Quick suggestion chips
    if not st.session_state.messages:
        suggestions = [
            "What does my glucose level mean?",
            "How can I lower my BMI effectively?",
            "Explain my risk factors in simple terms",
            "What lifestyle changes help prevent diabetes?",
        ]
        cols = st.columns(len(suggestions))
        for i, sug in enumerate(suggestions):
            if cols[i].button(sug, key=f"sug_{i}"):
                st.session_state._pending_message = sug
                st.rerun()

    # Process pending message from suggestion chips
    pending = st.session_state.pop("_pending_message", None)

    prompt = st.chat_input("Ask a health question…", key="chat_input_main")
    prompt = prompt or pending

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        if st.session_state.logged_in_user and st.session_state.selected_patient_email:
            save_chat_message(
                st.session_state.logged_in_user['id'],
                st.session_state.selected_patient_email,
                "user",
                prompt
            )

        rag_sources = []
        if st.session_state.patient_data is None:
            response = "👋 Welcome! Please run a **Risk Assessment** first so I can provide context-aware guidance based on your specific biomarkers."
        else:
            with st.spinner("HealthGuard AI is thinking…"):
                try:
                    # Build rich patient context
                    patient_ctx = {
                        "patient_profile":   st.session_state.patient_data,
                        "predicted_risks":   {"diabetes": st.session_state.risk_prob or 0.0},
                        "risk_level":        st.session_state.risk_level or "Unknown",
                        "top_risk_factors":  [],
                        "protective_factors": [],
                    }
                    history = st.session_state.messages[:-1]
                    resp_dict = health_llm_ask(
                        user_input=prompt,
                        patient_data=patient_ctx,
                        conversation_history=history,
                    )
                    response = resp_dict.get("assistant_response", str(resp_dict))
                    rag_sources = resp_dict.get("retrieved_context", [])

                    # Safety badge
                    if resp_dict.get("safety_flag"):
                        response = "🛡️ **Safety Alert**\n\n" + response
                    if resp_dict.get("escalation_required"):
                        response += "\n\n---\n**🚨 Please contact emergency services or your physician immediately.**"

                except Exception as e:
                    response = f"⚠️ LLM Error: {e}\n\nEnsure `GROQ_API_KEY` is set in your `.env` file."

        st.session_state.messages.append({
            "role": "assistant",
            "content": response,
            "rag_sources": rag_sources
        })
        if st.session_state.logged_in_user and st.session_state.selected_patient_email:
            save_chat_message(
                st.session_state.logged_in_user['id'],
                st.session_state.selected_patient_email,
                "assistant",
                response
            )
        st.rerun()

    if st.session_state.messages:
        if st.button("🗑️ Clear Conversation", use_container_width=False):
            if st.session_state.logged_in_user and st.session_state.selected_patient_email:
                clear_chat_history(
                    st.session_state.logged_in_user['id'],
                    st.session_state.selected_patient_email
                )
            st.session_state.messages = []
            st.rerun()
