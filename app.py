import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# Local imports
try:
    from backend.dependencies import load_model_and_scaler, get_model, get_scaler, biomarkers_to_df
    from health_llm_assistant.assistant import ask as health_llm_ask
    from phase3_recommendation_engine.hybrid_recommender import HybridRecommender
except ImportError as e:
    st.error(f"Import Error: {e}. Please ensure the python structure is correct.")

st.set_page_config(page_title="HealthGuard-XAI", page_icon="🛡️", layout="wide")

# Cached model loading
@st.cache_resource
def init_system():
    load_model_and_scaler()
    return get_model(), get_scaler(), None, HybridRecommender(get_model())

try:
    model, scaler, llm_assistant, recommender = init_system()
except Exception as e:
    st.error(f"System Initialization Error: {e}")
    st.stop()

st.title("🛡️ HealthGuard-XAI")
st.markdown("### Probabilistic Risk Prediction & Explainability Dashboard")

tab1, tab2, tab3, tab4 = st.tabs(["🩺 Risk Assessment", "🔍 Explainability", "📋 Personalized Plan", "🤖 AI Assistant"])

# Session state for storing current patient inputs
if 'patient_data' not in st.session_state:
    st.session_state.patient_data = None
if 'risk_prob' not in st.session_state:
    st.session_state.risk_prob = None

with tab1:
    st.header("Patient Biomarkers Form")
    col1, col2 = st.columns(2)
    
    with col1:
        pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=0)
        glucose = st.number_input("Glucose", min_value=0.0, max_value=300.0, value=120.0)
        blood_pressure = st.number_input("Blood Pressure", min_value=0.0, max_value=200.0, value=70.0)
        skin_thickness = st.number_input("Skin Thickness", min_value=0.0, max_value=100.0, value=20.0)
        
    with col2:
        insulin = st.number_input("Insulin", min_value=0.0, max_value=1000.0, value=80.0)
        bmi = st.number_input("BMI", min_value=0.0, max_value=100.0, value=25.0)
        dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5)
        age = st.number_input("Age", min_value=0, max_value=120, value=30)
        
    if st.button("Predict Risk", use_container_width=True):
        data = {
            "pregnancies": pregnancies,
            "glucose": glucose,
            "blood_pressure": blood_pressure,
            "skin_thickness": skin_thickness,
            "insulin": insulin,
            "bmi": bmi,
            "diabetes_pedigree_function": dpf,
            "age": age
        }
        st.session_state.patient_data = data
        df = biomarkers_to_df(data)
        
        X_scaled = scaler.transform(df)
        prob = float(model.predict_proba(X_scaled)[0][1])
        st.session_state.risk_prob = prob
        
        # Display Prediction Gauge
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = prob * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Risk Probability (%)"},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkred" if prob > 0.66 else "orange" if prob > 0.33 else "green"},
                'steps' : [
                    {'range': [0, 33], 'color': "lightgreen"},
                    {'range': [33, 66], 'color': "lightyellow"},
                    {'range': [66, 100], 'color': "salmon"}],
            }
        ))
        st.plotly_chart(fig, use_container_width=True)
        
        if prob <= 0.33:
            st.success("Low Risk")
        elif prob <= 0.66:
            st.warning("Medium Risk")
        else:
            st.error("High Risk")

with tab2:
    st.header("SHAP Feature Explainability")
    if st.session_state.patient_data is not None:
        if st.button("Generate SHAP Explanation"):
            with st.spinner("Calculating SHAP values..."):
                import shap
                df = biomarkers_to_df(st.session_state.patient_data)
                FEATURE_ORDER = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]
                X_scaled = pd.DataFrame(scaler.transform(df), columns=FEATURE_ORDER)
                
                explainer = shap.TreeExplainer(model)
                shap_vals = explainer.shap_values(X_scaled)
                if isinstance(shap_vals, list):
                    shap_vals = shap_vals[1]
                sv = shap_vals[0]
                
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(figsize=(8, 4))
                
                colors = ['red' if v > 0 else 'blue' for v in sv]
                bars = ax.barh(FEATURE_ORDER, sv, color=colors)
                ax.axvline(0, color='black', linewidth=0.8)
                ax.set_title("Feature Contribution to Risk")
                st.pyplot(fig)
    else:
        st.info("Please run a prediction first.")

with tab3:
    st.header("Hybrid Recommendations")
    if st.session_state.patient_data is not None:
        if st.button("Generate Recommendations"):
            with st.spinner("Synthesizing rules + LLM plan..."):
                rec = recommender.generate_recommendations(st.session_state.patient_data)
                st.write(rec)
    else:
        st.info("Please run a prediction first.")

with tab4:
    st.header("Chat with HealthGuard Assistant")
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask a health question..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            if st.session_state.patient_data is None:
                response = "Please run a Risk Assessment first so I can understand your health context!"
            else:
                try:
                    history = st.session_state.messages[:-1]
                    resp_dict = health_llm_ask(
                        user_input=prompt,
                        patient_data=st.session_state.patient_data,
                        conversation_history=history
                    )
                    response = resp_dict.get("assistant_response", str(resp_dict))
                except Exception as e:
                    response = f"LLM Error: {e}"
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
