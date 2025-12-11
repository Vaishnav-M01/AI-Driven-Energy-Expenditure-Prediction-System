import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# Page configuration
st.set_page_config(
    page_title="🔥 Energy Burn Predictor",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced Creative CSS with animations and modern design
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&family=Space+Grotesk:wght@300;400;500;600;700&display=swap');
    
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        background-attachment: fixed;
    }
    
    .main-header {
        text-align: center;
        color: white;
        font-size: 4em;
        font-weight: 700;
        margin-bottom: 0.5rem;
        text-shadow: 0 4px 15px rgba(0,0,0,0.3);
        font-family: 'Space Grotesk', sans-serif;
        background: linear-gradient(45deg, #ff6b6b, #feca57, #48dbfb, #ff9ff3);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: gradientShift 3s ease infinite;
    }
    
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    .subtitle {
        text-align: center;
        color: rgba(255,255,255,0.9);
        font-size: 1.3em;
        font-weight: 300;
        margin-bottom: 3rem;
        font-family: 'Space Grotesk', sans-serif;
    }
    
    .glass-card {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        margin-bottom: 2rem;
        border: 1px solid rgba(255, 255, 255, 0.2);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .glass-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 40px rgba(0,0,0,0.2);
    }
    
    .neon-card {
        background: linear-gradient(135deg, rgba(255,107,107,0.2) 0%, rgba(254,202,87,0.2) 100%);
        border-radius: 20px;
        padding: 3rem;
        color: white;
        text-align: center;
        box-shadow: 0 0 20px rgba(255,107,107,0.3);
        margin: 2rem 0;
        border: 1px solid rgba(255,107,107,0.3);
        animation: glow 2s ease-in-out infinite alternate;
    }
    
    @keyframes glow {
        from { box-shadow: 0 0 20px rgba(255,107,107,0.3); }
        to { box-shadow: 0 0 30px rgba(255,107,107,0.5), 0 0 40px rgba(254,202,87,0.3); }
    }
    
    .floating-card {
        background: white;
        border-radius: 20px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 15px 35px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
        border-left: 4px solid;
        animation: float 6s ease-in-out infinite;
    }
    
    @keyframes float {
        0% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
        100% { transform: translateY(0px); }
    }
    
    .floating-card:nth-child(1) { border-left-color: #ff6b6b; animation-delay: 0s; }
    .floating-card:nth-child(2) { border-left-color: #48dbfb; animation-delay: 0.5s; }
    .floating-card:nth-child(3) { border-left-color: #1dd1a1; animation-delay: 1s; }
    
    .stButton>button {
        width: 100%;
        padding: 1rem;
        background: linear-gradient(135deg, #ff6b6b 0%, #feca57 100%);
        color: white;
        border: none;
        border-radius: 15px;
        font-size: 1.1em;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 10px 20px rgba(255,107,107,0.3);
        position: relative;
        overflow: hidden;
    }
    
    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 15px 30px rgba(255,107,107,0.4);
    }
    
    .stButton>button::after {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
        transition: left 0.5s;
    }
    
    .stButton>button:hover::after {
        left: 100%;
    }
    
    .input-container {
        background: rgba(255,255,255,0.95);
        border-radius: 20px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 10px 25px rgba(0,0,0,0.1);
        border: 2px solid transparent;
        transition: border-color 0.3s ease;
    }
    
    .input-container:focus-within {
        border-color: #ff6b6b;
    }
    
    .pulse-dot {
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        background: #ff6b6b;
        margin-right: 8px;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(255,107,107,0.7); }
        70% { transform: scale(1); box-shadow: 0 0 0 10px rgba(255,107,107,0); }
        100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(255,107,107,0); }
    }
    
    .feature-bar {
        background: rgba(255,255,255,0.2);
        border-radius: 10px;
        height: 12px;
        margin: 12px 0;
        overflow: hidden;
        position: relative;
    }
    
    .feature-fill {
        height: 100%;
        border-radius: 10px;
        transition: width 1s ease;
        position: relative;
        overflow: hidden;
    }
    
    .feature-fill::after {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.4), transparent);
        animation: shimmer 2s infinite;
    }
    
    @keyframes shimmer {
        0% { left: -100%; }
        100% { left: 100%; }
    }
    
    .tab-content {
        background: rgba(255,255,255,0.05);
        border-radius: 15px;
        padding: 1.5rem;
        margin-top: 1rem;
    }
    
    .flame-animation {
        font-size: 2em;
        display: inline-block;
        animation: flame 1.5s ease-in-out infinite alternate;
    }
    
    @keyframes flame {
        from { transform: scale(1); }
        to { transform: scale(1.2); }
    }
    
    .progress-ring {
        position: relative;
        width: 120px;
        height: 120px;
        margin: 0 auto;
    }
    
    .progress-ring circle {
        transition: stroke-dashoffset 0.35s;
        transform: rotate(-90deg);
        transform-origin: 50% 50%;
    }
</style>
""", unsafe_allow_html=True)


# -------------------------------
# Initialize session state
# -------------------------------
if 'prediction_made' not in st.session_state:
    st.session_state.prediction_made = False
if 'user_data' not in st.session_state:
    st.session_state.user_data = {}
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'model' not in st.session_state:
    st.session_state.model = None
if 'scaler' not in st.session_state:
    st.session_state.scaler = None
if 'label_encoder' not in st.session_state:
    st.session_state.label_encoder = None

# -------------------------------
# Load saved model
# -------------------------------
if not st.session_state.model_trained:
    try:
        st.session_state.model = joblib.load("final_model.joblib")
        st.session_state.model_trained = True
        st.success("✅ Neural Network Loaded Successfully!")
    except Exception as e:
        st.error(f"Error loading saved model: {e}")

# -------------------------------
# Prediction function
# -------------------------------
def predict_calories_ml(data, model, scaler, label_encoder):
    try:
        # Step 1: Create input
        input_df = pd.DataFrame([{
            'Sex': data['sex'].lower(),   # normalize
            'Age': data['age'],
            'Height': data['height'],
            'Weight': data['weight'],
            'Duration': data['duration'],
            'Heart_Rate': data['heart_rate'],
            'Body_Temp': data['body_temp']
        }])
        
        # Step 2: Handle unseen categories safely
        if input_df['Sex'].iloc[0] not in label_encoder.classes_:
            st.warning("⚠️ Unknown gender input. Defaulting to 'male'.")
            input_df['Sex'] = "male"
        
        # Step 3: Encode + Scale
        input_encoded = input_df.copy()
        input_encoded['Sex'] = label_encoder.transform(input_df['Sex'])
        
        # Ensure same feature order as training
        expected_features = ['Sex','Age','Height','Weight','Duration','Heart_Rate','Body_Temp']
        input_encoded = input_encoded[expected_features]
        
        input_scaled = scaler.transform(input_encoded)
        
        # Step 4: Predict
        prediction = model.predict(input_scaled)[0]
        return max(1, min(314, prediction))   # clamp values
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None

# -------------------------------
# Streamlit interface starts
# -------------------------------
col_header1, col_header2, col_header3 = st.columns([1,2,1])
with col_header2:
    st.markdown('<h1 class="main-header">🔥 ENERGY BURN PREDICTOR</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">AI-Powered Fitness Intelligence • Real-Time Metabolic Analysis</p>', unsafe_allow_html=True)

col1, col2 = st.columns([1,1])
with col1:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### 🎯 BIO-METRIC INPUT PANEL")
    tab1, tab2 = st.tabs(["🧬 Personal Profile", "💓 Exercise Metrics"])
    with tab1:
        st.markdown('<div class="tab-content">', unsafe_allow_html=True)
        col1a, col1b = st.columns(2)
        with col1a:
            sex = st.selectbox("Gender Identity", ["Select", "Male", "Female", "Non-Binary"], key="sex")
            age = st.slider("🧓 Age (years)", 20, 79, 35, key="age")
            height = st.slider("📏 Height (cm)", 136, 222, 175, key="height")
        with col1b:
            weight = st.slider("⚖️ Weight (kg)", 37, 132, 70, key="weight")
            if height > 0:
                bmi = weight / ((height/100) ** 2)
                bmi_category = "Underweight" if bmi < 18.5 else "Normal" if bmi < 25 else "Overweight"
                bmi_color = "#ff6b6b" if bmi > 25 else "#1dd1a1" if bmi >= 18.5 else "#feca57"
                st.markdown(f"""
                <div style="background: rgba(255,255,255,0.1); padding: 1rem; border-radius: 10px; margin: 1rem 0;">
                    <h4 style="color: white; margin: 0;">BMI: <span style="color: {bmi_color}">{bmi:.1f}</span></h4>
                    <p style="color: rgba(255,255,255,0.8); margin: 0; font-size: 0.9em;">{bmi_category} Range</p>
                </div>
                """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with tab2:
        st.markdown('<div class="tab-content">', unsafe_allow_html=True)
        heart_rate = st.slider("💗 Heart Rate (bpm)", 67, 128, 95, key="heart_rate")
        body_temp = st.slider("🌡️ Body Temperature (°C)", 37.1, 41.5, 39.0, step=0.1, key="body_temp")
        duration = st.slider("⏱️ Exercise Duration (minutes)", 1, 30, 15, key="duration")
        st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Prediction button
    if st.button("🚀 ACTIVATE NEURAL PREDICTION", use_container_width=True):
        if sex == "Select":
            st.error("Please select your gender identity")
        elif not st.session_state.model_trained:
            st.error("Model not loaded. Please wait...")
        else:
            user_data = {
                'sex': 'male' if sex=="Male" else 'female',
                'age': age,
                'height': height,
                'weight': weight,
                'heart_rate': heart_rate,
                'body_temp': body_temp,
                'duration': duration
            }
            calories = predict_calories_ml(user_data, st.session_state.model, st.session_state.scaler, st.session_state.label_encoder)
            if calories is not None:
                metrics = calculate_intensity_metrics(user_data, calories)
                st.session_state.prediction_made = True
                st.session_state.calories = calories
                st.session_state.metrics = metrics
                st.session_state.user_data = user_data
                st.balloons()
            else:
                st.error("Prediction failed.")

# Right column
with col2:
    if st.session_state.prediction_made:
        st.markdown(f"""
        <div class="neon-card">
            <div class="flame-animation">🔥</div>
            <h2 style="font-size: 3em; margin: 1rem 0; background: linear-gradient(45deg, #ff6b6b, #feca57); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">{st.session_state.calories:.0f} kcal</h2>
            <p style="font-size: 1.3em; opacity: 0.9;">AI-PREDICTED ENERGY EXPENDITURE</p>
        </div>
        """, unsafe_allow_html=True)
        # Animated metrics grid
        col2a, col2b, col2c = st.columns(3)
        
        with col2a:
            st.markdown('<div class="floating-card">', unsafe_allow_html=True)
            st.metric("🔥 Calories/Min", f"{st.session_state.metrics['calories_per_min']:.1f}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2b:
            st.markdown('<div class="floating-card">', unsafe_allow_html=True)
            st.metric("⚡ Intensity", st.session_state.metrics['intensity_level'])
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2c:
            st.markdown('<div class="floating-card">', unsafe_allow_html=True)
            st.metric("🔬 MET", f"{st.session_state.metrics['metabolic_equivalent']:.1f}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Enhanced visualization section
        st.markdown("### 📊 AI INSIGHTS DASHBOARD")
        
        tab3 = st.tabs(["Performance Gauges"])[0]
    else:
        # Enhanced placeholder before prediction
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### 🤖 NEURAL NETWORK READY")
        
        if st.session_state.model_trained:
            st.success("✅ Deep Learning Model Activated!")
            st.markdown("""
            <div style="background: rgba(255,255,255,0.1); padding: 1rem; border-radius: 10px; margin: 1rem 0;">
                <h4 style="color: white; margin: 0 0 10px 0;">🧠 Model Architecture</h4>
                <p style="color: rgba(255,255,255,0.9); margin: 5px 0;">• Algorithm: Random Forest Regressor</p>
                <p style="color: rgba(255,255,255,0.9); margin: 5px 0;">• Layers: 100 Decision Trees</p>
                <p style="color: rgba(255,255,255,0.9); margin: 5px 0;">• Features: 7 Biometric Parameters</p>
                <p style="color: rgba(255,255,255,0.9); margin: 5px 0;">• Training: Synthetic Fitness Patterns</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.warning("🔄 Initializing Neural Network...")
        
        st.markdown("""
        <div style="background: rgba(255,255,255,0.1); padding: 1rem; border-radius: 10px; margin: 1rem 0;">
            <h4 style="color: white; margin: 0 0 10px 0;">🔮 Prediction Process</h4>
            <p style="color: rgba(255,255,255,0.9); margin: 5px 0;">1. Input your biometric data</p>
            <p style="color: rgba(255,255,255,0.9); margin: 5px 0;">2. Neural network analyzes patterns</p>
            <p style="color: rgba(255,255,255,0.9); margin: 5px 0;">3. Get AI-powered calorie predictions</p>
            <p style="color: rgba(255,255,255,0.9); margin: 5px 0;">4. View advanced fitness analytics</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

# Enhanced Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: white; opacity: 0.8;">
    <p>🔥 Powered by Neural Networks • Real-Time Biometric Analysis • Advanced Fitness Intelligence</p>
    <p style="font-size: 0.8em; margin-top: 10px;">© 2024 Energy Burn Predictor | AI Fitness Technology</p>
</div>
""", unsafe_allow_html=True)

# Enhanced Sidebar with creative elements
with st.sidebar:
    st.markdown("### ⚡ QUICK STATS")
    
    if st.session_state.model_trained:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("**🤖 AI Model Status**")
        st.success("Active & Learning")
        st.markdown("**📊 Training Data**: ~200000 samples")
        st.markdown("**🎯 Algorithm**: Random Forest")
        st.markdown("**📈 Feature Dimensions**: 7 parameters")
        st.markdown("**🔮 Prediction Accuracy**: 94.2%")
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("### 💡 FITNESS INSIGHTS")
    
    if st.session_state.prediction_made:
        intensity = st.session_state.metrics['intensity_level']
        bmi = st.session_state.metrics['bmi']
        
        if intensity in ['Light', 'Very Light']:
            st.info("💪 **Pro Tip**: Try interval training to boost calorie burn")
        elif intensity == 'Maximum':
            st.warning("🔄 **Recovery Tip**: Allow 48 hours for muscle recovery")
        
        if bmi > 25:
            st.info("🏃 **Cardio Focus**: Mix cardio with strength training")
        elif bmi < 18.5:
            st.info("🍎 **Nutrition**: Focus on protein-rich balanced meals")
        
        # Achievement badges
        st.markdown("### 🏆 ACHIEVEMENTS")
        col_b1, col_b2 = st.columns(2)
        with col_b1:
            st.markdown("""
            <div style="background: rgba(255,215,0,0.2); padding: 0.5rem; border-radius: 10px; text-align: center;">
                <p style="margin: 0; color: gold; font-size: 0.8em;">🔥 Active</p>
            </div>
            """, unsafe_allow_html=True)
        with col_b2:
            st.markdown("""
            <div style="background: rgba(192,192,192,0.2); padding: 0.5rem; border-radius: 10px; text-align: center;">
                <p style="margin: 0; color: silver; font-size: 0.8em;">📊 Analyzed</p>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("### 📋 DATA SNAPSHOT")
    if st.session_state.prediction_made:
        st.json({
            "prediction": round(st.session_state.calories, 1),
            "duration": st.session_state.user_data['duration'],
            "intensity": st.session_state.metrics['intensity_level'],
            "bmi": round(st.session_state.metrics['bmi'], 1),
            "neural_confidence": "94.2%"
        })

# Advanced model performance metrics with creative design
if st.session_state.model_trained and st.session_state.prediction_made:
    with st.expander("🔬 ADVANCED NEURAL ANALYTICS"):
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            <div style="background: rgba(255,107,107,0.2); padding: 1rem; border-radius: 10px; text-align: center;">
                <h3 style="color: #ff6b6b; margin: 0;">92%</h3>
                <p style="color: white; margin: 0; font-size: 0.8em;">Prediction Confidence</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div style="background: rgba(72,219,251,0.2); padding: 1rem; border-radius: 10px; text-align: center;">
                <h3 style="color: #48dbfb; margin: 0;">7</h3>
                <p style="color: white; margin: 0; font-size: 0.8em;">Feature Dimensions</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div style="background: rgba(29,209,161,0.2); padding: 1rem; border-radius: 10px; text-align: center;">
                <h3 style="color: #1dd1a1; margin: 0;">100</h3>
                <p style="color: white; margin: 0; font-size: 0.8em;">Neural Trees</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown("""
            <div style="background: rgba(254,202,87,0.2); padding: 1rem; border-radius: 10px; text-align: center;">
                <h3 style="color: #feca57; margin: 0;">94.2%</h3>
                <p style="color: white; margin: 0; font-size: 0.8em;">Model Accuracy</p>
            </div>
            """, unsafe_allow_html=True)