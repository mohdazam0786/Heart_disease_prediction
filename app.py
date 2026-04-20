import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="HeartCare AI",
    page_icon="❤️",
    layout="wide" 
)

st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #ff4b4b;
        color: white;
        font-weight: bold;
    }
    .stButton>button:hover {
        border: 1px solid #ff4b4b;
        color: #ff4b4b;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_assets():
    model = joblib.load("KNN_heart.pkl")
    scaler = joblib.load("scaler.pkl")
    expected_columns = joblib.load("columns.pkl")
    return model, scaler, expected_columns

try:
    model, scaler, expected_columns = load_assets()
except:
    st.error("Error loading models. Ensure .pkl files are in the directory.")

# sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/822/822118.png", width=100)
    st.title("About HeartCare AI")
    st.info("This tool uses a K-Nearest Neighbors (KNN) algorithm to predict the likelihood of heart disease based on clinical parameters.")
    st.warning("Disclaimer: This is for educational purposes only and not a substitute for professional medical advice.")

# main
st.title("❤️ Heart Disease Prediction")
st.write("Complete the clinical profile below to generate a risk report.")

with st.container():
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("👤 Patient Info")
        age = st.number_input("Age", 1, 120, 45)
        sex = st.selectbox("Biological Sex", ["M", "F"])
        max_hr = st.slider("Max Heart Rate Achieved", 60, 220, 150)

    with col2:
        st.subheader("🩺 Vital Signs")
        resting_bp = st.number_input("Resting BP (mm Hg)", 80, 200, 120)
        cholesterol = st.number_input("Serum Cholestoral (mg/dl)", 100, 600, 200)
        fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"])

    with col3:
        st.subheader("📊 Clinical Tests")
        chest_pain = st.selectbox("Chest Pain Type", ["ASY", "ATA", "NAP", "TA"])
        resting_ecg = st.selectbox("Resting ECG Results", ["Normal", "ST", "LVH"])
        exercise_angina = st.selectbox("Exercise Induced Angina", ["N", "Y"])
        st_slope = st.selectbox("ST Slope Segment", ["Up", "Flat", "Down"])
        oldpeak = st.number_input("Oldpeak (ST Depression)", 0.0, 6.0, 1.0, step=0.1)

st.divider()

# Prediction Logic
bs_val = 1 if fasting_bs == "Yes" else 0

if st.button("Generate Risk Report"):
    raw_input = {
        'Age': age,
        'RestingBP': resting_bp,
        'Cholesterol': cholesterol,
        'FastingBS': bs_val,
        'MaxHR': max_hr,
        'Oldpeak': oldpeak,
        'Sex_' + sex: 1,
        'ChestPainType_' + chest_pain: 1,
        'RestingECG_' + resting_ecg: 1,
        'ExerciseAngina_' + exercise_angina: 1,
        'ST_Slope_' + st_slope: 1
    }

    input_df = pd.DataFrame([raw_input])
    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[expected_columns]

    scaled_input = scaler.transform(input_df)
    prediction = model.predict(scaled_input)[0]
    prob = model.predict_proba(scaled_input)[0][1]

    # Results Display
    st.subheader("Analysis Results")
    
    res_col1, res_col2 = st.columns([1, 2])

    with res_col1:
        if prediction == 1:
            st.error("### HIGH RISK")
        else:
            st.success("### LOW RISK")
        
        st.metric(label="Risk Probability", value=f"{prob*100:.1f}%")
        st.progress(prob)

    with res_col2:
        if prediction == 1:
            st.markdown("""
            **Recommended Actions:**
            * Consult a cardiologist for a thorough examination.
            * Monitor blood pressure and cholesterol levels daily.
            * Review lifestyle factors such as diet and exercise.
            """)
        else:
            st.markdown("""
            **Health Maintenance:**
            * Continue regular physical activity.
            * Maintain a balanced diet low in saturated fats.
            * Schedule annual routine checkups.
            """)

    with st.expander("See Feature Weights"):
        st.write("This shows how your inputs compare to the model's training features.")
        st.dataframe(input_df)