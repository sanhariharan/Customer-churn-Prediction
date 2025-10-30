import streamlit as st
import numpy as np
import pandas as pd
import pickle
from tensorflow.keras.models import load_model

# =======================
# PAGE CONFIG
# =======================
st.set_page_config(
    page_title="Customer Churn Prediction - ANN Model",
    page_icon="💼",
    layout="centered",
)

# =======================
# CUSTOM CSS (Elegant UI)
# =======================
st.markdown("""
    <style>
        body {
            background-color: #f8fafc;
        }
        .main-title {
            text-align: center;
            font-size: 38px;
            font-weight: 800;
            color: #1f4e79;
            margin-bottom: 10px;
        }
        .sub-title {
            text-align: center;
            font-size: 18px;
            color: #5a5a5a;
            margin-bottom: 30px;
        }
        .form-card {
            background: white;
            padding: 25px 30px;
            border-radius: 18px;
            box-shadow: 0px 8px 20px rgba(0,0,0,0.08);
        }
        .result-card {
            background: linear-gradient(145deg, #eef6ff, #d6e9ff);
            border-radius: 18px;
            padding: 25px;
            margin-top: 25px;
            text-align: center;
            box-shadow: 0px 8px 20px rgba(0,0,0,0.08);
        }
        .prob-box {
            font-size: 22px;
            font-weight: 600;
            color: #154360;
            margin-top: 15px;
        }
        .positive {
            background: linear-gradient(135deg, #ffe5e0, #ffb3a7);
            color: #922B21;
            border: 1px solid #f5b7b1;
        }
        .negative {
            background: linear-gradient(135deg, #d5f5e3, #a9dfbf);
            color: #145A32;
            border: 1px solid #82e0aa;
        }
        .footer {
            text-align: center;
            font-size: 14px;
            color: gray;
            margin-top: 25px;
        }
        .predict-btn button {
            background: linear-gradient(90deg, #1f4e79, #2471a3);
            color: white !important;
            border-radius: 12px;
            font-weight: 600;
            padding: 0.6em 1em;
            border: none;
        }
        .predict-btn button:hover {
            background: linear-gradient(90deg, #154360, #1a5276);
        }
    </style>
""", unsafe_allow_html=True)

# =======================
# TITLE
# =======================
st.markdown('<div class="main-title">💼 Customer Churn Prediction</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Predict whether a customer will churn using our trained ANN model</div>', unsafe_allow_html=True)

# =======================
# LOAD MODEL & TRANSFORMERS
# =======================
model = load_model("model.h5")

with open("label_encoder_gender.pkl", "rb") as f:
    label_encoder_gender = pickle.load(f)
with open("onehot_encoder_geo.pkl", "rb") as f:
    onehot_encoder_geo = pickle.load(f)
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# =======================
# SIDEBAR
# =======================
st.sidebar.title("ℹ️ About This App")
st.sidebar.info("""
This interactive web app predicts whether a customer will **churn** 
based on profile and account details.

The model used: **Artificial Neural Network (ANN)**
""")
st.sidebar.markdown("---")
st.sidebar.write("👨‍💻 **Created by:** San Hariharan")
st.sidebar.write("🧠 **Model:** ANN | TensorFlow & Streamlit")

# =======================
# FORM INPUT
# =======================
st.markdown('<div class="form-card">', unsafe_allow_html=True)
st.markdown("### 🧾 Enter Customer Details")

col1, col2 = st.columns(2)
with col1:
    geography = st.selectbox('🌍 Geography', onehot_encoder_geo.categories_[0])
    gender = st.selectbox('👤 Gender', label_encoder_gender.classes_)
    age = st.slider('🎂 Age', 18, 92, 35)
    tenure = st.slider('📆 Tenure (Years)', 0, 10, 3)
    num_of_products = st.slider('🛍️ Number of Products', 1, 4, 1)

with col2:
    credit_score = st.number_input('💳 Credit Score', 300, 850, 650)
    balance = st.number_input('💰 Balance', 0.0, 250000.0, 10000.0, step=1000.0)
    estimated_salary = st.number_input('💼 Estimated Salary', 0.0, 200000.0, 50000.0, step=1000.0)
    has_cr_card = st.selectbox('💳 Has Credit Card', [0, 1])
    is_active_member = st.selectbox('✅ Is Active Member', [0, 1])

st.markdown('</div>', unsafe_allow_html=True)

# =======================
# PREDICTION LOGIC
# =======================
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

input_scaled = scaler.transform(input_data)

# =======================
# BUTTON + OUTPUT
# =======================
col_center = st.columns([1, 2, 1])[1]
with col_center:
    predict_clicked = st.button("🔍 Predict Churn", use_container_width=True)

if predict_clicked:
    with st.spinner("Analyzing with ANN model..."):
        prediction = model.predict(input_scaled)
        probability = float(prediction[0][0])
    
    st.markdown('<div class="result-card">', unsafe_allow_html=True)
    st.markdown(f"<div class='prob-box'>📈 Churn Probability: <strong>{probability:.2f}</strong></div>", unsafe_allow_html=True)

    if probability > 0.5:
        st.markdown(
            "<div class='result-card positive'>⚠️ The customer is <b>likely to churn</b>.<br>Consider retention offers and engagement strategies.</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            "<div class='result-card negative'>✅ The customer is <b>not likely to churn</b>.<br>Maintain current satisfaction and communication level.</div>",
            unsafe_allow_html=True,
        )

    st.markdown('</div>', unsafe_allow_html=True)

# =======================
# FOOTER
# =======================
st.markdown("""
<div class="footer">
© 2025 | Created by <b>San Hariharan</b> | Model Used: <b>Artificial Neural Network (ANN)</b> <br>
Powered by TensorFlow × Streamlit
</div>
""", unsafe_allow_html=True)
