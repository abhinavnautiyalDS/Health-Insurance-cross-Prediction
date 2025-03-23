import os

os.system("pip install -r requirements.txt")

import streamlit as st
import numpy as np
import pickle
import requests
import os

# Streamlit Page Configuration
st.set_page_config(page_title="Insurance Policy Predictor", page_icon="💰", layout="wide")

# Model Path (Temporary Local File)
MODEL_URL = "https://huggingface.co/abhinav965108/insurance_prediction/resolve/main/RandomForestModel.pkl"
MODEL_PATH = "RandomForestModel.pkl"

# Download the model if not exists
def download_model():
    if not os.path.exists(MODEL_PATH):
        st.info("⏳ Downloading model from Hugging Face...")
        try:
            response = requests.get(MODEL_URL, stream=True)
            response.raise_for_status()
            with open(MODEL_PATH, "wb") as file:
                for chunk in response.iter_content(chunk_size=1024):
                    file.write(chunk)
            st.success("✅ Model downloaded successfully!")
        except Exception as e:
            st.error(f"⚠️ Error downloading model: {e}")

# Load Model
@st.cache_resource
def load_model():
    try:
        with open(MODEL_PATH, "rb") as file:
            model = pickle.load(file)
        return model
    except Exception as e:
        st.error(f"⚠️ Error loading model: {e}")
        return None

# Download and Load Model
download_model()
model = load_model()

# Min-Max Scaling Constants
MIN_REGION, MAX_REGION = 0.0, 52.0
MIN_CHANNEL, MAX_CHANNEL = 1.0, 165.0
MIN_PREMIUM, MAX_PREMIUM = 2630.0, 540000.0

def min_max_scale(value, min_val, max_val):
    return (value - min_val) / (max_val - min_val)

# Function to make predictions
def predict_response(features):
    if model is None:
        return "⚠️ Model not available for prediction"
    try:
        input_data = np.array([features], dtype=np.float32)
        return model.predict(input_data)[0]
    except Exception as e:
        return f"⚠️ Prediction failed: {e}"

# UI Components
st.markdown("<h1 style='text-align: center;'>🚀 Vehicle Insurance Policy Predictor 💰</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: #FEB47B;'>Find out if a customer will take the policy!</h3>", unsafe_allow_html=True)

col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### 📊 Customer Information")
    age_category = st.selectbox("Select Age Category", ["<20", "20-40", "40-60", "60+"], index=1)
    gender = st.radio("Gender", ["Male", "Female"], horizontal=True)
    region_code = min_max_scale(st.slider("Region Code", 0, 52, 28), MIN_REGION, MAX_REGION)
    previously_insured = st.radio("Previously Insured", ["Yes", "No"], horizontal=True)

with col2:
    st.markdown("### 🚗 Vehicle & Policy Details")
    vehicle_age = st.radio("Vehicle Age", ["<1 year", "1-2 years", ">2 years"], horizontal=True)
    annual_premium = min_max_scale(st.slider("Annual Premium", 2630, 540000, 40000), MIN_PREMIUM, MAX_PREMIUM)
    policy_sales_channel = min_max_scale(st.slider("Policy Sales Channel", 1, 165, 26), MIN_CHANNEL, MAX_CHANNEL)
    customer_type = st.radio("Customer Type", ["Long-term", "Mid-term", "Short-term"], horizontal=True)

# Predict Button
st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)

if st.button("🔥 Predict Now 🔥"):
    user_input = [
        1 if age_category == "60+" else 0,
        1 if age_category == "40-60" else 0,
        1 if age_category == "20-40" else 0,
        1 if gender == "Male" else 0,  # Gender_Male
        float(region_code),
        1 if vehicle_age == "1-2 years" else 0, 
        1 if vehicle_age == "<1 year" else 0, 
        1 if vehicle_age == ">2 years" else 0,
        float(annual_premium),
        1 if previously_insured == "Yes" else 0,  
        float(policy_sales_channel), 
        1 if customer_type == "Long-term" else 0, 
        1 if customer_type == "Mid-term" else 0, 
        1 if customer_type == "Short-term" else 0
    ]
    
    prediction = predict_response(user_input)
    
    if prediction is not None:
        result = "✅ Likely to Take the Policy!" if int(prediction) == 1 else "❌ Not Interested in the Policy"
        st.success(result)
        if int(prediction) == 1:
            st.balloons()

# Custom Footer Styling
st.markdown(
    """
    <style>
        .footer {
            position: fixed;
            bottom: 10px;
            right: 10px;
            font-size: 14px;
            color: white;
            background-color: black;
            padding: 5px 10px;
            border-radius: 5px;
            opacity: 0.7;
        }
    </style>
    <div class="footer">
        Created by Abhinav Nautiyal
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("</div>", unsafe_allow_html=True)
