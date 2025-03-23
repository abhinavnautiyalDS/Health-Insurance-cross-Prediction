pip install -r requirements.txt

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
        st.success("✅ Model loaded successfully!")
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
    age_60_plus = st.checkbox("Age 60+")
    age_40_60 = st.checkbox("Age Between 40 and 60")
    age_20_40 = st.checkbox("Age Between 20 and 40")
    gender = st.radio("Gender", ["Male", "Female"], horizontal=True)
    region_code = min_max_scale(st.number_input("Region Code", min_value=0.0, value=28.0, step=1.0), MIN_REGION, MAX_REGION)
    previously_insured = st.radio("Previously Insured", ["Yes", "No"], horizontal=True)

with col2:
    st.markdown("### 🚗 Vehicle & Policy Details")
    vehicle_age = st.radio("Vehicle Age", ["<1 year", "1-2 years", ">2 years"], horizontal=True)
    annual_premium = min_max_scale(st.number_input("Annual Premium", min_value=0.0, value=40000.0), MIN_PREMIUM, MAX_PREMIUM)
    policy_sales_channel = min_max_scale(st.number_input("Policy Sales Channel", min_value=0.0, value=26.0
