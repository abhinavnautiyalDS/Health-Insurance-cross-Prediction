
import streamlit as st
import joblib
import numpy as np

# Load the trained model
model = joblib.load('/content/drive/MyDrive/learning/RandomForestModel1.joblib')

# Title of the Streamlit app
st.title('Insurance Renewal Prediction')

# Input features
st.header('Enter the details:')

# Age groups
age_group = st.selectbox('Age Group', ['60+', '40-60', '20-40'])
Age60_plus = 1 if age_group == '60+' else 0
AgeBetween40And60 = 1 if age_group == '40-60' else 0
AgeBetween20And40 = 1 if age_group == '20-40' else 0

# Gender
gender = st.selectbox('Gender', ['Male', 'Female'])
Gender_Male = 1 if gender == 'Male' else 0
Gender_Female = 1 if gender == 'Female' else 0

# Region code
Region_Code = st.number_input('Region Code', min_value=0, step=1)

# Vehicle age
vehicle_age = st.selectbox('Vehicle Age', ['1-2 years', '<1 year', '>2 years'])
Vehicle_age_1_2_years = 1 if vehicle_age == '1-2 years' else 0
Vehicle_age_less_1_years = 1 if vehicle_age == '<1 year' else 0
Vehicle_age_greater_2_years = 1 if vehicle_age == '>2 years' else 0

# Annual premium
Annual_Premium_Fixed = st.number_input('Annual Premium Fixed', min_value=0.0)

# Previously insured
previously_insured = st.selectbox('Previously Insured', ['Yes', 'No'])
Previously_Insured_yes = 1 if previously_insured == 'Yes' else 0
Previously_Insured_no = 1 if previously_insured == 'No' else 0

# Policy sales channel
Policy_Sales_Channel = st.number_input('Policy Sales Channel', min_value=0, step=1)

# Customer term
customer_term = st.selectbox('Customer Term', ['Long Term', 'Mid Term', 'Short Term'])
Long_term_customer = 1 if customer_term == 'Long Term' else 0
mid_term_customer = 1 if customer_term == 'Mid Term' else 0
short_term_customer = 1 if customer_term == 'Short Term' else 0

# Prediction button
if st.button('Predict'):
    # Input features as a numpy array
    features = np.array([
        Age60_plus, AgeBetween40And60, AgeBetween20And40, Gender_Male,
        Gender_Female, Region_Code, Vehicle_age_1_2_years,
        Vehicle_age_less_1_years, Vehicle_age_greater_2_years, Annual_Premium_Fixed,
        Previously_Insured_yes, Previously_Insured_no,
        Policy_Sales_Channel, Long_term_customer, mid_term_customer,
        short_term_customer
    ]).reshape(1, -1)

    # Make prediction
    prediction = model['model'].predict(features)

    # Display the result
    if prediction == 1:
        st.success('The policyholder is likely to renew their policy.')
    else:
        st.error('The policyholder is unlikely to renew their policy.')
