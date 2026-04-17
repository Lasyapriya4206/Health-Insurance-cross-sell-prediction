import streamlit as st
import pickle
import numpy as np

import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(BASE_DIR, "models", "saved_model.pkl")

model = pickle.load(open(model_path, "rb"))

st.title("Health Insurance Cross-Sell Prediction")

Gender = st.selectbox("Gender", ["Male", "Female"])

if Gender == "Male":
    Gender = 1
else:
    Gender = 0
age = st.number_input("Age", 18, 100)
driving_license = st.selectbox("Driving License", [0, 1])
Region_code=st.number_input("Region Code",1,50)
previously_insured = st.selectbox("Previously Insured", [0, 1])
Vehicle_Age = st.selectbox(
    "Vehicle Age",
    ["< 1 Year", "1-2 Year", "> 2 Years"]
)

vehicle_age_map = {
    "< 1 Year": 0,
    "1-2 Year": 1,
    "> 2 Years": 2
}

Vehicle_Age = vehicle_age_map[Vehicle_Age]
Vehicle_Damage = st.selectbox("Vehicle Damage", ["No", "Yes"])

if Vehicle_Damage == "Yes":
    Vehicle_Damage = 1
else:
    Vehicle_Damage = 0
annual_premium = st.number_input("Annual Premium", 1000, 100000)
policy_sales_channel= st.number_input("Policy sales channel",1,200)
vintage = st.number_input("Vintage", 0, 500)

if st.button("Predict"):
    input_data = np.array([[Gender, age, driving_license, Region_code,
                            previously_insured, Vehicle_Age, Vehicle_Damage,
                            annual_premium, policy_sales_channel, vintage]])

    prob = model.predict_proba(input_data)[0][1]

    st.write("Probability of Buying:", prob)

    if prob > 0.2:   # increase threshold
        st.success("Will Buy Insurance")
    else:
        st.error("Will Not Buy")
    