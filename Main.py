import streamlit as st
from prediction_helper import predict


st.header("Health Insurance Cost Predictor")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=30)

    gender = st.selectbox(
        "Gender",
        ['Male', 'Female']
    )

    marital_status = st.selectbox(
        "Marital Status",
        ['Unmarried', 'Married']
    )

with col2:
    number_of_dependants = st.number_input(
        "Number of Dependants",
        min_value=0,
        max_value=10,
        value=1
    )

    region = st.selectbox(
        "Region",
        ['Northwest', 'Southeast', 'Northeast', 'Southwest']
    )

    bmi_category = st.selectbox(
        "BMI Category",
        ['Normal', 'Obesity', 'Overweight', 'Underweight']
    )

with col3:
    income_lakhs = st.number_input(
        "Income (Lakhs)",
        min_value=0.0,
        max_value=100.0,
        value=10.0,
        step=0.5
    )

    genetical_risk = st.number_input(
        "Genetical Risk",
        min_value=0,
        max_value=5,
        value=2
    )

    insurance_plan = st.selectbox(
        "Insurance Plan",
        ['Bronze', 'Silver', 'Gold']
    )

st.divider()

col4, col5, col6 = st.columns(3)

with col4:
    smoking_status = st.selectbox(
        "Smoking Status",
        ['No Smoking', 'Regular', 'Occasional',
         'Does Not Smoke', 'Not Smoking', 'Smoking=0']
    )

with col5:
    employment_status = st.selectbox(
        "Employment Status",
        ['Salaried', 'Self-Employed', 'Freelancer']
    )

with col6:
    income_level = st.selectbox(
        "Income Level",
        ['<10L', '10L - 25L', '25L - 40L', '> 40L']
    )

medical_history = st.selectbox(
    "Medical History",
    ['Diabetes',
     'High blood pressure',
     'No Disease',
     'Diabetes & High blood pressure',
     'Thyroid',
     'Heart disease',
     'High blood pressure & Heart disease',
     'Diabetes & Thyroid',
     'Diabetes & Heart disease']
)

if st.button("Predict Premium"):

    input_dict = {
        'Age': age,
        'Number of Dependants': number_of_dependants,
        'Income in Lakhs': income_lakhs,
        'Genetical Risk': genetical_risk,
        'Insurance Plan': insurance_plan,
        'Employment Status': employment_status,
        'Gender': gender,
        'Marital Status': marital_status,
        'BMI Category': bmi_category,
        'Smoking Status': smoking_status,
        'Region': region,
        'Medical History': medical_history,
        'Income Level': income_level
    }

    prediction = predict(input_dict)

    st.success(
        f"Predicted Annual Premium: ₹{prediction:,.2f}"
    )










