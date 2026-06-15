import pandas as pd
import os
from joblib import load

# Safe loading
ARTIFACTS_DIR = 'artifacts'
model_rest = load(os.path.join(ARTIFACTS_DIR, 'model_rest.joblib'))
model_young = load(os.path.join(ARTIFACTS_DIR, 'model_young.joblib'))
scaler_rest = load(os.path.join(ARTIFACTS_DIR, 'scaler_rest.joblib'))
scaler_young = load(os.path.join(ARTIFACTS_DIR, 'scaler_young.joblib'))

# Model features (exactly as trained)
MODEL_FEATURES = [
    'age', 'number_of_dependants', 'income_lakhs',
    'insurance_plan', 'genetical_risk', 'normalized_risk_score',
    'gender_Male', 'region_Northwest', 'region_Southeast',
    'region_Southwest', 'marital_status_Unmarried',
    'bmi_category_Obesity', 'bmi_category_Overweight',
    'bmi_category_Underweight', 'smoking_status_Occasional Smoker',
    'smoking_status_Regular Smoker', 'employment_status_Salaried',
    'employment_status_Self-Employed'
]

SCALER_FEATURES = MODEL_FEATURES + ['income_level']

# Encodings
insurance_plan_encoding = {'Bronze': 0, 'Silver': 1, 'Gold': 2}
income_level_encoding = {
    '<10L': 0, '10L - 25L': 1,
    '25L - 40L': 2, '> 40L': 3
}

smoking_status_map = {
    'Regular': 'Regular Smoker',
    'Occasional': 'Occasional Smoker',
    'No Smoking': None,
    'Does Not Smoke': None,
    'Not Smoking': None,
    'Smoking=0': None
}

risk_scores = {
    "diabetes": 6, "heart disease": 8,
    "high blood pressure": 6, "thyroid": 5,
    "no disease": 0, "none": 0
}
MIN_SCORE = 0
MAX_SCORE = 14

def get_normalized_risk_score(medical_history):
    diseases = medical_history.lower().split(' & ')
    d1 = diseases[0]
    d2 = diseases[1] if len(diseases) > 1 else 'none'
    total = risk_scores[d1] + risk_scores[d2]
    return (total - MIN_SCORE) / (MAX_SCORE - MIN_SCORE)

def preprocessed_input(input_dict):
    df = pd.DataFrame([{col: 0 for col in SCALER_FEATURES}])

    df['age'] = input_dict['Age']
    df['number_of_dependants'] = input_dict['Number of Dependants']
    df['income_lakhs'] = input_dict['Income in Lakhs']
    df['genetical_risk'] = input_dict['Genetical Risk']
    df['insurance_plan'] = insurance_plan_encoding[input_dict['Insurance Plan']]
    df['income_level'] = income_level_encoding[input_dict['Income Level']]

    if input_dict['Gender'] == 'Male':
        df['gender_Male'] = 1

    region = input_dict['Region']
    if region == 'Northwest':
        df['region_Northwest'] = 1
    elif region == 'Southeast':
        df['region_Southeast'] = 1
    elif region == 'Southwest':
        df['region_Southwest'] = 1

    if input_dict['Marital Status'] == 'Unmarried':
        df['marital_status_Unmarried'] = 1

    bmi = input_dict['BMI Category']
    if bmi == 'Obesity':
        df['bmi_category_Obesity'] = 1
    elif bmi == 'Overweight':
        df['bmi_category_Overweight'] = 1
    elif bmi == 'Underweight':
        df['bmi_category_Underweight'] = 1

    smoke = input_dict['Smoking Status']
    mapped = smoking_status_map.get(smoke)
    if mapped == 'Regular Smoker':
        df['smoking_status_Regular Smoker'] = 1
    elif mapped == 'Occasional Smoker':
        df['smoking_status_Occasional Smoker'] = 1

    emp = input_dict['Employment Status']
    if emp == 'Salaried':
        df['employment_status_Salaried'] = 1
    elif emp == 'Self-Employed':
        df['employment_status_Self-Employed'] = 1

    df['normalized_risk_score'] = get_normalized_risk_score(
        input_dict['Medical History']
    )

    # Scaling: handle both dict or direct scaler
    df = handle_scaling(input_dict['Age'], df)

    df.drop(columns=['income_level'], inplace=True)
    return df[MODEL_FEATURES]

def handle_scaling(age, df):
    scaler_obj = scaler_young if age <= 25 else scaler_rest
    # Try to extract scaler and columns from dict, else assume it's the scaler itself
    if isinstance(scaler_obj, dict):
        cols = scaler_obj['col_to_scale']
        scaler = scaler_obj['scaler']
    else:
        # Assume the object is the scaler and we must scale all numerical columns
        cols = ['age', 'income_lakhs', 'number_of_dependants', 'genetical_risk',
                'insurance_plan', 'income_level', 'normalized_risk_score']
        scaler = scaler_obj
    df[cols] = scaler.transform(df[cols])
    return df

def predict(input_dict):
    input_df = preprocessed_input(input_dict)
    model = model_young if input_dict['Age'] <= 25 else model_rest
    pred = model.predict(input_df)
    return int(pred[0])