import streamlit as st
import pandas as pd
import joblib 

model = joblib.load('./KNN_heart.pkl')
scaler = joblib.load('./scaler.pkl')
columns = joblib.load('./columns.pkl')

st.title("Heart Stroke Prediction By Kaushal ❤️")
st.markdown("Provide The Following Details")


age = st.slider("Age", 18, 100, 40)
sex = st.selectbox("SEX",['Male','Female','Other'])
chest_pain = st.selectbox("Chest Pain Type",['ATA','NAP','TA','ASY'])
resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
cholesterol = st.number_input("Cholesterol (mg/DL)",100,600,200)
fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/DL",[0,1])
resting_ecg = st.selectbox("Resting ECG",['Normal','ST','LVH'])
max_hr = st.slider("Max heart Rate",60,220,150)
exercise_angina = st.selectbox("Exercise-Induced Angina",['Yes','No'])
oldpeak = st.slider("Oldpeak (ST Depression)",0.0,6.0,1.0) 
st_slope = st.selectbox("ST Slope",["Up","Flat","Down"])


if st.button("Predict"):

    # Create a raw input dictionary
    raw_input = {
        'age': age,
        'restingbp': resting_bp,
        'cholesterol': cholesterol,
        'fastingbs': fasting_bs,
        'maxhr': max_hr,
        'oldpeak': oldpeak,
        'sex_' + sex: 1,
        'chestpaintype_' + chest_pain: 1,
        'restingecg_' + resting_ecg: 1,
        'exerciseangina_' + exercise_angina: 1,
        'st_slope_' + st_slope: 1
    }

    # Create input dataframe
    input_df = pd.DataFrame([raw_input])

    # Fill in missing columns with 0s
    for col in columns:
        if col not in input_df.columns:
            input_df[col] = 0

    # Reorder columns
    input_df = input_df[columns]

    # Scale the input
    scaled_input = scaler.transform(input_df)

    # Make prediction
    prediction = model.predict(scaled_input)[0]

    # Show result
    if prediction == 1:
        st.error("⚠️ High Risk of Heart Disease")
    else:
        st.success("✅ Low Risk of Heart Disease")