import streamlit as st
import pandas as pd
import joblib

model = joblib.load("model_xgb.pkl")
scaler = joblib.load("scaler.pkl")
expected_columns = joblib.load("columns.pkl")

st.title("Employee Attrition Prediction")
st.markdown(
    "Enter the employee's information below to estimate the "
    "predicted risk of attrition."
)

numerical_cols = ['Age', 'DistanceFromHome', 'EnvironmentSatisfaction',
                   'JobInvolvement', 'JobLevel', 'JobSatisfaction',
                   'MonthlyIncome', 'RelationshipSatisfaction',
                   'StockOptionLevel', 'TrainingTimesLastYear', 'YearsAtCompany',
                   'YearsSinceLastPromotion', 'YearsWithCurrManager']

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 18, 65, 30)
    gender = st.selectbox("Gender", ["Male", "Female"])
    department = st.selectbox("Department", ['Human Resources', 'Sales', 'Research & Development'])
    jobrole = st.selectbox("Job Role", ['Healthcare Representative', 'Human Resources', 'Laboratory Technician',
                                          'Manager', 'Manufacturing Director', 'Research Director',
                                          'Research Scientist', 'Sales Executive', 'Sales Representative'])
    joblevel = st.selectbox("Job Level", [1, 2, 3, 4, 5])
    jobinvolvement = st.selectbox("Job Involvement", [1, 2, 3, 4, 5])
    jobsatisfaction = st.selectbox("Job Satisfaction", [1, 2, 3, 4, 5])
    environmentsatisfaction = st.selectbox("Environment Satisfaction", [1, 2, 3, 4, 5])
    overtime = st.selectbox("Overtime", ['Yes', 'No'])

with col2:
    distfromhome = st.slider("Distance From Home (in miles)", 1, 50, 10)
    businesstravel = st.selectbox("Business Travel", ['Non-Travel', 'Travel_Frequently', 'Travel_Rarely'])
    stockoption = st.selectbox("Stock Option Level", [0, 1, 2, 3])
    monthlyincome = st.number_input("Monthly Income", 1000, 50000, 5000)
    relationshipsatisfaction = st.selectbox("Relationship Satisfaction", [1, 2, 3, 4, 5])
    trainingtimeslastyear = st.slider("Training Times Last Year", 0, 6, 2)
    yearsatcompany = st.slider("Years at Company", 0, 50, 5)
    yearswithcurrmanager = st.slider("Years with Current Manager", 0, 50, 3)
    yearsincelastpromotion = st.slider("Years Since Last Promotion", 0, 30, 2)

if st.button("Predict Attrition", use_container_width=True):
    raw_input = {
        'Age': age,
        'DistanceFromHome': distfromhome,
        'EnvironmentSatisfaction': environmentsatisfaction,
        'Gender': 1 if gender == "Male" else 0,
        'JobInvolvement': jobinvolvement,
        'JobLevel': joblevel,
        'JobSatisfaction': jobsatisfaction,
        'MonthlyIncome': monthlyincome,
        'OverTime': 1 if overtime == "Yes" else 0,
        'RelationshipSatisfaction': relationshipsatisfaction,
        'StockOptionLevel': stockoption,
        'TrainingTimesLastYear': trainingtimeslastyear,
        'YearsAtCompany': yearsatcompany,
        'YearsSinceLastPromotion': yearsincelastpromotion,
        'YearsWithCurrManager': yearswithcurrmanager,
        'Department_' + department: 1,
        'JobRole_' + jobrole: 1,
        'BusinessTravel_' + businesstravel: 1,
    }

    input_df = pd.DataFrame([raw_input])

    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[expected_columns]

    input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])
    prediction = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.error(f"⚠️ High Attrition Risk ({proba:.1%} probability)")
    else:
        st.success(f"✅ Low Attrition Risk ({proba:.1%} probability)")