import streamlit as st
import pandas as pd
import joblib

# 1. Load the saved model and the column list
try:
    model = joblib.load('random_forest_model.pkl')
    model_columns = joblib.load('model_columns.pkl')
except:
    st.error("Model files not found! Make sure you ran the training script first.")
    st.stop()

# 2. App Title and Description
st.title("🏦 Bank Account Predictor")
st.write("Enter the details below to check if a person is likely to have a bank account.")

# 3. Create the Input Form
with st.form("prediction_form"):
    # Categorical Inputs (Dropdowns)
    # Note: You might want to update these lists with the exact unique values from your dataset if I missed any
    col1, col2 = st.columns(2)

    with col1:
        country = st.selectbox("Country", ["Kenya", "Rwanda", "Tanzania", "Uganda"])
        loc_type = st.selectbox("Location", ["Rural", "Urban"])
        cellphone = st.selectbox("Has Cellphone?", ["Yes", "No"])
        gender = st.selectbox("Gender", ["Male", "Female"])
        marital = st.selectbox("Marital Status",
                               ["Married/Living together", "Single/Never Married", "Widowed", "Divorced/Seperated"])

    with col2:
        edu = st.selectbox("Education Level", ["Primary education", "Secondary education", "Tertiary education",
                                               "Vocational/Specialised training", "No formal education"])
        job = st.selectbox("Job Type", ["Self employed", "Government Dependent", "Formally employed Private",
                                        "Formally employed Government", "Farming and Fishing", "Remittance Dependent",
                                        "Other Income", "Dont Know/Refuse to answer", "No Income"])
        rel = st.selectbox("Relationship with Head",
                           ["Head of Household", "Spouse", "Child", "Parent", "Other relative", "Other non-relatives"])

    # Numerical Inputs
    age = st.number_input("Age", min_value=16, max_value=100, value=30)
    household = st.number_input("Household Size", min_value=1, max_value=20, value=4)

    # Submit Button
    submitted = st.form_submit_button("Predict Status")

# 4. Logic when the button is clicked
if submitted:
    # A. Create a DataFrame from the inputs
    input_data = pd.DataFrame({
        'country': [country],
        'location_type': [loc_type],
        'cellphone_access': [cellphone],
        'gender_of_respondent': [gender],
        'relationship_with_head': [rel],
        'marital_status': [marital],
        'education_level': [edu],
        'job_type': [job],
        'household_size': [household],
        'age_of_respondent': [age],
        'year': [2018]  # Hardcoded as it's likely just a context feature
    })

    # B. One-Hot Encoding
    # We convert the text inputs to dummies
    input_dummies = pd.get_dummies(input_data)

    #  (filling missing ones with 0)
    input_final = input_dummies.reindex(columns=model_columns, fill_value=0)

    # Make Prediction
    prediction = model.predict(input_final)[0]
    probability = model.predict_proba(input_final)[0][1]

    # Display Result
    if prediction == 1:
        st.success(f"Prediction: YES, likely to have a bank account. (Confidence: {probability:.0%})")
    else:
        st.warning(f"Prediction: NO, unlikely to have a bank account. (Confidence: {1 - probability:.0%})")