
import pandas as pd
import streamlit as st
import mlflow.pyfunc

# Load the model — you can use the local URI or run ID
mlflow.set_tracking_uri("http://127.0.0.1:5000/")
model_uri = "models:/Credit Risk Model/2"  # Or use a registered model name
model = mlflow.pyfunc.load_model(model_uri)

st.title("Credit Risk Predictor (MLflow Version)")

# Inputs
age = st.number_input("Age", min_value=18, max_value=100, value=30)
income = st.number_input("Annual Income", value=50000)
loan_amount = st.number_input("Loan Amount", value=10000)
credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=650)
employment_status = st.selectbox("Employment Status", ["Employed", "Self-Employed", "Unemployed"])
marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])

# Input DataFrame
input_df = pd.DataFrame([{
    "age": age,
    "income": income,
    "loan_amount": loan_amount,
    "credit_score": credit_score,
    "employment_status": employment_status,
    "marital_status": marital_status
}])

# Predict
if st.button("Predict Credit Risk"):
    prediction = model.predict(input_df)
    
    prob = prediction[0] if isinstance(prediction, (list, np.ndarray)) else float(prediction)
    st.metric(label="Probability of Default", value=f"{prob:.2%}")
    
    if prob > 0.5:
        st.error("⚠️ High Risk")
    elif prob > 0.2:
        st.warning("🟠 Medium Risk")
    else:
        st.success("✅ Low Risk")
