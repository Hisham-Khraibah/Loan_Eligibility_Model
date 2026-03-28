import pickle
import pandas as pd
import streamlit as st

# Set the page title and description
st.title("Credit Loan Eligibility Predictor")
st.write("""
This app predicts whether a loan applicant is eligible for a loan 
based on various personal and financial characteristics.
""")

try:
    # Load the pre-trained model
    with open("models/RFmodel.pkl", "rb") as rf_file:
        rf_model = pickle.load(rf_file)

    # Load the scaler
    with open("models/scaler.pkl", "rb") as scaler_file:
        scaler = pickle.load(scaler_file)

    # Prepare the form to collect user inputs
    with st.form("user_inputs"):
        st.subheader("Loan Applicant Details")

        # Gender input
        Gender = st.selectbox("Gender", options=["Male", "Female"])

        # Marital Status
        Married = st.selectbox("Marital Status", options=["Yes", "No"])

        # Dependents
        Dependents = st.selectbox("Number of Dependents", options=["0", "1", "2", "3+"])

        # Education
        Education = st.selectbox("Education Level", options=["Graduate", "Not Graduate"])

        # Self Employment
        Self_Employed = st.selectbox("Self Employed", options=["Yes", "No"])

        # Applicant Income
        ApplicantIncome = st.number_input(
            "Applicant Monthly Income", min_value=0, max_value=100000, step=1000
        )

        # Coapplicant Income
        CoapplicantIncome = st.number_input(
            "Co-applicant Monthly Income", min_value=0, max_value=100000, step=1000
        )

        # Loan Amount
        LoanAmount = st.number_input(
            "Loan Amount", min_value=0, max_value=1000000, step=1000
        )

        # Loan Amount Term
        Loan_Amount_Term = st.selectbox(
            "Loan Amount Term (Months)", options=[360, 180, 240, 120, 60]
        )

        # Credit History
        Credit_History = st.selectbox("Credit History", options=[1, 0])

        # Property Area
        Property_Area = st.selectbox(
            "Property Area", options=["Urban", "Semiurban", "Rural"]
        )

        # Submit button
        submitted = st.form_submit_button("Predict Loan Eligibility")

    # Handle prediction after submit
    if submitted:
        try:
            # Rule-based rejection first
            total_income = ApplicantIncome + CoapplicantIncome

            if ApplicantIncome < 1500 or total_income < 2000 or Credit_History == 0:
                st.subheader("Prediction Result:")
                st.error("❌ Sorry, you are not eligible for the loan.")
                st.write("Approval Probability: 0.00")

            else:
                Gender_Female = 1 if Gender == "Female" else 0
                Gender_Male = 1 if Gender == "Male" else 0

                Married_No = 1 if Married == "No" else 0
                Married_Yes = 1 if Married == "Yes" else 0

                Dependents_0 = 1 if Dependents == "0" else 0
                Dependents_1 = 1 if Dependents == "1" else 0
                Dependents_2 = 1 if Dependents == "2" else 0
                Dependents_3 = 1 if Dependents == "3+" else 0

                Education_Graduate = 1 if Education == "Graduate" else 0
                Education_Not_Graduate = 1 if Education == "Not Graduate" else 0

                Self_Employed_No = 1 if Self_Employed == "No" else 0
                Self_Employed_Yes = 1 if Self_Employed == "Yes" else 0

                Property_Area_Rural = 1 if Property_Area == "Rural" else 0
                Property_Area_Semiurban = 1 if Property_Area == "Semiurban" else 0
                Property_Area_Urban = 1 if Property_Area == "Urban" else 0

                # Prepare the input in the same order as the training data
                prediction_input = pd.DataFrame([{
                    "ApplicantIncome": ApplicantIncome,
                    "CoapplicantIncome": CoapplicantIncome,
                    "LoanAmount": LoanAmount,
                    "Loan_Amount_Term": Loan_Amount_Term,
                    "Credit_History": Credit_History,
                    "Gender_Female": Gender_Female,
                    "Gender_Male": Gender_Male,
                    "Married_No": Married_No,
                    "Married_Yes": Married_Yes,
                    "Dependents_0": Dependents_0,
                    "Dependents_1": Dependents_1,
                    "Dependents_2": Dependents_2,
                    "Dependents_3+": Dependents_3,
                    "Education_Graduate": Education_Graduate,
                    "Education_Not Graduate": Education_Not_Graduate,
                    "Self_Employed_No": Self_Employed_No,
                    "Self_Employed_Yes": Self_Employed_Yes,
                    "Property_Area_Rural": Property_Area_Rural,
                    "Property_Area_Semiurban": Property_Area_Semiurban,
                    "Property_Area_Urban": Property_Area_Urban
                }])

                # Scale the input before prediction
                prediction_input_scaled = scaler.transform(prediction_input)

                # Make prediction
                new_prediction = rf_model.predict(prediction_input_scaled)
                prob = rf_model.predict_proba(prediction_input_scaled)[0][1]

                # Display result
                st.subheader("Prediction Result:")

                if new_prediction[0] == 1:
                    st.success("✅ You are eligible for the loan!")
                else:
                    st.error("❌ Sorry, you are not eligible for the loan.")

                st.write(f"Approval Probability: {prob:.2f}")

        except Exception as e:
            st.error(f"Error while making prediction: {e}")

    st.write(
        """We used a machine learning (Random Forest) model to predict your eligibility, 
        and the features used in this prediction are ranked by relative importance below."""
    )
    st.image("feature_importance.png")

except Exception as e:
    st.error(f"Error while loading the application: {e}")