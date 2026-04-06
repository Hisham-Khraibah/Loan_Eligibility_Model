# loan_eligibility_model
This app has been built using Streamlit and deployed with Streamlit Community Cloud.

https://khra0005-loan-eligibility-model.streamlit.app/

This application predicts whether a loan applicant is eligible for a loan based on personal and financial characteristics using supervised machine learning. The goal is to assist financial institutions in making faster and more accurate loan approval decisions.

## Features
* User-friendly interface powered by Streamlit.
* Input form to enter applicant details such as income, loan amount, credit history, etc.
* Predicts whether a loan will be approved.
* Visualizations including:
  * Correlation heatmap
  * Feature importance chart
  * Confusion matrix
* Model saved for reuse.

## Dataset
The application uses the **Loan Eligibility Dataset**, which includes the following features:
* Loan_ID (removed during preprocessing)
* Gender
* Married
* Dependents
* Education
* Self_Employed
* ApplicantIncome
* CoapplicantIncome
* LoanAmount
* Loan_Amount_Term
* Credit_History
* Property_Area
* Loan_Status (target variable)

## Machine Learning Approach
* **Algorithm Used:** Random Forest Classifier
* **Preprocessing:**
  * Missing value handling
  * One-hot encoding for categorical variables
  * Feature scaling using MinMaxScaler
* **Target Variable:**
  * Loan_Status:
    * Y → Approved
    * N → Not Approved
* **Evaluation:**
  * Accuracy Score
  * Confusion Matrix

## Project Structure
```
src/
│
├── data/
│   └── make_dataset.py
│
├── features/
│   └── build_features.py
│
├── models/
│   ├── train_model.py
│   └── predict_model.py
│
├── visualization/
│   └── visualize.py
│
main.py
streamlit.py
requirements.txt
```

## Technologies Used
* **Streamlit**: For building the web application.
* **Scikit-learn**: For classification (Random Forest) and evaluation.
* **Pandas** and **NumPy**: For data preprocessing.
* **Matplotlib** and **Seaborn**: For data visualization.

## How to Run the Project
### 1. Install dependencies
`pip install -r requirements.txt`

### 2. Run the pipeline
`python main.py`

### 3. Run the Streamlit app
`python -m streamlit run streamlit.py`

## Output
* Trained model saved as:
  `models/RFmodel.pkl`
* Scaler saved as:
  `models/scaler.pkl`
* Visualizations displayed during execution
