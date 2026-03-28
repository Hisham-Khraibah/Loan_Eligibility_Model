# loan_eligibility_prediction
This app has been built using Streamlit and can be deployed using Streamlit Community Cloud.



This application predicts whether a loan applicant is eligible for a loan based on various personal and financial characteristics. The model uses machine learning techniques to assist in decision-making for loan approvals.

## Features
- User-friendly interface powered by Streamlit.
- Input form to enter applicant details such as income, credit history, and employment status.
- Real-time prediction of loan eligibility.
- Feature importance visualization to understand key factors affecting loan approval.
- Data visualization including correlation heatmaps and confusion matrix.

## Dataset
The application uses a dataset containing features such as:
- Gender
- Marital Status
- Number of Dependents
- Education Level
- Self Employment Status
- Applicant Income
- Coapplicant Income
- Loan Amount
- Loan Term
- Credit History
- Property Area
- Loan Approval Status (Target Variable)

## Technologies Used
- **Streamlit**: For building the interactive web application.
- **Scikit-learn**: For model training and evaluation (Random Forest Classifier).
- **Pandas** and **NumPy**: For data preprocessing and manipulation.
- **Matplotlib** and **Seaborn**: For data visualization.
- **Pickle**: For saving and loading the trained model.