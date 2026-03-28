import pandas as pd

# create dummy features
def create_dummy_vars(df):
    
    try:
        # Convert target variable 'Loan_Approved' to numeric
        df['Loan_Approved'] = df['Loan_Approved'].map({'Y': 1, 'N': 0})

        # Create dummy variables for all 'object' type variables except 'Loan_Approved'
        df = pd.get_dummies(df, columns=['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area'])

        # store the processed dataset in data/processed
        df.to_csv('data/processed/Processed_Credit_Dataset.csv', index=None)

        # Separate the input features and target variable
        X = df.drop('Loan_Approved', axis=1)
        y = df['Loan_Approved']

        return X, y

    except Exception as e:
        print("Error while creating dummy variables:", e)