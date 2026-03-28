# Import accuracy score
from sklearn.metrics import accuracy_score, confusion_matrix

# Function to predict and evaluate
def evaluate_model(model, X_test_scaled, y_test):
    
    try:
        # Predict the loan eligibility on the testing set
        y_pred = model.predict(X_test_scaled)

        # Calculate the accuracy score
        accuracy = accuracy_score(y_test, y_pred)

        # Calculate the confusion matrix
        confusion_mat = confusion_matrix(y_test, y_pred)

        return accuracy, confusion_mat

    except Exception as e:
        print("Error while evaluating the model:", e)