from src.data.make_dataset import load_and_preprocess_data
from src.visualization.visualize import plot_correlation_heatmap, plot_feature_importance, plot_confusion_matrix
from src.features.build_features import create_dummy_vars
from src.models.train_model import train_RFmodel
from src.models.predict_model import evaluate_model

if __name__ == "__main__":

    try:
        # Load and preprocess the raw dataset
        df = load_and_preprocess_data("data/raw/credit.csv")

        # Build features and target
        X, y = create_dummy_vars(df)

        # Visualize feature correlations
        plot_correlation_heatmap(X)

        # Train the model
        model, X_test_scaled, y_test = train_RFmodel(X, y)

        # Evaluate the model
        accuracy, confusion_mat = evaluate_model(model, X_test_scaled, y_test)
        print("Accuracy:", accuracy)
        print("Confusion Matrix:\n", confusion_mat)

        # Plot model insights
        plot_feature_importance(model, X)

        # Plot confusion matrix
        y_pred = model.predict(X_test_scaled)
        plot_confusion_matrix(y_test, y_pred, classes=['Not Approved', 'Approved'])

    except Exception as e:
        print("Error while running the main pipeline:", e)