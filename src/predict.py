import pandas as pd
import joblib
from sklearn.metrics import classification_report, accuracy_score
from preprocessing import preprocess_teams  # Import the preprocessing function

# Load and preprocess the test data
def predict(model_path, test_data_path, train_data_path):
    # Load the model and team mapping
    model = joblib.load(model_path)
    mapping_path = model_path.replace('.pkl', '_team_mapping.pkl')
    team_mapping = joblib.load(mapping_path)

    # Load the training and test data
    train_df = pd.read_csv(train_data_path)
    test_df = pd.read_csv(test_data_path)

    # Preprocess the test data using the team mapping
    train_df, test_df, _ = preprocess_teams(train_df, test_df)

    # Define features (X)
    X_test = test_df[['Team A', 'Team B']]

    # Predict scores
    y_pred = model.predict(X_test)

    # Post-process predictions to derive outcomes
    test_df['Predicted Score A'] = y_pred[:, 0]
    test_df['Predicted Score B'] = y_pred[:, 1]
    test_df['Predicted Outcome'] = test_df.apply(
        lambda row: 1 if row['Predicted Score A'] > row['Predicted Score B'] else -1, axis=1
    )

    # Evaluate predictions
    accuracy = accuracy_score(test_df['Outcome'], test_df['Predicted Outcome'])
    print(f"Accuracy: {accuracy:.2f}")

    # Print predictions vs actual scores for debugging
    print("Sample Predictions:")
    print(test_df[['Team A', 'Team B', 'Predicted Score A', 'Predicted Score B', 'Outcome', 'Predicted Outcome']].head())

    return test_df


if __name__ == "__main__":
    # Example paths (adjust as needed)
    model_path = "models/random_forest_model.pkl"
    test_data_path = "data/2021_VCT_NA_Stage1Challengers3.csv"
    train_data_path = "data/combined_tournaments.csv"
    predict(model_path, test_data_path, train_data_path)