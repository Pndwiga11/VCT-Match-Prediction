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

    # Evaluate predictions and print accuracy with fraction
    accuracy = accuracy_score(test_df['Outcome'], test_df['Predicted Outcome'])
    total_predictions = len(test_df)
    correct_predictions = (test_df['Outcome'] == test_df['Predicted Outcome']).sum()
    print(f"\nOverall Accuracy: {accuracy:.2f} ({correct_predictions}/{total_predictions})\n")
    
    # Decode team names
    reverse_mapping = {v: k for k, v in team_mapping.items()}
    test_df['Decoded Team A'] = test_df['Team A'].map(reverse_mapping)
    test_df['Decoded Team B'] = test_df['Team B'].map(reverse_mapping)

    # Find incorrect predictions
    print("\nIncorrect Predictions:")
    incorrect_predictions = test_df[test_df['Outcome'] != test_df['Predicted Outcome']]
    #print(incorrect_predictions[['Decoded Team A', 'Decoded Team B', 'Score A', 'Score B', 'Predicted Score A', 'Predicted Score B', 'Outcome', 'Predicted Outcome']])
    print(incorrect_predictions[['Decoded Team A', 'Decoded Team B', 'Outcome', 'Predicted Outcome']])

    return test_df


if __name__ == "__main__":
    # Example paths (adjust as needed)
    model_path = "models/random_forest_model.pkl"
    test_data_path = "data/2021_VCT_NA_Stage1Challengers3.csv"
    train_data_path = "data/combined_tournaments.csv"
    predict(model_path, test_data_path, train_data_path)