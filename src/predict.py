import pandas as pd
import joblib
from sklearn.metrics import classification_report, accuracy_score
from preprocessing import preprocess_teams  # Import the preprocessing function

# Load and preprocess the test data
def predict(model_path, test_data_path, train_data_path):
    # Load the model and mappings
    model = joblib.load(model_path)
    team_mapping_path = model_path.replace('.pkl', '_team_mapping.pkl')
    team_mapping = joblib.load(team_mapping_path)

    map_mapping_path = model_path.replace('.pkl', '_map_mapping.pkl')
    map_mapping = joblib.load(map_mapping_path)
    reverse_map_mapping = {v: k for k, v in map_mapping.items()}  # Reverse mapping for decoding

    # Load the training and test data
    train_df = pd.read_csv(train_data_path)
    test_df = pd.read_csv(test_data_path)
    
    # Convert Date column to datetime and extract 'Day'
    train_df['Date'] = pd.to_datetime(train_df['Date'])
    train_df['Day'] = train_df['Date'].dt.day
    test_df['Date'] = pd.to_datetime(test_df['Date'])
    test_df['Day'] = test_df['Date'].dt.day

    # Preprocess the test data using the team mapping
    train_df, test_df, _ = preprocess_teams(train_df, test_df)
    
    # Preprocess maps using the loaded map mapping
    for col in ['Map 1', 'Map 2', 'Map 3', 'Map 4', 'Map 5']:
        test_df[col] = test_df[col].map(lambda x: map_mapping[x] if x in map_mapping else -1).astype(int)

    # Define features (X)
    X_test = test_df[['Team A', 'Team B', 'Day', 'Best Of']]

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
    reverse_team_mapping = {v: k for k, v in team_mapping.items()}
    test_df['Decoded Team A'] = test_df['Team A'].map(reverse_team_mapping)
    test_df['Decoded Team B'] = test_df['Team B'].map(reverse_team_mapping)

    # Decode predicted maps
    reverse_map_mapping = {v: k for k, v in map_mapping.items()}
    for col in ['Predicted Map 1', 'Predicted Map 2', 'Predicted Map 3', 'Predicted Map 4', 'Predicted Map 5']:
        if col in test_df:  # Ensure the column exists before decoding
            test_df[f'Decoded {col}'] = test_df[col].map(reverse_map_mapping)
    
    # Find incorrect predictions
    print("\nIncorrect Predictions:")
    incorrect_predictions = test_df[test_df['Outcome'] != test_df['Predicted Outcome']]
    #print(incorrect_predictions[['Decoded Team A', 'Decoded Team B', 'Score A', 'Score B', 'Predicted Score A', 'Predicted Score B', 'Outcome', 'Predicted Outcome']])
    print(incorrect_predictions[['Decoded Team A', 'Decoded Team B', 'Outcome', 'Predicted Outcome']])

    return test_df


if __name__ == "__main__":
    # Example paths (adjust as needed)
    model_path = "models/random_forest_model.pkl"
    test_data_path = "data/2021_VCT_NA_Stage1Masters.csv"
    train_data_path = "data/combined_tournaments.csv"
    predict(model_path, test_data_path, train_data_path)