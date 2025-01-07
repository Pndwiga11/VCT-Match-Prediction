import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error, accuracy_score
from preprocessing import preprocess_teams # Import the preprocessing function
import numpy as np

# Test the Outcome Prediction Model
def test_outcome_model(model_path, test_data_path, train_data_path):
    # Load the model and team mapping
    model = joblib.load(model_path)
    team_mapping_path = 'models/team_mapping.pkl'
    team_mapping = joblib.load(team_mapping_path)
    

    # Load the test data
    train_df = pd.read_csv(train_data_path)
    test_df = pd.read_csv(test_data_path)

    # Convert Date column to datetime and extract 'Day'
    train_df['Date'] = pd.to_datetime(train_df['Date'])
    train_df['Day'] = train_df['Date'].dt.day
    test_df['Date'] = pd.to_datetime(test_df['Date'])
    test_df['Day'] = test_df['Date'].dt.day

    # Preprocess the test data using the team mapping
    train_df, test_df, _ = preprocess_teams(train_df, test_df)

    # Convert the Predicted Map Pool from string to list
    test_df['Predicted Map Pool'] = test_df['Predicted Map Pool'].apply(eval)

    # Convert map pool predictions to binary features
    maps = ['Ascent', 'Bind', 'Haven', 'Icebox', 'Split']
    for map_name in maps:
        test_df[f'Map_{map_name}'] = test_df['Predicted Map Pool'].apply(lambda x: 1 if map_name in x else 0)

    # Define input features (X_test)
    X_test = test_df[['Team A', 'Team B', 'Day', 'Best Of'] + [f'Map_{map}' for map in maps]]

    # Predict scores
    y_pred = model.predict(X_test)

    # Post-process predictions to derive outcomes
    test_df['Predicted Score A'] = y_pred[:, 0]
    test_df['Predicted Score B'] = y_pred[:, 1]
    test_df['Predicted Outcome'] = test_df.apply(
        lambda row: 1 if row['Predicted Score A'] > row['Predicted Score B'] else -1, axis=1
    )

    # Decode team names
    reverse_team_mapping = {v: k for k, v in team_mapping.items()}
    test_df['Decoded Team A'] = test_df['Team A'].map(reverse_team_mapping)
    test_df['Decoded Team B'] = test_df['Team B'].map(reverse_team_mapping)

    
    # Evaluate predictions and print accuracy with fraction
    accuracy = accuracy_score(test_df['Outcome'], test_df['Predicted Outcome'])
    total_predictions = len(test_df)
    correct_predictions = (test_df['Outcome'] == test_df['Predicted Outcome']).sum()
    print(f"\nOverall Accuracy: {accuracy:.0%} ({correct_predictions}/{total_predictions})\n")

    # Calculate RMSE for score predictions
    # Calculate MSE and then take the square root to get RMSE
    mse = mean_squared_error(test_df[['Score A', 'Score B']], y_pred)
    rmse = mse ** 0.5
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

   
    
    # Find incorrect predictions
    print("\nIncorrect Predictions:")
    incorrect_predictions = test_df[test_df['Outcome'] != test_df['Predicted Outcome']]
    #print(incorrect_predictions[['Decoded Team A', 'Decoded Team B', 'Score A', 'Score B', 'Predicted Score A', 'Predicted Score B', 'Outcome', 'Predicted Outcome']])
    print(incorrect_predictions[['Decoded Team A', 'Decoded Team B', 'Outcome', 'Predicted Outcome']])
    

# Main entry point
if __name__ == "__main__":
    model_path = "models/outcome_model.pkl"
    test_data_path = "data/map_pool_predictions.csv"
    train_data_path = "data/combined_tournaments.csv"
    test_outcome_model(model_path, test_data_path, train_data_path)
