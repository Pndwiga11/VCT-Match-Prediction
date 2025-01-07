import pandas as pd
import joblib
from preprocessing import preprocess_teams # Import the preprocessing function
import numpy as np

# Test the Map Pool Prediction Model
def test_map_model(model_path, test_data_path, train_data_path):
    # Load the model and mappings
    model = joblib.load(model_path)
    team_mapping_path =  'models/team_mapping.pkl'
    team_mapping = joblib.load(team_mapping_path)
    reverse_team_mapping = {v: k for k, v in team_mapping.items()}

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

    # Define input features (X)
    X_test = test_df[['Team A', 'Team B', 'Day', 'Best Of']]

    # Predict map pools with confidence scores
    y_pred_proba = model.predict_proba(X_test)

    # Map names
    maps = ['Ascent', 'Bind', 'Haven', 'Icebox', 'Split'] # Current Map Pool as of Stage 1 Masters

    # Convert predictions to lists of maps
    predicted_map_pools = []
    for map_probs, best_of in zip(zip(*y_pred_proba), test_df['Best Of']):
        # Flatten the array to handle the multi-label output
        flattened_probs = [float(prob[1]) for prob in map_probs]

        # Sort maps by confidence and select top maps based on Best Of
        top_maps = sorted(
            zip(maps, flattened_probs),
            key=lambda x: x[1],
            reverse=True
        )[:best_of]

        # Extract only the map names
        predicted_map_pools.append([map_name for map_name, _ in top_maps])

    # Assign the predicted map pools to the DataFrame
    test_df['Predicted Map Pool'] = predicted_map_pools

    # Decode team names
    reverse_team_mapping = {v: k for k, v in team_mapping.items()}
    test_df['Team A'] = test_df['Team A'].map(reverse_team_mapping)
    test_df['Team B'] = test_df['Team B'].map(reverse_team_mapping)
    
    # Save the results to a CSV file
    test_df.to_csv("data/map_pool_predictions.csv", index=False)
    print("Predictions saved to data/map_pool_predictions.csv")
    
    return test_df

# Main entry point
if __name__ == "__main__":
    model_path = "models/map_pool_model.pkl"
    test_data_path = "data/2021_VCT_NA_Stage1Masters.csv"
    train_data_path = "data/combined_tournaments.csv"
    test_map_model(model_path, test_data_path, train_data_path)
