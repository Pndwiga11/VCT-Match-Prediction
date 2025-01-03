import joblib
import pandas as pd
from datetime import datetime

def predict_match(model_path, team_mapping_path, map_mapping_path, team_a, team_b, date, best_of):
    # Load the model and mappings
    model = joblib.load(model_path)
    team_mapping = joblib.load(team_mapping_path)
    map_mapping = joblib.load(map_mapping_path)

    # Encode the teams
    team_a_encoded = team_mapping.get(team_a, -1)
    team_b_encoded = team_mapping.get(team_b, -1)

    # Check for missing teams
    missing_teams = []
    if team_a_encoded == -1:
        missing_teams.append(f"Team A ({team_a})")
    if team_b_encoded == -1:
        missing_teams.append(f"Team B ({team_b})")

    # Handle cases with missing teams
    if missing_teams:
        print(f"\nWarning: The following team(s) are not in the training data: {', '.join(missing_teams)}")
        print("Predictions may not be accurate due to missing team data.")

    # Proceed with known or default team encodings
    if team_a_encoded == -1:
        print(f"Assigning default encoding for Team A ({team_a}).")
        team_a_encoded = len(team_mapping) + hash(team_a) % 1000
    if team_b_encoded == -1:
        print(f"Assigning default encoding for Team B ({team_b}).")
        team_b_encoded = len(team_mapping) + hash(team_b) % 1000


    # Extract the day from the date
    date_obj = datetime.strptime(date, "%Y-%m-%d")
    day = date_obj.day

    # Create the input vector
    input_vector = pd.DataFrame([[team_a_encoded, team_b_encoded, day, best_of]], columns=['Team A', 'Team B', 'Day', 'Best Of'])

    # Make the prediction
    prediction = model.predict(input_vector)
    predicted_score_a = prediction[0][0]
    predicted_score_b = prediction[0][1]
    predicted_maps = prediction[0][2:]

    # Decode the predicted maps
    reverse_map_mapping = {v: k for k, v in map_mapping.items()}
    decoded_maps = [reverse_map_mapping.get(int(map_code), "Unknown") for map_code in predicted_maps]

    # Output the results
    print(f"\nMatch Prediction: {team_a} vs {team_b}")
    print(f"Predicted Score: {team_a} {predicted_score_a:.2f} - {predicted_score_b:.2f} {team_b}")
    print(f"Predicted Maps: {', '.join(decoded_maps)}")
    print(f"Predicted Outcome: {'Team A Wins' if predicted_score_a > predicted_score_b else 'Team B Wins'}")
    
    

if __name__ == "__main__":
    # Example inputs
    model_path = "models/random_forest_model.pkl"
    team_mapping_path = "models/random_forest_model_team_mapping.pkl"
    map_mapping_path = "models/random_forest_model_map_mapping.pkl"
    
    team_a = "Immortals"
    team_b = "100 Thieves"
    date = "2021-03-12" #YYYY-MM-DD
    best_of = 3

    predict_match(model_path, team_mapping_path, map_mapping_path, team_a, team_b, date, best_of)