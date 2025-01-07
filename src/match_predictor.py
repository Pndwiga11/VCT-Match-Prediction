import joblib
import pandas as pd
from datetime import datetime

def predict_match(map_model_path, outcome_model_path, team_mapping_path, team_a, team_b, date, best_of):
    # Load the models and mappings
    map_model = joblib.load(map_model_path)
    outcome_model = joblib.load(outcome_model_path)
    team_mapping = joblib.load(team_mapping_path)

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

    # Extract the day from the date
    date_obj = datetime.strptime(date, "%Y-%m-%d")
    day = date_obj.day

    # Create the input vector for the map model
    input_vector = pd.DataFrame([[team_a_encoded, team_b_encoded, day, best_of]], columns=['Team A', 'Team B', 'Day', 'Best Of'])

    # Predict the map pool
    map_probs = map_model.predict_proba(input_vector)
    maps = ['Ascent', 'Bind', 'Haven', 'Icebox', 'Split']
    predicted_maps = sorted(
        zip(maps, [prob[0][1] for prob in map_probs]),
        key=lambda x: x[1],
        reverse=True
    )[:best_of]
    predicted_maps = [map_name for map_name, _ in predicted_maps]


    # Create the input vector for the outcome model
    for map_name in maps:
        input_vector[f'Map_{map_name}'] = 1 if map_name in predicted_maps else 0

    # Predict the match outcome
    outcome_pred = outcome_model.predict(input_vector)
    predicted_score_a = outcome_pred[0][0]
    predicted_score_b = outcome_pred[0][1]

    # Output the results
    print(f"Match Prediction: {team_a} vs {team_b}")
    print(f"Predicted Score: {team_a} {predicted_score_a:.2f} - {predicted_score_b:.2f} {team_b}")
    print(f"Predicted Maps: {', '.join(predicted_maps)}")
    print(f"Predicted Outcome: {'Team A Wins' if predicted_score_a > predicted_score_b else 'Team B Wins'}")

if __name__ == "__main__":
    # Example inputs
    map_model_path = "models/map_pool_model.pkl"
    outcome_model_path = "models/outcome_model.pkl"
    team_mapping_path = "models/team_mapping.pkl"

    team_a = "Sentinels"
    team_b = "100 Thieves"
    date = "2021-03-14"  # YYYY-MM-DD
    best_of = 3

    predict_match(map_model_path, outcome_model_path, team_mapping_path, team_a, team_b, date, best_of)
