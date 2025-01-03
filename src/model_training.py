import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib
from preprocessing import preprocess_teams  # Import the preprocessing function
from preprocessing import preprocess_maps  # Import the preprocessing function

# Train the model
from sklearn.ensemble import RandomForestRegressor

def train_model(data_path, model_path):
    # Load and preprocess data
    df = pd.read_csv(data_path)
    
    df['Date'] = pd.to_datetime(df['Date'])  # Convert Date to datetime
    df['Day'] = df['Date'].dt.day  # Extract the day
    
    # Preprocess Teams and Maps
    train_df, _, team_mapping = preprocess_teams(df, df)
    train_df, _, map_mapping = preprocess_maps(train_df)

    # Define features (X) and targets (y)
    X = train_df[['Team A', 'Team B', 'Day', 'Best Of']]
    y = train_df[['Score A', 'Score B', 'Map 1', 'Map 2', 'Map 3', 'Map 4', 'Map 5']]

    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Random Forest Regressor
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)

    # Evaluate model (R^2 score)
    score = model.score(X_val, y_val)
    print(f"Validation R^2 Score: {score:.2f}")

    # Save the model
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

    # Save the team mapping for future use
    team_mapping_path = model_path.replace('.pkl', '_team_mapping.pkl')
    joblib.dump(team_mapping, team_mapping_path)
    print(f"Team mapping saved to {team_mapping_path}")
    
    map_mapping_path = model_path.replace('.pkl', '_map_mapping.pkl')
    joblib.dump(map_mapping, map_mapping_path)
    print(f"Map mapping saved to {map_mapping_path}")

# Main entry point
if __name__ == "__main__":
    train_model("data/combined_tournaments.csv", "models/random_forest_model.pkl")
