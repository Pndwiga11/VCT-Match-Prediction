import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
import joblib
import ast
from preprocessing import preprocess_teams # Import the preprocessing function

# Train the Map Pool Prediction Model
def train_map_model(data_path, model_path):
    # Load and preprocess data
    df = pd.read_csv(data_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df['Day'] = df['Date'].dt.day
    df['Map Pool'] = df['Map Pool'].apply(ast.literal_eval)  # Convert string to list

    # Preprocess teams
    train_df, _, team_mapping = preprocess_teams(df, df)

    # Create map labels
    maps = ['Ascent', 'Bind', 'Haven', 'Icebox', 'Split'] # Current Map Pool as of Stage 1 Masters
    for map_name in maps:
        df[f'Map_{map_name}'] = df['Map Pool'].apply(lambda x: 1 if map_name in x else 0)

    # Define features (X) and targets (y)
    X = train_df[['Team A', 'Team B', 'Day', 'Best Of']]
    y = df[[f'Map_{map_name}' for map_name in maps]]

    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the multi-label classifier
    classifier = RandomForestClassifier(random_state=42)
    model = MultiOutputClassifier(classifier)
    model.fit(X_train, y_train)

    # Evaluate model (R^2 score)
    score = model.score(X_val, y_val)
    print(f"Validation R^2 Score: {score:.2f}")

    # Save the model
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

    # Save the team mapping for future use
    mapping_path = 'models/team_mapping.pkl'
    joblib.dump(team_mapping, mapping_path)
    print(f"Team mapping saved to {mapping_path}")

# Main entry point
if __name__ == "__main__":
    train_map_model("data/combined_tournaments.csv", "models/map_pool_model.pkl")