import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib
import ast
from preprocessing import preprocess_teams # Import the preprocessing function

# Train the Match Outcome Prediction Model
def train_outcome_model(data_path, model_path):
    # Load and preprocess data
    df = pd.read_csv(data_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df['Day'] = df['Date'].dt.day
    df['Predicted Map Pool'] = df['Predicted Map Pool'].apply(ast.literal_eval)  # Convert string to list

    # Preprocess teams
    train_df, _, team_mapping = preprocess_teams(df, df)

    # Convert map pool predictions to binary features
    maps = ['Ascent', 'Bind', 'Haven', 'Icebox', 'Split'] # Current Map Pool as of Stage 1 Masters
    for map_name in maps:
        df[f'Map_{map_name}'] = df['Predicted Map Pool'].apply(lambda x: 1 if map_name in x else 0)

    # Define features (X) and targets (y)
    X = train_df[['Team A', 'Team B', 'Day', 'Best Of'] + [f'Map_{map}' for map in maps]]
    y = train_df[['Score A', 'Score B']]

    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the Random Forest Regressor
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)

    # Evaluate model (R^2 score)
    score = model.score(X_val, y_val)
    print(f"Validation R^2 Score: {score:.2f}")

    # Save the model
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

# Main entry point
if __name__ == "__main__":
    train_outcome_model("data/map_pool_predictions.csv", "models/outcome_model.pkl")