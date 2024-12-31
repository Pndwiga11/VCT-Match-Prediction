import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib
from preprocessing import preprocess_teams  # Import the preprocessing function

# Train the model
from sklearn.ensemble import RandomForestRegressor

def train_model(data_path, model_path):
    # Load and preprocess data
    df = pd.read_csv(data_path)
    train_df, _, team_mapping = preprocess_teams(df, df)

    # Define features (X) and targets (y)
    X = train_df[['Team A', 'Team B']]
    y = train_df[['Score A', 'Score B']]

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
    mapping_path = model_path.replace('.pkl', '_team_mapping.pkl')
    joblib.dump(team_mapping, mapping_path)
    print(f"Team mapping saved to {mapping_path}")

# Main entry point
if __name__ == "__main__":
    train_model("data/2021_VCT_NA_Stage1Challengers1.csv", "models/random_forest_model.pkl")
