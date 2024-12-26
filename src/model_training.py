import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
from preprocessing import preprocess_teams  # Import the preprocessing function

# Train the model
def train_model(data_path, model_path):
    # Load the dataset
    df = pd.read_csv(data_path)

    # Preprocess training data
    train_df, _, team_mapping = preprocess_teams(df, df)  # Create mapping for training data only

    # Define features (X) and target (y)
    X = train_df[['Team A', 'Team B', 'Score A', 'Score B']]
    y = train_df['Outcome']

    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the Random Forest model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Evaluate the model on the validation set
    accuracy = model.score(X_val, y_val)
    print(f"Validation Accuracy: {accuracy:.2f}")

    # Save the model and team mapping
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

    # Save the team mapping to a file for reuse in prediction
    mapping_path = model_path.replace('.pkl', '_team_mapping.pkl')
    joblib.dump(team_mapping, mapping_path)
    print(f"Team mapping saved to {mapping_path}")

# Main entry point
if __name__ == "__main__":
    train_model("data/2021_VCT_NA_Stage1Challengers1.csv", "models/random_forest_model.pkl")
