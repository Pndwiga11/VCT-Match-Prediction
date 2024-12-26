import pandas as pd
import joblib
from sklearn.metrics import classification_report, accuracy_score
from preprocessing import preprocess_teams  # Import the preprocessing function

# Load and preprocess the test data
def load_test_data(test_data_path, team_mapping):
    # Load the test dataset
    df = pd.read_csv(test_data_path)
    
    # Preprocess the test data
    _, test_df, _ = preprocess_teams(pd.DataFrame(team_mapping.keys(), columns=['Team']), df)  # Map test data using team mapping
    return test_df

# Predict outcomes and evaluate
def predict(model_path, test_data_path, train_data_path):
    # Load the trained model
    model = joblib.load(model_path)

    # Load the team mapping
    mapping_path = model_path.replace('.pkl', '_team_mapping.pkl')
    team_mapping = joblib.load(mapping_path)

    # Load the training and test data
    train_df = pd.read_csv(train_data_path)
    test_df = pd.read_csv(test_data_path)

    # Preprocess the test data using the team mapping
    train_df, test_df, _ = preprocess_teams(train_df, test_df)

    # Define features (X) and target (y)
    X_test = test_df[['Team A', 'Team B', 'Score A', 'Score B']]
    y_test = test_df['Outcome']

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate the predictions
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")


if __name__ == "__main__":
    # Example paths (adjust as needed)
    model_path = "models/random_forest_model.pkl"
    test_data_path = "data/2021_VCT_NA_Stage1Challengers2.csv"
    train_data_path = "data/2021_VCT_NA_Stage1Challengers1.csv"
    predict(model_path, test_data_path, train_data_path)


import pandas as pd

test_data_path = "data/2021_VCT_NA_Stage1Challengers2.csv"
df = pd.read_csv(test_data_path)
print(df.columns)