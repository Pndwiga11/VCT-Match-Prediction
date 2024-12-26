import pandas as pd

def preprocess_teams(train_df, test_df):
    print("Train Columns:", train_df.columns)
    print("Test Columns:", test_df.columns)
    # Create mapping from training data
    team_mapping = {team: idx for idx, team in enumerate(pd.concat([train_df['Team A'], train_df['Team B']]).unique())}

    # Map training data teams
    train_df['Team A'] = train_df['Team A'].map(team_mapping)
    train_df['Team B'] = train_df['Team B'].map(team_mapping)

    # Handle test data
    test_df['Team A'] = test_df['Team A'].apply(lambda x: team_mapping[x] if x in team_mapping else len(team_mapping) + hash(x) % 1000)
    test_df['Team B'] = test_df['Team B'].apply(lambda x: team_mapping[x] if x in team_mapping else len(team_mapping) + hash(x) % 1000)

    return train_df, test_df, team_mapping