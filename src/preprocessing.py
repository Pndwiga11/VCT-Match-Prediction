import pandas as pd

def preprocess_teams(train_df, test_df):
    #print("Train Columns:", train_df.columns)
    #print("Test Columns:", test_df.columns)
    
    # Create mapping from training data
    team_mapping = {team: idx for idx, team in enumerate(pd.concat([train_df['Team A'], train_df['Team B']]).unique())}

    # Map training data teams
    train_df['Team A'] = train_df['Team A'].map(team_mapping)
    train_df['Team B'] = train_df['Team B'].map(team_mapping)

    # Handle test data
    test_df['Team A'] = test_df['Team A'].apply(lambda x: team_mapping[x] if x in team_mapping else len(team_mapping) + hash(x) % 1000)
    test_df['Team B'] = test_df['Team B'].apply(lambda x: team_mapping[x] if x in team_mapping else len(team_mapping) + hash(x) % 1000)

    return train_df, test_df, team_mapping

def preprocess_maps(train_df, test_df=None):
    # Columns representing the maps
    map_columns = ['Map 1', 'Map 2', 'Map 3', 'Map 4', 'Map 5']
    
    # Create a mapping for maps from the training data
    all_maps = pd.concat([train_df[col] for col in map_columns]).dropna().unique()
    map_mapping = {map_name: idx for idx, map_name in enumerate(all_maps)}

    # Encode maps in the training data
    for col in map_columns:
        train_df[col] = train_df[col].map(map_mapping).fillna(-1).astype(int)

    # If test data is provided, encode maps in the test data
    if test_df is not None:
        for col in map_columns:
            test_df[col] = test_df[col].map(lambda x: map_mapping[x] if x in map_mapping else -1).astype(int)

    return train_df, test_df, map_mapping