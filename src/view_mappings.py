import joblib

def view_team_mapping(team_mapping_path):
    try:
        team_mapping = joblib.load(team_mapping_path)
        print("\nTeam Mapping (Team Name -> Code):")
        for team, code in team_mapping.items():
            print(f"{team} -> {code}")
    except FileNotFoundError:
        print(f"Error: The file {team_mapping_path} does not exist.")
    except Exception as e:
        print(f"An error occurred while loading the team mapping: {e}")

def view_map_mapping(map_mapping_path):
    try:
        map_mapping = joblib.load(map_mapping_path)
        print("\nMap Mapping (Map Name -> Code):")
        for map_name, code in map_mapping.items():
            print(f"{map_name} -> {code}")
    except FileNotFoundError:
        print(f"Error: The file {map_mapping_path} does not exist.")
    except Exception as e:
        print(f"An error occurred while loading the map mapping: {e}")

if __name__ == "__main__":
    team_mapping_path = "models/random_forest_model_team_mapping.pkl"
    map_mapping_path = "models/random_forest_model_map_mapping.pkl"
    
    # View mappings
    view_team_mapping(team_mapping_path)
    view_map_mapping(map_mapping_path)
