# Deactivate distracting warnings
import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.getcwd())))
import os
import shutil
import json
import pandas as pd
from tqdm.notebook import tqdm
from datatools.preprocess import (
    load_event_data, 
    load_position_data, 
    load_team_sheets,
    extract_match_id, 
    extract_event_pos,
    convert_locations,
    add_position_info,
    calc_player_velocities,
)
from PlayerImputer.datatools.visualization import plot_event_count, plot_kde, plot_goal_positions

# Define the path to the dataset
path = os.path.join(os.path.dirname(os.getcwd()), "data", "DFL")
path = os.path.join(os.path.dirname(os.getcwd()), "dfl-confidential")
match_ids = [extract_match_id(filename) for filename in os.listdir(path)]


for match_id in tqdm(match_ids):
    match_path = os.path.join(path, match_id)

    # xy_objects
    # 1, X, Y:  x- and y-coordinates (XY/ in m),
    # 2. D: distance covered since the preceding frame (D in cm),
    # 3. S: speed (S in km/h)
    # 4. A: acceleration (A in m/s²)
    # 5. M: minute of play (M)
    events = load_event_data(match_path)
    events["game_id"] = match_id # SPADL game_id
    
    positions, team_sheets = load_position_data(match_path)
    positions = convert_locations(positions)
    positions = calc_player_velocities(positions, team_sheets)
    
    events = extract_event_pos(events, positions, team_sheets)

    # Rename columns(format: SPADL)
    team_sheets = team_sheets.rename(columns={"tID": "team_id",
                                              "pID": "player_id"
    })
                                              
    events = events.rename(columns={"tID": "team_id", 
                                    "pID": "player_id", 
                                    "eID": "type_name",
                                    "gameclock": "time_seconds"
    })
    events= add_position_info(events, team_sheets)

    events.to_csv(os.path.join(match_path, "events.csv"), index=False)
    positions.to_csv(os.path.join(match_path, "positions.csv"), index=False)
    team_sheets.to_csv(os.path.join(match_path, "teams.csv"), index=False)