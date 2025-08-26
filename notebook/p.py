import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..", "..")))

# Define the path to the dataset
base_path = os.path.join(os.path.dirname(os.getcwd()))
base_path

from tqdm import tqdm
from datatools.Bepro_preprocessor import OlderPreprocesssor, NewPreprocessor
from datatools.utils import compute_camera_coverage, is_inside
from datatools.trace_snapshot import TraceSnapshot
import imputer.config as config

data_path = os.path.join(base_path, "data", "BEPRO", "2024")
game_ids = sorted([f for f in os.listdir(data_path)])
print("Total games: ", len(game_ids))

def extract_event_pos(events, tracking_data, team_sheets):
    '''
    Linearly interpolate the position of a player at a given time.
    '''
    events = events.copy()
    team_sheets = team_sheets.copy()
    tracking_data = tracking_data.copy()

    team_dict = {row.player_id: f"{row.team[0]}{row.xID:02d}" for row in team_sheets.itertuples()} #
    tracking_by_period = {
        period: df.reset_index(drop=True)
        for period, df in tracking_data.groupby("period_id")
    }

    events[["trace_start_x", "trace_start_y", "trace_related_x", "trace_related_y", "trace_time"]] = None
    for idx, row in tqdm(events.iterrows()):
        copy_tracking_data = tracking_by_period[row.period_id]
        closest_idx = copy_tracking_data["time"].sub(row.time_seconds).abs().idxmin()

        events.at[idx, "trace_time"] = copy_tracking_data.at[closest_idx, "time"]
        if pd.isna(row.player_id):
            events.at[idx, "trace_start_x"] = None
            events.at[idx, "trace_start_y"] = None
        else:
            x_col = team_dict[row.player_id] + "_x"
            y_col = team_dict[row.player_id] + "_y"
            
            x, y = copy_tracking_data.loc[closest_idx, [x_col, y_col]]
            events.at[idx, "trace_start_x"] = x
            events.at[idx, "trace_start_y"] = y

        if pd.isna(row.related_id):
            events.at[idx, "trace_related_x"] = None
            events.at[idx, "trace_related_y"] = None
        else:
            x_col = team_dict[row.related_id] + "_x"
            y_col = team_dict[row.related_id] + "_y"
            
            x, y = copy_tracking_data.loc[closest_idx, [x_col, y_col]]
            events.at[idx, "trace_related_x"] = x
            events.at[idx, "trace_related_y"] = y

    return events

# Load the data
# 32번 경기 제외: 출전시간이 적은 Gabriel선수는 이벤트 기록이 있는데도 tracking-data에 기록되어 있지 않음
for game_id in tqdm(game_ids[:1]):
    save_path = os.path.join(data_path, game_id)
    preprocessor = NewPreprocessor(game_id=game_id, data_path=data_path)

    teams = preprocessor.teams
    teams = teams.rename(columns={"tID": "team_id", "pID": "player_id"})

    events = preprocessor.build_events(left_to_right=True)
    events = events.merge(teams[["player_id", "team_id"]], on="player_id", how="left")

    positions = preprocessor.build_traces(left_to_right=True)
    positions = positions.rename(columns={"time_seconds": "time"})

    events = extract_event_pos(events, positions, teams)
    
    # save the data
    positions.to_csv(os.path.join(save_path, "positions.csv"), index=False)
    events.to_csv(os.path.join(save_path, "events.csv"), index=False)
    teams.to_csv(os.path.join(save_path, "teams.csv"), index=False)
