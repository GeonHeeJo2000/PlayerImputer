"""bepro data to vaep framework converter."""

from typing import Any, cast

import numpy as np
import pandas as pd  # type: ignore
from pandera.typing import DataFrame
from scipy.optimize import linear_sum_assignment # Hungarian algorithm

import vaep.config as config
from vaep.schema import Schema
HEIGHT_POST = 2.5
TOUCH_LINE_LENGTH = 105
GOAL_LINE_LENGTH = 68

LEFT_POST = 0.449 * GOAL_LINE_LENGTH # convert ratio to meter
RIGHT_POST = 0.551 * GOAL_LINE_LENGTH # convert ratio to meter
CENTER_POST = (LEFT_POST+RIGHT_POST) / 2

Eighteen_YARD = 16.4592 # 18yard = 16.4592meter

def convert_to_actions(events: pd.DataFrame, game_id: int):
    """
    Convert bepro events to VAEP framework.
    """

    events[["type_id", "bodypart_id", "result_id", "end_x", "end_y"]] = events.apply(_parse_event, axis=1, result_type="expand")
    events = events[events["type_id"] != config.actiontypes.index("non_action")].reset_index(drop=True)

    # 끝 위치를 정의해야하는 액션: 패스, 드리블, 슛, 클리어런스
    events = _fix_clearance(events)
    events = _fix_pass(events)
    events = _fix_dribble(events)
    events = _fix_shot(events)

    actions = pd.DataFrame()

    actions["game_id"] = [int(game_id)] * len(events)
    actions["event_id"] = events.event_id.astype(object)
    actions["period_id"] = events.period_id.astype(int)
    actions["team_id"] = events.team_id
    actions["player_id"] = events.player_id

    actions["team"] = events.team
    actions["position"] = events.position
    # time_seconds: First half kick-off: 0(ms), second half kick-off: 2,700,000(ms)
    # 후반전도 0s부터 시작하도록 조정(값의 스케일 제한)
    # actions["time_seconds"] = (
    #     events["time_seconds"]
    #     - ((events.period_id > 1) * 45 * 60)
    #     - ((events.period_id > 2) * 45 * 60)
    #     - ((events.period_id > 3) * 15 * 60)
    #     - ((events.period_id > 4) * 15 * 60)
    # )
    actions["time_seconds"] = events.time_seconds

    actions["start_x"] = events.start_x
    actions["start_y"] = events.start_y
    actions["end_x"] = events.end_x
    actions["end_y"] = events.end_y
    actions["related_x"] = events.related_x
    actions["related_y"] = events.related_y
    actions["related_id"] = events.related_id

    actions["type_id"] = events.type_id.astype(int)
    actions["bodypart_id"] = events.bodypart_id.astype(int)
    actions["result_id"] = events.result_id.astype(int)

    # convert ball control after receive to dribble.
    actions = _add_dribbles_after_receive(actions, selector_receive = actions["type_id"].isin([config.actiontypes.index("Receive"), config.actiontypes.index("Recovery")]))
    actions = _find_duel_pairs(actions)
    actions = _find_foul_pairs(actions)

    actions["type_name"] = actions.type_id.apply(lambda x: config.actiontypes[x])
    actions["bodypart_name"] = actions.bodypart_id.apply(lambda x: config.bodyparts[x])
    actions["result_name"] = actions.result_id.apply(lambda x: config.results[x])

    actions["action_id"] = range(len(actions))
    actions["event_id"] = range(len(actions))

    columns = Schema.to_schema().columns.keys()
    return actions
    return cast(DataFrame[Schema], actions[columns])

def _find_duel_pairs(events: pd.DataFrame) -> pd.DataFrame:
    def _pair_duel_events(period_group: pd.DataFrame) -> pd.DataFrame:
        has_duel = period_group["type_id"] == config.actiontypes.index("Duel")
        duel_events = period_group[has_duel].reset_index(drop=False) # drop=False: index를 기준으로 병합및 정렬
        other_events = period_group[~has_duel].reset_index(drop=False)
         
        if len(duel_events) < 2:
            return period_group

        # Create a cost matrix based on event_time differences: Hungarian algorithm
        cost_matrix = abs(duel_events["time_seconds"].values[:, None] - duel_events["time_seconds"].values).astype(np.float64)
        np.fill_diagonal(cost_matrix, np.inf) # diagonal: 자기 자신과의 거리는 무한대로 설정(not mapping to itself)
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        paired_indices = [[i, j] for i, j in zip(row_ind, col_ind) if i < j]
        paired_indices.sort(key=lambda x: abs(duel_events.at[x[0], "time_seconds"] - duel_events.at[x[1], "time_seconds"])) 

        # 시간차이가 가장 작은 조합 순으로 정렬
        selected_pairs = []
        for i, j in paired_indices:
            if i not in selected_pairs and j not in selected_pairs:
                if (duel_events.at[i, "result_id"] == config.results.index("success")) and (duel_events.at[j, "result_id"] == config.results.index("fail")):
                    duel_events.at[i, "related_id"] = duel_events.at[j, "player_id"]
                    duel_events.at[i, "related_x"] = duel_events.at[j, "start_x"]
                    duel_events.at[i, "related_y"] = duel_events.at[j, "start_y"]
                elif (duel_events.at[i, "result_id"] == config.results.index("fail")) and (duel_events.at[j, "result_id"] == config.results.index("success")):
                    duel_events.at[j, "related_id"] = duel_events.at[i, "player_id"]
                    duel_events.at[j, "related_x"] = duel_events.at[i, "start_x"]
                    duel_events.at[j, "related_y"] = duel_events.at[i, "start_y"]
                else:
                    col = ['event_id', 'result_name', 'type_name' ,'player_name']
                    print(f"Unexpected duel event pair: {duel_events.loc[i, col]}, {duel_events.loc[j, col]}")

                selected_pairs.extend([i, j])
            else:
                raise ValueError(f"Duplicate indices found in duel events: {i}, {j}")
        # Handle unselected duel events (if any)
        unselected_indices = set(duel_events.index) - set(selected_pairs)
        duel_events.loc[list(unselected_indices), "related_id"] = np.nan

        # Set related_x and related_y to NaN for unselected events
        duel_events = duel_events[duel_events["related_id"].notna()].reset_index(drop=True)

        # Update the original group with paired duel events
        # 주의 사항: 동일한 시점에 발생한 이벤트는 sort_values함수 사용시 순서가 변경될 수 있음
        period_group = pd.concat(
            [other_events, duel_events], 
            axis=0, ignore_index=True, sort=False
            ).sort_values("index").drop(columns=["index"]).reset_index(drop=True)

        return period_group
    
    events = events.groupby("period_id").apply(_pair_duel_events)

    return events.reset_index(drop=True)

def _find_foul_pairs(events: pd.DataFrame, tolerance: float = 5.0) -> pd.DataFrame:
    '''
    tolerance: 3.0 seconds: foul과 foul won의 시간차가 tolerance보다 크면 매칭하지 않음
    '''
    def _pair_foul_events(period_group: pd.DataFrame) -> pd.DataFrame:
        fouls = period_group[period_group["type_id"] == config.actiontypes.index("Foul")].reset_index(drop=False)
        foulwons = period_group[period_group["type_id"] == config.actiontypes.index("Foul Won")].reset_index(drop=False)
        other_events = period_group[~period_group["type_id"].isin([config.actiontypes.index("Foul"), config.actiontypes.index("Foul Won")])].reset_index(drop=False)


        if len(foulwons) == 0:
            # foul만 있는 경우: 병합필요X
            return period_group
        
        if (len(fouls) == 0) and (len(foulwons) > 0):
            # foul은 없는데, foul won만 있는 경우: 불가능한 경우: 데이터 오류
            raise ValueError(f"Foul won without foul in period {period_group['period_id'].values[0]}")
        
        # Hungarian algorithm: 최소 시간차로 매칭
        cost_matrix = np.abs(fouls["time_seconds"].values[:, None] - foulwons["time_seconds"].values).astype(np.float64)
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # 매칭 결과를 foul 기준으로 기록
        selected_foul = []
        selected_foulwon = []
        for i, j in zip(row_ind, col_ind):
            # 매칭 기준: 시간차가 너무 크면 매칭하지 않음 (예: 5초 이상은 무시)
            if abs(fouls.at[i, "time_seconds"] - foulwons.at[j, "time_seconds"]) > tolerance:
                continue

            if i not in selected_foul and j not in selected_foulwon:
                fouls.at[i, "related_id"] = foulwons.at[j, "player_id"]
                fouls.at[i, "related_x"] = foulwons.at[j, "start_x"]
                fouls.at[i, "related_y"] = foulwons.at[j, "start_y"]
            else:
                print(fouls.loc[i, ["event_id", "result_name", "type_name", "player_nba"]])
                print(foulwons.loc[j, ["event_id", "result_name", "type_name", "player_name"]])
                raise ValueError(f"Duplicate indices found in foul events: {i}, {j}")
            
            selected_foul.append(i)
            selected_foulwon.append(j)

        # foul만 남기고, foul won은 모두 제거
        merged = pd.concat([other_events, fouls], axis=0, ignore_index=True, sort=False)
        merged = merged.sort_values("index").drop(columns=["index"]).reset_index(drop=True)
        return merged

    # period_id별로 적용
    events = events.groupby("period_id").apply(_pair_foul_events)
    return events.reset_index(drop=True)

def shift_with_edge_fix(actions: pd.DataFrame, shift_value: int) -> pd.DataFrame:
    """
    Shift each group by a specified value and fill NaN only in the first or last row.
    Does not alter NaN values in the middle of the group to avoid affecting naturally missing data.
    """

    shift_action = actions.groupby("period_id").shift(shift_value)

    if shift_value < 0:
        # When shifting upwards, last row gets NaN, so fill it with the original last value
        fill_indices = actions.groupby("period_id").tail(abs(shift_value)).index
    else:
        # When shifting downwards, first row gets NaN, so fill it with the original first value
        fill_indices = actions.groupby("period_id").head(abs(shift_value)).index

    # Fill the NaN rows with the corresponding original values
    shift_action.loc[fill_indices] = actions.loc[fill_indices]
    shift_action["period_id"] = actions["period_id"]

    return shift_action

def _fix_clearance(actions: pd.DataFrame) -> pd.DataFrame:
    next_actions = shift_with_edge_fix(actions, shift_value=-1)

    clearance_idx = actions.type_id == config.actiontypes.index("Clearance")

    actions.loc[clearance_idx, "end_x"] = next_actions.loc[clearance_idx, "start_x"].values
    actions.loc[clearance_idx, "end_y"] = next_actions.loc[clearance_idx, "start_y"].values

    return actions

def _fix_pass(actions: pd.DataFrame) -> pd.DataFrame:
    next_actions = shift_with_edge_fix(actions, shift_value=-1)

    pass_type = [config.actiontypes.index("Pass"), config.actiontypes.index("Cross"), 
                 config.actiontypes.index("Throw-In"), config.actiontypes.index("Goal Kick"), 
                 config.actiontypes.index("Pass_Freekick"), config.actiontypes.index("Pass_Corner"),]
    pass_idx = (
        actions.type_id.isin(pass_type) &
        actions["end_x"].isna() &
        actions["end_y"].isna()
     )
    actions.loc[pass_idx, "end_x"] = next_actions.loc[pass_idx, "start_x"].values
    actions.loc[pass_idx, "end_y"] = next_actions.loc[pass_idx, "start_y"].values

    return actions

# _fix_dribble : Update the end position of dribble events based on their success or failure.
# If the dribble failed, the end position is set to the position of the next event.
# If the dribble succeeded, the end position is set to the position of the next event that is not a tackle or interception.
def _fix_dribble(df_actions: pd.DataFrame) -> pd.DataFrame:
    next_actions = shift_with_edge_fix(df_actions, shift_value=-1)

    failed_tackle = (
        (next_actions['type_id'] == config.actiontypes.index("Tackle")) &
        (next_actions['result_id'] == config.results.index("fail"))
    )
    failed_interception = (
        (next_actions['type_id'] == config.actiontypes.index("Interception")) &
        (next_actions['result_id'] == config.results.index("fail"))
    )

    same_team = df_actions.team_id == next_actions.team_id
    failed_defensive = (failed_tackle | failed_interception) & ~same_team

    # next_actions : 실패한 수비가 아닌 이벤트의 위치를 드리블의 끝 위치로 활용하고자하는 목적
    next_actions = next_actions.mask(failed_defensive)[["start_x", "start_y"]].bfill()

    selector_dribble = df_actions.type_name == "Carry"

    df_actions.loc[selector_dribble, "end_x"] = next_actions.loc[selector_dribble, "start_x"].values
    df_actions.loc[selector_dribble, "end_y"] = next_actions.loc[selector_dribble, "start_y"].values

    return df_actions

def _fix_shot(df_events: pd.DataFrame) -> pd.DataFrame:
    away_idx = df_events["team"] == "Away"

    shot_type = [config.actiontypes.index("Shot"), config.actiontypes.index("Shot_Freekick"), 
                config.actiontypes.index("Shot_Corner"), config.actiontypes.index("Penalty Kicks")]
    shot = df_events["type_id"].isin(shot_type)
    owngoal = df_events["result_id"] == config.results.index("owngoal")

    # 왼쪽 측면에서의 슛은 왼쪽 골 포스트 바깥으로 설정
    out_left_idx = (
        df_events["start_x"] < (LEFT_POST - Eighteen_YARD)
    )
    df_events.loc[shot & out_left_idx, "end_x"] = TOUCH_LINE_LENGTH
    df_events.loc[shot & out_left_idx, "end_y"] = LEFT_POST - Eighteen_YARD
    df_events.loc[shot & out_left_idx & away_idx, "end_x"] = config.field_length - df_events.loc[shot & out_left_idx & away_idx, "end_x"].values   
    df_events.loc[shot & out_left_idx & away_idx, "end_y"] = config.field_width - df_events.loc[shot & out_left_idx & away_idx, "end_y"].values 

    # 오른쪽 측면에서의 슛은 오른쪽 골 포스트 바깥으로 설정
    out_right_idx = (
        df_events["start_x"] > (RIGHT_POST + Eighteen_YARD)
    )
    df_events.loc[shot & out_right_idx, "end_x"] = TOUCH_LINE_LENGTH 
    df_events.loc[shot & out_right_idx, "end_y"] = RIGHT_POST + Eighteen_YARD
    df_events.loc[shot & out_right_idx & away_idx, "end_x"] = config.field_length - df_events.loc[shot & out_right_idx & away_idx, "end_x"].values
    df_events.loc[shot & out_right_idx & away_idx, "end_y"] = config.field_width - df_events.loc[shot & out_right_idx & away_idx, "end_y"].values

    # 중앙에서의 슛은 중앙 골 포스트 방향으로 설정
    out_center_idx = (
        (df_events["start_x"] >= (LEFT_POST - Eighteen_YARD))
        & (df_events["start_x"] <= (RIGHT_POST + Eighteen_YARD))
    )
    df_events.loc[shot & out_center_idx, "end_x"] = TOUCH_LINE_LENGTH 
    df_events.loc[shot & out_center_idx, "end_y"] = CENTER_POST
    df_events.loc[shot & out_center_idx & away_idx, "end_x"] = config.field_length - df_events.loc[shot & out_center_idx & away_idx, "end_x"].values
    df_events.loc[shot & out_center_idx & away_idx, "end_y"] = config.field_width - df_events.loc[shot & out_center_idx & away_idx, "end_y"].values

    # 자책골의 경우, 우리 팀 진영의 중앙 골 포스트로 설정
    df_events.loc[owngoal, "end_x"] = CENTER_POST
    df_events.loc[owngoal, "end_y"] = 0
    df_events.loc[owngoal & away_idx, "end_x"] = config.field_length - df_events.loc[owngoal & away_idx, "end_x"].values
    df_events.loc[owngoal & away_idx, "end_y"] = config.field_width - df_events.loc[owngoal & away_idx, "end_y"].values

    # 블로킹된 슛의 경우, 다음 블로킹 이벤트의 위치로 설정
    # 주의: 블로킹한 액션은 수비팀의 액션이기 때문에 위치는 수비 진영을 기준으로 기록되어있음
    # outcome: 'Shots Off Target', 'Shots On Target', 'Blocked Shots', 'Goals', 'Keeper Rush-outs'
    blocked =  df_events['qualifier'].apply(lambda ets: any((e.get("event_name") == "Shot") and (e["property"]["Outcome"] == "Blocked Shots") for e in ets))
    blocked_idx = shot & blocked

    df_events_next = shift_with_edge_fix(df_events, shift_value=-1) 
    df_events.loc[blocked_idx, "end_x"] = df_events_next.loc[blocked_idx, "start_x"].values
    df_events.loc[blocked_idx, "end_y"] = df_events_next.loc[blocked_idx, "start_y"].values

    goalkeeper_type = [config.actiontypes.index("Aerial Control"), config.actiontypes.index("Defensive Line Supports"), config.actiontypes.index("Save")]   
    goalkeeper_idx = shot & df_events_next["type_id"].isin(goalkeeper_type)
    df_events.loc[goalkeeper_idx, "end_x"] = df_events_next.loc[goalkeeper_idx, "start_x"].values
    df_events.loc[goalkeeper_idx, "end_y"] = df_events_next.loc[goalkeeper_idx, "start_y"].values

    return df_events

def _add_dribbles_after_receive(actions: pd.DataFrame, selector_receive: pd.Series) -> pd.DataFrame:
    """ Adds dribbles after receiving the ball if the player moves a certain distance."""

    min_dribble_length: float = 3.0

    next_actions = shift_with_edge_fix(actions, shift_value=-1)
    same_player = actions.player_id == next_actions.player_id

    dx = actions.start_x - next_actions.start_x
    dy = actions.start_y - next_actions.start_y
    far_enough = dx**2 + dy**2 >= min_dribble_length**2

    dribble_idx = (
        selector_receive
        & same_player
        & far_enough
    )

    receive = actions[dribble_idx]
    dribbles = receive.copy() 
    next = next_actions[dribble_idx]

    if not dribbles.empty:
        dribbles["event_id"] = np.nan
        dribbles["time_seconds"] = receive.time_seconds + 1e-3 # Dribbles occur right after receiving the ball

        dribbles["start_x"] = receive.start_x
        dribbles["start_y"] = receive.start_y
        dribbles["end_x"] = next.start_x
        dribbles["end_y"] = next.start_y

        dribbles["bodypart_id"] = config.bodyparts.index("foot")
        dribbles["type_id"] = config.actiontypes.index("Carry")
        dribbles["result_id"] = config.results.index("success")

        actions = pd.concat([dribbles, actions], ignore_index=True, sort=False)
        actions = actions.sort_values(["period_id", "time_seconds"], kind="mergesort").reset_index(drop=True)

    return actions

def _parse_event(event : pd.Series) -> tuple[int, int, float, float]:
    # 23 possible values : pass, cross, throw-in, 
    # crossed free kick, short free kick, crossed corner, short corner, 
    # take-on, foul, tackle, interception, 
    # shot, penalty shot, free kick shot, 
    # keeper save, keeper claim, keeper punch, keeper pick-up, 
    # clearance, bad touch, dribble and goal kick, block.
    events = {
        "Pass": _parse_pass_event,
        "Cross": _parse_pass_event,
        
        "Throw-In": _parse_set_piece_event,
        "Goal Kick": _parse_set_piece_event,
        "Pass_Corner": _parse_set_piece_event,
        "Shot_Corner": _parse_set_piece_event,
        "Pass_Freekick": _parse_set_piece_event,
        "Penalty Kicks": _parse_set_piece_event,
        "Shot_Freekick": _parse_set_piece_event,

        "Take-On": _parse_take_on_event,
        "Carry": _parse_carry_event,
        "Foul": _parse_foul_event,
        "Foul Won": _parse_foul_won_event, # 추후 foul won과 통합하면서 제거되는 액션

        "Tackle" : _parse_tackle_event,

        "Interception": _parse_interception_event,

        "Shot": _parse_shot_event,


        "Aerial Control" : _parse_goalkeeper_event,
        "Defensive Line Supports" : _parse_goalkeeper_event,
        "Save": _parse_save_event,

        "Block": _parse_block_event,
        "Clearance" : _parse_clearance_event,
        "Mistake" : _parse_bad_touch_event,
        "Own Goal": _parse_bad_touch_event,

        "Receive": _parse_receive_event,
        "Recovery": _parse_recovery_event,
        "Duel" : _parse_duel_event,

        "Offside" : _parse_offside_event
    }

    # non_action, foul won은 추후에 제거할 예정
    parser = events.get(event["type_name"], lambda e: ("non_action", "foot", "fail", None, None))
    if event["type_name"] not in events:
        print(f"Unknown event type: {event['type_name']}")

    actiontype, bodypart, result, end_x, end_y = parser(event)
    
    return config.actiontypes.index(actiontype), config.bodyparts.index(bodypart), config.results.index(result), end_x, end_y

def _parse_pass_event(event):

    actiontype = event.type_name # Pass or Cross
    bodypart = "foot"

    outcome =  next(
        (e['property']['Outcome'] for e in event['qualifier'] if e.get('event_name') == actiontype), None
    )
    if outcome == "Succeeded":
        result = "success"
    elif outcome == "Failed":
        result = "fail"  # Offside situations are handled in _fix_offside
    else:
        raise ValueError(f"Unexpected outcome value: {outcome}")

    # 실패한 패스의 경우 추후 보정
    end_x, end_y = event[["related_x", "related_y"]]

    return actiontype, bodypart, result, end_x, end_y

def _parse_set_piece_event(event):
    # actiontype: Throw-Ins, Freekicks, Goal Kicks, Corners, Penalty Kicks
    actiontype = event.type_name
    bodypart = "foot"

    outcome =  next(
        (e['property']['Outcome'] for e in event['qualifier'] if e.get('event_name') == actiontype), None
    )
    if outcome in ["Succeeded", "Goals"]: # 패스인 경우 성공, 슛인 경우 Goals일 때 성공한 세트피스
        result = "success"
    elif outcome in ['Failed', 'Shots Off Target', 'Shots On Target', 'Blocked Shots', 'Keeper Rush-outs']:
        result = "fail"
    else:
        raise ValueError(f"Unexpected outcome value: {outcome}")
    
    end_x, end_y = event[["related_x", "related_y"]]
    return actiontype, bodypart, result, end_x, end_y

def _parse_take_on_event(event):
    actiontype = event.type_name
    bodypart = "foot"

    outcome =  next(
        (e['property']['Outcome'] for e in event['qualifier'] if e.get('event_name') == actiontype), None
    )
    if outcome == "Succeeded":
        result = "success"
    elif outcome == "Failed":
        result = "fail"  # Offside situations are handled in _fix_offside
    else:
        raise ValueError(f"Unexpected outcome value: {outcome}")
    
    end_x, end_y = event[["start_x", "start_y"]]
    return actiontype, bodypart, result, end_x, end_y

def _parse_carry_event(event):
    actiontype = event.type_name
    bodypart = "foot"
    result = "success"

     # 드리블의 경우 추후 보정
    end_x, end_y = np.nan, np.nan

    return actiontype, bodypart, result, end_x, end_y
 
def _parse_foul_event(event):
    actiontype = event.type_name
    bodypart = "other"

    outcome =  next(
        (e['property']['Type'] for e in event['qualifier'] if e.get('event_name') == actiontype), None
    )
    if outcome == "Yellow Cards":
        result = "yellow_card"
    elif outcome == "Red Cards":
        result = "red_card"
    else:
        result = "fail"

    end_x, end_y = event[["start_x", "start_y"]]
    return actiontype, bodypart, result, end_x, end_y


def _parse_foul_won_event(event):
    actiontype = event.type_name
    bodypart = "foot"
    result =  "success"  
    end_x, end_y = event[["start_x", "start_y"]]
    
    return actiontype, bodypart, result, end_x, end_y

def _parse_tackle_event(event):
    actiontype = event.type_name
    bodypart = "foot"

    outcome =  next(
        (e['property']['Outcome'] for e in event['qualifier'] if e.get('event_name') == actiontype), None
    )
    if outcome == "Tackle Succeeded: Possession":
        result = "success"
    elif outcome in ["Tackle Succeeded: No Possession", "Tackle Failed"]:
        result = "fail"
    else:
        raise ValueError(f"Unexpected outcome value: {outcome}")    

    end_x, end_y = event[["start_x", "start_y"]]
    return actiontype, bodypart, result, end_x, end_y

def _parse_interception_event(event):
    actiontype = event.type_name
    bodypart = "foot"
    result =  "success"  
    end_x, end_y = event[["start_x", "start_y"]]

    return actiontype, bodypart, result, end_x, end_y

def _parse_shot_event(event):
    actiontype = event.type_name
    bodypart = "foot"

    # outcome: 'Shots Off Target', 'Shots On Target', 'Blocked Shots', 'Goals', 'Keeper Rush-outs'
    outcome =  next(
        (e['property']['Outcome'] for e in event['qualifier'] if e.get('event_name') == actiontype), None
    )
    if outcome == "Goals":
        result = "success"
    elif outcome in ['Shots Off Target', 'Shots On Target', 'Blocked Shots', 'Keeper Rush-outs']:
        result = "fail"
    else:
        raise ValueError(f"Unexpected outcome value: {outcome}")    

     # 슛의 경우 추후 보정
    end_x, end_y = np.nan, np.nan

    return actiontype, bodypart, result, end_x, end_y

def _parse_goalkeeper_event(event):
    actiontype = event.type_name
    bodypart = "other"

    outcome =  next(
        (e['property']['Outcome'] for e in event['qualifier'] if e.get('event_name') == actiontype), None
    )
    if outcome == "Succeeded":
        result = "success"
    elif outcome == "Failed":
        result = "fail"
    else:
        print(event[["event_id", "type_name", "qualifier"]].values)
        raise ValueError(f"Unexpected outcome value: {outcome}")    
    
    end_x, end_y = event[["start_x", "start_y"]]

    return actiontype, bodypart, result, end_x, end_y

def _parse_save_event(event):
    actiontype = event.type_name
    bodypart = "other"

    outcome =  next(
        (e['property'] for e in event['qualifier'] if e.get('event_name') == actiontype), None
    )
    if "Type" in outcome:
        if outcome["Type"] in ["Catches", "Parries"]:
            result = "success"
    else:
        result = "fail"
    
    end_x, end_y = event[["start_x", "start_y"]]

    return actiontype, bodypart, result, end_x, end_y

def _parse_block_event(event):
    actiontype = event.type_name
    bodypart = "other"
    result =  "success"  
    end_x, end_y = event[["start_x", "start_y"]]

    return actiontype, bodypart, result, end_x, end_y

def _parse_clearance_event(event):
    actiontype = event.type_name
    bodypart = "foot"
    result =  "success"

    # 클리어런스의 경우 추후 보정
    end_x, end_y = np.nan, np.nan

    return actiontype, bodypart, result, end_x, end_y

def _parse_bad_touch_event(event):
    actiontype = "bad_touch"
    bodypart = "other"
    
    if event.type_name == "Mistake":
        result = "fail"
    elif event.type_name == "Own Goal":
        result ="owngoal"
    else:
        raise ValueError(f"Unexpected outcome value: {event.type_name}")  
    
    end_x, end_y = event[["start_x", "start_y"]]

    return actiontype, bodypart, result, end_x, end_y

def _parse_receive_event(event):
    actiontype = event.type_name
    bodypart = "foot"
    result =  "success"  
    end_x, end_y = event[["start_x", "start_y"]]

    return actiontype, bodypart, result, end_x, end_y

def _parse_recovery_event(event):
    actiontype = event.type_name
    bodypart = "foot"
    result =  "success"  
    end_x, end_y = event[["start_x", "start_y"]]

    return actiontype, bodypart, result, end_x, end_y

def _parse_duel_event(event):
    actiontype = event.type_name
    bodypart = "other"

    outcome =  next(
        (e['property']['Outcome'] for e in event['qualifier'] if e.get('event_name') == actiontype), None
    )
    if outcome == "Succeeded":
        result = "success"
    elif outcome == "Failed":
        result = "fail"
    else:
        raise ValueError(f"Unexpected outcome value: {outcome}")    
    
    end_x, end_y = event[["start_x", "start_y"]]

    return actiontype, bodypart, result, end_x, end_y

def _parse_offside_event(event):
    actiontype = event.type_name
    bodypart = "other"
    result =  "fail"  
    end_x, end_y = event[["start_x", "start_y"]]

    return actiontype, bodypart, result, end_x, end_y


