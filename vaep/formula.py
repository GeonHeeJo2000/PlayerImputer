"""Implements the formula of the VAEP framework."""
import numpy as np
import pandas as pd  # type: ignore
from pandera.typing import DataFrame, Series

import vaep.config as config
from vaep.schema import Schema

def _prev(x: pd.Series) -> pd.Series:
    prev_x = x.shift(1)
    prev_x[:1] = x.values[0] # interpolation about first value
    return prev_x

_samephase_nb: int = 15

def offensive_value(
    actions: DataFrame[Schema], scores: Series[float], concedes: Series[float]
) -> Series[float]:
    r"""Compute the offensive value of each action.

    VAEP defines the *offensive value* of an action as the change in scoring
    probability before and after the action.

    .. math::

      \Delta P_{score}(a_{i}, t) = P^{k}_{score}(S_i, t) - P^{k}_{score}(S_{i-1}, t)

    where :math:`P_{score}(S_i, t)` is the probability that team :math:`t`
    which possesses the ball in state :math:`S_i` will score in the next 10
    actions.

    Parameters
    ----------
    actions : pd.DataFrame
        SPADL action.
    scores : pd.Series
        The probability of scoring from each corresponding game state.
    concedes : pd.Series
        The probability of conceding from each corresponding game state.

    Returns
    -------
    pd.Series
        The offensive value of each action.
    """
    same_team = _prev(actions.team_id) == actions.team_id
    # Prevents carrying over previous scores at the start of a new period (first/second half)
    same_period = _prev(actions.period_id) == actions.period_id 
    prev_scores = (_prev(scores) * (same_team & same_period) + _prev(concedes) * (~same_team & same_period)).astype(float)

    # if the previous action was too long ago, the odds of scoring are now 0
    # In set-pieces or goal-kick situations, the action is treated as independent. so previous score changes are not considered.
    toolong_idx = abs(actions.time_seconds - _prev(actions.time_seconds)) > _samephase_nb
    prev_scores[toolong_idx] = 0.0

    # if the previous action was a goal, the odds of scoring are now 0
    # prevgoal_idx = (_prev(actions.type_name).isin(["shot", "shot_freekick", "shot_penalty"])) & (
    #     _prev(actions.result_name) == "success"
    # )
    # prev_scores[prevgoal_idx] = 0

    # if the previous action was a goal, the odds of scoring(e.g. tackle, kick off...) are now 0
    for target in ["goal", "owngoal"]:
        _, kickoff_indices, _ = find_kickoff_indices(actions, target)
        prev_scores.loc[kickoff_indices] = 0.0

    # fixed odds of scoring when penalty
    penalty_idx = actions.type_name == "Penalty Kicks"
    prev_scores[penalty_idx] = 0.792453

    # fixed odds of scoring when corner
    corner_idx = actions.type_name.isin(["Corners"])
    prev_scores[corner_idx] = 0.046500

    return scores - prev_scores


def defensive_value(
    actions: DataFrame[Schema], scores: Series[float], concedes: Series[float]
) -> Series[float]:
    r"""Compute the defensive value of each action.

    VAEP defines the *defensive value* of an action as the change in conceding
    probability.

    .. math::

      \Delta P_{concede}(a_{i}, t) = P^{k}_{concede}(S_i, t) - P^{k}_{concede}(S_{i-1}, t)

    where :math:`P_{concede}(S_i, t)` is the probability that team :math:`t`
    which possesses the ball in state :math:`S_i` will concede in the next 10
    actions.

    Parameters
    ----------
    actions : pd.DataFrame
        SPADL action.
    scores : pd.Series
        The probability of scoring from each corresponding game state.
    concedes : pd.Series
        The probability of conceding from each corresponding game state.

    Returns
    -------
    pd.Series
        The defensive value of each action.
    """
    sameteam = _prev(actions.team_id) == actions.team_id
    # Prevents carrying over previous scores at the start of a new period (first/second half)
    same_period = _prev(actions.period_id) == actions.period_id 
    prev_concedes = (_prev(concedes) * (sameteam & same_period) + _prev(scores) * (~sameteam & same_period)).astype(float)

    # In set-pieces or goal-kick situations, the action is treated as independent. so previous score changes are not considered.
    toolong_idx = abs(actions.time_seconds - _prev(actions.time_seconds)) > _samephase_nb
    prev_concedes[toolong_idx] = 0.0

    # if the previous action was a goal, the odds of conceding are now 0
    # prevgoal_idx = (_prev(actions.type_name).isin(["shot", "shot_freekick", "shot_penalty"])) & (
    #     _prev(actions.result_name) == "success"
    # )
    # prev_concedes[prevgoal_idx] = 0.0

    # if the previous action was a goal, the odds of conceding(e.g. tackle, kick off...) are now 0
    for target in ["goal", "owngoal"]:
        _, kickoff_indices, _ = find_kickoff_indices(actions, target)
        prev_concedes.loc[kickoff_indices] = 0.0

    return -(concedes - prev_concedes)


def value(
    actions: DataFrame[Schema], Pscores: Series[float], Pconcedes: Series[float]
) -> pd.DataFrame:
    r"""Compute the offensive, defensive and VAEP value of each action.

    The total VAEP value of an action is the difference between that action's
    offensive value and defensive value.

    .. math::

      V_{VAEP}(a_i) = \Delta P_{score}(a_{i}, t) - \Delta P_{concede}(a_{i}, t)

    Parameters
    ----------
    actions : pd.DataFrame
        SPADL action.
    Pscores : pd.Series
        The probability of scoring from each corresponding game state.
    Pconcedes : pd.Series
        The probability of conceding from each corresponding game state.

    Returns
    -------
    pd.DataFrame
        The 'offensive_value', 'defensive_value' and 'vaep_value' of each action.

    See Also
    --------
    :func:`~socceraction.vaep.formula.offensive_value`: The offensive value
    :func:`~socceraction.vaep.formula.defensive_value`: The defensive value
    """
    v = pd.DataFrame()
    
    v["offensive_value"] = offensive_value(actions, Pscores, Pconcedes)
    v["defensive_value"] = defensive_value(actions, Pscores, Pconcedes)
    v["vaep_value"] = v["offensive_value"] + v["defensive_value"]

    return v

#original version function
def find_kickoff_indices(actions, target):
    if target == "goal":
        target_idx = (actions.type_name.isin(["Shot", "shot_freekick", "shot_penalty"])) & (
            actions.result_name == "success"
        )
    elif target == "owngoal":
        target_idx = (actions.result_name == "owngoal")
    else:
        raise ValueError(f"Invalid task: target={target}")

    target_indices = actions[target_idx].index.values
    pass_indices = actions[actions.type_name.isin(["pass"])].index.values

    # 이진탐색(searchsorted) : if side='right', array[i-1] <= value < array[i]이다. 
    # j번째 득점이후 킥오프를 찾을 때, pass_idx[i-1] <= 득점[j] < pass_idx[i]인 i가 kick off의 인덱스이다.
    positions = np.searchsorted(pass_indices, target_indices, side='right') # 득점 이후 첫번째 패스 인덱스

    # 킥오프 인덱스 & 킥오프 이전 인덱스 
    kickoff_indices = []
    prev_kickoff_indices = []
    for pos in positions:
        if pos < len(pass_indices):
            kickoff_idx = pass_indices[pos]
            kickoff_indices.append(kickoff_idx)
            prev_kickoff_indices.append(kickoff_idx - 1)
        elif pos == len(pass_indices):
            prev_kickoff_indices.append(actions.index[-1]) # 예외) 득점 이후 킥오프 액션이 없는 경우: 득점과 동시에 경기가 끝나는 경우
            #print(actions["game_id"].unique()) # game_id = [85890, 114462, 85789, 85931]
        else:
            raise ValueError(f"Invalid pos: pos={pos}")

    return target_indices, kickoff_indices, prev_kickoff_indices

def calculate_total_vaep(vaep):
    stats = (
        vaep[["player_id", "vaep_value", "offensive_value", "defensive_value"]]
        .groupby(["player_id"])
        .sum()
        .reset_index()
    )

    return stats

def calculate_vaep_per_90(vaep, player_info):
    stats = calculate_total_vaep(vaep)
    stats_per_90 = stats.merge(player_info, on="player_id", how="left")

    stats_per_90["vaep_value_per_90"] = stats_per_90.vaep_value * 90 / stats_per_90.total_minutes_played
    stats_per_90["offensive_value_per_90"] = stats_per_90.offensive_value * 90 / stats_per_90.total_minutes_played
    stats_per_90["defensive_value_per_90"] = stats_per_90.defensive_value * 90 / stats_per_90.total_minutes_played
    
    return stats_per_90

# 첫 번째 방법 (총 VAEP / 총 경기 수)
def calculate_simple_vaep_per_game(vaep, player_info):
    stats = calculate_total_vaep(vaep)
    stats_per_game = stats.merge(player_info, on="player_id", how="left")
    
    stats_per_game["simple_vaep_value_per_game"] = stats_per_game.vaep_value / stats_per_game.games_played
    stats_per_game["simple_offensive_value_per_game"] = stats_per_game.offensive_value / stats_per_game.games_played
    stats_per_game["simple_defensive_value_per_game"] = stats_per_game.defensive_value / stats_per_game.games_played
    
    return stats_per_game

# 두 번째 방법 (경기별 90분 단위 환산 후 산출)
# 경기별 VAEP의 합을 구하되 출전시간이 적은 선수도 공정한 기회를 보정하기 위해 90분당 VAEP의 총합으로 변환한 후에 경기당 지표로 계산.
def calculate_normalized_vaep_per_game(vaep, player_info, player_games):
    # 각 경기별 선수의 총 VAEP합
    playersR_by_game = (
        vaep[["player_id", "game_id", "vaep_value", "offensive_value", "defensive_value"]]
        .groupby(["player_id", "game_id"])
        .sum()
        .reset_index()
    )
    playersR_by_game = playersR_by_game.merge(
        player_games[["player_id", "game_id", "minutes_played"]], on=["player_id", "game_id"], how="left")
        
    # 각 경기별 선수의 90분당 VAEP로 변환: 풀타임으로 뛴 선수와 교체로 짧게 뛴 선수를 동일하게 평가하기 위함을 목적
    playersR_by_game['vaep_value_by_game'] = playersR_by_game['vaep_value'] * 90 / playersR_by_game['minutes_played']
    playersR_by_game['offensive_value_by_game'] = playersR_by_game['offensive_value'] * 90 / playersR_by_game['minutes_played']
    playersR_by_game['defensive_value_by_game'] = playersR_by_game['defensive_value'] * 90 / playersR_by_game['minutes_played']

    # 선수별 90분 VAEP의 총합
    stats_per_game = (
        playersR_by_game[["player_id", "vaep_value_by_game", "offensive_value_by_game", "defensive_value_by_game"]]
        .groupby(["player_id"])
        .sum()
        .reset_index()
    )
    stats_per_game = stats_per_game.merge(player_info, on="player_id", how="left")

    # 정규화된 선수별 경기당 VAEP
    stats_per_game['normalized_vaep_value_per_game'] = stats_per_game['vaep_value_by_game']  / stats_per_game['games_played']
    stats_per_game['normalized_offensive_value_per_game'] = stats_per_game['offensive_value_by_game']  / stats_per_game['games_played']
    stats_per_game['normalized_defensive_value_per_game'] = stats_per_game['defensive_value_by_game']  / stats_per_game['games_played']

    return stats_per_game

def calculate_vaep_per_game_and_90(vaep, player_games):
    # actor
    actor_by_game = (
        vaep[["player_id", "game_id", "vaep_value", "offensive_value", "defensive_value"]]
        .groupby(["player_id", "game_id"])
        .sum()
        .reset_index()
    )
    actor_by_game = actor_by_game[actor_by_game["player_id"] != -1]


    stats = actor_by_game.copy()
    stats_per_90 = stats.merge(player_games, on=["game_id", "player_id"], how="left")

    # 정규화된 선수별 경기당 VAEP
    stats_per_90['vaep_value_per_90'] = stats_per_90['vaep_value'] * 90 / stats_per_90['minutes_played']
    stats_per_90['offensive_value_per_90'] = stats_per_90['offensive_value'] * 90 / stats_per_90['minutes_played']
    stats_per_90['defensive_value_per_90'] = stats_per_90['defensive_value'] * 90 / stats_per_90['minutes_played']

    return stats_per_90