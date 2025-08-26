"""Configuration of the SPADL language.

Attributes
----------
field_length : float
    The length of a pitch (in meters).
field_width : float
    The width of a pitch (in meters).
bodyparts : list(str)
    The bodyparts used in the SPADL language.
results : list(str)
    The action results used in the SPADL language.
actiontypes : list(str)
    The action types used in the SPADL language.

"""

import pandas as pd  # type: ignore

field_length: float = 105.0  # unit: meters
field_width: float = 68.0  # unit: meters

# bodyparts: list[str] = ["foot", "head", "other", "head/other", "foot_left", "foot_right"]
bodyparts: list[str] = [
    "foot", 
    "other"
]
results: list[str] = [
    "fail",
    "success",
    #"offside",
    "owngoal",
    "yellow_card",
    "red_card",
]

actiontypes: list[str] = [
    "Pass",
    "Cross",

    "Throw-In",
    "Pass_Freekick",
    "Pass_Corner",
    "Shot_Freekick",
    "Shot_Corner",
    "Penalty Kicks",
    "Goal Kick",

    "Take-On",
    "Carry",
    "Foul",
    "Tackle",
    "Interception",
    "Shot",
    "Aerial Control",
    "Defensive Line Supports",
    "Save",
    "Block",
    "Clearance",
    "bad_touch",
    "Receive",
    "Recovery",
    "Duel",
    "Offside",

    "Foul Won",
    "non_action"
]


def actiontypes_df() -> pd.DataFrame:
    """Return a dataframe with the type id and type name of each SPADL action type.

    Returns
    -------
    pd.DataFrame
        The 'type_id' and 'type_name' of each SPADL action type.
    """
    return pd.DataFrame(list(enumerate(actiontypes)), columns=["type_id", "type_name"])


def results_df() -> pd.DataFrame:
    """Return a dataframe with the result id and result name of each SPADL action type.

    Returns
    -------
    pd.DataFrame
        The 'result_id' and 'result_name' of each SPADL action type.
    """
    return pd.DataFrame(list(enumerate(results)), columns=["result_id", "result_name"])


def bodyparts_df() -> pd.DataFrame:
    """Return a dataframe with the bodypart id and bodypart name of each SPADL action type.

    Returns
    -------
    pd.DataFrame
        The 'bodypart_id' and 'bodypart_name' of each SPADL action type.
    """
    return pd.DataFrame(list(enumerate(bodyparts)), columns=["bodypart_id", "bodypart_name"])