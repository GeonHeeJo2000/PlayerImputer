"""Schema for SPADL actions."""

from typing import Any, Optional

import pandera as pa
from pandera.typing import Series

import vaep.config as config

class Schema(pa.SchemaModel):
    """Definition of a SPADL dataframe."""

    game_id: Series[Any] = pa.Field()
    event_id: Series[Any] = pa.Field(nullable=True)
    action_id: Series[int] = pa.Field()
    period_id: Series[int] = pa.Field(ge=1, le=5)
    time_seconds: Series[float] = pa.Field(ge=0) # Seconds since the start of the period.
    team_id: Series[Any] = pa.Field()
    player_id: Series[Any] = pa.Field()
    start_x: Series[float] = pa.Field(ge=0, le=config.field_length)
    start_y: Series[float] = pa.Field(ge=0, le=config.field_width)
    end_x: Series[float] = pa.Field(ge=0, le=config.field_length)
    end_y: Series[float] = pa.Field(ge=0, le=config.field_width)
    bodypart_id: Series[int] = pa.Field(isin=config.bodyparts_df().bodypart_id)
    bodypart_name: Optional[Series[str]] = pa.Field(isin=config.bodyparts_df().bodypart_name)
    type_id: Series[int] = pa.Field(isin=config.actiontypes_df().type_id)
    type_name: Optional[Series[str]] = pa.Field(isin=config.actiontypes_df().type_name)
    result_id: Series[int] = pa.Field(isin=config.results_df().result_id)
    result_name: Optional[Series[str]] = pa.Field(isin=config.results_df().result_name)

    class Config:  # noqa: D106
        strict = True
        coerce = True
