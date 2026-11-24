"""School Director environment package."""

from gymnasium.envs.registration import register

from environment.custom_env import (
    ACTION_NAMES,
    CHILD_NAMES,
    N_ACTIONS,
    OBS_NAMES,
    SchoolEnv,
    make_config,
    obs_from_state,
)

register(id="SchoolDirector-v0", entry_point="environment.custom_env:SchoolEnv")

__all__ = [
    "ACTION_NAMES", "CHILD_NAMES", "N_ACTIONS", "OBS_NAMES",
    "SchoolEnv", "make_config", "obs_from_state",
]
