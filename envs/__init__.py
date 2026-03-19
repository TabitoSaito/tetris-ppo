from gymnasium.envs.registration import register
from gymnasium.wrappers.flatten_observation import FlattenObservation

register(
    id="GridWorld-v0",
    entry_point="envs.grid_world:GridWorldEnv",
    max_episode_steps=30,
    additional_wrappers=(FlattenObservation.wrapper_spec(),)
)