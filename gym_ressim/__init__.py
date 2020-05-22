from gym.envs.registration import register

register(
    id='ResSim-v1',
    entry_point='gym_ressim.envs:ResSimEnv',
    max_episode_steps=5,
)