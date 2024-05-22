from gymnasium.envs.registration import register

register(id='MIMoSaccadeVergence-v0',
         entry_point='mimoAEC.envs:MIMoSaccadeVergenceEnv',
         max_episode_steps=32, 
         )

register(id='MIMoPlayRoom-v0',
         entry_point='mimoAEC.envs:MIMoPlayRoomEnv',
         max_episode_steps=1000, 
         )

register(id='MIMoFlatTargetsOne-v0',
         entry_point='mimoAEC.envs:MIMoFlatTargetsOneEnv',
         max_episode_steps=1000, 
         )

