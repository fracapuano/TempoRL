"""
Test a random policy on the laser environment.
Built to test out different parts of the environment.
"""
from tqdm import tqdm
from env.LaserEnv_v1 import LaserEnv_v1
from env.env_utils import EnvParametrization

params = EnvParametrization()
compressor_params, bounds, B_integral = params.get_parametrization()

env = LaserEnv_v1(
    bounds = bounds, 
    compressor_params = compressor_params, 
    B_integral = B_integral)

print('State space:', env._observation_space)
print('Action space:', env.action_space)

n_episodes = 5
render = True  # TODO: Implement custom renderer.  

for episode in range(n_episodes):
    done = False
    observation = env.reset()	# Reset environment to initial state
    episode_length = 0
    while not done:  # Until the episode is over
        action = env.action_space.sample()	# Sample random action
        observation, reward, done, info = env.step(action)	# Step the simulator to the next timestep
        episode_length += 1

        if render:
            env.render()
        
    print(f"Episode Length: {episode_length}")
    print(info["L1Loss"])