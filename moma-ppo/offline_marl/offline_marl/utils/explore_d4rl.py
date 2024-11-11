import gym
import d4rl # Import required to register environments
from offline_marl.utils.env_and_dataset import make_env_and_dataset

# Create the environment
env, dataset = make_env_and_dataset('antmaze-umaze-diverse-v2', seed=1000, record_frames=False)

obs = dataset.observations

for o in obs:
    if any(o[27:]):
        print("found contact forces")

print('stop')