import os.path as osp
from pathlib import Path
import argparse
import numpy as np
import json
import pickle
from tqdm import tqdm
import gym 

from multiagent_mujoco import MujocoMulti

# TODO: Note that many mappings are still partial observable even when considering all the agents! Meaning that 
# some part of the state is not observed by any agent and thus one cannot reconstruct the state from the observations
# even when considering all the agents


def get_argparser():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--scenario", type=str, default='HalfCheetah-v2', help="Mujoco env")
    parser.add_argument("--agent_conf", type=str, default='6x1', help="how to split single agent env into multiple-agent: n_agents X action_dim")
    parser.add_argument("--agent_obsk", type=int, default=0, help="multi-agent mujoco observation distance. 0 = observe just the correspinding agent, None = observe global state")
    parser.add_argument("--max_episode", type=int, default=10, help="number of episodes to collect")

    return parser

name_dict = {'ant-v2':'Ant-v2',
            'halfcheetah-v2': 'HalfCheetah-v2'}

def parse_agent_obsk(agent_obsk):
    if agent_obsk == 'None':
        return None
    else:
        return int(agent_obsk)

def collect_mapping_data(args):
    env = MujocoMulti(env_args={"scenario": name_dict.get(args.scenario, args.scenario),
                         "agent_conf": args.agent_conf,
                        "agent_obsk": parse_agent_obsk(args.agent_obsk),
                        "episode_limit": 1000})

    # contact forces are not computed for some versions of mujoco (such as in d4rl)
    if 'ant' in args.scenario.lower():
        msg = "contact forces are always 0 in ant so they will be removed"
        gym.logger.warn(msg)
        remove_zeros = True
    else:
        remove_zeros = False

    env_info = env.get_env_info()

    n_actions = env_info["n_actions"]
    n_agents = env_info["n_agents"]

    mapping_data = {'obs': [],
                    'states': [],
                    'actions': []}

    for e in tqdm(range(args.max_episode), desc="collects mapping data"):
        env.reset()
        terminated = False

        while not terminated:
            obs = env.get_obs()
            state = env.get_state()

            if remove_zeros:
                obs = [o[o != 0] for o in obs]
                state = state[state != 0]

            mapping_data['obs'].append(obs)
            mapping_data['states'].append(state)

            actions = []
            for agent_id in range(n_agents):
                action = np.random.uniform(-1.0, 1.0, n_actions)
                actions.append(action)
            
            mapping_data['actions'].append(actions)

            _, terminated, _ = env.step(actions)

    return mapping_data

def estimate_mapping(mapping_data):
    states = mapping_data['states']
    obs = mapping_data['obs']

    n_transition = len(obs)
    n_agent = len(obs[0])
    
    mappings = []

    # for each agent and transition we match obs components to state components
    for agent_i in tqdm(range(n_agent), desc="estimating agent's mapping"):
        temp_matchings = []
        for t in tqdm(range(n_transition)):
            current_state = states[t]
            current_agent_obs = obs[t][agent_i]

            # we compute the matching but ONLY on non-zeros components true zero components are always zero 
            current_matching = []
            for o in current_agent_obs:
                for i, s in enumerate(current_state):
                    if o == s :
                        current_matching.append(i)
                        break

            # we assert that we can reconstruct the obs from the state
            reconstructed_obs = current_state[current_matching]
            assert all(reconstructed_obs == current_agent_obs)

            # we transform to tuple because we want to make a set afterwards
            temp_matchings.append(tuple(current_matching))
    
        # we make sure that all matchings are the same so that it defines a mapping
        mapping = set(temp_matchings)
        assert len(mapping) == 1
        mappings.append(list(mapping.pop())) # back to list because we want to iterate over indexes
    
    return mappings

def get_mappings(args):
    root = Path(__file__).absolute().parent

    env_name = args.scenario.lower() + "_" + args.agent_conf + "_" + str(args.agent_obsk)

    mapping_file = root / env_name / 'mapping.json'

    if mapping_file.exists():
        # just load mapping
        with open(str(mapping_file), 'r') as f:
            mappings = json.load(f)
    
    else: 
        # we estimate obs mapping from data
        mapping_data_file = root / env_name / 'mapping_data.pkl'

        if not mapping_data_file.exists():
            # collect env interactions to estimate obs mapping
            mapping_data_file.parent.mkdir(parents=True, exist_ok=True)
            mapping_data = collect_mapping_data(args)

            with open(str(mapping_data_file), 'wb') as f:
                pickle.dump(mapping_data, f)
        else:

            with open(str(mapping_data_file), 'rb') as f:
                mapping_data = pickle.load(f)

        mappings = estimate_mapping(mapping_data)

        with open(str(mapping_file), 'w') as f:
            json.dump(mappings, f)
    
    return mappings

if __name__ == "__main__":

    args = get_argparser().parse_args()

    mappings = get_mappings(args)

    print(mappings)


