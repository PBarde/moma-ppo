import argparse
import torch
import numpy as np
import gym

from multiagent_mujoco import MujocoMulti

from alfred.utils.directory_tree import DirectoryTree
from alfred.utils.config import load_config_from_json

from offline_marl.single_agent.iql import IQLLearner
from dzsc.ma_d4rl.utils.env_and_dataset import make_env
import offline_marl.utils.ml as ml
from tqdm import tqdm

import mapping 

# NOTE: mappings for ant are only safe if states and obs do not contain contact forces (always zero due to mujoco bug)
# scenario_agent-conf_agent-obsk
SAFE_MAPPINGS = ['halfcheetah-v2_2x3_0', 'halfcheetah-v2_2x3_None', 'halfcheetah-v2_6x1_0', 'halfcheetah-v2_6x1_None',
                'ant-v2_4x2_0', 'ant-v2_4x2_None', 'ant-v2_2x4_0', 'ant-v2_2x4_None', 'ant-v2_2x4d_0', 'ant-v2_2x4d_None']

def get_loading_args():
    parser = argparse.ArgumentParser()

    # env args
    parser.add_argument("--scenario", type=str, default='Ant-v2', help="Mujoco env")
    parser.add_argument("--agent_conf", type=str, default="2x4", help="how to split single agent env into multiple-agent: n_agents X action_dim")
    parser.add_argument("--agent_obsk", type=int, default=None, help="multi-agent mujoco observation distance. 0 = observe just the correspinding agent, None = observe global state")
    
    # model args
    parser.add_argument("--storage_name", type=str, default="No73_2dfb5ed-4aff6f8_iql_ant-medium-v2_cutoff_obs", help="name of storage, relative to root_dir")
    parser.add_argument("--experiment_num", type=int, default=1)
    parser.add_argument("--seed_num", type=str, default=131214)
    parser.add_argument("--model_name", type=str, default="iql_model_temporary.pt", help="full model name, ex: iql_model_best.pt")
    parser.add_argument("--eval_seed", type=int, default=10101)
    parser.add_argument("--max_eval_episode", type=int, default=10)
    parser.add_argument("--root_dir", type=str, default=None)
    
    return parser.parse_args()

def load_and_evaluate(args):
    dir_tree = DirectoryTree.init_from_branching_info(root_dir=args.root_dir, storage_name=args.storage_name,
                                                      experiment_num=args.experiment_num, seed_num=args.seed_num)

    config = load_config_from_json(dir_tree.seed_dir / "config.json")

    if args.model_name is None:
        model_name = config.alg_name + '_model.pt'
    else:
        model_name = args.model_name

    model_path = dir_tree.seed_dir / model_name

    learner = IQLLearner.init_from_save(model_path, device=torch.device('cpu'))

    assert args.scenario.lower() == config.task_name.replace('-medium','')


    env_multi = MujocoMulti(env_args={"scenario": args.scenario,
                         "agent_conf": args.agent_conf,
                        "agent_obsk": args.agent_obsk,
                        "episode_limit": 1000})

    # contact forces are not computed for some versions of mujoco (such as in d4rl)
    if 'ant' in args.scenario.lower():
        msg = "contact forces are always 0 in ant so they will be removed"
        gym.logger.warn(msg)
        remove_zeros = True
    else:
        remove_zeros = False
    
    env_info = env_multi.get_env_info()

    n_actions = env_info["n_actions"]
    n_agents = env_info["n_agents"]

    env_d4rl = make_env(config.task_name, args.eval_seed, record_frames=False)

    # we load the mappings for the corresponding env
    mapping_args, _ = mapping.get_argparser().parse_known_args()
    mapping_args.__dict__.update(vars(args)) # updates mapping default args with current script args
    mappings = mapping.get_mappings(mapping_args)

    returns = []

    for e in tqdm(range(args.max_eval_episode)):
        ret = 0
        done = False 
        env_multi.reset()

        while not done:
            state = env_multi.get_state()
            obs = env_multi.get_obs()

            if remove_zeros:
                obs = [o[o != 0] for o in obs]
                state = state[state != 0]

            state = update_state_from_obs(state, obs, mappings)
            action = learner.policy.act(env_d4rl.to_torch(state, device=torch.device('cpu')), sample=False, return_log_pi=False).detach().cpu().numpy()
            actions = action.reshape(n_agents, n_actions)

            # we shuffle actions to see if it affects performance
            # shuffled_actions = actions[::-1,:]
            # r, done, info = env_multi.step(shuffled_actions)
            r, done, info = env_multi.step(actions)

            ret += r
        
        print(ret)
        returns.append(ret)
    
    score = np.mean(returns)
    score = env_d4rl.get_normalized_score(score)*100

    print(score)
    
def update_state_from_obs(state, obs, mappings):

    assert len(obs) == len(mappings)

    for agent_obs, agent_map in zip(obs, mappings):
        assert len(agent_obs) == len(agent_map)

        for o, idx in zip(agent_obs, agent_map):
            state[idx] = o

    return state

if __name__ == "__main__":
    args = get_loading_args()
    load_and_evaluate(args)