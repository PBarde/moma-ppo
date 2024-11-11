import argparse
import torch
import wandb
from pathlib import Path
import sys
import numpy as np
from tqdm import tqdm
import pickle
import os

def set_up_import():
    alfred_folder = (Path(__file__).resolve().parents[2] / 'alfred_omarl').resolve()
    alg_folder = (Path(__file__).resolve().parents[2] / 'offline_marl').resolve()
    env_folder = (Path(__file__).resolve().parents[2] / 'dzsc').resolve()
    
    for p in [alg_folder, env_folder, alfred_folder]:
        if p not in sys.path:
            sys.path.append(str(p))

set_up_import()

from alfred.utils.directory_tree import DirectoryTree
from alfred.utils.config import load_config_from_json, parse_bool


from offline_marl import trainer # this sets up correct default storage and all
from offline_marl.train import make_trainer
from offline_marl import train_rl
from offline_marl.single_agent.ppo import PPOBuffer
from dzsc.make_env_module import get_make_env_module
from offline_marl.utils.misc import save_video_to_wandb, get_timestamp
import offline_marl.utils.ml as ml

def get_loading_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-storage_name", type=str, default="", help="name of storage, relative to root_dir")
    parser.add_argument("-experiment_num", type=int, default=1)
    parser.add_argument("-seed_num", type=str, default=1)
    parser.add_argument("--model_name", type=str, default=None, help="full model name, ex: iql_model_best.pt")
    parser.add_argument("--eval_seed", type=int, default=10101)
    parser.add_argument("--max_transitions", type=int, default=int(1e6))
    parser.add_argument("--use_gpu", type=parse_bool, default=True)
    parser.add_argument("--root_dir", type=str, default="")
    
    return parser.parse_args()

def generate_dataset(args):

    dir_tree = DirectoryTree.init_from_branching_info(root_dir=args.root_dir, storage_name=args.storage_name,
                                                      experiment_num=args.experiment_num, seed_num=args.seed_num)

    config = load_config_from_json(dir_tree.seed_dir / "config.json")

    # gets training and rollouts devices
    train_device, rollout_device = ml.get_computing_devices(args.use_gpu, torch, do_rollouts_on_cpu=False, logger=None)

    if args.model_name is None:
        model_name = config.alg_name + '_model.pt'
    else:
        model_name = args.model_name

    model_path = dir_tree.seed_dir /  model_name
    
    learner = make_trainer(config.alg_name).get_make_learner(config.alg_name, config.task_name).init_from_save(model_path, device=rollout_device)

    env = get_make_env_module(config.task_name).make_env(config.task_name, args.eval_seed, record_gif=False)

    # we deal with episode termination ourselves
    if not any([task in config.task_name for task in ['toy', 'reacher']]):
            env.env.env.env.env.env.env._max_episode_steps = np.inf


    done = True
    time_out = True
    observations = None
    n_collected_transitions = 0
    pbar = tqdm(range(args.max_transitions), desc="collected transitions")

    collect_stats = PPOBuffer()

    dataset = PPOBuffer()

    while n_collected_transitions < args.max_transitions:

        ### we collect data for ppo
        batches, observations, done, time_out, collect_stats_dict = train_rl.collect_transition_batches(env, learner, observations, done, time_out, rollout_device, config, training_step=1)

        # update the number of env steps we did
        n_collected_transitions += config.ppo_transitions_between_update
        pbar.update(config.ppo_transitions_between_update)
        
        batches = train_rl.process_training_batches_to_save(batches)
        dataset.extend(batches)
        collect_stats.extend(collect_stats_dict, wrapped=True)
    
    dataset = dataset.flush()
    #we clip actions to env bounds (ppo policy is not restricted or clipped)
    low = np.asarray([s.low for s in env.action_space]).reshape(1,-1)
    high = np.asarray([s.high for s in env.action_space]).reshape(1,-1)
    dataset['actions'] = np.clip(a=dataset['actions'], a_min=low, a_max=high)

    collect_stats = collect_stats.flush()
    collect_stats.pop('collect_frames')
    collect_stats = {key: {'min': np.min(val), 'max': np.max(val), 'mean': np.mean(val), 'std': np.std(val), 'median': np.median(val)} for key, val in collect_stats.items()}


    d_path = dir_tree.seed_dir / 'generated_datasets'
    path = d_path / f'eval_seed_{args.eval_seed}_n_{args.max_transitions}_{model_name}'
    os.makedirs(path, exist_ok=True)

    dataset_path = str(path / 'dataset.pkl')
    with open(dataset_path, 'wb') as fh:
        pickle.dump(dataset, fh)

    summary_path = str(path / 'summary.txt')
    with open(summary_path, 'w') as fh:
        print(collect_stats, file=fh)

    print(f"Saved dataset {dataset_path}")
    print(collect_stats)
    

if __name__ == "__main__":
    args = get_loading_args()
    generate_dataset(args)