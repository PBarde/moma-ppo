
from pathlib import Path
import sys
import re
import os

def set_up_alfred_and_import():
    alfred_folder = (Path(__file__).resolve().parents[2] / 'alfred').resolve()
    alg_folder = (Path(__file__).resolve().parents[2] / 'offline_marl').resolve()
    env_folder = (Path(__file__).resolve().parents[2] / 'dzsc').resolve()

    sys.path.append(str(alg_folder))
    sys.path.append(str(env_folder))
    sys.path.append(str(alfred_folder))
    
set_up_alfred_and_import()

import argparse
import torch
import numpy as np

from offline_marl.train_worldmodel import WorldModelEnsemble
from offline_marl.utils import ml
from offline_marl.train_worldmodel import WorldTraining

def get_loading_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name", type=str, default="hanabi-human_data") # dataset that will be loaded
    parser.add_argument("--wm_overwrite_task_name", type=str, default="hanabi-iql_selfplay") # folder of the wm
    parser.add_argument("--model_name", type=str, default="world-model_model_wm_6.pt")
    parser.add_argument("--k_steps_rollouts", type=int, default=5, help="number of steps world model rollouts")
    parser.add_argument("--memory_len", type=int, default=10)
    parser.add_argument("--use_gpu", default=True)
    return parser.parse_args()

def load_wm_ensemble(args, device):
    from offline_marl import train

    code_root = Path('/home/mila/b/bardepau/code/offline_marl/scratch')

    if args.wm_overwrite_task_name is not None:
        wm_task_name = args.wm_overwrite_task_name
    else:
        wm_task_name = args.task_name

    model_path = code_root/ 'world_models' / wm_task_name / args.model_name
    
    wm_ensemble = WorldModelEnsemble.init_from_save(str(model_path), device)
    return wm_ensemble

def quick_eval(args):
    device, _ = ml.get_computing_devices(args.use_gpu, torch, do_rollouts_on_cpu=False, logger=None)
    

    wm_ensemble = load_wm_ensemble(args, device)
    make_env_and_dataset =  WorldTraining.get_make_env_module(args.task_name).make_env_and_dataset
    env, dataset, dataset_valid = make_env_and_dataset(task_name=args.task_name,
                                                            centralized_training=True, 
                                                            seed=1212, 
                                                            train_device=device,
                                                            logger=None, 
                                                            memory_len=args.memory_len,
                                                            context_len=0,
                                                            make_worldmodel_dataset=True)

    assert env.action_space == wm_ensemble.init_dict['act_space']
    assert env.observation_space == wm_ensemble.init_dict['obs_space']


    # Evaluation phase 
    stats = wm_ensemble.evaluate(dataset, dataset_valid, 50, 256)

    return stats


def eval_accoss_models_and_datasets(args, metrics):
    device, _ = ml.get_computing_devices(args.use_gpu, torch, do_rollouts_on_cpu=False, logger=None)

    models_folders = ['hanabi-human_data', 'hanabi-iql_selfplay', 'hanabi-iql_crossplay']
    dataset_folders = ['hanabi-human_data', 'hanabi-iql_selfplay', 'hanabi-iql_crossplay']

    stats_across = {}

    for dataset_f in dataset_folders:
        stats_across[dataset_f] = {}

        args.task_name = dataset_f

        make_env_and_dataset =  WorldTraining.get_make_env_module(args.task_name).make_env_and_dataset
        env, dataset, dataset_valid = make_env_and_dataset(task_name=args.task_name,
                                                            centralized_training=True, 
                                                            seed=1212, 
                                                            train_device=device,
                                                            logger=None, 
                                                            memory_len=args.memory_len,
                                                            context_len=0,
                                                            make_worldmodel_dataset=True)

        for model_f in models_folders:
            args.wm_overwrite_task_name = model_f
            
            wm_ensemble = load_wm_ensemble(args, device)

            
            stats_across[dataset_f][model_f] = wm_ensemble.evaluate(dataset, dataset_valid, 500, 256)

    stat_aggreg = {}
    for dataset_f in dataset_folders:
        stat_aggreg[dataset_f] = {}
        for model_f in models_folders:
            stat_aggreg[dataset_f][model_f] = {}
            for m in metrics:
                exp = re.compile(m)
                l = []
                for key, val in stats_across[dataset_f][model_f].items():
                    match = exp.match(key)
                    if match:
                        l.append(val)

                stat_aggreg[dataset_f][model_f][m] = np.mean(l)



    big_space = '            '
    small_space = '   '
    for metric in metrics:
        s = metric + '\n'
        s += big_space
        for dataset_f in dataset_folders:
            s += dataset_f
            s += small_space
        
        s += '\n'

        for model_f in models_folders:
            s += model_f
            s += small_space
            for dataset_f in dataset_folders:
                s += f'{stat_aggreg[dataset_f][model_f][metric]}'
                s += small_space
            
            s += '\n'
        
        print(s)




if __name__ == "__main__":
    args = get_loading_args()

    eval_accoss_models_and_datasets(args, metrics=[r'wm_\d+_R_dist_means_valid', r'wm_\d+_T_dist_means_valid'])