from pathlib import Path
import sys
import os

from torch.multiprocessing import set_start_method
try:
     set_start_method('spawn')
except RuntimeError:
    pass


def set_up_alfred(alg_folder, env_folder, alfred_folder):
    from alfred.utils.directory_tree import DirectoryTree

    DirectoryTree.default_root = str((Path(__file__).resolve().parents[2] / 'scratch').resolve())
    DirectoryTree.git_repos_to_track[alg_folder.name] = str(alg_folder)
    DirectoryTree.git_repos_to_track[env_folder.name] = str(env_folder)

def set_up_import():
    alfred_folder = (Path(__file__).resolve().parents[2] / 'alfred_omarl').resolve()
    alg_folder = (Path(__file__).resolve().parents[2] / 'offline_marl').resolve()
    env_folder = (Path(__file__).resolve().parents[2] / 'dzsc').resolve()
    
    for p in [alg_folder, env_folder, alfred_folder]:
        if p not in sys.path:
            sys.path.append(str(p))

    return alg_folder, env_folder, alfred_folder

set_up_alfred(*set_up_import())

import argparse
from dzsc.ma_d4rl.constants import MULTI_AGENT_TASKS
from nop import NOP
import logging
import torch
import datetime
import submitit
from pathlib import Path
import traceback
import warnings

from alfred.utils.config import save_config_to_json, load_config_from_json
from alfred.utils.directory_tree import DirectoryTree

import offline_marl.utils.ml as ml

class Training:

    @staticmethod
    def get_training_argsparser():
        parser = argparse.ArgumentParser()

        # Alfred args
        parser.add_argument('--desc', type=str, default='', help="description of the run")
        parser.add_argument('--alg_name', type=str, default='', help="name of the algo, currently only used by alfred")
        parser.add_argument('--task_name', type=str, default='', help='d4rl task name with agent decomposition')
        parser.add_argument('--seed', type=int, default=131, help='random seed that seeds everything')

        return parser

    def __init__(self) -> None:
        self.learner = None
    
    def __call__(self, checkpointpath: str):
        # checkpointpath should be a seed_dir in terms of alfred
        seed_dir = Path(checkpointpath)
        config = load_config_from_json(str(seed_dir / 'config.json'))
        dir_tree = DirectoryTree.init_from_seed_path(seed_dir, root=str(seed_dir.parents[2]))

        try:
            # we start back from a preemption
            if (seed_dir / 'PREEMPTED').exists():

                os.remove(str(seed_dir / 'PREEMPTED'))

                preemted_model_path = seed_dir / (config.alg_name + '_model_preempt.pt')
                regular_model_path = seed_dir / (config.alg_name + '_model.pt')

                # if we didn't manage to save a model at checkpoint we take the regularly saved model
                if preemted_model_path.exists():
                    model_path = preemted_model_path
                else:
                    model_path = regular_model_path

                # we can load the model on the cpu and it will be moved on gpu by the .train() afterwards
                # actually this does not work with optimizers better to load directly on train_device
                train_device, rollout_device = ml.get_computing_devices(config.use_gpu, torch, do_rollouts_on_cpu=True)
                warnings.warn(f'train_device: {train_device}')
                learner = self.get_make_learner(config.alg_name, config.task_name).init_from_save(model_path, device=train_device)
                # we delete the preemt model for next preemption
                if preemted_model_path.exists():
                    ml.remove([learner], seed_dir, suffix="model_preempt.pt")

                return self.train(config, learner=learner, dir_tree=dir_tree)
            
            elif (seed_dir / 'UNHATCHED').exists():
                os.remove(str(seed_dir / 'UNHATCHED'))
                # if UNHATCHED we make sure that we start from 0 
                config.training_step = 0
                if 'wandb_run_id' in config.__dict__:
                    del config.__dict__['wandb_run_id']

                return self.train(config, dir_tree=dir_tree)

            else:
                warnings.warn('Trying to run a job that is neither UNHATCHED not PREEMPTED')
        
        except Exception as e:
                with open(str(seed_dir / 'CRASH.txt'), 'w+') as f:
                    f.write(f'Crashed at: {datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S")}.')
                    f.write(f'Error: {e}\n')
                    f.write(traceback.format_exc())
                raise e


    def checkpoint(self, checkpointpath):
        
        # we had enough time to build dataset and save initial model
        if hasattr(self, 'to_save'):
            # we add the preempted flag
            open(str(self.dir_tree.seed_dir / 'PREEMPTED'), 'w+').close()
            
            warnings.warn("Created PREEMPTED flag")

            # we save the model
            ml.save(self.to_save, self.dir_tree.seed_dir, suffix="model_preempt.pt")

            warnings.warn(f"Saved Model in {self.dir_tree.seed_dir}")

            # we save the config
            self.config.training_step = self.__dict__.get('training_step', self.config.__dict__.get('training_step', 0))
            save_config_to_json(self.config, str(self.dir_tree.seed_dir / 'config.json'))

        else:
            # it's like we did nothing 
            open(str(self.dir_tree.seed_dir / 'UNHATCHED'), 'w+').close()
            
        # we define the relaunch train function class
        seed_dir = self.dir_tree.seed_dir
        training_callable = self.__class__()


        warnings.warn(f'Delayed submission')
        # Resubmission to the queue is performed through the DelayedSubmission object
        # note that we overide the checkpointpath here to be sure it is the seed_dir
        return submitit.helpers.DelayedSubmission(training_callable, str(seed_dir))


    def train(self, config, learner=None, dir_tree=None, pbar="default_pbar", logger=None):
        raise NotImplementedError

