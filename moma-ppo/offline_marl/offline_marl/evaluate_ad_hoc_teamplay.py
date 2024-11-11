import argparse
from multiprocessing import Pool
import multiprocessing
import multiprocessing.pool
import sys

import torch
import wandb
import numpy as np
from pathlib import Path
import pickle
import warnings
from nop import NOP

def set_up_import():
    alfred_folder = (Path(__file__).resolve().parents[2] / 'alfred_omarl').resolve()
    alg_folder = (Path(__file__).resolve().parents[2] / 'offline_marl').resolve()
    env_folder = (Path(__file__).resolve().parents[2] / 'dzsc').resolve()
    
    for p in [alg_folder, env_folder, alfred_folder]:
        if p not in sys.path:
            sys.path.append(str(p))

set_up_import()

from alfred.utils.directory_tree import DirectoryTree, sanity_check_exists
from alfred.utils.config import load_config_from_json, parse_bool, save_config_to_json
from alfred.utils.recorder import Aggregator, Recorder
from alfred.utils.misc import Bunch, select_storage_dirs

from offline_marl.train import make_trainer
from offline_marl.utils.ml import set_seeds, get_computing_devices
from dzsc.make_env_module import get_make_env_module
from offline_marl.utils.misc import save_video_to_wandb, get_timestamp

from offline_marl.train_rl import evaluate
from offline_marl.single_agent.constants import SINGLE_AGENT_ALGOS

from gym.wrappers.time_limit import TimeLimit
from dzsc.ma_d4rl.utils.env_wrapper import EpisodeMonitor


class NoDaemonProcess(multiprocessing.Process):
    @property
    def daemon(self):
        return False

    @daemon.setter
    def daemon(self, value):
        pass


class NoDaemonContext(type(multiprocessing.get_context())):
    Process = NoDaemonProcess

# We sub-class multiprocessing.pool.Pool instead of multiprocessing.Pool
# because the latter is only a wrapper function, not a proper class.
class MyPool(multiprocessing.pool.Pool):
    def __init__(self, *args, **kwargs):
        kwargs['context'] = NoDaemonContext()
        super(MyPool, self).__init__(*args, **kwargs)


def get_experiment_dirs_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--from_file', type=str, default='', help="Path containing all the storage_names to launch")
    parser.add_argument("--storage_name", type=str, nargs='+', default=None, help="name of storage, relative to root_dir")
    parser.add_argument("--only_expe_dir", type=str, default=None)
    parser.add_argument("--root_dir", type=str, default='')
    return parser.parse_args()



def get_experiment_dirs(experiment_dirs_args):
    storage_dirs = select_storage_dirs(experiment_dirs_args.from_file, experiment_dirs_args.storage_name, experiment_dirs_args.root_dir)
    storage_dirs = [storage_dir for storage_dir in storage_dirs if sanity_check_exists(storage_dir, None)]

    experiment_dirs = []
    for storage_dir in storage_dirs:
        experiment_dirs += DirectoryTree.get_all_experiments(storage_dir)

    filtered_experiment_dirs = []
    for experiment_dir in experiment_dirs:
        if experiment_dirs_args.only_expe_dir is not None:
            if not experiment_dir.name == experiment_dirs_args.only_expe_dir:
                continue
        filtered_experiment_dirs.append(experiment_dir)

    return filtered_experiment_dirs


class Evaluator(object):

    # ARGS = {"cross_play_eval": True,
    #             "model_name": None,
    #             "eval_seed": 10101,
    #             "max_eval_episode": 2,
    #             "env_rollout_length_eval": 1000,
    #             "n_cross_eval": 4,
    #             "record_gif": True,
    #             'n_ep_in_gif': 2,
    #             "extend_old_pkl": False,
    #             "skip_if_pkl_exists": False,
    #             "use_gpu": True,
    #             "n_processes": 3}

    # ARGS = {"cross_play_eval": True,
    #         "model_name": None,
    #         "eval_seed": 10101,
    #         "max_eval_episode": 10,
    #         "env_rollout_length_eval": int(1e3),
    #         "n_cross_eval": 10,
    #         "record_gif": False,
    #         'n_ep_in_gif': 2,
    #         "extend_old_pkl": False,
    #         "skip_if_pkl_exists": False,
    #         "use_gpu": True,
    #         "n_processes": 3}
    
    # ARGS = {"cross_play_eval": False,
    #         "model_name": 'ma-ppo_model_init.pt',
    #         "eval_seed": 10101,
    #         "max_eval_episode": 100,
    #         "env_rollout_length_eval": 50,
    #         "n_cross_eval": 10,
    #         "record_gif": False,
    #         'n_ep_in_gif': 2,
    #         "extend_old_pkl": False,
    #         "skip_if_pkl_exists": False,
    #         "use_gpu": True,
    #         "n_processes": 1}
    
    ARGS = {"cross_play_eval": False,
                "model_name": None,
                "eval_seed": 10101,
                "max_eval_episode": 10,
                "env_rollout_length_eval": 'env_dependent',
                "n_cross_eval": 10,
                "record_gif": False,
                'n_ep_in_gif': 2,
                "extend_old_pkl": False,
                "skip_if_pkl_exists": True,
                "use_gpu": True,
                "n_processes": 3}

    
    def __init__(self):
        self.evaluator = None
    
    def __call__(self, checkpointpath: str):
        # checkpointpath should be a experiment_dir in terms of alfred
        experiment_dir = Path(checkpointpath)
        args = Bunch(self.ARGS.copy())
        return self._load_and_evaluate_experiment_dir(experiment_dir, args)
    
    # this is for when we do not use a submitit script to launch the eval
    def load_and_evaluate(self, experiment_dirs):
        args = Bunch(self.ARGS.copy())
        for experiment_dir in experiment_dirs:
            self._load_and_evaluate_experiment_dir(experiment_dir, args)
        warnings.warn('Evaluation is done.')

    def _load_and_evaluate_experiment_dir(self, experiment_dir, args):
        save_config_to_json(args, str(experiment_dir / 'eval_config.json'))

        list_of_args = []
        seed_dirs = DirectoryTree.get_all_seeds(experiment_dir)
        for i, seed_dir in enumerate(seed_dirs): 
            seed_args = dict(**args.__dict__)
            seed_args['seed_dir'] = seed_dir
            seed_args['eval_seed'] = args.eval_seed + 1 + i
            list_of_args.append(Bunch(seed_args))

        if args.n_processes > 1:
            with MyPool(args.n_processes) as p:
                results = p.map(self._load_and_evaluate_seed_dir, list_of_args)
        else:
            results = [self._load_and_evaluate_seed_dir(a) for a in list_of_args]

        aggregated_results = {'return_self-play_evaluation': {'mean':None, 'std': None, 'n': None}, 
                                'return_cross-play_evaluation': {'mean':None, 'std': None, 'n': None},
                                'model_name': None}
        
        for metric in aggregated_results.keys():
            values = []
            for r in results:
                values += [v for v in r.tape[metric] if v is not None]
            
            if metric == 'model_name':
                values = set(values)
                assert len(values) == 1
                aggregated_results[metric] = values.pop()
            
            else:
                values = np.array(values).flatten()
                aggregated_results[metric]['mean'] = np.mean(values)
                aggregated_results[metric]['std'] = np.std(values)
                aggregated_results[metric]['n'] = len(values)
        
        warnings.warn(f"Experiment dir : {str(experiment_dir)}: {aggregated_results}")
        
        with open(str(experiment_dir / 'evaluation_aggregated_results.pkl'), 'wb') as fh:
            pickle.dump(aggregated_results, fh)
        
        with open(str(experiment_dir / 'evaluation_summary.txt'), 'w') as fh:
            print(aggregated_results, file=fh)

    def _load_and_evaluate_seed_dir(self, args):
        with torch.no_grad():
            seed_dir = args.seed_dir
            experiment_dir = args.seed_dir.parent

            warnings.warn(f'Evaluating {seed_dir}')
            recorders_dir = seed_dir / "recorders"

            if (recorders_dir / 'eval_recorder.pkl').exists() and args.skip_if_pkl_exists:
                eval_recorder = Recorder.init_from_pickle_file(str(recorders_dir / 'eval_recorder.pkl'))
                warnings.warn(f"Eval recorder found and loaded {str(recorders_dir / 'eval_recorder.pkl')}\n")
                return eval_recorder

            config = load_config_from_json(seed_dir / "config.json")

            eval_device, _ = get_computing_devices(args.use_gpu, torch, do_rollouts_on_cpu=False)

            try:
                raise NotImplementedError
                import wandb
            except Exception as e:
                warnings.warn("Error with wandb continuing without it")
                wandb = NOP()

            if (recorders_dir / 'eval_recorder.pkl').exists() and args.extend_old_pkl:
                eval_recorder = Recorder.init_from_pickle_file(str(recorders_dir / 'eval_recorder.pkl'))
            else:
                eval_recorder = Recorder(metrics_to_record=['model_name', 'return_self-play_evaluation', 'return_cross-play_evaluation'])

            if args.model_name is None:
                model_name = config.alg_name + '_model.pt'
            else: 
                model_name = args.model_name

            model_path = seed_dir /  model_name
            
            learner = make_trainer(config.alg_name).get_make_learner(config.alg_name, config.task_name).init_from_save(model_path, device=eval_device)

            config.seed = args.eval_seed
            config.record_frames = True
            config.record_gif = True
            env = get_make_env_module(config.task_name).make_env(**config.__dict__)
            
            
            if not config.alg_name in SINGLE_AGENT_ALGOS:
                # we deal with episode termination ourselves
                if not any([task in config.task_name for task in ['toy', 'reacher']]):
                    env.env.env.env.env.env.env._max_episode_steps = np.inf
            else:
                if 'reacher' in config.task_name:
                    env = EpisodeMonitor(TimeLimit(env, 50))

            set_seeds(args.eval_seed, env)

            #### SELF-PLAY EVAL
            # we multiply by n_cross_eval to be consistent in the number of episodes that are used for self-play anc cross-play evals
            config.max_eval_episode = args.max_eval_episode*args.n_cross_eval
            config.record_gif_every = 1
            config.eval_frequency = 1
            config.training_step = 1
            config.record_gif = args.record_gif
            config.n_ep_in_gif = args.n_ep_in_gif
            config.rollout_device = eval_device

            if args.env_rollout_length_eval=='env_dependent':
                if 'reacher' in config.task_name:
                    r_len = 50
                else:
                    r_len = 1000
                config.env_rollout_length_eval = r_len

            else:
                assert isinstance(args.env_rollout_length_eval, int)
                config.env_rollout_length_eval = args.env_rollout_length_eval

            if config.alg_name in SINGLE_AGENT_ALGOS:
                stats = learner.evaluate(env, config.max_eval_episode)
            else:
                stats, time_out, done = evaluate(learner, env, config, training_step=0, rollout_device=eval_device)

            frames = stats.pop('frames')
            if len(frames) > 0:
                frames = np.stack(frames)
                save_video_to_wandb(wandb, f"self-play_evaluation_{model_name}_{get_timestamp()}", frames)

            to_record = {'return_self-play_evaluation': stats['return'], 
                        'model_name': model_name}
            
            eval_recorder.write_to_tape(to_record)
            eval_recorder.save(recorders_dir / 'eval_recorder.pkl')
            wandb.log(to_record)
            warnings.warn(f"self-play evaluation with model_name {model_name} : {stats['return']}")

            #### CROSS-PLAY EVAL
            if (not config.alg_name in SINGLE_AGENT_ALGOS) and args.cross_play_eval:
                # we create a new agent that we will use to stitch different parts together
                cross_play_agent = make_trainer(config.alg_name).get_make_learner(config.alg_name, config.task_name).init_from_save(model_path, device=eval_device)

                # we get all the other agents from other trainings
                other_seeds = DirectoryTree.get_all_seeds(experiment_dir)
                other_seeds = [s for s in other_seeds if not s == seed_dir]
                other_models = [s / model_name for s in other_seeds]
                other_agents = [make_trainer(config.alg_name).get_make_learner(config.alg_name, config.task_name).init_from_save(model, device=eval_device) for model in other_models]

                eval_aggregator = Aggregator()

                config.max_eval_episode = args.max_eval_episode

                for _ in range(args.n_cross_eval):
                    # we select teammates from the pool of other agents
                    team_idx = np.random.randint(low=0, high=len(other_agents), size=cross_play_agent.n_learner-1)
                    teammates = [other_agents[idx] for idx in team_idx]
                    agents = [learner] + teammates

                    # we get each robot's part from a different model
                    parts_to_agent_idx = np.random.permutation(cross_play_agent.n_learner)

                    stitched_parts = [agents[agent_idx].learners[part_idx] for part_idx, agent_idx in enumerate(parts_to_agent_idx)]

                    cross_play_agent.learners = torch.nn.ModuleList(stitched_parts)

                    stats, time_out, done = evaluate(cross_play_agent, env, config, training_step=0, rollout_device=eval_device)

                    eval_aggregator.update(stats)

                frames = [] 
                for f in eval_aggregator.pop('frames'):
                    frames += f

                returns = eval_aggregator.pop('return')

                if len(frames) > 0:
                    frames = np.stack(frames)
                    save_video_to_wandb(wandb, f"cross-play_evaluation_{model_name}_{get_timestamp()}", frames)
                
                to_record = {'return_cross-play_evaluation': returns,
                            'model_name': model_name}
                eval_recorder.write_to_tape(to_record)
                eval_recorder.save(recorders_dir / 'eval_recorder.pkl')
                wandb.log(to_record)
                warnings.warn(f"cross-play evaluation with model_name {model_name} : {returns}")

            return eval_recorder

if __name__ == "__main__":
    evaluator = Evaluator()
    experiment_dirs_args = get_experiment_dirs_args()
    experiment_dirs = get_experiment_dirs(experiment_dirs_args)
    evaluator.load_and_evaluate(experiment_dirs)

