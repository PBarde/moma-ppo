
from pathlib import Path
import sys

from torch.multiprocessing import set_start_method

try:
     set_start_method('spawn')
except RuntimeError:
    pass


def set_up_import():
    alfred_folder = (Path(__file__).resolve().parents[2] / 'alfred_omarl').resolve()
    alg_folder = (Path(__file__).resolve().parents[2] / 'offline_marl').resolve()
    env_folder = (Path(__file__).resolve().parents[2] / 'dzsc').resolve()
    
    for p in [alg_folder, env_folder, alfred_folder]:
        if p not in sys.path:
            sys.path.append(str(p))


set_up_import()

import argparse
from dzsc.ma_d4rl.constants import MULTI_AGENT_TASKS
from nop import NOP
import os
import logging
import torch
import datetime
import submitit
from pathlib import Path
import traceback
import warnings

import d4rl

from alfred.utils.recorder import Aggregator, Recorder
from alfred.utils.misc import create_management_objects
from alfred.utils.config import parse_bool, parse_log_level, save_config_to_json

from offline_marl.trainer import Training
from offline_marl.utils.misc import save_video_to_wandb, get_timestamp
import offline_marl.utils.ml as ml
from offline_marl.single_agent.iql import IQLLearner, IQLLearnerDiscrete
from offline_marl.multi_agent.iql_independent import IQLIndependentLearner, IQLIndependentLearnerDiscrete
from offline_marl.multi_agent.itd3 import ITD3Learner
from offline_marl.multi_agent.ma_iql import MaIQL, MaIQLDiscrete
from offline_marl.single_agent.constants import SINGLE_AGENT_ALGOS
from offline_marl.multi_agent.constants import MULTI_AGENT_ALGOS, CENTRALIZED_ALGOS

import dzsc.ma_d4rl.utils.constants as sa_d4rl_constants
import dzsc.ma_d4rl.constants as ma_d4rl_constants


SINGLE_AGENT_TASKS = sa_d4rl_constants.SINGLE_AGENT_TASKS
MULTI_AGENT_TASKS = ma_d4rl_constants.MULTI_AGENT_TASKS

from dzsc.make_env_module import get_make_env_module

class AgentTraining(Training):

    @staticmethod
    def get_training_argsparser():
        parser = argparse.ArgumentParser()

        # Alfred args
        parser.add_argument('--desc', type=str, default='', help="description of the run")
        parser.add_argument('--alg_name', type=str, default='ma-iql', help="name of the algo, currently only used by alfred")
        parser.add_argument('--task_name', type=str, default='reacher-expert-mix-v0_2x1first_0_DEV', help='d4rl task name')
        parser.add_argument('--seed', type=int, default=132231, help='random seed that seeds everything')

        # Algorithm args (single agent and multi-agent)
        parser.add_argument('--batch_size', type=int, default=256, help="mini-batch size for all models and updates")
        parser.add_argument('--discount_factor', type=float, default=0.99, help="TD learning discount factor")
        parser.add_argument('--expectile', type=float, default=0.7, help="IQL V learning expectile (tau)")
        parser.add_argument('--awr_temperature', type=float, default=3.0, help="temperature in AWR policy learning (beta)")
        parser.add_argument('--target_update_coef', type=float, default=0.005, help="soft updates coefficient for target networks")
        parser.add_argument('--hidden_size', type=int, default=256, help="number of hidden units in fully connected layers")
        parser.add_argument('--lr_q', type=float, default=3e-4, help="learning rate for Q network updates")
        parser.add_argument('--lr_v', type=float, default=3e-4, help="learning rate for V network updates")
        parser.add_argument('--lr_pi', type=float, default=3e-4, help="learning rate for policy network updates")
        parser.add_argument('--iql_lr', type=float, default=3e-4, help="if not None, overwrites lr_q, lr_pi and lr_v")
        parser.add_argument('--grad_clip_norm', type=float, default=1., help="the norm of the gradient clipping for all the losses")

        #### TD3 args
        parser.add_argument('--td3_policy_freq', type=int, default=2, help="number of steps between policy updates")
        parser.add_argument('--td3_policy_noise', type=float, default=0.2, help="std of gaussian noise added to policy")
        parser.add_argument('--td3_noise_clip', type=float, default=0.5, help="clip of gaussian noise added to target policy")
        ## TD3 + BC
        parser.add_argument('--td3_bc_coeff', type=float, default=2.5, help="coefficient for the behavior cloning loss")
        ## TD3 + OMAR
        parser.add_argument('--td3_omar_coeff', type=float, default=0.9, help="coefficient for the OMAR loss")
        # CQL of OMAR
        parser.add_argument('--td3_omar_cql_alpha', type=float, default=1.0, help="")
        parser.add_argument('--td3_omar_cql_lse_temp', type=float, default=1., help="")
        parser.add_argument('--td3_omar_cql_num_sampled_actions', type=int, default=10, help="")
        parser.add_argument('--td3_omar_cql_sample_noise_level', type=float, default=0.2, help="")
        # OMAR
        parser.add_argument('--td3_omar_iters', default=2, type=int)
        parser.add_argument('--td3_init_omar_mu', default=0., type=float)
        parser.add_argument('--td3_init_omar_sigma', default=2.0, type=float)
        parser.add_argument('--td3_omar_num_samples', default=20, type=int)
        parser.add_argument('--td3_omar_num_elites', default=5, type=int)

        parser.add_argument('--td3_shortcut_omar', default=False, type=parse_bool)

        # Memory encoder
        parser.add_argument('--memory_len', type=int, default=10, help="number of frames to consider for building a memory, if 0 no encoder is used")
        parser.add_argument('--lr_memory', type=float, default=3e-4, help="learning rate for memory network updates")
        parser.add_argument('--memory_backprop_actor', type=parse_bool, default=True, help="PPO always to true, whether or not to backprop the actor loss into the memory encoder")
        parser.add_argument('--memory_out_size', type=int, default=128, help="embedding size outputed by memory encoder")
        parser.add_argument('--memory_op', type=str, default='self-attention', choices=['mean', 'product', 'self-attention'], help="reduce operation for memory encoder")

        # Multi-agent args
        parser.add_argument('--centralized_training', type=parse_bool, default=True, help="if each agent's batch correspond to same tranisition when sampling dataset")
        parser.add_argument('--train_in_parallel', type=parse_bool, default=False, help="if we use multiprocessing to train independent learners")
        parser.add_argument('--independent_awr', type=parse_bool, default=False, help="Compute independent loss for each agent in AWR")
        parser.add_argument('--action_aware_mixer', type=parse_bool, default=True, help="mixer for Q has access to joint actions")
        # Monitoring args
        parser.add_argument('--max_training_step', type=int, default=int(1e6 + 1), help="number of updates during training") # int(1e6 + 1)
        parser.add_argument('--log_frequency', type=int, default=50, help='number of learning steps before writing training stats (losses) to disk') #20000
        parser.add_argument('--eval_frequency', type=int, default=50, help='number of training steps between evaluation') #20000
        parser.add_argument('--max_eval_episode', type=int, default=5, help='number of episodes used for evaluation') #10
        parser.add_argument('--save_incremental_model', type=parse_bool, default=True, help="does not delete models")
        parser.add_argument('--incremental_save_frequency', type=int, default=50, help="frequency to save models incrementally, must be a multiple of eval_frequency")
        parser.add_argument('--record_gif', type=parse_bool, default=False, help="records gifs to wandb")
        parser.add_argument('--record_gif_every', type=int, default=10, help='records a gif only every x evals')
        parser.add_argument('--n_ep_in_gif', type=int, default=2)
        parser.add_argument('--use_gpu', type=parse_bool, default=True, help="flag to use GPU for backprop if available")
        parser.add_argument('--use_wandb', type=parse_bool, default=True, help="flag to record to wandb")
        parser.add_argument('--sync_wandb', type=parse_bool, default=True, help="flag to sync to wandb server")
        parser.add_argument('--log_level', type=parse_log_level, default=logging.INFO)
        parser.add_argument('--root_dir', type=str, default=None, help="to overwrite default storage directory")

        return parser

    def sanity_check_config(self, config, logger):
        old_dict = config.__dict__.copy()

        if 'ppo' in config.alg_name:
            if not config.memory_backprop_actor:
                raise NotImplementedError("ppo actor always backprops")
        if not config.iql_lr is None:
            logger.warn('overwriting learning-rates with iql_lr')
            config.lr_q = config.iql_lr
            config.lr_v = config.iql_lr
            config.lr_pi = config.iql_lr
            
        if config.task_name in SINGLE_AGENT_TASKS:
            assert config.alg_name in SINGLE_AGENT_ALGOS, f"task {config.task_name} is single agent and not compatible with algo {config.alg_name}"
        elif config.task_name in MULTI_AGENT_TASKS:
            assert config.alg_name in MULTI_AGENT_ALGOS, f"task {config.task_name} is multi agent and not compatible with algo {config.alg_name}"
        else :
            raise NotImplementedError(f"unknown task_name {config.task_name}")

        if config.train_in_parallel:
            if not config.alg_name in MULTI_AGENT_ALGOS:
                
                logger.warn('train_in_parallel is only for multi-agent algos, setting it to False')
                config.train_in_parallel = False

        if config.alg_name in CENTRALIZED_ALGOS:
            if not config.centralized_training:
                logger.warn(f'centralized algo {config.alg_name} requires centralized_training, setting it to True')
                config.centralized_training = True

            if config.train_in_parallel:
                logger.warn(f'centralized algo {config.alg_name} cannot train in parallel, setting it to False')
                config.train_in_parallel = False

        if 'toy' in config.task_name:
            if not config.memory_len == 0:
                logger.warn('setting memory_len to 0 because we are training on toy environment')
                config.memory_len = 0 

        # if we modified the config we redo a sanity check
        if old_dict != config.__dict__:
            config = self.sanity_check_config(config, logger)

        return config

    @staticmethod
    def get_make_learner(alg_name, task_name):
        assert (alg_name in SINGLE_AGENT_ALGOS) or (alg_name in MULTI_AGENT_ALGOS), f"unknown alg name {alg_name}, known are {SINGLE_AGENT_ALGOS + MULTI_AGENT_ALGOS}"

        if ('hanabi' in task_name) or ('toy' in task_name):
            if alg_name == 'iql':
                assert 'toy' in task_name
                # for now we do not support single-agent learning in hanabi
                return IQLLearnerDiscrete
            elif alg_name in ['iql-independent', 'bc']:
                return IQLIndependentLearnerDiscrete
            elif alg_name == 'ma-iql':
                return MaIQLDiscrete
            else:
                raise NotImplementedError(f'unknown learner class for alg {alg_name}')
        else:
            if alg_name == 'iql':
                return IQLLearner
            elif alg_name in ['iql-independent', 'bc']:
                return IQLIndependentLearner
            elif alg_name == 'ma-iql':
                return MaIQL
            elif alg_name == 'itd3':
                return ITD3Learner
            else:
                raise NotImplementedError(f'unknown learner class for alg {alg_name}')

    def train(self, config, learner=None, dir_tree=None, pbar="default_pbar", logger=None):

        self.config = config
        # creates management objects and saves config
        self.dir_tree, logger, pbar = create_management_objects(dir_tree=dir_tree, logger=logger, pbar=pbar, config=self.config, 
        # if we are restarting we need to increase the training_step to when it was stopped
        pbar_total=(self.config.__dict__.get('training_step', 0), self.config.max_training_step))

        self.config = self.sanity_check_config(self.config, logger)
        self.config.experiment_name = str(self.dir_tree.experiment_dir)
        os.makedirs(self.dir_tree.recorders_dir, exist_ok=True)

        # gets training and rollouts devices
        train_device, rollout_device = ml.get_computing_devices(self.config.use_gpu, torch, do_rollouts_on_cpu=True, logger=logger)

        # registers run to wandb
        if not self.config.sync_wandb:
            os.environ['WANDB_MODE'] = 'dryrun'

        if self.config.use_wandb:
            try:
                import wandb
                os.environ["WANDB_DIR"] = str(self.dir_tree.seed_dir.absolute())
                dt = get_timestamp()
                # we do this for requeeded jobs
                if not 'wandb_run_id' in self.config.__dict__:
                    run_id = self.dir_tree.get_run_name() + '_' + dt
                    if len(run_id) > 128:
                        warnings.warn(f'run_id is too long for wandb init, wandb will fail, reduce name length')
                    self.config.__dict__['wandb_run_id'] = run_id

                wandb.init(id=self.config.wandb_run_id, project='mila_omarl', reinit=True, resume="allow", entity='paul-b-barde', config=self.config)

            except Exception as e:
                logger.error(f'Some wandb error: {e}')
                wandb = NOP()
        else:
            wandb = NOP()
        
        print('before make env module')
        make_env_and_dataset = get_make_env_module(self.config.task_name).make_env_and_dataset
        print('after make env module')
        make_learner = self.get_make_learner(self.config.alg_name, self.config.task_name)

        # makes env and wraps it, normalizes dataset scores, removes timeouts resets transitions
        env, dataset = make_env_and_dataset(make_worldmodel_dataset=False,
                                            **self.config.__dict__)

        ml.set_seeds(self.config.seed, env)

        # we have to deal with the episode termination for reacher (we handle it differently between ppo and iql)
        if 'reacher' in config.task_name:
            from gym.wrappers.time_limit import TimeLimit
            from dzsc.ma_d4rl.utils.env_wrapper import EpisodeMonitor

            env = EpisodeMonitor(TimeLimit(env, 50))

        # makes iql learner (if not preempted and requeede) and saves it
        if learner is None:
            learner = make_learner(**dict(vars(self.config), **ml.get_env_dims(env)), train_device=train_device)
            ml.save([learner], self.dir_tree.seed_dir, suffix="model_init.pt")

        self.to_save = [learner]


        if (self.dir_tree.recorders_dir / 'train_recorder.pkl').exists():
            train_recorder = Recorder.init_from_pickle_file(str(self.dir_tree.recorders_dir / 'train_recorder.pkl'))
            logger.info(f'train_recorder: {train_recorder}')
        else:
            train_recorder = Recorder(metrics_to_record=learner.metrics_to_record)

        train_aggregator = Aggregator()
        ml.save(self.to_save, self.dir_tree.seed_dir, suffix="model.pt")
        
        save_config_to_json(self.config, str(self.dir_tree.seed_dir / 'config.json'))

        # training loop
        for training_step in pbar:

            losses = learner.update(dataset, self.config.batch_size, train_device, train_in_parallel=self.config.train_in_parallel)
            
            train_aggregator.update(losses)

            # logging step
            if ((training_step + 1) % self.config.log_frequency == 0) or training_step == 0: 
                if self.config.train_in_parallel:
                    to_record = dict(learner.get_stats(), training_step=training_step)
                else:
                    to_record = dict(train_aggregator.pop_all_means(), training_step = training_step)

                train_recorder.write_to_tape(to_record)
                train_recorder.save(self.dir_tree.recorders_dir / 'train_recorder.pkl')
                wandb.log(to_record)
            
            # evaluation step
            if ((training_step + 1) % self.config.eval_frequency == 0) or training_step == 0:
                if self.config.max_eval_episode > 0:
                    
                    if hasattr(env, 'record_frames'):
                        if self.config.record_gif and ((training_step + 1) % (self.config.record_gif_every * self.config.eval_frequency)) == 0:
                            env.record_frames = True
                            recording_frames = True
                        else:
                            env.record_frames = False
                            recording_frames = False
                    else:
                        recording_frames = False

                    stats = learner.evaluate(env, self.config.max_eval_episode, rollout_device)
                    
                    # we deal with frames in a different way than other "stats"
                    frames = stats.pop('frames', None)
                    if recording_frames:
                        save_video_to_wandb(wandb, f"evaluation_train_step_{training_step}", frames)
                    
                    to_record = dict(stats, training_step = training_step)
                    train_recorder.write_to_tape(to_record)
                    train_recorder.save(self.dir_tree.recorders_dir / 'train_recorder.pkl')
                    wandb.log(to_record)
                
                    # save current model 
                    # IMPORTANT to save after evaluation (and not before) for multiprocessing because it is the evaluation function that gathers the workers models
                    # and loads them into the main process (this one)
                    ml.remove(self.to_save, self.dir_tree.seed_dir, suffix="model.pt")
                    ml.save(self.to_save, self.dir_tree.seed_dir, suffix="model.pt")

                    if self.config.save_incremental_model:
                        if ((training_step + 1) % self.config.incremental_save_frequency == 0) or training_step == 0:
                            ml.save(self.to_save, self.dir_tree.seed_dir / "incrementals", suffix=f'{training_step+1}_model.pt')

                    # we update training_step here because we actually updated the main thread models
                    self.training_step = training_step

                    # we update training step in config and save the config
                    self.config.training_step = self.training_step
                    save_config_to_json(self.config, str(self.dir_tree.seed_dir / 'config.json'))
            
        if self.config.train_in_parallel:
            learner.terminate_processes()

        open(str(self.dir_tree.seed_dir / 'COMPLETED'), 'w+').close()

if __name__ == '__main__':

    trainer = AgentTraining()

    config = AgentTraining.get_training_argsparser().parse_args()

    trainer.train(config)