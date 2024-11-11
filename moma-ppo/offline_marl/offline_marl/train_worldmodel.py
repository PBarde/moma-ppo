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
from pathlib import Path
import warnings

from alfred.utils.recorder import Aggregator, Recorder, TrainingIterator
from alfred.utils.misc import create_management_objects, keep_two_signif_digits
from alfred.utils.config import parse_bool, parse_log_level, save_config_to_json, load_config_from_json
from alfred.utils.directory_tree import DirectoryTree

from offline_marl.utils.misc import save_video_to_wandb, get_timestamp
import offline_marl.utils.ml as ml
from offline_marl.trainer import Training
from offline_marl.multi_agent.worldmodel_ensemble import WorldModelEnsemble

import dzsc.ma_d4rl.utils.constants as sa_d4rl_constants
import dzsc.ma_d4rl.constants as ma_d4rl_constants


SINGLE_AGENT_TASKS = sa_d4rl_constants.SINGLE_AGENT_TASKS
MULTI_AGENT_TASKS = ma_d4rl_constants.MULTI_AGENT_TASKS

from dzsc.make_env_module import get_make_env_module

class WorldTraining(Training):

    @staticmethod
    def get_training_argsparser():
        parser = argparse.ArgumentParser()

        # Alfred args
        parser.add_argument('--alg_name', type=str, default='world-model', help="name of the algo")
        parser.add_argument('--desc', type=str, default='', help="description of the run")
        parser.add_argument('--task_name', type=str, default='reacher-expert-mix-v0_2x1first_0_DEV', help='d4rl task name with agent decomposition')
        parser.add_argument('--seed', type=int, default=131, help='')

        # World Models args (multi-agent)
        parser.add_argument('--batch_size', type=int, default=256, help="mini-batch size for all models and updates")
        parser.add_argument('--lr', type=float, default=3e-5, help="learning rate")
        parser.add_argument('--hidden_size', type=int, default=1024, help="number of hidden units in fully connected layers")
        parser.add_argument('--n_wm_trained', type=int, default=7, help="number of models in ensemble")
        parser.add_argument('--n_wm_used', type=int, default=5, help="number of models that are kept based on their validation accuracy")
        parser.add_argument('--weight_decay', type=float, default=0., help="value to clip gradients from context and reconstruction losses" )
        parser.add_argument('--grad_clip_norm', type=float, default=1., help="how to compute distances and losses")
        parser.add_argument('--k_mixture_gaussian', type=int, default=1)
        parser.add_argument('--loss_weights', type=eval, default="{'T':1., 'R':1., 'masks':1., 'legal_moves':1.}")
        parser.add_argument('--deterministic_wm', type=parse_bool, default=False, help="whether to use guassian or MSE model")
        parser.add_argument('--binary_reward_sigmoid_state', type=parse_bool, default=False, help="whether to use binary reward and sigmoid on state output")
        parser.add_argument('--spectral_norm', type=parse_bool, default=False, help="whether or not to use spectral norm for gaussian wm")
        parser.add_argument('--four_layers_wm', type=parse_bool, default=True)

        # VAE world-model (below options apply only if vae_wm is true)
        parser.add_argument('--vae_wm', type=parse_bool, default=False, help="whether or not to use a vae as world-model, if true overwrite other wm selection options")
        parser.add_argument('--latent_size', type=int, default=256, help="size of the latent variable")
        parser.add_argument('--state_sigmoid_output', type=parse_bool, default=True, help="whether or not to use a sigmoid activation function on the state decoder output")
        parser.add_argument('--beta_vae', type=float, default=1., help="normalized weight for the kl term in the vae-loss")
        parser.add_argument('--vae_loss', type=str, default='l2', choices=['l1', 'l2'], help="reconstruction loss for reward and state")

        # Memory encoder (each world model in ensemble has its own memory encoder)
        parser.add_argument('--memory_len', type=int, default=0, help="number of frames to consider for building a memory, if 0 no encoder is used")
        parser.add_argument('--lr_memory', type=float, default=3e-4, help="learning rate for memory network updates")
        parser.add_argument('--memory_out_size', type=int, default=128, help="embedding size outputed by memory encoder")
        parser.add_argument('--memory_op', type=str, default='self-attention', choices=['mean', 'product', 'self-attention'], help="reduce operation for memory encoder")

        # Multi-processing args
        parser.add_argument('--train_in_parallel', type=parse_bool, default=False, help="if we use multiprocessing to train independent learners")

        # Monitoring args
        parser.add_argument('--max_training_step', type=int, default=int(1e4 + 1), help="number of updates during training") # int(1e6 + 1)
        parser.add_argument('--log_frequency', type=int, default=1, help='number of learning steps before writing training stats (losses) to disk') #20000
        parser.add_argument('--eval_frequency', type=int, default=5, help='number of training steps between evaluation') #20000
        parser.add_argument('--save_incremental_model', type=parse_bool, default=True, help="does not delete models")
        parser.add_argument('--incremental_save_frequency', type=int, default=int(10), help="frequency to save models incrementally, must be a multiple of eval_frequency")
        parser.add_argument('--max_eval_episode', type=int, default=50, help='number of episodes used for evaluation') #10
        parser.add_argument('--use_gpu', type=parse_bool, default=True, help="flag to use GPU for backprop if available")
        parser.add_argument('--use_wandb', type=parse_bool, default=True, help="flag to record to wandb")
        parser.add_argument('--sync_wandb', type=parse_bool, default=True, help="flag to sync to wandb server")
        parser.add_argument('--log_level', type=parse_log_level, default=logging.INFO)
        parser.add_argument('--root_dir', type=str, default=None, help="to overwrite default storage directory")

        return parser


    @staticmethod
    def sanity_check_config(config, logger):
        old_dict = config.__dict__.copy()
        # if we modified the config we redo a sanity check

        if config.spectral_norm:
            assert not config.deterministic_wm
            assert config.k_mixture_gaussian == 1

        if hasattr(config, 'centralized_training'):
            assert config.centralized_training

        if config.save_incremental_model:
            assert config.incremental_save_frequency % config.eval_frequency == 0

        if config.train_in_parallel:
            raise NotImplementedError("Not working at the moment, memory leak or something")

        if config.vae_wm:
            config.binary_reward_sigmoid_state = False
            config.deterministic_wm = False

        if config.binary_reward_sigmoid_state:
            config.deterministic_wm = True

        if old_dict != config.__dict__:
            config = WorldTraining.sanity_check_config(config, logger)

        return config

    @staticmethod
    def get_make_env_module(task_name):
        raise NotImplementedError(f"unknown task_name {task_name}")

    @staticmethod
    def get_make_learner(alg_name, task_name):
        assert alg_name == 'world-model'
        return WorldModelEnsemble

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
        train_device, rollout_device = ml.get_computing_devices(self.config.use_gpu, torch, do_rollouts_on_cpu=False, logger=logger)

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
        
        make_env_and_dataset = get_make_env_module(self.config.task_name).make_env_and_dataset
        make_learner = self.get_make_learner(self.config.alg_name, self.config.task_name)

        # makes env and wraps it, normalizes dataset scores, removes timeouts resets transitions
        env, dataset, dataset_valid = make_env_and_dataset(**dict(make_worldmodel_dataset=True, centralized_training=True, **self.config.__dict__))

        ml.set_seeds(self.config.seed, env)

        # makes learner (if not preempted and requeede) and saves it
        if learner is None:
            learner = make_learner(**dict(vars(self.config), **ml.get_env_dims(env)), train_device=train_device)
            ml.save([learner], self.dir_tree.seed_dir, suffix="model_init.pt")

        self.learner = learner
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
            if ((training_step + 1) % self.config.log_frequency == 0) or (training_step == 0):
                if self.config.train_in_parallel:
                    to_record = dict(learner.get_stats(), training_step=training_step)
                else:
                    to_record = dict(train_aggregator.pop_all_means(), training_step = training_step)

                train_recorder.write_to_tape(to_record)
                train_recorder.save(self.dir_tree.recorders_dir / 'train_recorder.pkl')
                wandb.log(to_record)
            
            # evaluation step
            if ((training_step + 1) % self.config.eval_frequency == 0) or (training_step == 0):
                if self.config.max_eval_episode > 0:
                    stats = learner.evaluate(dataset, dataset_valid, self.config.max_eval_episode, self.config.batch_size)
                    
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
                        if ((training_step + 1) % self.config.incremental_save_frequency == 0) or (training_step == 0):
                            ml.save(self.to_save, self.dir_tree.seed_dir / "incrementals", suffix=f'{training_step}_model.pt')

                    # we update training_step here because we actually updated the main thread models
                    self.training_step = training_step

                    # we update training step in config and save the config
                    self.config.training_step = self.training_step
                    save_config_to_json(self.config, str(self.dir_tree.seed_dir / 'config.json'))
            
        if self.config.train_in_parallel:
            learner.terminate_processes()

        open(str(self.dir_tree.seed_dir / 'COMPLETED'), 'w+').close()

if __name__ == '__main__':

    trainer = WorldTraining()

    config = WorldTraining.get_training_argsparser().parse_args()

    trainer.train(config)