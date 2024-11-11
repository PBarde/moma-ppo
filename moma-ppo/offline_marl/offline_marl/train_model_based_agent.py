
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
import numpy as np

import d4rl

from alfred.utils.recorder import Aggregator, Recorder, TrainingIterator
from alfred.utils.misc import create_management_objects, Bunch
from alfred.utils.config import parse_bool, parse_log_level, save_config_to_json, load_config_from_json
from alfred.utils.directory_tree import DirectoryTree

from offline_marl.train_agents import AgentTraining
from offline_marl.utils.misc import save_video_to_wandb, get_timestamp
import offline_marl.utils.ml as ml
from offline_marl.single_agent.iql import IQLLearner, IQLLearnerDiscrete
from offline_marl.multi_agent.iql_independent import IQLIndependentLearner, IQLIndependentLearnerDiscrete
from offline_marl.multi_agent.ma_iql import MaIQL, MaIQLDiscrete
from offline_marl.single_agent.zeus_iql import ZeusIQLLearner
from offline_marl.multi_agent.zeus_iql_independent import ZeusIQLIndependentLearner, ZeusIQLIndependentLearnerDiscrete
from offline_marl.single_agent.constants import SINGLE_AGENT_ALGOS, SINGLE_ZEUS_ALGOS
from offline_marl.multi_agent.constants import MULTI_AGENT_ALGOS, CENTRALIZED_ALGOS, MULTI_ZEUS_ALGOS
from offline_marl import load_worldmodel

from offline_marl.utils.ml import batch_as_tensor, batch_to_device, check_if_nan_in_batch

import dzsc.ma_d4rl.utils.constants as sa_d4rl_constants
import dzsc.ma_d4rl.constants as ma_d4rl_constants

import dzsc.hanabi.constants as hanabi_constants

import dzsc.toy_experiment.constants as toy_constants

SINGLE_AGENT_TASKS = sa_d4rl_constants.SINGLE_AGENT_TASKS + toy_constants.SINGLE_AGENT_TASKS
MULTI_AGENT_TASKS = ma_d4rl_constants.MULTI_AGENT_TASKS + hanabi_constants.MULTI_AGENT_TASKS + toy_constants.MULTI_AGENT_TASKS

from dzsc.make_env_module import get_make_env_module
import dzsc.hanabi.multiagent_env_and_dataset as replay_buffer_module

class ModelBasedAgentTraining(AgentTraining):

    @staticmethod
    def get_training_argsparser():
        parser = argparse.ArgumentParser()

        # Alfred args
        parser.add_argument('--desc', type=str, default='', help="description of the run")
        parser.add_argument('--alg_name', type=str, default='mb-ma-iql', help="name of the algo, currently only used by alfred")
        parser.add_argument('--task_name', type=str, default='reacher-expert-mix-v0_2x1first_0_DEV', help='d4rl task name with agent decomposition')
        parser.add_argument('--seed', type=int, default=131, help='random seed that seeds everything')

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

        # Memory encoder
        parser.add_argument('--memory_len', type=int, default=10, help="number of frames to consider for building a memory, if 0 no encoder is used")
        parser.add_argument('--lr_memory', type=float, default=3e-4, help="learning rate for memory network updates")
        parser.add_argument('--memory_backprop_actor', type=parse_bool, default=True, help="whether or not to backprop the actor loss into the memory encoder")
        parser.add_argument('--memory_out_size', type=int, default=128, help="embedding size outputed by memory encoder")
        parser.add_argument('--memory_op', type=str, default='self-attention', choices=['mean', 'product', 'self-attention'], help="reduce operation for memory encoder")

        # Multi-agent args
        parser.add_argument('--centralized_training', type=parse_bool, default=True, help="if each agent's batch correspond to same tranisition when sampling dataset")
        parser.add_argument('--train_in_parallel', type=parse_bool, default=False, help="if we use multiprocessing to train independent learners")
        parser.add_argument('--independent_awr', type=parse_bool, default=False, help="Compute independent loss for each agent in AWR")
        parser.add_argument('--action_aware_mixer', type=parse_bool, default=True, help="mixer for Q has access to joint actions")
        
        # Model-Based args
        parser.add_argument("--wm_overwrite_task_name", type=str, default=None)
        parser.add_argument("--model_name", type=str, default="world-model_model.pt")
        parser.add_argument("--replay_buffer_size", type=int, default=int(1e3))
        parser.add_argument("--replay_buffer_min_fill", type=int, default=int(1e3), help="wait to sample in buffer until")
        parser.add_argument("--populate_frequency", type=int, default=10, help='adds synthetic data to replay buffer every populate_frequency training_steps')
        parser.add_argument("--batch_mixture_proportion", type=float, default=.05, help="proportion of real data, batch = proportion x real + (1-proportion) x synthetic")
        parser.add_argument("--k_steps_rollouts", type=int, default=10, help="number of steps world model rollouts")
        parser.add_argument("--n_wait_before_mb", type=int, default=0)
        parser.add_argument("--mb_finetuning", type=parse_bool, default=True)
        parser.add_argument("--mb_batch_size", type=int, default=256, help="number of states sampled in the dataset to start mb-rollouts")
        parser.add_argument("--perfect_wm", type=parse_bool, default=True, help="Uses hanabi simulator as worldmodel")
        parser.add_argument('--n_wm_used', type=int, default=7, help="number of models that are kept based on their validation accuracy")
        parser.add_argument('--wm_use_min_rewards', type=parse_bool, default=True, help="Uses min reward accros ensemble instead of randomly sampled")
        parser.add_argument('--wm_use_all_min', type=parse_bool, default=False, help="Use model that is most pessimistic wrt to the reward instead of randomly sampling it")
        parser.add_argument('--use_reward_cov', type=parse_bool, default=True, help="whether to use cov on reward instead than on full reward and nextstate")
        parser.add_argument("--reward_uncertainty_penalty", type=float, default=1., help="constant for MOPO penalty, usually between 0.5 and 5")
        parser.add_argument("--ensemble_cov_norm_threshold", type=float, default=10, help="if above this we cut the model rollout")
        
        # Monitoring args
        parser.add_argument('--max_training_step', type=int, default=int(1e6 + 1), help="number of updates during training") # int(1e6 + 1)
        parser.add_argument('--log_frequency', type=int, default=5, help='number of learning steps before writing training stats (losses) to disk') #20000
        parser.add_argument('--eval_frequency', type=int, default=5, help='number of training steps between evaluation') #20000
        parser.add_argument('--max_eval_episode', type=int, default=1, help='number of episodes used for evaluation') #10
        parser.add_argument('--record_gif', type=parse_bool, default=False, help="records gifs to wandb")
        parser.add_argument('--record_gif_every', type=int, default=1, help='records a gif only every x evals')
        parser.add_argument('--n_ep_in_gif', type=int, default=2)
        parser.add_argument('--use_gpu', type=parse_bool, default=True, help="flag to use GPU for backprop if available")
        parser.add_argument('--use_wandb', type=parse_bool, default=True, help="flag to record to wandb")
        parser.add_argument('--sync_wandb', type=parse_bool, default=True, help="flag to sync to wandb server")
        parser.add_argument('--log_level', type=parse_log_level, default=logging.INFO)
        parser.add_argument('--root_dir', type=str, default=None, help="to overwrite default storage directory")

        return parser
    
    @staticmethod
    def get_make_learner(alg_name, task_name):
        no_mb_alg_name = alg_name.replace('mb-','')
        return AgentTraining.get_make_learner(no_mb_alg_name, task_name)


    def __call__(self, checkpointpath: str):
        # checkpointpath should be a seed_dir in terms of alfred
        seed_dir = Path(checkpointpath)
        config = load_config_from_json(str(seed_dir / 'config.json'))
        dir_tree = DirectoryTree.init_from_seed_path(seed_dir, root=str(seed_dir.parents[2]))

        try:
            # we start back from a preemption
            if (seed_dir / 'PREEMPTED').exists():

                os.remove(str(seed_dir / 'PREEMPTED'))

                # we can load the model on the cpu and it will be moved on gpu by the .train() afterwards
                # actually this does not work with optimizers better to load directly on train_device
                train_device, rollout_device = ml.get_computing_devices(config.use_gpu, torch, do_rollouts_on_cpu=True)
                warnings.warn(f'train_device: {train_device}')

                preemted_model_path = seed_dir / (config.alg_name + '_model_preempt.pt')
                regular_model_path = seed_dir / (config.alg_name + '_model.pt')

                 ## ADDED this to reload replay-buffer
                preemted_replay_path = seed_dir / ('replay-buffer' + '_model_preempt.pt')
                regular_replay_path = seed_dir / ('replay-buffer' + '_model.pt')

                # if we didn't manage to save a model at checkpoint we take the regularly saved model
                if preemted_model_path.exists() and preemted_replay_path.exists():
                    model_path = preemted_model_path
                    replay_path = preemted_replay_path
                else:
                    model_path = regular_model_path
                    replay_path = regular_replay_path

                learner = self.get_make_learner(config.alg_name, config.task_name).init_from_save(model_path, device=train_device)

                # we delete the preemt model for next preemption
                if preemted_model_path.exists():
                    ml.remove([learner], seed_dir, suffix="model_preempt.pt")

                replay_buffer = replay_buffer_module.init_replay_from_save(replay_path, device=torch.device('cpu'))

                # we delete the preemt model for next preemption
                if preemted_replay_path.exists():
                    ml.remove([replay_buffer], seed_dir, suffix="model_preempt.pt")

                return self.train(config, learner=learner, replay_buffer=replay_buffer, dir_tree=dir_tree)
            
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

    def sanity_check_config(self, config, logger):
        if config.perfect_wm == False:
            assert config.wm_overwrite_task_name is None

        else:
            assert not ('toy' in config.task_name)
        
        if config.wm_use_all_min:
            if not config.wm_use_min_rewards:
                warnings.warn(f'Cannot wm_use_all_min if not wm_use_min_rewards, setting wm_use_all_min to False')
                config.wm_use_all_min = False

        if config.mb_finetuning:
            warnings.warn(f'mb_finetuning is true, overwriting parameters accordingly')
            config.mb_batch_size = 1
            config.populate_frequency = config.k_steps_rollouts

            
            config.n_wait_before_mb = int(1e6)
            
            config.replay_buffer_size = int(1e6)

            # config.n_wait_before_mb = 100
            
            # config.replay_buffer_size = 100
            
            config.replay_buffer_min_fill = 1
            config.batch_mixture_proportion = "finetuning"

        return super().sanity_check_config(config, logger)

    def train(self, config, learner=None, replay_buffer=None, dir_tree=None, pbar="default_pbar", logger=None):

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
        make_replay_buffer = replay_buffer_module.make_replay_buffer

        make_learner = self.get_make_learner(self.config.alg_name, self.config.task_name)

        # makes env and wraps it, normalizes dataset scores, removes timeouts resets transitions
        env, dataset = make_env_and_dataset(logger=logger, 
                                            make_worldmodel_dataset=False,
                                            **self.config.__dict__)


        # we have to deal with the episode termination for reacher (we handle it differently between ppo and iql)
        if 'reacher' in config.task_name:
            from gym.wrappers.time_limit import TimeLimit
            from dzsc.ma_d4rl.utils.env_wrapper import EpisodeMonitor

            env = EpisodeMonitor(TimeLimit(env, 50))

        # loads world-models
        if not self.config.perfect_wm:
            wm_ensemble = load_worldmodel.load_wm_ensemble(self.config, device=train_device)
        else:
            wm_ensemble = None

        ml.set_seeds(self.config.seed, env)

        # makes iql learner (if not preempted and requeede) and saves it
        if learner is None:
            learner = make_learner(**dict(vars(self.config), **ml.get_env_dims(env)), train_device=train_device)
            ml.save([learner], self.dir_tree.seed_dir, suffix="model_init.pt")
        
        self.learner = learner
        
        # makes replay buffer
        if replay_buffer is None:
            replay_buffer = make_replay_buffer(env, self.config.memory_len, self.config.replay_buffer_size, self.config.task_name, torch.device('cpu'))
            ml.save([replay_buffer], self.dir_tree.seed_dir, suffix="model_init.pt")

        self.replay_buffer = replay_buffer
   
        self.to_save = [learner, replay_buffer]

        if (self.dir_tree.recorders_dir / 'train_recorder.pkl').exists():
            train_recorder = Recorder.init_from_pickle_file(str(self.dir_tree.recorders_dir / 'train_recorder.pkl'))
            logger.info(f'train_recorder: {train_recorder}')
        else:
            train_recorder = Recorder(metrics_to_record=learner.metrics_to_record)

        train_aggregator = Aggregator()
        ml.save(self.to_save, self.dir_tree.seed_dir, suffix="model.pt")
        save_config_to_json(self.config, str(self.dir_tree.seed_dir / 'config.json'))

        replay_min_filled_flag = False
        # training loop
        for training_step in pbar:

            ### Model-Based phase
            # we start by populating the buffer
            if ((training_step + 1) % config.populate_frequency == 0) and ((training_step + 1) >= config.n_wait_before_mb):
                if not config.batch_mixture_proportion == 1.:
                    populate_replay_buffer(self.config, learner, dataset, replay_buffer, wm_ensemble, training_step=training_step, device=train_device)
                # just to display some info
                if (not replay_min_filled_flag) and (replay_buffer.filled >= config.replay_buffer_min_fill):
                    print(f'training-step: {training_step}: replay_min_filled_reached')
                    replay_min_filled_flag = True

            # get mixed data (AND reward penalty!)
            batches = get_mixed_data(self.config, dataset, replay_buffer)

            ## we update the learners
            losses = learner.update_from_batch(batches, train_device)
            
            train_aggregator.update(losses)

            # logging step
            if ((training_step + 1) % self.config.log_frequency == 0) or (training_step == 0):
                if self.config.train_in_parallel:
                    to_record = dict(learner.get_stats(), training_step=training_step)
                else:
                    to_record = dict(train_aggregator.pop_all_means(), training_step=training_step)

                train_recorder.write_to_tape(to_record)
                train_recorder.save(self.dir_tree.recorders_dir / 'train_recorder.pkl')
                wandb.log(to_record)
            
            # evaluation step
            if ((training_step + 1) % self.config.eval_frequency == 0) or (training_step == 0):
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

                    # we update training_step here because we actually updated the main thread models
                    self.training_step = training_step

                    # we update training step in config and save the config
                    self.config.training_step = self.training_step
                    save_config_to_json(self.config, str(self.dir_tree.seed_dir / 'config.json'))

            
        if self.config.train_in_parallel:
            learner.terminate_processes()

        open(str(self.dir_tree.seed_dir / 'COMPLETED'), 'w+').close()


def get_mixed_data(config, dataset, replay_buffer):


    if replay_buffer.filled >= config.replay_buffer_min_fill:

        if config.batch_mixture_proportion == "finetuning":
            batch_mixture_proportion = dataset.n_data / (dataset.n_data + replay_buffer.filled)
        
        else:
            batch_mixture_proportion = config.batch_mixture_proportion
        
        sample_proportion = batch_mixture_proportion * config.batch_size

        real_batch_size = int(sample_proportion)+ np.random.binomial(1, sample_proportion-int(sample_proportion))
        synthetic_batch_size = config.batch_size - real_batch_size
    else:
        real_batch_size = config.batch_size
        synthetic_batch_size = 0

    # get real data
    if real_batch_size > 0:
        real_batches = dataset.sample(real_batch_size)
    else:
        real_batches = None

    # get synthetic data
    if synthetic_batch_size > 0:
        synthetic_samples = replay_buffer.sample(synthetic_batch_size)


        for samples in synthetic_samples:
            # modify reward with model uncertainty
            if config.use_reward_cov:
                coeff = samples['rewards_cov']
            else:
                coeff = samples['ensemble_cov_norm']

            samples['rewards'] = samples['rewards'] - config.reward_uncertainty_penalty*(coeff+samples['model_max_cov_norm'])
    
    else:
        synthetic_samples = None

    # mix data
    batches = mix_batches_and_samples(real_batches, synthetic_samples)

    return batches

def mix_batches_and_samples(batches, samples):
    if batches is None:
        return [Bunch(s) for s in samples]
    
    if samples is None:
        return batches

    assert len(samples) == len(batches)

    mixes=[]
    for batch, sample in zip(batches, samples):
        batch = batch_to_device(batch_as_tensor(batch), sample['observations'].device)
        mix = {}
        for key, val in batch.items():
            mix[key] = torch.cat((val, sample[key]), dim=0)
        mixes.append(batch.__class__(**mix))
    
    return mixes


def populate_replay_buffer(config, learner, dataset, replay_buffer, wm_ensemble, device, training_step, n_sub_batches=1):

    with torch.no_grad():
        
        learner.to(device)

        # we cannot do it with full batch_size because then co-variance matrices accross batch are too large
        for n in range(n_sub_batches):

            data_idx = dataset.sample_indexes(config.mb_batch_size//n_sub_batches)

            data = dataset.sample_from_idx(data_idx)

            if config.perfect_wm:
                # we spawn a env for each transition to rollout from and get it to that state
                dataset.spawn_corresponding_envs(data_idx)
            
            for k in range(config.k_steps_rollouts):
                
                # take some new actions from the current policies
                for i, batch, learner_i in zip(range(len(data)), data, learner.learners):

                    batch =  batch_to_device(batch_as_tensor(batch), device)

                    observations = learner_i.concat_embedings_to_obs(batch, target=False)

                    if 'legal_moves' in batch:
                        new_actions = learner_i.policy.act(observations, batch['legal_moves'], sample=True, return_log_pi=False).detach()
                    else:
                        new_actions = learner_i.policy.act(observations, sample=True, return_log_pi=False).detach()
                    
                    # we modify the batch with the new action, in order to do so we have to make it a dict
                    batch['actions'] = new_actions

                    # we update the list of batches (per agent) accordingly
                    data[i] = batch
        
                # generate corresponding synthetic data
                # convert to bunch to access dict key as attributes (we do not have all the batch's attribute so we cannot convert to batch_class)
                
                if config.perfect_wm:
                    synthetic_data = dataset.transition_envs([d['actions'] for d in data])
                    # if the dataset rewards are normalized we have to normalize the reward comming out of the simulator as well
                    # same env kind for all
                    if hasattr(dataset, 'reward_normalization_factor'):
                        for synth_d in synthetic_data:
                            synth_d['rewards'] *= dataset.reward_normalization_factor
                
                else:
                    synthetic_data = wm_ensemble.sample([Bunch(d) for d in data], config, device)

                # difference_in_next_states = (synthetic_data[0]['next_observations'] - data[0]['next_observations'].cpu()).abs().mean()
                # difference_in_reward = (synthetic_data[0]['rewards'] - (data[0]['rewards']/dataset.reward_normalization_factor).cpu()).abs().mean()

                # we update the synthetic transitions with real data 
                for synthetic_d, d in zip(synthetic_data, data):
                    for key in ['observations', 'actions', 'legal_moves', 'history_memories']:
                        if key in d:
                            synthetic_d[key] = d[key]

                    if 'toy' in config.task_name:
                        synthetic_d['legal_moves'] = torch.ones((len(synthetic_d['observations']), 2), device=synthetic_d['observations'].device)

                # adds it to replay buffer
                replay_buffer.extend(synthetic_data)

                if config.k_steps_rollouts > 1:
                    
                    # we have to move to the next step
                    next_batches = []

                    # we will remove transitions that ended up in dones
                    # we compute it once because all agents have same masks
                    not_dones = (synthetic_data[0]['masks'] == 1.).flatten()

                    if config.perfect_wm:
                        dataset.keep_only_not_done_envs(not_dones)

                    if not any(not_dones):
                        return

                    # we step in terms of observations and memory
                    for i, synthetic_d in enumerate(synthetic_data):

                        observations = synthetic_d['observations'][not_dones]
                        actions = synthetic_d['actions'][not_dones]
                        next_observations = synthetic_d['next_observations'][not_dones]

                        agent_next_batch = {'observations': next_observations}
                        
                        if config.memory_len > 0:
                            history_memories = synthetic_d['history_memories'][not_dones]

                            next_history_memories = replay_buffer.next_histories(history_memories, observations, actions)
                            agent_next_batch['history_memories'] = next_history_memories

                        next_batches.append(agent_next_batch)


                    # we get the corresponding legal move
                    if 'hanabi' in config.task_name:
                        if config.perfect_wm:
                            next_legal_moves = dataset.temp_next_legal_moves
                        else:
                            next_legal_moves = wm_ensemble.get_legal_move([Bunch(b) for b in next_batches], device)
                        for agent_next_batch, agent_next_legal_moves in zip(next_batches, next_legal_moves):
                            agent_next_batch['legal_moves'] = agent_next_legal_moves
                    elif 'toy' in config.task_name:
                        for agent_next_batch in next_batches:
                            agent_next_batch['legal_moves'] = torch.ones((len(agent_next_batch['observations']), 2), device=agent_next_batch['observations'].device)

                    # convert to bunch to access dict key as attributes (we do not have all the batch's attribute so we cannot convert to batch_class)
                    data = [Bunch(next_b) for next_b in next_batches]

if __name__ == '__main__':

    trainer = ModelBasedAgentTraining()

    config = ModelBasedAgentTraining.get_training_argsparser().parse_args()

    trainer.train(config)