
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
from nop import NOP
import os
import logging
import torch
import numpy as np
import datetime
import traceback
import warnings
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle
import json

from alfred.utils.recorder import Aggregator, Recorder
from alfred.utils.misc import create_management_objects
from alfred.utils.config import parse_bool, parse_log_level, save_config_to_json, load_config_from_json
from alfred.utils.directory_tree import DirectoryTree

from offline_marl.trainer import Training
from offline_marl.single_agent.ppo import PPOBuffer
from offline_marl.multi_agent.mappo import MAPPO
from offline_marl.utils.misc import save_video_to_wandb, get_timestamp
import offline_marl.utils.ml as ml
from offline_marl.single_agent.actors import epsilon

from dzsc.make_env_module import get_make_env_module

class RLAgentTraining(Training):

    @staticmethod
    def get_training_argsparser():
        parser = argparse.ArgumentParser()

        # Alfred args
        parser.add_argument('--desc', type=str, default='init', help="description of the run")
        parser.add_argument('--alg_name', type=str, default='ma-ppo', help="name of the algo, currently only used by alfred")
        parser.add_argument('--task_name', type=str, default='reacher-dataset-v0_2x1first_0_DEV', help='')
        parser.add_argument('--seed', type=int, default=131, help='random seed that seeds everything')

        # Algorithm args (single agent and multi-agent)
        # common
        parser.add_argument('--batch_size', type=int, default=256, help="mini-batch size for all models and updates")
        parser.add_argument('--discount_factor', type=float, default=0.99, help="TD learning discount factor")
        parser.add_argument('--lr_v', type=float, default=3e-4, help="learning rate for V network updates")
        parser.add_argument('--lr_pi', type=float, default=3e-4, help="learning rate for policy network updates")
        parser.add_argument('--grad_clip_norm', type=float, default=1., help="the norm of the gradient clipping for all the losses")
        parser.add_argument('--state_dependent_std', type=parse_bool, default=True)
        parser.add_argument('--hidden_size', type=int, default=256, help="number of hidden units in fully connected layers")
        parser.add_argument('--env_rollout_length_train', type=int, default=50)
        parser.add_argument('--env_rollout_length_eval', type=int, default=50)
        # parser.add_argument('--env_rollout_length_train', type=int, default=1000)
        # parser.add_argument('--env_rollout_length_eval', type=int, default=1000)
        parser.add_argument('--artificial_rollout_length', type=int, default=0)

        # ppo like
        parser.add_argument('--ppo_epochs_per_update', type=int, default=5)
        parser.add_argument('--ppo_transitions_between_update', type=int, default=2000)
        parser.add_argument('--ppo_update_clip_param', type=float, default=0.2)
        parser.add_argument('--ppo_lamda', type=float, default=0.98)
        parser.add_argument('--ppo_critic_loss_coeff', type=float, default=0.5, help="weight between critic and actor loss, useful because they share the same memory encoder and mixer")
        parser.add_argument('--ppo_entropy_bonus_coeff', type=float, default=0.001, help="entropy bonus coeff in total loss, ignored if 0.")
        parser.add_argument('--ppo_general_lr', type=float, default=5e-5, help="ignored if None, otherwise overwrites lr_pi, lr_v and lr_memory")
        parser.add_argument('--ppo_actor_squashing', type=str, default='none', choices=['none', 'tanh'])
        parser.add_argument('--ppo_entropy_target', type=float, default=-2)
        parser.add_argument('--ppo_action_penalty_coeff', type=float, default=1., help="coefficient to penalized mean squared action in loss")
        parser.add_argument('--ppo_double_V', type=parse_bool, default=False, help="whether or not to use two V networks and take the min between them")

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

        # Monitoring args
        parser.add_argument('--max_training_step', type=int, default=int(1e6 + 1), help="number of updates during training") # int(1e6 + 1)
        parser.add_argument('--log_frequency', type=int, default=2, help='number of learning steps before writing training stats (losses) to disk') #20000
        parser.add_argument('--eval_frequency', type=int, default=5, help='number of training steps between evaluation') #20000
        parser.add_argument('--max_eval_episode', type=int, default=5, help='number of episodes used for evaluation') #10
        parser.add_argument('--save_incremental_model', type=parse_bool, default=False, help="does not delete models")
        parser.add_argument('--incremental_save_frequency', type=int, default=int(20), help="frequency to save models incrementally, must be a multiple of eval_frequency")
        parser.add_argument('--save_training_batches', type=parse_bool, default=False)
        parser.add_argument('--save_training_batches_frequency', type=int, default=10, help="we save training batches every freq training steps (downsample)")
        parser.add_argument('--record_gif', type=parse_bool, default=False, help="records gifs to wandb")
        parser.add_argument('--plot_traj', type=parse_bool, default=False)
        parser.add_argument('--record_gif_every', type=int, default=5, help='records a gif only every x evals')
        parser.add_argument('--n_ep_in_gif', type=int, default=2)
        parser.add_argument('--use_gpu', type=parse_bool, default=True, help="flag to use GPU for backprop if available")
        parser.add_argument('--use_wandb', type=parse_bool, default=True, help="flag to record to wandb")
        parser.add_argument('--sync_wandb', type=parse_bool, default=True, help="flag to sync to wandb server")
        parser.add_argument('--log_level', type=parse_log_level, default=logging.INFO)
        parser.add_argument('--root_dir', type=str, default=None, help="to overwrite default storage directory")

        return parser
    
    @staticmethod
    def get_make_learner(alg_name, task_name):
        if alg_name == 'ma-ppo':
            return MAPPO
        else:
            raise NotImplementedError


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

                # if we didn't manage to save a model at checkpoint we take the regularly saved model
                if preemted_model_path.exists():
                    model_path = preemted_model_path

                else:
                    model_path = regular_model_path

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

    def sanity_check_config(self, config, logger):
        old_dict = config.__dict__.copy()
        if 'ppo' in config.alg_name:
            if not config.memory_backprop_actor:
                raise NotImplementedError("ppo actor always backprops")

        if not config.ppo_general_lr is None:
            warnings.warn("Using ppo_general_lr, overwriting lr_pi, lr_v and lr_memory")
            config.lr_pi = config.ppo_general_lr
            config.lr_v = config.ppo_general_lr
            config.lr_memory = config.ppo_general_lr

            
        if config.artificial_rollout_length > 0: 
            assert config.memory_len == 0, "artificial_rollout_length do not account for memory"

        if config.env_rollout_length_eval != config.env_rollout_length_train:
            if config.memory_len > 0:
                assert "navigation" in config.task_name, f"Using memory with different learning and eval episode lengths, only ok for debuggin in navigation tasks"
                warnings.warn(f"Using memory with different learning and training episode lengths")
                
        if not any([task in config.task_name for task in ['hanabi', 'toy', 'navigation', 'reacher']]):
            # this means we are using a d4rl env and the timeouts should be set to 1000
            if not ((config.env_rollout_length_train == 1000) and (config.env_rollout_length_eval == 1000)):
                raise ValueError("env_rollout_length_train and env_rollout_length_eval should be set to 1000 to match timelimit mujoco envs")

        if 'reacher' in config.task_name: 
            if not ((config.env_rollout_length_train == 50) and (config.env_rollout_length_eval == 50)):
                raise ValueError("env_rollout_length_train and env_rollout_length_eval should be set to 50 to match timelimit mujoco envs for REACHER")

        if config.__dict__.get('save_training_batches', False): 
            assert 'full' in config.task_name, "we only support fully observable recoding for now (we just concat actions and keep first agent's quantities)"

        # if we modified the config we redo a sanity check
        if old_dict != config.__dict__:
            return self.sanity_check_config(config, logger)
        else:
            return config

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
        

        #### TRAINING SCRIPT STARTS 

        make_env = get_make_env_module(self.config.task_name).make_env
        make_learner = self.get_make_learner(self.config.alg_name, self.config.task_name)

        # makes env
        env = make_env(**self.config.__dict__)

        # we deal with episode termination ourselves
        if not any([task in config.task_name for task in ['toy', 'reacher']]):
            env.env.env.env.env.env.env._max_episode_steps = np.inf

        ml.set_seeds(self.config.seed, env)

        # makes iql learner (if not preempted and requeede) and saves it
        if learner is None:
            learner = make_learner(**dict(vars(self.config), **ml.get_env_dims(env)), train_device=train_device)
            ml.save([learner], self.dir_tree.seed_dir, suffix="model_init.pt")
        
        self.to_save = [learner]

        if (self.dir_tree.recorders_dir / 'train_recorder.pkl').exists():
            train_recorder = Recorder.init_from_pickle_file(str(self.dir_tree.recorders_dir / 'train_recorder.pkl'))
            logger.info(f'train_recorder: {train_recorder}')
        else:
            train_recorder = Recorder(metrics_to_record=get_metrics_to_record(learner.metrics_to_record))

        train_aggregator = Aggregator()
        ml.save(self.to_save, self.dir_tree.seed_dir, suffix="model.pt")
        save_config_to_json(self.config, str(self.dir_tree.seed_dir / 'config.json'))


        env_step = config.__dict__.get('env_step', 0)
        done = True
        time_out = True
        observations = None
        for training_step in pbar:

            ### we collect data for ppo
            batches, observations, done, time_out, collect_stats_dict = collect_transition_batches(env, learner, observations, done, time_out, rollout_device, config, training_step)

            # update the number of env steps we did
            env_step += config.ppo_transitions_between_update

            if config.artificial_rollout_length > 0:
                batches = add_artificial_rollout_length(batches, config)

            # we save training batches to build replay datasets
            if config.save_training_batches and (training_step + 1) % self.config.save_training_batches_frequency == 0:
                batch_folder = self.dir_tree.incrementals_dir / "training_batches"
                save_training_batches(batches, batch_folder, training_step)

            ## we update the learners
            train_stats_dict = learner.update_from_batch(batches, train_device)

            collect_frames = collect_stats_dict.pop('collect_frames')
            if len(collect_frames)>0:
                collect_frames = np.stack(collect_frames)
                save_video_to_wandb(wandb, f"collection_train_step_{training_step}", collect_frames)
            
            train_aggregator.update(collect_stats_dict)
            train_aggregator.update(train_stats_dict)

            ## logging step
            if ((training_step + 1) % self.config.log_frequency == 0) or training_step == 0:                
                to_record = dict(train_aggregator.pop_all_means(), training_step=training_step, env_step=env_step)
                train_recorder.write_to_tape(to_record)
                train_recorder.save(self.dir_tree.recorders_dir / 'train_recorder.pkl')
                wandb.log(to_record)
            
            ## evaluation step
            if ((training_step + 1) % self.config.eval_frequency == 0) or training_step == 0:
                if self.config.max_eval_episode > 0:

                    stats, time_out, done = evaluate(learner, env, config, training_step, rollout_device)
                    assert (time_out or done), "we need to reset env before resuming collecting data after eval"

                    traj_list = stats.pop('traj_list')
                    if config.plot_traj:
                        plot_traj_quiver(traj_list, self.dir_tree.recorders_dir, training_step)

                    frames = stats.pop('frames')
                    if len(frames)>0:
                        frames = np.stack(frames)
                        save_video_to_wandb(wandb, f"evaluation_train_step_{training_step}", frames)

                    to_record = dict(stats, training_step=training_step)
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
                    self.env_step = env_step

                    # we update training step in config and save the config
                    self.config.training_step = self.training_step
                    self.config.env_step = self.env_step
                    save_config_to_json(self.config, str(self.dir_tree.seed_dir / 'config.json'))

        open(str(self.dir_tree.seed_dir / 'COMPLETED'), 'w+').close()

def save_training_batches(batches, batch_folder, training_step):
    if not batch_folder.exists():
        os.makedirs(str(batch_folder), exist_ok=True)

    # we take all the quantities from first agent (fully observable) except for the actions

    to_save = process_training_batches_to_save(batches)

    with open(str(batch_folder / f"train_batches_{training_step}.pkl"), "wb") as fh:
        pickle.dump(to_save, fh)

def process_training_batches_to_save(batches):
    to_save = {key: val.copy() for key, val in batches[0].items()}      
    to_save['actions'] = np.concatenate([b['actions'] for b in batches], axis=1)
    return to_save

def get_metrics_to_record(metrics):
    return list(metrics) + ['training_step', 'env_step', 'return', 'ep_length', 'collect_ep_length', 'collect_return', 'collect_mean_entropy']


def reset_agents_history(learner):
    _ = [agent.reset_histories() for agent in learner.learners]


def append_history_agents(learner, observations, actions): 
    # we clip actions to be coherent with replay buffer
    _ = [agent.append_histories(obs, np.clip(a, a_min=-1. + epsilon, a_max=1. - epsilon)) for agent, obs, a in zip(learner.learners, observations, actions)]


def process_current_histories(learner, torch_observations):
    return [agent.process_current_histories(obs) for agent, obs in zip(learner.learners, torch_observations)]


def act(learner, torch_observations, sample, config):
    actions = []
    log_pis = []
    
    for agent, obs in zip(learner.learners, torch_observations):
        if 'toy' in config.task_name:
            action, log_pi = agent.policy.act(obs, legal_move=torch.ones((obs.shape[0],2), device=obs.device), sample=sample, return_log_pi=True)
        else:
            action, log_pi = agent.policy.act(obs, sample=sample, return_log_pi=True)

        actions.append(action.squeeze(0).detach().cpu().numpy())
        
        if log_pi is not None:
            log_pis.append(log_pi.squeeze(0).detach().cpu().numpy())
        else:
            log_pis.append(None)

    return actions, log_pis


def append_to_buffer(data_buffers, data_dict):
    for key, data in data_dict.items():
        if len(data) == 1:
            # fully cooperative so agents share reward and masks
            data = [data[0] for _ in range(len(data_buffers))]
        
        assert len(data) == len(data_buffers)
        
        for buffer, d in zip(data_buffers, data):
            buffer.append_item(key, d)


def run_one_step(env, learner, observations, reset, sample, device, config):
    # if finished we reset
    if reset:
        observations, infos = env.reset()
        reset_agents_history(learner)

    # convert observation to torch
    torch_observations = [torch.as_tensor(obs, device=device).unsqueeze(0) for obs in observations]

    # add memory embedding
    torch_observations = process_current_histories(learner, torch_observations)

    # get actions
    actions, log_pis = act(learner, torch_observations, sample=sample, config=config)

    # step env 
    next_observations, reward, done, infos = env.step(actions)

    # update memory
    append_history_agents(learner, observations, actions)

    # entropy as sampled E[-pi logpi]
    infos['log_pis'] = log_pis
    infos['actions'] = actions

    return observations, actions, reward, next_observations, done, infos

def add_artificial_rollout_length(batches, config):
    rollout_length = config.artificial_rollout_length

    for batch in batches:
        init_mask = batch["time_out_masks"][0].copy()
        batch["time_out_masks"][::rollout_length] = 0
        batch["time_out_masks"][0] = init_mask

    return batches

def collect_transition_batches(env, learner, observations, done, time_out, rollout_device, config, training_step):

    learner.to(rollout_device)

    # do some stuff to record gifs
    if hasattr(env, 'frames'):
        if config.record_gif and ((training_step + 1) % (config.record_gif_every * config.eval_frequency)) == 0:
            env.record_frames = True
            recording_frames = True
        else:
            env.record_frames = False
            recording_frames = False
    else:
        recording_frames = False

    running_buffers = [PPOBuffer() for _ in range(learner.n_learner)]
    env.record_frames = False
    returns = []
    ep_lengths = []
    ep_t = 0
    running_return = 0
    running_stats = PPOBuffer()
    frames = []
    n_ep_rec = 0
    for t in range(config.ppo_transitions_between_update):
        
        observations, actions, reward, next_observations, done, infos = run_one_step(env, learner, observations, reset=(done or time_out), sample=True, device=rollout_device, config=config)
        
        if any([np.isnan(a).any() for a in actions]):
            raise ValueError("nan found in actions during env rollout")

        if recording_frames:
            if n_ep_rec < config.n_ep_in_gif:
                frames.append(env.render('rgb_array'))

        # check for time limit
        time_out = ((t + 1) % config.env_rollout_length_train == 0) 

        # append data to buffers TODO: deal with this to put it agents ordered
        
        append_to_buffer(running_buffers, {'observations': observations, 'actions': actions, 'rewards': [reward], 'next_observations': next_observations,
                                'masks': [1. - np.float32(done)], 'time_out_masks': [1. - np.float32(time_out)]})

        # next step 
        observations = next_observations

        ep_t += 1
        running_return += reward
        
        if done or time_out:
            ep_lengths.append(ep_t)
            returns.append(running_return)
            ep_t = 0
            running_return = 0
            n_ep_rec = min(config.n_ep_in_gif, n_ep_rec + 1)
        
        running_stats.extend({'log_pis': infos['log_pis']}, wrapped=True)

    running_stats = running_stats.flush()
    # action are sampled so the mean on dim=1 is an expectation over one agent's policy and mean on dim = 0 is average over agents
    entropies = - running_stats['log_pis']
    mean_entropy = np.mean(np.sum(entropies, axis=1), axis=0)

    stats = {'collect_ep_length': np.mean(ep_lengths), 'collect_return': np.mean(returns), 'collect_mean_entropy': mean_entropy, 'collect_frames': frames}
    
    return [b.flush() for b in running_buffers] , observations, done, time_out, stats

def evaluate(learner, env, config, training_step, rollout_device):
    
    learner.to(rollout_device)

    # do some stuff to record gifs
    if hasattr(env, 'frames'):
        if config.record_gif and ((training_step + 1) % (config.record_gif_every * config.eval_frequency)) == 0:
            env.record_frames = True
            recording_frames = True
        else:
            env.record_frames = False
            recording_frames = False
    else:
        recording_frames = False

    # we reset env 
    eval_ep = 0
    t_eval = 0
    done = True
    time_out = True
    observations = None
    eval_pbar = tqdm(range(config.max_eval_episode), desc='Evaluation')
    # ep_pbar = tqdm(range(config.env_rollout_length_eval), desc='Episode time')
    eval_returns_list = []
    eval_ep_len = []
    running_return = 0
    frames = []
    traj_list = {'observations': [], 'actions': []}
    traj = {'observations': [], 'actions': []}
    n_ep_rec = 0
    while eval_ep < config.max_eval_episode:
        _, _, reward, observations, done, infos = run_one_step(env, learner, observations, reset=(done or time_out), sample=False, device=rollout_device, config=config)
        # ep_pbar.update(1)
        traj['observations'].append(observations)
        traj['actions'].append(infos['actions'])
        t_eval += 1 
        running_return += reward
        
        if recording_frames:
            if n_ep_rec < config.n_ep_in_gif:
                frames.append(env.render('rgb_array'))

        time_out = (t_eval >= config.env_rollout_length_eval)


        if (time_out or done):

            if hasattr(env, 'get_normalized_score'):
                running_return = env.get_normalized_score(running_return)

            eval_returns_list.append(running_return)
            eval_ep_len.append(t_eval)
            traj_list['observations'].append(traj['observations'])
            traj_list['actions'].append(traj['actions'])
            traj = {'observations': [], 'actions': []}
            running_return = 0
            t_eval = 0
            eval_ep += 1 
            n_ep_rec = min(config.n_ep_in_gif, n_ep_rec + 1)
            eval_pbar.update(1)
            # ep_pbar = tqdm(range(config.env_rollout_length_eval), desc='Episode time')
    
    stats = {'return': np.mean(eval_returns_list), 'frames': frames, 'ep_length': np.mean(eval_ep_len), 'traj_list': traj_list}

    return stats, time_out, done

                        
if __name__ == '__main__':

    trainer = RLAgentTraining()

    config = RLAgentTraining.get_training_argsparser().parse_args()

    trainer.train(config)