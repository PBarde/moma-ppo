
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
from collections import deque

from alfred.utils.recorder import Aggregator, Recorder
from alfred.utils.misc import create_management_objects
from alfred.utils.config import parse_bool, parse_log_level, save_config_to_json, load_config_from_json
from alfred.utils.directory_tree import DirectoryTree

from offline_marl import load_worldmodel
from offline_marl import train_rl
from offline_marl.trainer import Training
from offline_marl.single_agent.ppo import PPOMultiEnvRolloutBuffer
from offline_marl.train_rl import RLAgentTraining
from offline_marl.utils.histories import next_histories
from offline_marl.utils.misc import save_video_to_wandb, get_timestamp
import offline_marl.utils.ml as ml
from offline_marl.utils.ml import batch_as_tensor, batch_to_device, convert_to_numpy

from dzsc.make_env_module import get_make_env_module

class MBRLAgentTraining(Training):

    @staticmethod
    def get_training_argsparser():
        parser = argparse.ArgumentParser()

        # Alfred args
        parser.add_argument('--desc', type=str, default='test_clip', help="description of the run")
        parser.add_argument('--alg_name', type=str, default='mb-ma-ppo', help="name of the algo, currently only used by alfred")
        parser.add_argument('--task_name', type=str, default='reacher-expert-mix-v0_2x1first_0_DEV', help='d4rl task name with agent decomposition')
        parser.add_argument('--seed', type=int, default=1, help='random seed that seeds everything')

        # Algorithm args (single agent and multi-agent)
        # common
        parser.add_argument('--batch_size', type=int, default=64, help="mini-batch size for all models and updates")
        parser.add_argument('--discount_factor', type=float, default=0.99, help="TD learning discount factor")
        parser.add_argument('--lr_v', type=float, default=3e-4, help="learning rate for V network updates")
        parser.add_argument('--lr_pi', type=float, default=3e-4, help="learning rate for policy network updates")
        parser.add_argument('--grad_clip_norm', type=float, default=1., help="the norm of the gradient clipping for all the losses")
        parser.add_argument('--state_dependent_std', type=parse_bool, default=True)
        parser.add_argument('--hidden_size', type=int, default=256, help="number of hidden units in fully connected layers")
        parser.add_argument('--env_rollout_length_train', type=int, default=50)
        parser.add_argument('--env_rollout_length_eval', type=int, default=50)
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

        # Model-Based args
        parser.add_argument("--wm_overwrite_task_name", type=str, default=None)
        parser.add_argument("--model_name", type=str, default="world-model_model.pt")
        parser.add_argument("--k_steps_rollouts", type=int, default=10, help="number of steps world model rollouts")
        parser.add_argument("--perfect_wm", type=parse_bool, default=False, help="Uses hanabi simulator as worldmodel")
        parser.add_argument('--n_wm_used', type=int, default=5, help="number of models that are kept based on their validation accuracy")
        parser.add_argument('--wm_use_min_rewards', type=parse_bool, default=False, help="Uses min reward accros ensemble instead of randomly sampled")
        parser.add_argument('--wm_use_all_min', type=parse_bool, default=False, help="Use model that is most pessimistic wrt to the reward instead of randomly sampling it")
        parser.add_argument("--reward_penalty_gaussian_std", type=float, default=0.0, help="constant for MOPO penalty, usually between 0.5 and 5")
        parser.add_argument("--reward_penalty_ensemble_std", type=float, default=0.1, help="constant for reward penalty but multiplies the ensemble covariance instead of the max gaussian std")
        parser.add_argument("--reward_penalty_ensemble_reward_only_std", type=float, default=0.05, help="like above but ensemble covariance is only computed on the reward prediction")
        parser.add_argument("--ensemble_cov_norm_threshold", type=float, default=2, help="if above this we cut the model rollout")
        parser.add_argument("--clip_wm_to_dataset", type=parse_bool, default=True, help="if true we clip the world model outputs to the minimum bounding box of the dataset https://en.wikipedia.org/wiki/Minimum_bounding_box")
        # Monitoring args
        parser.add_argument('--max_training_step', type=int, default=int(1e5 + 1), help="number of updates during training") # int(1e6 + 1)
        parser.add_argument('--eval_frequency', type=int, default=50, help='number of training steps between evaluation') #20000
        parser.add_argument('--log_frequency', type=int, default=10, help='number of learning steps before writing training stats (losses) to disk') #20000
        parser.add_argument('--max_eval_episode', type=int, default=2, help='number of episodes used for evaluation') #10
        parser.add_argument('--record_gif', type=parse_bool, default=False, help="records gifs to wandb")
        parser.add_argument('--record_gif_every', type=int, default=50, help='records a gif only every x evals')
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
        return RLAgentTraining.get_make_learner(no_mb_alg_name, task_name)


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

        if config.wm_use_min_rewards or config.wm_use_all_min: 
            raise NotImplementedError

        if not config.centralized_training:
            raise ValueError("We only consider centralized algos in this project !!!")

        if config.perfect_wm == False:
            if 'DEV' in config.task_name:
                if config.wm_overwrite_task_name is None:
                    wm_task_name = config.task_name.replace('_DEV', '')
                    warnings.warn(f"Over writing wm_task_name to load wm to {wm_task_name}")
                    config.wm_overwrite_task_name = wm_task_name
            else:
                assert config.wm_overwrite_task_name is None

        if 'toy' in config.task_name:
            if config.perfect_wm: 
                warnings.warn("setting perfect_wm to false because toy problem")
                config.perfect_wm = False

            if not config.k_steps_rollouts == 1:
                warnings.warn("setting k_steps_rollouts to 1 because toy problem")
                config.k_steps_rollouts = 1

            if not config.memory_len == 0:
                warnings.warn("setting memory_len to 0 because toy problem")
                config.memory_len = 0
        
        config = RLAgentTraining().sanity_check_config(config, logger)

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

        make_env_and_dataset = get_make_env_module(self.config.task_name).make_env_and_dataset
        make_learner = self.get_make_learner(self.config.alg_name, self.config.task_name)

        # makes env and wraps it, normalizes dataset scores, removes timeouts resets transitions
        env, dataset = make_env_and_dataset(**dict(make_worldmodel_dataset=False, **self.config.__dict__))

        # we deal with episode termination ourselves
        if not any([task in config.task_name for task in ['toy', 'reacher']]):
            env.env.env.env.env.env.env._max_episode_steps = np.inf


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
        for training_step in pbar:

            ### we collect data for ppo
            batches, collect_stats_dict = generate_transition_batches(config, learner, dataset, wm_ensemble, rollout_device)

            # update the number of env steps we did
            samples_len = set([len(b['rewards']) for b in batches])
            assert len(samples_len) == 1
            samples_len = samples_len.pop()
            env_step += samples_len


            ## we update the learners
            train_stats_dict = learner.update_from_batch(batches, train_device)
            
            train_aggregator.update(collect_stats_dict)
            train_aggregator.update(train_stats_dict)

            ## logging step
            if ((training_step + 1) % self.config.log_frequency == 0) or (training_step == 0):               
                to_record = dict(train_aggregator.pop_all_means(), training_step=training_step, env_step=env_step)
                train_recorder.write_to_tape(to_record)
                train_recorder.save(self.dir_tree.recorders_dir / 'train_recorder.pkl')
                wandb.log(to_record)
            
            ## evaluation step
            if ((training_step + 1) % self.config.eval_frequency == 0) or (training_step == 0):
                if self.config.max_eval_episode > 0:

                    stats, time_out, done = train_rl.evaluate(learner, env, config, training_step, rollout_device)
                    assert (time_out or done), "we need to reset env before resuming collecting data after eval"

                    traj_list = stats.pop('traj_list')

                    frames = stats.pop('frames')
                    if len(frames) > 0:
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

                    # we update training_step here because we actually updated the main thread models
                    self.training_step = training_step
                    self.env_step = env_step

                    # we update training step in config and save the config
                    self.config.training_step = self.training_step
                    self.config.env_step = self.env_step
                    save_config_to_json(self.config, str(self.dir_tree.seed_dir / 'config.json'))

        open(str(self.dir_tree.seed_dir / 'COMPLETED'), 'w+').close()


def generate_transition_batches(config, learner, dataset, wm_ensemble, device, n_sub_batches=1):

    with torch.no_grad():
        
        learner.to(device)

        sub_batch_buffer_list = deque([])

        n_collected = 0

        collect_stats_dict = {'k_rollout_lengths': []}

        while n_collected < config.ppo_transitions_between_update:
            
            n_initial_states = ((config.ppo_transitions_between_update - n_collected) // config.k_steps_rollouts) + 1

            # we cannot do it with full batch_size because then co-variance matrices accross batch are too large
            for n in range(n_sub_batches):
                    
                sub_batch_buffers = [PPOMultiEnvRolloutBuffer() for _ in range(learner.n_learner)] 

                data_idx = dataset.sample_indexes(n_initial_states//n_sub_batches + 1)

                live_envs_idx = np.arange(len(data_idx[0]))

                data = dataset.sample_from_idx(data_idx)

                if config.perfect_wm:
                    # we spawn a env for each transition to rollout from and get it to that state
                    dataset.spawn_corresponding_envs(data_idx)
                
                for k in range(config.k_steps_rollouts):

                    # take some new actions from the current policies
                    for i, batch, learner_i in zip(range(len(data)), data, learner.learners):

                        batch = batch_to_device(batch_as_tensor(batch), device)

                        observations = learner_i.concat_embedings_to_obs(batch)

                        if 'legal_moves' in batch:
                            new_actions, log_pis = learner_i.policy.act(observations, batch['legal_moves'], sample=True, return_log_pi=True)
                            new_actions, log_pis = new_actions.detach(), log_pis.detach()
                        else:
                            new_actions, log_pis = learner_i.policy.act(observations, sample=True, return_log_pi=True)
                            new_actions, log_pis = new_actions.detach(), log_pis.detach()
                        
                        # we modify the batch with the new action, in order to do so we have to make it a dict
                        batch = batch._asdict() if not type(batch) == dict else batch
                        
                        batch['actions'] = new_actions
                        batch['log_pis'] = log_pis
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
                        synthetic_data = wm_ensemble.sample(data, config, device)
                        if 'toy' in config.task_name:
                            pass
                        else:
                            if config.clip_wm_to_dataset:
                                synthetic_data = clip_to_dataset(synthetic_data, dataset)
                            synthetic_data = correct_rewards(synthetic_data, **config.__dict__)

                    # difference_in_next_states = (synthetic_data[0]['next_observations'] - data[0]['next_observations'].cpu()).abs().mean()
                    # difference_in_reward = (synthetic_data[0]['rewards'] - (data[0]['rewards']).cpu()).abs().mean()

                    # we update the synthetic transitions with real data
                    # for the first k-step the history memories do not change
                    for synthetic_d, d in zip(synthetic_data, data):
                        for key in ['observations', 'actions', 'legal_moves', 'history_memories', 'log_pis']:
                            if key in d:
                                synthetic_d[key] = d[key]

                        if 'toy' in config.task_name:
                            synthetic_d['legal_moves'] = np.ones((len(synthetic_d['observations']), 2))

                    for synthetic_d in synthetic_data:
                        # we add the timeouts that if we reached the last k step
                        synthetic_d['time_out_masks'] = np.asarray([(k+1) != config.k_steps_rollouts for _ in synthetic_d['masks']], dtype=np.float32)[:, None]

                        if 'ensemble_cov_norm' in synthetic_d:
                            # we also add a timeout if we strayed too far from the model training distribution and ensemble is confused
                            out_of_distrib = convert_to_numpy(synthetic_d['ensemble_cov_norm'] > config.ensemble_cov_norm_threshold)
                            synthetic_d['time_out_masks'][out_of_distrib] = 0.


                    # adds it to the buffers
                    _ = [buffer.extend(synthetic_d, live_envs_idx) for buffer, synthetic_d in zip(sub_batch_buffers, synthetic_data)]

                    if config.k_steps_rollouts > 1:
                        
                        # we have to move to the next step
                        next_batches = []

                        # we will remove transitions that ended up in dones
                        # we compute it once because all agents have same masks
                        not_dones = (convert_to_numpy((synthetic_data[0]['masks'] == 1.)) * convert_to_numpy((synthetic_data[0]['time_out_masks'] == 1.))).flatten()
                        envs_done = not_dones == False

                        if not any(not_dones):
                            collect_stats_dict['k_rollout_lengths'].append(k+1)
                            break

                        if any(envs_done):
                            collect_stats_dict['k_rollout_lengths'].append(k+1)
                            live_envs_idx = np.delete(live_envs_idx, convert_to_numpy(envs_done))


                        if config.perfect_wm:
                            dataset.keep_only_not_done_envs(not_dones)

                        # we step in terms of observations and memory
                        for i, synthetic_d in enumerate(synthetic_data):

                            observations = synthetic_d['observations'][not_dones]
                            actions = synthetic_d['actions'][not_dones]
                            next_observations = synthetic_d['next_observations'][not_dones]

                            agent_next_batch = {'observations': next_observations}
                            
                            if config.memory_len > 0:
                                history_memories = synthetic_d['history_memories'][not_dones]

                                next_history_memories = next_histories(history_memories, observations, actions)
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
                                agent_next_batch['legal_moves'] = np.ones((len(agent_next_batch['observations']), 2))

                        # convert to bunch to access dict key as attributes (we do not have all the batch's attribute so we cannot convert to batch_class)
                        data = next_batches

                # we store the traj that were generated during that sub batch
                sub_batch_buffer_list.append([b.flush() for b in sub_batch_buffers])

                n_collected += len(sub_batch_buffer_list[-1][0]['masks'])


        # we merge the trajs in each sub_batch for each agent
        return_buffers = [{key: np.concatenate([buffer[agent][key] for buffer in sub_batch_buffer_list]) for key in sub_batch_buffer_list[0][0].keys()} for agent in range(learner.n_learner)]
        # # we compute mean entropy and remove it from batch, dim=0 is now agent
        # log_pis = np.stack([buffer.pop('log_pis') for buffer in return_buffers])
        # # action are sampled so the mean on dim=1 is an expectation over one agent's policy and mean on dim = 0 is average over agents
        # mean_entropy = np.mean(-np.mean(log_pis, axis=1), axis=0)

        # Entropy as sampled E[-plogp]
        log_pis = np.stack([buffer.pop('log_pis') for buffer in return_buffers])

        #dims are agent, batch 
        mean_entropy = np.mean(np.sum(-log_pis, axis=0), axis=0)

        n_transitions = set([len(b['masks']) for b in return_buffers])
        assert len(n_transitions) == 1
        n_transitions = n_transitions.pop()
        assert n_transitions == n_collected

        collect_stats_dict['k_rollout_lengths'] = np.mean(collect_stats_dict['k_rollout_lengths'])
        collect_stats_dict.update({'n_collected_transitions': n_transitions, 'collect_mean_entropy': mean_entropy})

        collect_stats_dict.update({'ensemble_cov_norm_mean': convert_to_numpy(synthetic_data[0]['ensemble_cov_norm'].mean()),
                                     'ensemble_cov_norm_min': convert_to_numpy(synthetic_data[0]['ensemble_cov_norm'].min()),
                                     'ensemble_cov_norm_max': convert_to_numpy(synthetic_data[0]['ensemble_cov_norm'].max()),
                                     'model_max_cov_norm_mean': convert_to_numpy(synthetic_data[0]['model_max_cov_norm'].mean()),
                                     'model_max_cov_norm_min': convert_to_numpy(synthetic_data[0]['model_max_cov_norm'].min()),
                                     'model_max_cov_norm_max': convert_to_numpy(synthetic_data[0]['model_max_cov_norm'].max()),
                                     'rewards_cov_mean': convert_to_numpy(synthetic_data[0]['rewards_cov'].mean()),
                                     'rewards_cov_min': convert_to_numpy(synthetic_data[0]['rewards_cov'].min()),
                                     'rewards_cov_max': convert_to_numpy(synthetic_data[0]['rewards_cov'].max()),
                                     'rewards_penalty_mean': convert_to_numpy(synthetic_data[0].get('rewards_penalty', torch.tensor([0.])).mean()),
                                     'rewards_penalty_min': convert_to_numpy(synthetic_data[0].get('rewards_penalty', torch.tensor([0.])).min()),
                                     'rewards_penalty_max': convert_to_numpy(synthetic_data[0].get('rewards_penalty', torch.tensor([0.])).max()),})


    return return_buffers, collect_stats_dict

def correct_rewards(data, **kwargs):
    for d in data:
        d['rewards_penalty'] = kwargs['reward_penalty_gaussian_std'] * d['model_max_cov_norm'] + kwargs['reward_penalty_ensemble_std'] * d['ensemble_cov_norm'] + kwargs['reward_penalty_ensemble_reward_only_std'] * d['rewards_cov']
        d['rewards'] = d['rewards'] - d['rewards_penalty']
    return data

def clip_to_dataset(data, dataset):
    if not hasattr(dataset, 'clip_dicts'):
        clip_dicts = []
        for dset in dataset.datasets:
            tmp_dict = {}
            
            tmp_dict['rewards'] = {'min': dset.rewards.min(0), 'max': dset.rewards.max(0)}
            
            # for the obs the limit is the limits of both obs and next obs
            tmp_dict['observations'] = {'min': np.minimum(dset.observations.min(0), dset.next_observations.min(0)),
                                        'max': np.maximum(dset.observations.max(0), dset.next_observations.max(0))}
            tmp_dict['next_observations'] = tmp_dict['observations']

            clip_dicts.append(tmp_dict)
        
        dataset.clip_dicts = clip_dicts
    
    assert len(data) == len(dataset.clip_dicts)

    for d, clip_dict in zip(data, dataset.clip_dicts):
        for key, clip_limits in clip_dict.items():
            # Usually obsevations is not in synthetic data
            if key in d: 
                d[key] = np.clip(d[key], a_min=clip_limits['min'], a_max=clip_limits['max'])
    
    return data

def get_metrics_to_record(metrics):
    return train_rl.get_metrics_to_record(metrics) + ['n_collected_transitions', 'k_rollout_lengths', 'ensemble_cov_norm_mean', 'ensemble_cov_norm_min', 'ensemble_cov_norm_max',
            'model_max_cov_norm_mean', 'model_max_cov_norm_min', 'model_max_cov_norm_max',
            'rewards_cov_mean', 'rewards_cov_min', 'rewards_cov_max',
            'rewards_penalty_mean', 'rewards_penalty_min', 'rewards_penalty_max']

if __name__ == '__main__':

    trainer = MBRLAgentTraining()

    config = MBRLAgentTraining.get_training_argsparser().parse_args()

    trainer.train(config)