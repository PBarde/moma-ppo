import numpy as np
import torch
from tqdm import tqdm
from typing import List

from alfred.utils.misc import Bunch

import dzsc.ma_d4rl.utils.env_and_dataset as single_agent_env_and_dataset
from dzsc.ma_d4rl.obs_mapping.mapping import get_mappings
from dzsc.ma_d4rl.utils.env_wrapper import MAActionMapping, MAObsMapping, InfoResetWrapper


def make_env(task_name, seed, record_gif, n_ep_in_gif=2, **kwargs):

    scenario, agent_conf, agent_obsk = split_task_name(task_name)

    env = single_agent_env_and_dataset.make_env(task_name=scenario, 
                            seed=seed, record_gif=record_gif, n_ep_in_gif=n_ep_in_gif)
    
    mappings = get_mappings(Bunch({'scenario': scenario, 
                                    'agent_conf': agent_conf,
                                    'agent_obsk': agent_obsk}))
    
    env = MAObsMapping(env, mappings['obs_indexes'])
    env = MAActionMapping(env, agent_conf, mappings['action_indexes'])
    env.q_info = mappings['q_info']

    # reset returns empty info dict
    env = InfoResetWrapper(env)

    return env

def split_task_name(task_name):
    task_name = task_name.replace('_DEV', '')
    scenario, agent_conf, agent_obsk = task_name.split('_')
    return scenario, agent_conf, agent_obsk

def get_split_idx(n_data, valid_proportion=0.1):
    n_valid = int(valid_proportion * n_data)
    rng = np.random.default_rng(12345)
    idx = np.arange(n_data)
    rng.shuffle(idx)
    valid_idx = idx[:n_valid]
    train_idx = idx[n_valid:]
    return valid_idx, train_idx

# makes env and wraps it, normalizes dataset scores, removes timeouts resets transitions
def make_env_and_dataset(task_name, centralized_training, seed, train_device=torch.device('cpu'), record_gif=False, logger=None, **kwargs):

    scenario, agent_conf, agent_obsk = split_task_name(task_name)

    env = make_env(task_name, seed, record_gif, kwargs.get('n_ep_in_gif', 2))

    normalize_rewards = single_agent_env_and_dataset.check_algo_for_reward_normalization(**kwargs)

    dataset = single_agent_env_and_dataset.make_dataset(env, scenario, train_device=torch.device('cpu'), normalize_rewards_if_needed=normalize_rewards)

    # If DEV we just consider the 1000 first datapoints to be faster
    if 'DEV' in task_name:
        dataset = single_agent_env_and_dataset.extract_dataset(dataset, np.arange(10000))
        
    if kwargs['make_worldmodel_dataset']:

        valid_idx, train_idx = get_split_idx(n_data=dataset.size)

        valid_dataset = single_agent_env_and_dataset.extract_dataset(dataset, valid_idx)
        train_dataset = single_agent_env_and_dataset.extract_dataset(dataset, train_idx)

        valid_dataset = MADataset(valid_dataset, centralized_training, observation_mapping_fct=env.observation, action_mapping_fct=env.reverse_action, memory_len=kwargs['memory_len'], train_device=train_device)
        train_dataset = MADataset(train_dataset, centralized_training, observation_mapping_fct=env.observation, action_mapping_fct=env.reverse_action, memory_len=kwargs['memory_len'], train_device=train_device)

        return env, train_dataset, valid_dataset
    
    else:
        dataset = MADataset(dataset, centralized_training, observation_mapping_fct=env.observation, action_mapping_fct=env.reverse_action, memory_len=kwargs['memory_len'], train_device=train_device, task_name=task_name, keep_true_obs=kwargs.get('perfect_wm', False))

        return env, dataset


class MADataset(object):
    def __init__(self, dataset: single_agent_env_and_dataset.D4RLDataset, centralized_training, observation_mapping_fct, action_mapping_fct, memory_len,  train_device, keep_true_obs=False, task_name=None) -> List[single_agent_env_and_dataset.Dataset]:
        # splits a single agent dataset into multiple agent
        # organizes data into a list of single agent d4rl datasets that can be sampled independently or not
        
        self.keep_true_obs = keep_true_obs

        if self.keep_true_obs: 
            self.task_name = task_name
            self.true_obs = torch.tensor(dataset.observations)
        
        self.datasets = []
        self.centralized_training = centralized_training
        self.observation_mapping_fct = observation_mapping_fct
        self.action_mapping_fct = action_mapping_fct

        self.n_agent = len(self.observation_mapping_fct(dataset.observations[0]))
        self.num_player = self.n_agent
        self.n_data = len(dataset.observations)
        self.reward_normalization_factor = dataset.reward_normalization_factor

        split_obs = [[] for i in range(self.n_agent)]
        split_actions = [[] for i in range(self.n_agent)]
        split_rewards = [[] for i in range(self.n_agent)]
        split_masks = [[] for i in range(self.n_agent)]
        split_dones_float = [[] for i in range(self.n_agent)]
        split_next_observations = [[] for i in range(self.n_agent)]

        # it is easier to have all datasets in ram
        device = torch.device('cpu')
        

        for obs, action, reward, mask, done_float, next_obs, i in zip(dataset.observations, 
                                                                        dataset.actions, 
                                                                        dataset.rewards,
                                                                        dataset.masks,
                                                                        dataset.dones_float,
                                                                        dataset.next_observations, 
                                                                        tqdm(range(len(dataset.observations)), desc='mapping single agent dataset to multi-agent')):
            
            mapped_obs = self.observation_mapping_fct(obs)
            mapped_action = self.action_mapping_fct(action)
            mapped_next_obs = self.observation_mapping_fct(next_obs)

            for i in range(self.n_agent):
                split_obs[i].append(mapped_obs[i])
                split_actions[i].append(mapped_action[i])
                split_rewards[i].append(reward)
                split_masks[i].append(mask)
                split_dones_float[i].append(done_float)
                split_next_observations[i].append(mapped_next_obs[i])
            
        self.convert = lambda x: np.asarray(x, dtype=np.float32)
        self.rand_idx = lambda x: np.random.randint(self.n_data, size=x)

        for i in tqdm(range(self.n_agent), desc='creating independent datasets'):

            sa_dataset = single_agent_env_and_dataset.Dataset(observations=self.convert(split_obs[i]),
                                                    actions=self.convert(split_actions[i]),
                                                    rewards=self.convert(split_rewards[i]),
                                                    masks=self.convert(split_masks[i]),
                                                    dones_float=self.convert(split_dones_float[i]),
                                                    next_observations=self.convert(split_next_observations[i]),
                                                    size=self.n_data)
            if memory_len > 0: 
                sa_dataset = single_agent_env_and_dataset.MemoryWrapper(sa_dataset, memory_len)

            self.datasets.append(sa_dataset)

    def sample(self, batch_size) -> List[single_agent_env_and_dataset.Dataset]:

        indexes = self.sample_indexes(batch_size)

        return self.sample_from_idx(indexes)
    
    def sample_indexes(self, batch_size):

        if self.centralized_training:
            indexes = [self.rand_idx((batch_size,))]*self.n_agent
        else:
            indexes = self.rand_idx((self.n_agent, batch_size))

        return indexes
    
    def sample_from_idx(self, indexes):
        return [dataset.sample_from_idx(idx) for dataset, idx in zip(self.datasets, indexes)]

    def spawn_corresponding_envs(self, idxs):

        # this is centralized training so all the indexes are the sames
        idxs = idxs[0]
        self.envs = [make_env(self.task_name, seed=i.item(), record_gif=False) for i in idxs]

        true_obs = self.true_obs[idxs]
        # all envs are same kind so they all have same q-mapping
        q_info = self.envs[0].q_info
        
        if isinstance(q_info, dict):
            q_pos = lambda x: x[q_info['qpos_idxs']]
            q_vel = lambda x : x[q_info['qvel_idxs']]
            if q_info['n_append_left'] > 0:
                # we add default values for the qpos not present in the state
                true_obs = torch.cat((torch.ones((true_obs.shape[0], q_info['n_append_left'])), true_obs), dim=1)

        else:
            q_pos = q_info.q_pos
            q_vel = q_info.q_vel

        # we set each state to its corresponding state    
        for env, obs in zip(self.envs, true_obs):
            env.reset()
            env.set_state(q_pos(obs), q_vel(obs))

    def transition_envs(self, agents_actions):
        data = [{'next_observations':[], 'rewards':[], 'masks':[], 'ensemble_cov_norm':[], 'model_max_cov_norm':[], 'rewards_cov': []} for _ in range(self.num_player)]
        
        for env, actions in zip(self.envs, zip(*agents_actions)):
            observations, reward, done, infos = env.step(actions)

            for i, d in enumerate(data):
                d['next_observations'].append(torch.tensor(observations[i], dtype=torch.float32))
                d['rewards'].append(torch.tensor(reward, dtype=torch.float32))
                d['masks'].append(torch.tensor(1.- done, dtype=torch.float32))
                d['ensemble_cov_norm'].append(torch.tensor(0., dtype=torch.float32))
                d['model_max_cov_norm'].append(torch.tensor(0., dtype=torch.float32))
                d['rewards_cov'].append(torch.tensor(0., dtype=torch.float32))
        
        for d in data:
            for key, val in d.items():
                d[key] = torch.vstack(val)

        return data
    
    def keep_only_not_done_envs(self, not_dones):
        self.envs = [self.envs[i] for i, not_done in enumerate(not_dones) if not_done]
