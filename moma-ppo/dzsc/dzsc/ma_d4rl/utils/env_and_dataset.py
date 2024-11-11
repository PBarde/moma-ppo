from tqdm import tqdm
import numpy as np
import gym
import d4rl
import torch
import warnings
import pickle

from alfred.utils.misc import Bunch

from dzsc.ma_d4rl.obs_mapping.mapping import get_mappings
from dzsc.ma_d4rl.utils.env_wrapper import ActionClipping, SinglePrecision, EpisodeMonitor, ObsCutoffWrapper, ObsMapping, ANT_CONTACT_FORCE_IDX
from dzsc.ma_d4rl.utils.constants import ADDED_TASKS
from dzsc.ma_d4rl.generated_datasets import dataset_path

from offline_marl.utils.histories import MemoryWrapper

def make_env(**kwargs):
    env_name, seed, record_frames, n_ep_per_gifs = kwargs['task_name'], kwargs['seed'], kwargs['record_gif'], kwargs['n_ep_in_gif']
    
    splited_env_name = env_name.split("_")

    if len(splited_env_name) == 2: 
        env_name, obsk = splited_env_name
        mapping = get_mappings(Bunch({'scenario': env_name, 
                                    'agent_conf': 'sa',  # single agent conf
                                    'agent_obsk': obsk}))
    elif len(splited_env_name) == 1:
        env_name = splited_env_name[0]
        mapping = None
    else:
        raise NotImplementedError

    if 'reacher' in env_name:
        env = ActionClipping(gym.envs.mujoco.reacher.ReacherEnv())
    else:
        env = gym.make(env_name)

    env = EpisodeMonitor(env, record_frames, n_ep_per_gifs)
    env = SinglePrecision(env)

    if 'ant-' in env_name:
        msg = "ant env have zero contact forces in observations, cutting-off zero observations"
        gym.logger.warn(msg)
        env = ObsCutoffWrapper(env, cutoff_idx=ANT_CONTACT_FORCE_IDX)
        
        # we color legs and torso differently
        from dzsc.ma_d4rl.obs_mapping.ant import color_ant_env
        color_ant_env(env)
        
    elif 'reacher' in env_name:
        # we color legs and torso differently
        from dzsc.ma_d4rl.obs_mapping.reacher import color_reacher_env
        color_reacher_env(env)

    env.seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)

    if mapping is not None:
        env = ObsMapping(env, mapping['obs_indexes'][0])
    return env

def make_dataset(env, env_name, train_device, normalize_rewards_if_needed=False):

    if env_name in ADDED_TASKS: 
        dataset = D4RLDataset.from_env_name(env_name)
    else:
        dataset = D4RLDataset.from_env(env)

    # we handle contact force problem problem with mujoco ant env
    if 'ant-' in env_name:
        msg = "ant dataset have zero contact forces in observations,  cutting-off zero observations"
        gym.logger.warn(msg)
        dataset = cutoff_obs_in_dataset(dataset, ANT_CONTACT_FORCE_IDX)

    if 'antmaze' in env_name:
        dataset.rewards -= 1.0
        # See https://github.com/aviralkumar2907/CQL/blob/master/d4rl/examples/cql_antmaze_new.py#L22
        # but I found no difference between (x - 0.5) * 4 and x - 1.0

    if normalize_rewards_if_needed:
        if ('halfcheetah' in env_name) or ('walker2d' in env_name) or ('ant-' in env_name): # we do not normalize rewards for reacher
            dataset.normalize_rewards(1000.0)

    return dataset

def extract_dataset(dset, indexes):
    extracted = {}
    if not isinstance(dset, dict):
        dset = dset.__dict__
        
    for key, val in dset.items():
        if hasattr(val, '__len__'):
            extracted[key] = val[indexes]
        else:
            if key == 'size':
                extracted[key] = len(indexes)
            elif key =='reward_normalization_factor':
                # note that we keep the same normalization, we do not renormalize the rewards
                extracted[key] = val
            else:
                raise NotImplementedError

    return D4RLDataset(extracted)

def make_env_and_dataset(**kwargs):

    env = make_env(**kwargs)

    normalize_rewards = check_algo_for_reward_normalization(**kwargs)

    dataset = make_dataset(env, kwargs['task_name'], train_device=torch.device('cpu'), normalize_rewards_if_needed=normalize_rewards)

    if kwargs.get('memory_len', 0) > 0:
        dataset = MemoryWrapper(dataset, memory_len=kwargs['memory_len'])
    
    return env, dataset

def check_algo_for_reward_normalization(**kwargs):

    # this accounts for 'ma-iql' and other algos on top of iql

    reward_norm_algos_tags = ['iql', 'world_model', 'bc']
    tag_in_name = [tag in kwargs['alg_name'] for tag in reward_norm_algos_tags]

    if any(tag_in_name):
        return True
    else:
        return False
    
class Dataset(object):
    def __init__(self, observations: np.ndarray, actions: np.ndarray,
                 rewards: np.ndarray, masks: np.ndarray,
                 dones_float: np.ndarray, next_observations: np.ndarray,
                 size: int, reward_normalization_factor = 1.):

        self.observations = observations
        self.actions = actions
        self.rewards = rewards
        # masks are 1-terminal where terminals are NOT from timeouts
        self.masks = masks
        # dones are end of episodes BOTH from timeouts AND terminals
        self.dones_float = dones_float
        self.next_observations = next_observations
        self.size = size
        self.reward_normalization_factor = reward_normalization_factor

    def sample_from_idx(self, indx):
        return dict(observations=self.observations[indx],
                     actions=self.actions[indx],
                     rewards=self.rewards[indx],
                     masks=self.masks[indx],
                     next_observations=self.next_observations[indx])

    def sample_idx(self, batch_size):
        return np.random.randint(self.size, size=batch_size)

    def sample(self, batch_size: int):
        indx = self.sample_idx(batch_size)
        return self.sample_from_idx(indx)   

class D4RLDataset(Dataset):

    def __init__(self, datadict):
    
        super().__init__(**datadict)

    @classmethod
    def from_env_name(cls, env_name):
        data_path = dataset_path.datasets[env_name]

        with open(data_path, 'rb') as fh:
            dataset = pickle.load(fh)

        time_out_masks = dataset.pop('time_out_masks')
        # dones are end of episodes BOTH from timeouts AND terminals
        mask_dones_floats = time_out_masks * dataset['masks']
        dataset['dones_float'] = 1.- mask_dones_floats

        dataset['size'] = len(dataset['observations'])
        
        return cls(dataset)

    @classmethod
    def from_env(cls,
                 env: gym.Env,
                 clip_to_eps: bool = True,
                 eps: float = 1e-5):

        # this dataset doesn't consider timeouts as terminals states
        dataset = d4rl.qlearning_dataset(env)

        if clip_to_eps:
            lim = 1 - eps
            dataset['actions'] = np.clip(dataset['actions'], -lim, lim)

        dones_float = np.zeros_like(dataset['rewards'])

        for i in range(len(dones_float) - 1):
            # The following line detects resets from timeouts as well as from terminals
            # dones_floats mark end of trajectories, either from timeouts or terminals
            if np.linalg.norm(dataset['observations'][i + 1] -
                              dataset['next_observations'][i]
                              ) > 1e-6 or dataset['terminals'][i] == 1.0:
                dones_float[i] = 1
            else:
                dones_float[i] = 0

        dones_float[-1] = 1

        if hasattr(env, 'mapping'):
            warnings.warn('Maps dataset observations')
            dataset['observations'] = np.take(dataset['observations'], env.mapping, axis=1)
            dataset['next_observations'] = np.take(dataset['next_observations'], env.mapping, axis=1)

        data_dict = {'observations': dataset['observations'].astype(np.float32),
                    'actions' : dataset['actions'].astype(np.float32),
                    'rewards' : (dataset['rewards'].astype(np.float32)).reshape(-1,1),
                    'masks' : (1.0 - dataset['terminals'].astype(np.float32)).reshape(-1,1),
                    'dones_float' : dones_float.astype(np.float32).reshape(-1,1),
                    'next_observations' : dataset['next_observations'].astype(np.float32),
                    'size' : len(dataset['observations'])}
        
        return cls(data_dict)

    def normalize_rewards(self, mean_ep_len):
        trajs = split_into_trajectories(self.observations, self.actions,
                                        self.rewards, self.masks,
                                        self.dones_float,
                                        self.next_observations)

        def compute_returns(traj):
            episode_return = 0
            for _, _, rew, _, _, _ in traj:
                episode_return += rew

            return episode_return

        trajs.sort(key=compute_returns)

        self.reward_normalization_factor = (mean_ep_len / (compute_returns(trajs[-1]) - compute_returns(trajs[0]))).item()

        self.rewards *= self.reward_normalization_factor

    
def split_into_trajectories(observations, actions, rewards, masks, dones_float,
                            next_observations):
    # todo: check this function more in depth 
    trajs = [[]]

    for i in tqdm(range(len(observations)), desc='normalizes reward'):
        trajs[-1].append((observations[i], actions[i], rewards[i], masks[i],
                          dones_float[i], next_observations[i]))
                          
        # splits trajs on both timeouts and terminals
        if dones_float[i] == 1.0 and i + 1 < len(observations):
            trajs.append([])

    return trajs

def cutoff_obs_in_dataset(dataset: D4RLDataset, cutoff_idx: int) -> D4RLDataset:
    dataset.observations = dataset.observations[:,:cutoff_idx]
    dataset.next_observations = dataset.next_observations[:,:cutoff_idx]
    return dataset
