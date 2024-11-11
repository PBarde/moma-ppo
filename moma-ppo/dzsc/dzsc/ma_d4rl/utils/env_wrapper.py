from typing import Tuple
import time
import gym
import numpy as np
from gym.spaces import Box, Dict
import copy
import torch

TimeStep = Tuple[np.ndarray, float, bool, dict]
ANT_CONTACT_FORCE_IDX = 27

class EpisodeMonitor(gym.ActionWrapper):
    """A class that computes episode returns and lengths.
        Also records frames to do gifs"""
    def __init__(self, env: gym.Env, record_frames=True, n_ep_per_gifs=2):
        super().__init__(env)
        self.total_timesteps = 0
        self.record_frames = record_frames
        self.n_ep_per_gifs = n_ep_per_gifs
        self._reset_stats()

    def _reset_stats(self):
        self.reward_sum = 0.0
        self.episode_length = 0
        self.frames = []
        self.start_time = time.time()

    def step(self, action: np.ndarray) -> TimeStep:
        observation, reward, done, info = self.env.step(action)

        self.reward_sum += reward
        self.episode_length += 1
        self.total_timesteps += 1
        if self.record_frames:
            f = self.render('rgb_array')
            self.frames.append(f)

        info['total'] = {'timesteps': self.total_timesteps}

        if done:
            info['episode'] = {}
            info['episode']['return'] = self.reward_sum
            info['episode']['length'] = self.episode_length
            info['episode']['duration'] = time.time() - self.start_time
            info['episode']['frames'] = self.frames

            if hasattr(self, 'get_normalized_score'):
                info['episode']['return'] = self.get_normalized_score(
                    info['episode']['return']) * 100.0

        return observation, reward, done, info

    def reset(self) -> np.ndarray:
        self._reset_stats()
        observation = self.env.reset()

        if self.record_frames:
            f = self.render('rgb_array')
            self.frames.append(f)

        return observation

class InfoResetWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
    
    def reset(self):
        return super().reset(), {}

class SinglePrecision(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

        if isinstance(self.observation_space, Box):
            obs_space = self.observation_space
            self.observation_space = Box(obs_space.low, obs_space.high,
                                         obs_space.shape)
        elif isinstance(self.observation_space, Dict):
            obs_spaces = copy.copy(self.observation_space.spaces)
            for k, v in obs_spaces.items():
                obs_spaces[k] = Box(v.low, v.high, v.shape)
            self.observation_space = Dict(obs_spaces)
        else:
            raise NotImplementedError

    def observation(self, observation: np.ndarray) -> np.ndarray:
        if isinstance(observation, np.ndarray):
            return observation.astype(np.float32)
        elif isinstance(observation, dict):
            observation = copy.copy(observation)
            for k, v in observation.items():
                observation[k] = v.astype(np.float32)
            return observation

class ObsCutoffWrapper(gym.ObservationWrapper):
    def __init__(self, env, cutoff_idx):
        super().__init__(env)
        self.cutoff_idx = cutoff_idx # Index after which observations are discarded

        assert isinstance(self.observation_space, Box)
        
        obs_space = self.observation_space
        self.observation_space = Box(obs_space.low[:self.cutoff_idx], obs_space.high[:self.cutoff_idx], (ANT_CONTACT_FORCE_IDX,))

    def observation(self, observation: np.ndarray) -> np.ndarray:
        cut_off_obs = observation[:self.cutoff_idx]
        return cut_off_obs

class ObsMapping(gym.ObservationWrapper):
    def __init__(self, env, mapping):
        super().__init__(env)
        self.mapping = mapping

        if isinstance(self.observation_space, Box):
            obs_space = self.observation_space
            mapping = self.mapping
            self.observation_space = Box(obs_space.low[mapping], obs_space.high[mapping], (len(mapping),))
        else:
            raise NotImplementedError

    def observation(self, observation):
        # return multi-agent individual observations from global state
        return observation[self.mapping]

class MAObsMapping(gym.ObservationWrapper):
    def __init__(self, env, mappings):
        super().__init__(env)
        self.mappings = mappings

        if isinstance(self.observation_space, Box):
            obs_space = self.observation_space
            self.observation_space = [Box(obs_space.low[mapping], obs_space.high[mapping], (len(mapping),))
             for mapping in self.mappings]
        else:
            raise NotImplementedError

    def observation(self, observation):
        # return multi-agent individual observations from global state
        return [observation[mapping] for mapping in self.mappings]

class MAActionMapping(gym.ActionWrapper):
    # following MaMujoco there is no action mapping, actions are just flattened and not reorganized, 
    # all agents are assumed to have the same action space.
    def __init__(self, env, agent_conf, action_indexes):
        super().__init__(env)
        self.agent_conf = agent_conf.lower()
        
        self.action_indexes = [np.asarray(indexes) for indexes in action_indexes]

        self.action_shuffle = np.argsort(np.concatenate(self.action_indexes))

        conf = ''.join(c for c in self.agent_conf if (c.isdigit() or c=='x'))
        n_agent, n_action = [int(n) for n in conf.split('x')]

        self.n_agent = n_agent
        self.n_action = n_action

        if isinstance(self.action_space, Box):
            act_space = self.action_space
            assert self.n_agent * self.n_action == act_space.shape[0]
            split_idx = [np.arange(self.n_action*i, self.n_action*(i+1)) 
                            for i in range(self.n_agent)]
            self.action_space = [Box(act_space.low[idx], act_space.high[idx], (self.n_action,)) 
                                    for idx in split_idx]
        else:
            raise NotImplementedError

    def action(self, action):
        assert (type(action) == list) or (type(action) == tuple)

        action = [a.detach().cpu().numpy() if type(a)==torch.Tensor else a for a in action]
        action = np.asarray(action)
        return np.concatenate(action)[self.action_shuffle]
    
    def reverse_action(self, action):
        return [action[indexes] for indexes in self.action_indexes]
    
class ActionClipping(gym.ActionWrapper):

    def __init__(self, env):
        super().__init__(env)
    
    def action(self, action):

        action = np.clip(a=action, a_min=self.action_space.low, a_max=self.action_space.high)

        try:
            return super().action(action)
        except NotImplementedError: 
            return action