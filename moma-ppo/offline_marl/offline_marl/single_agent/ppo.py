import numpy as np
import torch
from torch import nn
from gym.spaces import Box, Discrete
from nop import NOP

from collections import deque

from offline_marl.utils.base_algo import BaseAlgo
from offline_marl.single_agent.critics import V, DoubleV
from offline_marl.single_agent.actors import GaussianPolicy, CategoricalPolicy
from offline_marl.utils.histories import HistoryEncoder
import offline_marl.utils.histories as histories
from offline_marl.utils.ml import convert_to_numpy

def append_item(data, key, val, convert_fct, wrapped=True):

    val = convert_fct(val)

    if wrapped:
        val = deque([val])

    if key in data:
        data[key].append(val)
    else:
        data[key] = deque([val])

class PPOMultiEnvRolloutBuffer(object):
    """
    In this buffer the dim=0 is across different environment (and initial states) and dim=1 is accross trajectory time step
    """
    def __init__(self):
        self._data = {}

    def extend(self, data_dict, live_envs_idx):
        for key, val in data_dict.items():
            if key not in self._data:
                _ = [append_item(self._data, key, v, convert_to_numpy, wrapped=True) for v in val]
            else:
                assert len(live_envs_idx) == len(val)
                
                for idx, v in zip(live_envs_idx, val):
                    self._data[key][idx].append(convert_to_numpy(v))
    
    def flush(self):
        return {key: np.concatenate([np.stack(v) for v in val]).astype(np.float32) for key, val in self._data.items()}

        

class PPOBuffer(object):
    """
    In this buffer the dim=0 dimension is the timestep (trajectories are one after the other)
    """
    def __init__(self):
        self._data = {}
    
    def extend(self, data_dict, wrapped=False):
        for key, val in data_dict.items():
            append_item(self._data, key, val, convert_to_numpy, wrapped=wrapped)

    def append_item(self, key, val, wrapped=True):
        append_item(self._data, key, val, convert_to_numpy, wrapped=wrapped)

    def flush(self):
        # we convert lists to np.arrays
        to_return = {key: np.concatenate(val).astype(np.float32) for key, val in self._data.items()}
        # we add the batch dim for lists of scalars
        to_return = {key: val[:, None] if len(val.shape) == 1 else val for key, val in to_return.items()}
        self._data = {}
        return to_return

class PPOLearner(BaseAlgo, nn.Module):
    def __init__(self, **kwargs):

        nn.Module.__init__(self)
        BaseAlgo.__init__(self)

        actions_dim = kwargs['act_space'].shape[0] if isinstance(kwargs['act_space'], Box) else kwargs['act_space'].n

        # Memory encoder
        self.memory_len = kwargs['memory_len']
        if kwargs.get('memory_len', 0) > 0:
            self.memory, memory_embedding_vec = self.make_history_encoder(num_in=kwargs['obs_space'].shape[0] + actions_dim,
                                                                                                    num_out=kwargs['memory_out_size'], 
                                                                                                    hidden_size=kwargs['hidden_size'],
                                                                                                    lr=kwargs['lr_memory'],
                                                                                                    reduce_op=kwargs['memory_op'],
                                                                                                    history_len=kwargs['memory_len'],
                                                                                                    train_device=kwargs['train_device'])
                                                                                                    
            obs_space = Box(np.concatenate((kwargs['obs_space'].low, memory_embedding_vec)), np.concatenate((kwargs['obs_space'].high, memory_embedding_vec)))
        else:
            self.memory = NOP()
            obs_space = kwargs['obs_space']

        # Policy
        params = kwargs.copy()
        params.update({'lr': kwargs['lr_pi'], 'obs_space': obs_space})

        if isinstance(kwargs['act_space'], Discrete):
            assert 'toy' in kwargs['task_name']
            self.policy = CategoricalPolicy(**params)
            
        elif isinstance(kwargs['act_space'], Box):
            self.policy = GaussianPolicy(**params)
            
        else:
            raise NotImplementedError

        # Critic
        V_params = kwargs.copy()
        V_params.update({'lr': kwargs['lr_v'], 'obs_space': obs_space})
        if kwargs['double_V']:
            self.double_V = True
            self.V = DoubleV(**V_params)
        else:
            self.double_V = False
            self.V = V(**V_params)


        # to save and reload
        self.init_dict = kwargs
        self.name = 'ppo'

        # data to monitor
        self.train_metrics = {'loss_v', 'loss_pi'}
        self.evaluation_metrics = {'return', 'length', 'training_step', 'mean_actions'}
        self.metrics_to_record = self.train_metrics | self.evaluation_metrics

    @staticmethod
    def make_history_encoder(num_in, num_out, hidden_size, lr, reduce_op, history_len, train_device, weight_decay=0, grad_norm_clip=1e7):
        encoder = HistoryEncoder(num_in=num_in, num_out=num_out,
        hidden_size=hidden_size, lr=lr, reduce_op=reduce_op, max_history_len=history_len,
        train_device=train_device, weight_decay=weight_decay)

        # we extend observation space with memory embedding size 
        embeding_vec = np.ones(num_out)
        
        return encoder, embeding_vec

    def concat_embedings_to_obs(self, batch, next_obs=False):
        return histories.concat_embedings_to_obs(batch, self.memory, self.memory_len, next_obs)

    def process_current_histories(self, observation):
        # add memory embedding
        current_memory = self.memory.get_history()
        memory, memory_len = current_memory[0], current_memory[1]
        observation = histories.concact_memory_to_obs_from_obs(self.memory, observation, memory, memory_len)
        return observation
    
    def reset_histories(self):
        self.memory.reset_history()

    def append_histories(self, observation, action):
        self.memory.append_history((observation, action))

    def build_memories(self, data):
        if self.memory_len > 0:
            return histories.build_memories(data, self.memory_len)
        else:
            return {}

    def get_state_dict(self):
        return {'policy': self.policy.get_state_dict(), 
                'memory': self.memory.get_state_dict(),
                'V': self.V.get_state_dict()}
    
    def do_load_state_dict(self, state_dict):
        self.policy.do_load_state_dict(state_dict['policy'])
        self.memory.do_load_state_dict(state_dict['memory'])
        self.V.do_load_state_dict(state_dict['V'])
    