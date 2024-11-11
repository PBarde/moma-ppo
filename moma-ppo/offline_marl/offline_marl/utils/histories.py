import numpy as np
from collections import deque
import torch.nn as nn
import torch
from torch.optim import Adam
from nop import NOP

import sys
from pathlib import Path

def set_up_import():
    alfred_folder = (Path(__file__).resolve().parents[3] / 'alfred').resolve()
    alg_folder = (Path(__file__).resolve().parents[3] / 'offline_marl').resolve()
    env_folder = (Path(__file__).resolve().parents[3] / 'dzsc').resolve()
    
    for p in [alg_folder, env_folder, alfred_folder]:
        if p not in sys.path:
            sys.path.append(str(p))


set_up_import()

from offline_marl.utils.networks import MLPNetwork
from offline_marl.utils.networks import SelfAttention
from offline_marl.utils.ml import convert_to_numpy
from offline_marl.single_agent.actors import epsilon

def default_value_array(shape):
    return np.zeros(shape, dtype=np.float32)

def combine(to_combine):
    # we check that all the elements to combine have the same shape
    shapes_len = set([len(elem.shape) for elem in to_combine])
    assert len(shapes_len) == 1

    elem_shape_len = shapes_len.pop()

    if elem_shape_len == 1:
        return np.concatenate(to_combine)
    elif elem_shape_len == 2:
        return np.concatenate(to_combine, axis=1)
    else:
        raise NotImplementedError

def precompute_memories(to_combine):
    combined = combine(to_combine) # we combine different elements into one (obs, actions) or (obs, actions, rewards)

    default_shape = sum([elem.shape[-1] for elem in to_combine])
    default_value = default_value_array(default_shape)[None] # extra dim 

    combined = np.concatenate((combined, default_value), axis=0) # we add default value so we can access it fast
    default_index = len(combined) - 1

    return combined, default_index

def precompute_indexes(dones_float, history_len, default_index):
    history_indexes = []
    next_history_indexes = []
    for i in range(len(dones_float)):
        di = 1 
        history_idx = deque([])
        while (i - di >= 0 ) and (not dones_float[i - di] == 1.) and (di <= history_len):
            history_idx.appendleft(i - di)
            di += 1
        
        # all memory must be same size so we extend the partial memory idx with the idx that points to the default value for the memory
        while len(history_idx) < history_len:
            history_idx.appendleft(default_index)

        # next obs memory is like the current one but without the oldest obs and with the current obs
        next_history_idx = deque([c for c in history_idx])
        next_history_idx.popleft()
        next_history_idx.append(i)

        history_indexes.append(history_idx)
        next_history_indexes.append(next_history_idx)
    
    history_indexes = np.stack(history_indexes)
    next_history_indexes = np.stack(next_history_indexes)

    return history_indexes, next_history_indexes

def build_memories(data, memory_len):

    # Model-Based case (initial state has histories)
    if 'history_memories' in data:
        assert 'next_history_memories' not in data
        to_return = {'history_memories': data['history_memories'], 'next_history_memories': next_histories(data['history_memories'], data['observations'], data['actions'])}
        return to_return

    # Model-free case initial state has no history 
    else:
        memories, default_index = precompute_memories(to_combine=(data['observations'], data['actions']))
        # histories are discontinued both by timeouts or env being done
        traj_is_done = (((1. - data['time_out_masks']) + (1. - data['masks'])) > 0.).astype(np.float32) 
        history_indexes, next_history_indexes = precompute_indexes(dones_float=traj_is_done, history_len=memory_len, default_index=default_index)
        return {'history_memories': memories[history_indexes], 'next_history_memories': memories[next_history_indexes]}

def next_histories(histories, observations, actions):
    # we have to move all the history one step in time
    # history seq is 2nd dim

    histories = convert_to_numpy(histories)
    observations = convert_to_numpy(observations)
    actions = convert_to_numpy(actions)

    if 'int' in str(actions.dtype):
        raise NotImplementedError("must convert int action with something like self.batched_actions_one_hot(actions, self.n_actions)")

    # actions have to be in the correct range
    # TODO: refactor this into a "process action" method to account for discrete actions and all
    
    actions = np.clip(actions, a_min=-1. + epsilon, a_max=1. - epsilon)

    batch_size, seq_len, sa_len = histories.shape
    next_histories = histories.copy()

    shifted_index = np.clip(a=np.arange(seq_len) + 1, a_min=None, a_max=seq_len-1)
    next_histories = np.take(next_histories, shifted_index, axis=1)

    current_sa = np.concatenate((observations, actions), axis=1)
    next_histories[:, -1, :] = current_sa

    return next_histories

def concat_embedings_to_obs(batch, memory_encoder, memory_len, next_obs=False):
    if next_obs:
        if isinstance(memory_encoder, NOP):
                return batch['next_observations']
        else:
            return concact_memory_to_obs_from_obs(memory_encoder, batch['next_observations'], batch['next_history_memories'], memory_len)

    else:
        if isinstance(memory_encoder, NOP):
            return batch['observations']
        else:
            return concact_memory_to_obs_from_obs(memory_encoder, batch['observations'], batch['history_memories'], memory_len)

def concact_memory_to_obs_from_obs(memory_encoder, observations, memories, memory_len):
    if isinstance(memory_encoder, NOP):
        return observations
    else:
        memories = memory_encoder(memories, memory_len)
        return torch.cat((memories, observations), dim=1)

class HistoryEncoder(nn.Module):
    def __init__(self, num_in, num_out, hidden_size, lr, reduce_op, max_history_len, train_device, weight_decay):
        super().__init__()

        self.reduce_op = reduce_op

        if self.reduce_op == 'self-attention':
            self.network = SelfAttention(num_inputs=num_in, num_outputs=num_out, seq_len=max_history_len)
        else:
            self.network = MLPNetwork(num_inputs=num_in, num_outputs=num_out, hidden_size=hidden_size)

        # we put the parameters on the right device before creating the optimizer
        self.to(train_device)

        self.optim = Adam(params=self.parameters(), lr=lr, weight_decay=weight_decay)

        self.max_history_len = max_history_len

        # buffer contains np.arrays
        self.reset_history()

        self.default_value = default_value_array(num_in)
    
    def reset_history(self):
        self.buffer = deque(maxlen=self.max_history_len)
    
    def append_history(self, to_combine):
        self.buffer.append(combine(to_combine))
    
    def get_history(self):
        while len(self.buffer) < self.max_history_len:
            self.buffer.appendleft(self.default_value)
        
        memory_len = self.max_history_len

        memory = torch.as_tensor(np.stack(self.buffer), device=self.device).unsqueeze(0)

        return memory, memory_len

    def __call__(self, histories, histories_len):
        
        # precomputed embeddings so always same size
        if isinstance(histories_len, int):
            batch_size = histories.shape[0]
            memory_dim = histories.shape[2]

            if self.reduce_op == 'self-attention':
                reduced_embeddings = self.network(histories)

            elif self.reduce_op in ['mean', 'product']:
                batched_embeddings = self.network(histories.reshape(batch_size*histories_len, memory_dim))
                splitted_embeddings = batched_embeddings.reshape(batch_size, histories_len, -1)

                if self.reduce_op == 'mean':
                    reduced_embeddings = splitted_embeddings.mean(1, keepdim=True)
                elif self.reduce_op == 'product':
                    reduced_embeddings = splitted_embeddings.prod(1, keepdim=True)
                else:
                    raise NotImplementedError
            else:
                raise NotImplementedError

        return reduced_embeddings

    @property
    def device(self):
        return next(self.parameters()).device

    def get_state_dict(self):
        return {'network': self.network.state_dict(),
                'optim': self.optim.state_dict()}

    def do_load_state_dict(self, state_dict):
        self.network.load_state_dict(state_dict['network'])
        self.optim.load_state_dict(state_dict['optim'])


class MemoryWrapper(object):
    def __init__(self, dataset, memory_len):
        # this links the dataset object to the one created here
        self._dataset = dataset

        self.memory_len = memory_len
        self.memories, self.default_index = precompute_memories(to_combine=(self.observations, self.actions))
        self.memory_indexes, self.next_memory_indexes = precompute_indexes(dones_float=self.dones_float, history_len=self.memory_len, default_index=self.default_index)

    # overwrites sample function
    def sample(self, batch_size):
        indx = self._dataset.sample_idx(batch_size)
        return self.sample_from_idx(indx)

    @property
    def observations(self):
        return self._dataset.observations
    
    @property
    def actions(self):
        return self._dataset.actions

    @property
    def rewards(self):
        return self._dataset.rewards
    
    @property
    def masks(self):
        return self._dataset.masks

    @property
    def dones_float(self):
        return self._dataset.dones_float
    
    @property
    def next_observations(self):
        return self._dataset.next_observations

    @property
    def size(self):
        return self._dataset.size

    def sample_from_idx(self, indx):
        return dict(observations=self.observations[indx],
                    actions=self.actions[indx],
                    rewards=self.rewards[indx],
                    masks=self.masks[indx],
                    next_observations=self.next_observations[indx], 
                    history_memories=self.memories[self.memory_indexes[indx]],
                    next_history_memories=self.memories[self.next_memory_indexes[indx]]) # here it is just a int !      

if __name__=="__main__":
    batch_size = 25
    seq_len = 7
    obs_len = 3 
    a_len = 2


    prev_obs = torch.arange(7).unsqueeze(0) + torch.arange(25).unsqueeze(1)*100.
    prev_obs = prev_obs.unsqueeze(2).repeat((1, 1, 3))

    prev_actions = torch.arange(7).unsqueeze(0) + torch.arange(25).unsqueeze(1)*10.
    prev_actions = prev_actions.unsqueeze(2).repeat((1,1,2))

    histories = torch.cat((prev_obs, prev_actions), dim=2)

    observations = -torch.arange(25)*100.
    observations = observations.unsqueeze(1).repeat((1, 3))

    actions = -torch.arange(25)*10.
    actions = actions.unsqueeze(1).repeat((1, 2))

    hist_next = next_histories(histories, observations, actions)
    hist_next_torch = next_histories_torch(histories, observations, actions)

    assert (hist_next_torch.numpy() == hist_next).all()
    print('oh')