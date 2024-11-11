from torch import nn
from torch.optim import Adam
from gym.spaces import Box, Discrete
import torch

from offline_marl.utils.networks import MLPNetwork
from offline_marl.utils.ml import soft_update



class V(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        
        assert type(kwargs['obs_space']) == Box
        assert len(kwargs['obs_space'].shape) == 1

        self.network = MLPNetwork(num_inputs=kwargs['obs_space'].shape[0], num_outputs=1, hidden_size=kwargs['hidden_size'])

        # we put the parameters on the right device before creating the optimizer
        self.to(kwargs['train_device'])
        
        self.optim = Adam(params=self.parameters(), lr=kwargs['lr'])

    def __call__(self, input):
        return self.network(input)

    @property
    def device(self):
        return next(self.parameters()).device

    def get_state_dict(self):
        return {'network': self.network.state_dict(),
                'optim': self.optim.state_dict()}

    def do_load_state_dict(self, state_dict):
        self.network.load_state_dict(state_dict['network'])
        self.optim.load_state_dict(state_dict['optim'])


class DoubleV(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        
        self.v1 = V(**kwargs)
        self.v2 = V(**kwargs)

    def __call__(self, inputs):
        return self.v1(inputs), self.v2(inputs)
    
    @property
    def device(self):
        assert self.v1.device == self.v2.device
        return self.v1.device

    def get_state_dict(self):
        return {'v1': self.v1.get_state_dict(), 
                'v2': self.v2.get_state_dict()}
    
    def do_load_state_dict(self, state_dict):
        self.v1.do_load_state_dict(state_dict['v1'])
        self.v2.do_load_state_dict(state_dict['v2'])


class Q(nn.Module):
    def __init__(self, obs_space, act_space, hidden_size, lr, train_device):
        super().__init__()

        self.init_(obs_space, act_space, hidden_size, lr, train_device)
    
    def init_(self, obs_space, act_space, hidden_size, lr, train_device):
        assert type(obs_space) == Box and type(act_space) == Box
        assert len(obs_space.shape) == len(act_space.shape) == 1

        self.network = MLPNetwork(num_inputs=obs_space.shape[0] + act_space.shape[0], num_outputs=1, hidden_size=hidden_size)
        # we put the parameters on the right device before creating the optimizer
        self.to(train_device)
        self.optim = Adam(params=self.parameters(), lr=lr)

    def __call__(self, obs, actions):
        return self.network(torch.cat((obs, actions), dim=1))

    @property
    def device(self):
        return next(self.parameters()).device
    
    @staticmethod
    def update_target_soft(target, source, tau):
        soft_update(target=target.network, source=source.network, tau=tau)

    @staticmethod
    def update_target_hard(target, source):
        Q.update_target_soft(target, source, tau=1.)

    def get_state_dict(self):
        return {'network': self.network.state_dict(),
                'optim': self.optim.state_dict()}

    def do_load_state_dict(self, state_dict):
        self.network.load_state_dict(state_dict['network'])
        self.optim.load_state_dict(state_dict['optim'])
    
class QDiscrete(Q):
    def __init__(self, obs_space, act_space, hidden_size, lr, train_device):
        super().__init__(obs_space, act_space, hidden_size, lr, train_device)

    
    def init_(self, obs_space, act_space, hidden_size, lr, train_device):
        assert type(obs_space) == Box and type(act_space) == Discrete
        assert len(obs_space.shape) == 1

        self.network = MLPNetwork(num_inputs=obs_space.shape[0], num_outputs=act_space.n, hidden_size=hidden_size)
        # we put the parameters on the right device before creating the optimizer
        self.to(train_device)
        self.optim = Adam(params=self.parameters(), lr=lr)

    
    def __call__(self, obs, actions):
        return self.network(obs).gather(dim=1, index=actions)

    def action_values(self, obs):
        return self.network(obs)

class DoubleQ(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.init_(**kwargs)

    def init_(self, **kwargs):
        self.q1 = Q(obs_space=kwargs['obs_space'], act_space=kwargs['act_space'], hidden_size=kwargs['hidden_size'],
                     lr=kwargs['lr'], train_device=kwargs['train_device'])
        self.q2 = Q(obs_space=kwargs['obs_space'], act_space=kwargs['act_space'], hidden_size=kwargs['hidden_size'],
                     lr=kwargs['lr'], train_device=kwargs['train_device'])
    
    def __call__(self, obs, actions):
        return self.q1(obs, actions), self.q2(obs, actions)

    @property
    def device(self):
        assert self.q1.device == self.q2.device
        return self.q1.device

    @staticmethod
    def update_target_soft(target, source, tau):
        Q.update_target_soft(target=target.q1, source=source.q1, tau=tau)
        Q.update_target_soft(target=target.q2, source=source.q2, tau=tau)
    
    @staticmethod
    def update_target_hard(target, source):
        Q.update_target_hard(target=target.q1, source=source.q1)
        Q.update_target_hard(target=target.q2, source=source.q2)

    def get_state_dict(self):
        return {'q1': self.q1.get_state_dict(), 
                'q2': self.q2.get_state_dict()}
    
    def do_load_state_dict(self, state_dict):
        self.q1.do_load_state_dict(state_dict['q1'])
        self.q2.do_load_state_dict(state_dict['q2'])

class DoubleQDiscrete(DoubleQ):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def init_(self, obs_space, act_space, hidden_size, lr, train_device):
        self.q1 = QDiscrete(obs_space=obs_space, act_space=act_space, hidden_size=hidden_size, lr=lr, train_device=train_device)
        self.q2 = QDiscrete(obs_space=obs_space, act_space=act_space, hidden_size=hidden_size, lr=lr, train_device=train_device)

    def action_values(self, obs):
        return self.q1.action_values(obs), self.q2.action_values(obs)