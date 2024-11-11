import torch.nn.functional as F
import torch
from torch import logit, nn, optim
import math
from nop import NOP

from offline_marl.utils import ml
from offline_marl.utils.ml import soft_update

MIN_LOG_STD = -6
epsilon = 1e-5
MAX_LOG_STD = 0


class TD3Actor(nn.Module):
        
    def __init__(self, **kwargs):
        super().__init__()
                
        self.l1 = nn.Linear(kwargs['obs_space'].shape[0], kwargs['hidden_size'])
        self.l2 = nn.Linear(kwargs['hidden_size'], kwargs['hidden_size'])
        self.l3 = nn.Linear(kwargs['hidden_size'], kwargs['act_space'].shape[0])

        self.to(kwargs['train_device'])
        self.optim = optim.Adam(params=self.parameters(), lr=kwargs['lr'])

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        return  torch.tanh(self.l3(a))

    def act(self, state, sample=False, return_log_pi=False):

        if sample:
            raise NotImplementedError
        
        if return_log_pi:
            return self.forward(state), None
        else:
            return self.forward(state)
    
    @staticmethod
    def update_target_soft(target, source, tau):
        soft_update(target=target, source=source, tau=tau)

    @staticmethod
    def update_target_hard(target, source):
        TD3Actor.update_target_soft(target, source, tau=1.)

    @property
    def device(self):
        return next(self.parameters()).device
        
    def get_state_dict(self):
        return {'self': self.state_dict(), 
                'optim': self.optim.state_dict()}

    def do_load_state_dict(self, state_dict):
        self.load_state_dict(state_dict['self'])
        self.optim.load_state_dict(state_dict['optim'])


class GaussianPolicy(nn.Module):

    # This a continuous policy

    def __init__(self, **kwargs):
        super().__init__()

        self.state_dependent_std = kwargs['state_dependent_std']

        self.fc1 = nn.Linear(kwargs['obs_space'].shape[0], kwargs['hidden_size'])
        self.fc2 = nn.Linear(kwargs['hidden_size'], kwargs['hidden_size'])

        self.mean_layer = nn.Linear(kwargs['hidden_size'], kwargs['act_space'].shape[0])

        if kwargs['set_final_bias']:
            self._set_weight_and_bias(self.mean_layer)

        # seems that https://github.com/ikostrikov/implicit_q_learning/blob/3f4b63498583015ff5f54140bd68248dffc668ee/policy.py#L56
        # multiplies log_stds with temperature
        if self.state_dependent_std:
            self.log_std_layer = nn.Linear(kwargs['hidden_size'], kwargs['act_space'].shape[0])
            
            if kwargs['set_final_bias']:
                self._set_weight_and_bias(self.log_std_layer)
        
        else:
            self.log_std = nn.Parameter(torch.zeros(1, kwargs['act_space'].shape[0]))

        # we put the parameters on the right device before creating the optimizer
        self.to(kwargs['train_device'])

        self.optim = optim.Adam(params=self.parameters(), lr=kwargs['lr'])

        if kwargs.get('cosine_scheduler_max_steps', None) is not None:
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optim, T_max=kwargs['cosine_scheduler_max_steps'])
        else:
            self.scheduler = NOP()

        self.action_squashing = kwargs['action_squashing']
        if self.action_squashing == 'tanh':
            self.squash_action = lambda x: torch.tanh(torch.clamp(x, min=math.atanh(-1. + epsilon), max=math.atanh(1. - epsilon)))
            self.inverse_squash_action = lambda x: torch.atanh(torch.clamp(x, min=-1. + epsilon, max=1. - epsilon))

        elif self.action_squashing == 'none':
            self.squash_action = lambda x: x
        else:
            raise NotImplementedError

    def _set_weight_and_bias(self, layer):
        layer.weight.data.mul_(0.1)
        layer.bias.data.mul_(0.0)

    def __call__(self, obs):
        # this returns logits for mean and std
        h = F.relu(self.fc1(obs))
        h = F.relu(self.fc2(h))
        mean_logits = self.mean_layer(h)

        if self.state_dependent_std:
            log_std_logits = self.log_std_layer(h)
        else:
            log_std_logits = self.log_std


        return (mean_logits, log_std_logits)

    def act(self, obs, sample, return_log_pi, return_entropy=False):
        return self.act_from_logits(self(obs), sample, return_log_pi, return_entropy)

    def get_log_prob_from_obs_action_pairs(self, action, obs, return_entropy=False):

        mean, log_std = GaussianPolicy.get_mean_logstd_from_logits(self(obs))
        if self.action_squashing == 'tanh':
            action = self.inverse_squash_action(action)

        log_prob = self.log_prob_density(action, mean, log_std)

        if return_entropy:
            entropy = GaussianPolicy.entropy(log_std)
            return log_prob, entropy
        else:
            return log_prob

    def act_from_logits(self, logits, sample, return_log_pi, return_entropy):

        return self.get_action(logits, sample, return_log_pi, return_entropy)

        # if return_log_pi:
        #     return self.get_action_and_log_prob(logits, sample)
        # else:
        #     return self.get_action(logits, sample)

    @staticmethod
    def entropy(log_std):
        if self.action_squashing == 'tanh':
            return NotImplementedError("Not sure there is a closed form entropy expression for tanh gaussians")

        assert len(log_std.shape) == 2

        # we drop all the constants
        entropy = log_std.sum(1, keepdim=True)

        return entropy

    @staticmethod
    def get_mean_logstd_from_logits(logits):
        mean, log_std = logits
        log_std = torch.clamp(log_std, min=MIN_LOG_STD, max=MAX_LOG_STD)
        return mean, log_std

    @staticmethod
    def action_from_mean_log_std(mean, log_std, sample):

        if not sample:
            action = mean
        else:
            noise = torch.normal(
                torch.zeros_like(mean), torch.ones_like(mean)
            ) * torch.exp(log_std)
            action = mean + noise
        
        return action


    def log_prob_density(self, action, mean, log_std):
        z = (action - mean) / log_std.exp()

        if self.action_squashing == 'tanh':
            log_prob = (- 0.5 * (2 * log_std + z ** 2 + math.log(2 * math.pi)) \
                        - torch.log(1 - self.squash_action(action) ** 2 + epsilon)).sum(dim=-1, keepdim=True)
        elif self.action_squashing == 'none':
            log_prob = (- 0.5 * (2 * log_std + z ** 2 + math.log(2 * math.pi))).sum(dim=-1, keepdim=True)
        else:
            raise NotImplementedError

        if torch.isnan(log_prob).any():
            raise ValueError("nan found in policy log-prob")

        return log_prob

    def get_action(self, logits, sample, retrun_log_pi, return_entropy):
        mean, log_std = GaussianPolicy.get_mean_logstd_from_logits(logits)
        action = GaussianPolicy.action_from_mean_log_std(mean, log_std, sample)

        if retrun_log_pi:
            log_prob = self.log_prob_density(action, mean, log_std)

            if return_entropy:

                entropy = GaussianPolicy.entropy(log_std)

                return self.squash_action(action), log_prob, entropy
            
            else:
                return self.squash_action(action), log_prob
        else:
            return self.squash_action(action)

    @property
    def device(self):
        return next(self.parameters()).device
        
    def get_state_dict(self):
        return {'self': self.state_dict(), 
                'optim': self.optim.state_dict(),
                'scheduler': self.scheduler.state_dict()}
    
    def do_load_state_dict(self, state_dict):
        self.load_state_dict(state_dict['self'])
        self.optim.load_state_dict(state_dict['optim'])
        self.scheduler.load_state_dict(state_dict['scheduler'])


class CategoricalPolicy(nn.Module):
    # this is a discrete policy

    def __init__(self, **kwargs):
        super().__init__()

        self.fc1 = nn.Linear(kwargs['obs_space'].shape[0], kwargs['hidden_size'])
        self.fc2 = nn.Linear(kwargs['hidden_size'], kwargs['hidden_size'])
        self.out_layer = nn.Linear(kwargs['hidden_size'], kwargs['act_space'].n)

        # we put the parameters on the right device before creating the optimizer
        self.to(kwargs['train_device'])

        self.optim = optim.Adam(params=self.parameters(), lr=kwargs['lr'])

        if kwargs.get('cosine_scheduler_max_steps', None) is not None:
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optim, T_max=kwargs['cosine_scheduler_max_steps'])
        else:
            self.scheduler = NOP()
    
    def __call__(self, obs):
        h = F.relu(self.fc1(obs))
        h = F.relu(self.fc2(h))
        logits = self.out_layer(h)
        return logits

    def act(self, obs, legal_move = None, sample=True, return_log_pi=False):
        logits = self(obs)
        
        if legal_move is None:
            legal_move = torch.ones_like(logits)
            
        # we remove illegal moves
        assert logits.size() == legal_move.size()
        logits = logits - (1 - legal_move) * 1e10

        return self.act_from_logits(logits=logits, sample=sample, return_log_pi=return_log_pi)

    def get_log_prob_from_obs_action_pairs(self, action, obs):
        logits = self(obs=obs)
        return self.log_prob_density(x=action, logits=logits)

    def act_from_logits(self, logits, sample, return_log_pi):
        action = self.get_action(logits=logits, sample=sample)
        if return_log_pi:
            log_pi = self.log_prob_density(x=action, logits=logits)
            return action, log_pi
        else:
            return action

    def get_action(self, logits, sample):
        if not sample:
            action = logits.argmax(-1, keepdim=True)
        else:
            action = ml.gumbel_softmax(logits=logits)
            action = action.argmax(-1, keepdim=True)
        return action

    def log_prob_density(self, x, logits):
        log_probas = F.log_softmax(logits, dim=-1)

        if not 'int' in str(x.dtype):
            x = x.type(torch.int64)

        log_proba_of_sample = log_probas.gather(dim=1, index=x)
        return log_proba_of_sample

    @property
    def device(self):
        return next(self.parameters()).device
        
    def get_state_dict(self):
        return {'self': self.state_dict(), 
                'optim': self.optim.state_dict(),
                'scheduler': self.scheduler.state_dict()}
    
    def do_load_state_dict(self, state_dict):
        self.load_state_dict(state_dict['self'])
        self.optim.load_state_dict(state_dict['optim'])
        self.scheduler.load_state_dict(state_dict['scheduler'])