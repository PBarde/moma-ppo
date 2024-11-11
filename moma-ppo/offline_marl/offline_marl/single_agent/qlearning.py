from gym.spaces import Box
from nop import NOP
import numpy as np
import torch

from offline_marl.single_agent.iql import IQLLearnerDiscrete
from offline_marl.single_agent.critics import DoubleQDiscrete


class QLearner(IQLLearnerDiscrete):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def _init(self, kwargs):
        self.discount_factor = kwargs['discount_factor']
        self.target_update_coef = kwargs['target_update_coef']
        self.memory_backprop_actor = kwargs['memory_backprop_actor']

        actions_dim = kwargs['act_space'].shape[0] if isinstance(kwargs['act_space'], Box) else kwargs['act_space'].n

        # Memory encoder
        self.memory_len = kwargs['memory_len']
        if kwargs.get('memory_len', 0) > 0:
            self.memory, self.memorytarget, memory_embedding_vec = self.make_history_encoder(num_in=kwargs['obs_space'].shape[0] + actions_dim,
                                                                                                    num_out=kwargs['memory_out_size'], 
                                                                                                    hidden_size=kwargs['hidden_size'],
                                                                                                    lr=kwargs['lr_memory'],
                                                                                                    reduce_op=kwargs['memory_op'],
                                                                                                    history_len=kwargs['memory_len'],
                                                                                                    train_device=kwargs['train_device'])
            obs_space = Box(np.concatenate((kwargs['obs_space'].low, memory_embedding_vec)), np.concatenate((kwargs['obs_space'].high, memory_embedding_vec)))
        else:
            self.memory = NOP()
            self.memorytarget = NOP()
            obs_space = kwargs['obs_space']

        # Actor-Critic
        self.doubleQ, self.doubleQtarget = self.make_actor_critic(obs_space=obs_space,
                                                                act_space=kwargs['act_space'],
                                                                hidden_size=kwargs['hidden_size'],
                                                                lr_q=kwargs['lr_q'],
                                                                train_device=kwargs['train_device'])
        
        # to save and reload
        self.init_dict = kwargs
        self.name = kwargs.get('learner_name', 'iql')

        # data to monitor
        self.train_metrics = {'loss_q', 'mean_q', 'min_q', 'max_q'}
        self.evaluation_metrics = {'return', 'length', 'training_step', 'mean_actions'}
        self.metrics_to_record = self.train_metrics | self.evaluation_metrics

    @staticmethod
    def make_actor_critic(obs_space, act_space, hidden_size, lr_q, train_device):
        doubleQ = DoubleQDiscrete(obs_space=obs_space, act_space=act_space, hidden_size=hidden_size, lr=lr_q, train_device=train_device)
        with torch.no_grad():
            doubleQtarget = DoubleQDiscrete(obs_space=obs_space, act_space=act_space, hidden_size=hidden_size, lr=lr_q, train_device=train_device)
            DoubleQDiscrete.update_target_hard(doubleQtarget, doubleQ)

        return doubleQ, doubleQtarget

    def update_v(self, batch):
        return NotImplementedError

    def update_pi(self, batch):
        return NotImplementedError
    
    def update_q(self, batch):

        observations = self.concat_embedings_to_obs(batch, target=False)
        
        # computes Temporal Difference target r(s,a) + gamma*V(s')
        with torch.no_grad():
            target_next_observations = self.concat_embedings_to_obs(batch, target=True, next_obs=True)
            target_next_q1, target_next_q2 = self.doubleQtarget.action_values(target_next_observations)
            target_next_q = torch.min(target_next_q1, target_next_q2)
            max_target_next_q = torch.max(target_next_q, dim=1).values.unsqueeze(0)
            target = batch.rewards + self.discount_factor*batch.masks*max_target_next_q
        
        observations = self.concat_embedings_to_obs(batch, target=False)

        # computes Q(s,a) with double clipped Q
        Qsa1, Qsa2 = self.doubleQ(observations, batch.actions)
        
        # loss is computed for both Q: (r(s,a) + gamma*V(s') - Q(s,a))**2
        loss = ((target - Qsa1)**2 + (target - Qsa2)**2).mean(0)

        # clean potentials left-over gradients and updates networks
        self.doubleQ.q1.optim.zero_grad()
        self.doubleQ.q2.optim.zero_grad()
        self.memory.optim.zero_grad()
        loss.backward()
        self.doubleQ.q1.optim.step()
        self.doubleQ.q2.optim.step()
        self.memory.optim.step()

        return loss.detach().cpu().numpy()

    def update_from_batch(self, batch, train_device, **kwargs):
        # from https://github.com/ikostrikov/implicit_q_learning/blob/master/learner.py
        # with same batch the update goes
        # 1. update V-network with L_V
        # 2. use new V-network to update policy with AWR
        # 3. update Q-network with new V-network and L_Q
        # 4. update target Q-network

        # put models on training device
        self.to(train_device)

        batch = self.move_batch_to_device(batch, train_device)

        # update Q-network with new V-network and L_Q
        loss_q = self.update_q(batch)

        # 4. update target networks
        self.doubleQ.update_target_soft(target=self.doubleQtarget, source=self.doubleQ, tau=self.target_update_coef)
        self.memory.update_target_soft(target=self.memorytarget, source=self.memory, tau=self.target_update_coef)

        return {'loss_q': loss_q}

    @property
    def policy(self):
        return self

    def act(self, obs, legal_move, sample, return_log_pi):
        q1, q2 = self.doubleQ.action_values(obs)

        q_values = torch.min(q1,q2)
        
        assert q_values.shape == legal_move.shape
        q_values = q_values - (1 - legal_move) * 1e10
        return torch.max(q_values, dim=1).indices.unsqueeze(1)

    def get_state_dict(self):
        return {'doubleQ': self.doubleQ.get_state_dict(),
                'doubleQtarget': self.doubleQtarget.get_state_dict(),
                'memory': self.memory.get_state_dict(),
                'memorytarget': self.memorytarget.get_state_dict()}
    
    def do_load_state_dict(self, state_dict):
        self.doubleQ.do_load_state_dict(state_dict['doubleQ'])
        self.doubleQtarget.do_load_state_dict(state_dict['doubleQtarget'])
        self.memory.do_load_state_dict(state_dict['memory'])
        self.memorytarget.do_load_state_dict(state_dict['memorytarget'])
