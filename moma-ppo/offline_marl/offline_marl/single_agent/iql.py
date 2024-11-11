import warnings
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from gym.spaces import Box
from nop import NOP


from offline_marl.utils.base_algo import BaseAlgo
from offline_marl.single_agent.critics import DoubleQ, V, DoubleQDiscrete
from offline_marl.single_agent.actors import GaussianPolicy, CategoricalPolicy
import offline_marl.utils.histories as histories
from offline_marl.utils.histories import HistoryEncoder
from offline_marl.utils.ml import batch_as_tensor, batch_to_device
from offline_marl.utils.ml import hard_update as update_target_hard
from offline_marl.utils.ml import soft_update as update_target_soft


class IQLLearner(BaseAlgo, nn.Module):
    def __init__(self, **kwargs):

        nn.Module.__init__(self)
        BaseAlgo.__init__(self)
        params = kwargs.copy()
        params.update({'set_final_bias': False, 'action_squashing': 'tanh', 'state_dependent_std': True})
        self._init(params)
    
    def _init(self, kwargs):
        
        if kwargs['alg_name'] == 'bc':
            self.bc_only = True
        else:
            self.bc_only = False

        self.discount_factor = kwargs['discount_factor']
        self.expectile = kwargs['expectile']
        self.awr_temperature = kwargs['awr_temperature']
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
        params = kwargs.copy()
        params.update({'obs_space': obs_space})
        self.doubleQ, self.doubleQtarget, self.V, self.policy = self.make_actor_critic(**params)
        
        # to save and reload
        self.init_dict = kwargs
        self.name = kwargs.get('learner_name', 'iql')

        # data to monitor
        self.train_metrics = {'loss_v', 'loss_pi', 'loss_q', 'mean_v', 'min_v', 'max_v', 'mean_q', 'min_q', 'max_q'}
        self.evaluation_metrics = {'return', 'length', 'training_step', 'mean_actions'}
        self.metrics_to_record = self.train_metrics | self.evaluation_metrics

    @staticmethod
    def make_history_encoder(num_in, num_out, hidden_size, lr, reduce_op, history_len, train_device, weight_decay=0, grad_norm_clip=1e7):
        encoder = HistoryEncoder(num_in=num_in, num_out=num_out,
        hidden_size=hidden_size, lr=lr, reduce_op=reduce_op, max_history_len=history_len,
        train_device=train_device, weight_decay=weight_decay)

        with torch.no_grad():
            encodertarget = HistoryEncoder(num_in=num_in, num_out=num_out,
            hidden_size=hidden_size, lr=lr, reduce_op=reduce_op, max_history_len=history_len,
            train_device=train_device, weight_decay=weight_decay)

            update_target_hard(encodertarget, encoder)

        # we extend observation space with memory embedding size 
        embeding_vec = np.ones(num_out)
        
        return encoder, encodertarget, embeding_vec

    @staticmethod
    def make_actor_critic(**kwargs):
        
        params = kwargs.copy()
        params.update({'lr': kwargs['lr_q']})
        
        doubleQ = DoubleQ(**params)
        with torch.no_grad():
            doubleQtarget = DoubleQ(**params)
            DoubleQ.update_target_hard(doubleQtarget, doubleQ)

        params = kwargs.copy()
        params.update({'lr': kwargs['lr_v']})
        Vnet = V(**params)

        params = kwargs.copy()
        params.update({'lr': kwargs['lr_pi']})
        policy = GaussianPolicy(**params)

        return doubleQ, doubleQtarget, Vnet, policy

    def concat_embedings_to_obs(self, batch, target, next_obs=False):
        if target:
            memory_encoder = self.memorytarget
        else:
            memory_encoder = self.memory
        return histories.concat_embedings_to_obs(batch, memory_encoder, self.memory_len, next_obs)

    def update_v(self, batch):
        
        if self.bc_only:
            return 0. 

        # comuptes target Q(s,a) with double clipped Q
        with torch.no_grad():
            target_observations = self.concat_embedings_to_obs(batch, target=True)
            target_q_val1, target_q_val2 = self.doubleQtarget(target_observations, batch['actions'])
            target_q_val = torch.minimum(target_q_val1, target_q_val2).detach()

        observations = self.concat_embedings_to_obs(batch, target=False)

        # Computes V(s)
        vs = self.V(observations)

        # Computes loss |tau - I(Q(s,a) - V(s)<0)|(Q(s,a)-V(s))^2
        loss = (torch.abs(self.expectile - (target_q_val < vs).float())*(target_q_val - vs)**2).mean(0)

        # cleans potentials left-over gradients and updates networks
        self.V.optim.zero_grad()
        self.memory.optim.zero_grad()
        loss.backward()
        self.V.optim.step()
        self.memory.optim.step()

        return loss.detach().cpu().numpy()

    def update_q(self, batch):

        if self.bc_only:
            return 0. 

        observations = self.concat_embedings_to_obs(batch, target=False)
        
        # computes Temporal Difference target r(s,a) + gamma*V(s')
        with torch.no_grad():
            target_next_observations = self.concat_embedings_to_obs(batch, target=True, next_obs=True)
            target = batch['rewards'] + self.discount_factor*batch['masks']*self.V(target_next_observations)
        

        observations = self.concat_embedings_to_obs(batch, target=False)

        # computes Q(s,a) with double clipped Q
        Qsa1, Qsa2 = self.doubleQ(observations, batch['actions'])
        
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

    def update_pi(self, batch):

        # computes advantage weights -exp(beta(Q(s,a)-V(s)))
        with torch.no_grad():

            if self.bc_only:
                weight = -1.
            else:
                target_observations = self.concat_embedings_to_obs(batch, target=True)
                target_q1, target_q2 = self.doubleQtarget(target_observations, batch['actions'])
                target_q = torch.minimum(target_q1, target_q2)
                vs = self.V(target_observations)
                # we clip advantages to 100
                weight = -torch.clamp(torch.exp(self.awr_temperature*(target_q-vs)), max=100.0)
        
        observations = self.concat_embedings_to_obs(batch, target=False)

        log_pi = self.policy.get_log_prob_from_obs_action_pairs(batch['actions'], observations)

        loss = (weight * log_pi).mean(0)

        # clean potentials left-over gradients and updates networks
        self.policy.optim.zero_grad()
        self.memory.optim.zero_grad()
        loss.backward()
        self.policy.optim.step()

        if self.memory_backprop_actor:
            self.memory.optim.step()

        # updates cosine lr schedule
        self.policy.scheduler.step()

        return loss.detach().cpu().numpy()

    def update(self, dataset, batch_size, train_device, **kwargs):
        batch = dataset.sample(batch_size)
        return self.update_from_batch(batch, train_device, **kwargs)

    def update_from_batch(self, batch, train_device, **kwargs):
        # from https://github.com/ikostrikov/implicit_q_learning/blob/master/learner.py
        # with same batch the update goes
        # 1. update V-network with L_V
        # 2. use new V-network to update policy with AWR
        # 3. update Q-network with new V-network and L_Q
        # 4. update target Q-network

        # put models on training device
        self.to(train_device)

        batch = batch_to_device(batch_as_tensor(batch), train_device)
        
        # 1. update v_network with L_V
        loss_v = self.update_v(batch)

        # 2. use new V-network to update policy with AWR
        loss_pi = self.update_pi(batch)

        # 3. update Q-network with new V-network and L_Q
        loss_q = self.update_q(batch)

        # 4. update target networks
        update_target_soft(target=self.doubleQtarget, source=self.doubleQ, tau=self.target_update_coef)
        update_target_soft(target=self.memorytarget, source=self.memory, tau=self.target_update_coef)

        return {'loss_v': loss_v, 'loss_pi': loss_pi, 'loss_q': loss_q}

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

    def evaluate(self, env, max_eval_episode, rollout_device=torch.device('cpu')):

        # puts model on rollout device
        self.to(rollout_device)

        stats = {'return': [], 'length': [], 'frames': []}


        for ep in tqdm(range(max_eval_episode), desc="evaluation"):
            
            observation, done = env.reset(), False
            self.reset_histories()

            # env have a timelimit wrapper of 1000 max-steps
            # it also has a wrapper that computes episode return and length
            while not done:
                torch_observation = torch.as_tensor(observation, device=rollout_device).unsqueeze(0)

                # add memory and context embeddings
                torch_observation = self.process_current_histories(torch_observation)

                # take action
                action = self.policy.act(torch_observation, sample=False, return_log_pi=False).squeeze(0).detach().cpu().numpy()

                # get reward
                next_observation, reward, done, info = env.step(action)

                # update histories with obs, action and reward
                self.append_histories(observation, action)

                # moves to next interaction
                observation = next_observation
                
            for k in stats.keys():
                if k == 'frames':
                    if len(stats[k]) >= env.n_ep_per_gifs:
                        continue
                stats[k].append(info['episode'][k])

        for k, v in stats.items():
            if k == 'frames':
                v = [np.asarray(ep) for ep in v]
                stats[k] = np.concatenate(v, axis=0)
            else:
                stats[k] = np.mean(v)
        
        return stats
    
    def get_state_dict(self):
        return {'doubleQ': self.doubleQ.get_state_dict(),
                'doubleQtarget': self.doubleQtarget.get_state_dict(),
                'V': self.V.get_state_dict(),
                'policy': self.policy.get_state_dict(), 
                'memory': self.memory.get_state_dict(),
                'memorytarget': self.memorytarget.get_state_dict()}
    
    def do_load_state_dict(self, state_dict):
        self.doubleQ.do_load_state_dict(state_dict['doubleQ'])
        self.doubleQtarget.do_load_state_dict(state_dict['doubleQtarget'])
        self.V.do_load_state_dict(state_dict['V'])
        self.policy.do_load_state_dict(state_dict['policy'])
        self.memory.do_load_state_dict(state_dict['memory']) 
        self.memorytarget.do_load_state_dict(state_dict['memorytarget'])

class IQLLearnerDiscrete(IQLLearner):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    @staticmethod
    def make_actor_critic(obs_space, act_space, hidden_size, lr_q, lr_v, lr_pi, train_device, max_training_step, **kwargs):
        doubleQ = DoubleQDiscrete(obs_space=obs_space, act_space=act_space, hidden_size=hidden_size, lr=lr_q, train_device=train_device)
        with torch.no_grad():
            doubleQtarget = DoubleQDiscrete(obs_space=obs_space, act_space=act_space, hidden_size=hidden_size, lr=lr_q, train_device=train_device)
            DoubleQDiscrete.update_target_hard(doubleQtarget, doubleQ)
        
        Vnet = V(obs_space=obs_space, hidden_size=hidden_size, lr=lr_v, train_device=train_device)
        
        cosine_max_step = None

        policy = CategoricalPolicy(obs_space=obs_space, act_space=act_space, hidden_size=hidden_size, lr=lr_pi, 
                                    cosine_scheduler_max_steps=cosine_max_step, train_device=train_device)

        return doubleQ, doubleQtarget, Vnet, policy

    def append_histories(self, observation, action):
        one_hot_action = np.zeros(self.init_dict['act_space'].n)
        one_hot_action[action] = 1.
        self.memory.append_history((observation, one_hot_action))

    # The policy update is different because we have legal_moves
    def update_pi(self, batch):

        if self.bc_only:
            weight = -1.
        else:
            # compus advantage weights -exp(beta(Q(s,a)-V(s)))
            with torch.no_grad():
                target_observations = self.concat_embedings_to_obs(batch, target=True)
                target_q1, target_q2 = self.doubleQtarget(target_observations, batch['actions'])
                target_q = torch.minimum(target_q1, target_q2)
                vs = self.V(target_observations)
                # we clip advantages to 100
                weight = -torch.clamp(torch.exp(self.awr_temperature*(target_q-vs)), max=100.0)
        
        observations = self.concat_embedings_to_obs(batch, target=False)

        logits = self.policy(observations)

        # we make non-legal moves very unlikely
        legal_moves = batch['legal_moves']
        assert logits.size() == legal_moves.size()
        logits = logits - (1 - legal_moves) * 1e10

        log_pi = torch.nn.functional.log_softmax(logits, dim=1).gather(dim=1, index=batch['actions'])

        loss = (weight * log_pi).mean(0)

        # clean potentials left-over gradients and updates networks
        self.policy.optim.zero_grad()
        self.memory.optim.zero_grad()
        loss.backward()
        self.policy.optim.step()

        if self.memory_backprop_actor:
            self.memory.optim.step()

        # updates cosine lr schedule
        self.policy.scheduler.step()

        return loss.detach().cpu().numpy()

    def evaluate(self, env, max_eval_episode, rollout_device=torch.device('cpu')):

        # puts model on rollout device
        self.to(rollout_device)

        stats = {'return': [], 'length': []}          

        for ep in tqdm(range(max_eval_episode), desc="evaluation"):

            observations, infos = env.reset()
            done = False

            self.reset_histories()

            # env has a timelimit wrapper of 1000 max-steps
            # it also has a wrapper that computes episode return and length
            while not done:
                torch_observation = env.to_torch(observations, device=rollout_device).unsqueeze(0)
                torch_legal_move = env.to_torch(infos['legal_moves'], device=rollout_device).unsqueeze(0)

                # add memory embedding
                torch_observation = self.process_current_histories(torch_observation)

                action = self.policy.act(torch_observation, legal_move=torch_legal_move, sample=False, return_log_pi=False).squeeze(0).detach().cpu().numpy()


                next_observations, reward, done, infos = env.step(action)
                self.append_histories(observations, action)

                observations = next_observations

            for k in stats.keys():
                stats[k].append(infos['episode'][k])

        for k, v in stats.items():
            stats[k] = np.mean(v)
        
        return stats
    