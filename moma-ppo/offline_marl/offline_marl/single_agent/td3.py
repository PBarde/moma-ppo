
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from gym.spaces import Box
from nop import NOP
import torch.nn.functional as F

from offline_marl.utils.base_algo import BaseAlgo
from offline_marl.single_agent.critics import DoubleQ
from offline_marl.single_agent.actors import TD3Actor
import offline_marl.utils.histories as histories
from offline_marl.utils.histories import HistoryEncoder
from offline_marl.utils.ml import batch_as_tensor, batch_to_device
from offline_marl.utils.ml import hard_update as update_target_hard
from offline_marl.utils.ml import soft_update as update_target_soft


class TD3Learner(BaseAlgo, nn.Module):
    def __init__(self, **kwargs):

        nn.Module.__init__(self)
        BaseAlgo.__init__(self)
        params = kwargs.copy()

        self.discount_factor = kwargs['discount_factor']
        self.target_update_coef = kwargs['target_update_coef']
        self.memory_backprop_actor = kwargs['memory_backprop_actor']

        self.policy_noise = kwargs['td3_policy_noise']
        self.noise_clip = kwargs['td3_noise_clip']
        self.policy_freq = kwargs['td3_policy_freq']
        self.total_it = 0

        self.bc_coeff = kwargs['td3_bc_coeff']

        self.omar_coeff = kwargs['td3_omar_coeff']
        self.cql_alpha = kwargs['td3_omar_cql_alpha']
        self.cql_lse_temp = kwargs['td3_omar_cql_lse_temp']
        self.cql_num_sampled_actions = kwargs['td3_omar_cql_num_sampled_actions']
        self.cql_sample_noise_level = kwargs['td3_omar_cql_sample_noise_level']

        self.omar_iters = kwargs['td3_omar_iters']
        self.init_omar_mu = kwargs['td3_init_omar_mu']
        self.init_omar_sigma = kwargs['td3_init_omar_sigma']
        self.omar_num_samples = kwargs['td3_omar_num_samples']
        self.omar_num_elites = kwargs['td3_omar_num_elites']

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
        self.doubleQ, self.doubleQtarget, self.policy, self.policytarget = self.make_actor_critic(**params)
        
        # to save and reload
        self.init_dict = kwargs
        self.name = kwargs.get('learner_name', 'td3')

        # data to monitor
        self.train_metrics = { 'loss_q', 'mean_q', 'min_q', 'max_q', 'loss_pi', 'loss_td3', 'loss_bc', 'loss_omar', 'lmbda', 'loss_cql'}
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
        params.update({'lr': kwargs['lr_pi']})
        policy = TD3Actor(**params)
        with torch.no_grad():
            policytarget = TD3Actor(**params)
            policy.update_target_hard(policytarget, policy)

        return doubleQ, doubleQtarget, policy, policytarget

    def concat_embedings_to_obs(self, batch, target, next_obs=False):
        if target:
            memory_encoder = self.memorytarget
        else:
            memory_encoder = self.memory
        return histories.concat_embedings_to_obs(batch, memory_encoder, self.memory_len, next_obs)

    def cql_calc_gaussian_pdf(self, samples, mu=0):
        pdfs = 1 / (self.cql_sample_noise_level * np.sqrt(2 * np.pi)) * torch.exp( - (samples - mu)**2 / (2 * self.cql_sample_noise_level**2) )
        pdf = torch.prod(pdfs, dim=-1)
        return pdf

    def cql_get_policy_actions(self, state, network):
        if not self.cql_alpha > 0:
            raise NotImplementedError
        if not self.omar_coeff > 0:
            raise NotImplementedError
         
        action = network(state)

        formatted_action = action.unsqueeze(1).repeat(1, self.cql_num_sampled_actions, 1).view(action.shape[0] * self.cql_num_sampled_actions, action.shape[1])

        random_noises = torch.FloatTensor(formatted_action.shape[0], formatted_action.shape[1])

        random_noises = random_noises.normal_() * self.cql_sample_noise_level
        random_noises_log_pi = self.cql_calc_gaussian_pdf(random_noises).view(action.shape[0], self.cql_num_sampled_actions, 1).to(self.device)
        random_noises = random_noises.to(self.device)

        noisy_action = (formatted_action + random_noises).clamp(-1., 1.)

        return noisy_action, random_noises_log_pi
    
    def update_q(self, batch):

        observations = self.concat_embedings_to_obs(batch, target=False)
        
        # computes Temporal Difference target r(s,a) + gamma*V(s')
        with torch.no_grad():
            target_next_observations = self.concat_embedings_to_obs(batch, target=True, next_obs=True)
            
            # Select action according to policy and add clipped noise
            noise = (
                    torch.randn_like(batch['actions']) * self.policy_noise
                ).clamp(-self.noise_clip, self.noise_clip)
                
            next_action = (
                self.policytarget(target_next_observations) + noise
            ).clamp(-1., 1.)

            target_Q1, target_Q2 = self.doubleQtarget(target_next_observations, next_action)
            target_Q = torch.min(target_Q1, target_Q2)

            target = batch['rewards'] + self.discount_factor*batch['masks']*target_Q
        

        observations = self.concat_embedings_to_obs(batch, target=False)

        # computes Q(s,a) with double clipped Q
        Qsa1, Qsa2 = self.doubleQ(observations, batch['actions'])
        
        # loss is computed for both Q: (r(s,a) + gamma*V(s') - Q(s,a))**2
        loss = ((target - Qsa1)**2 + (target - Qsa2)**2).mean(0)


        if self.omar_coeff > 0: 
            if self.cql_alpha > 0:

                actions = batch['actions']

                # CQL copied from https://github.com/ling-pan/OMAR/blob/master/algorithms/maddpg.py

                formatted_obs = observations.unsqueeze(1).repeat(1, self.cql_num_sampled_actions, 1).view(-1, observations.shape[1])

                random_action = (torch.FloatTensor(actions.shape[0] * self.cql_num_sampled_actions, actions.shape[1]).uniform_(-1, 1)).to(self.device)
                random_action_log_pi = np.log(0.5 ** random_action.shape[-1])

                curr_action, curr_action_log_pi = self.cql_get_policy_actions(observations, self.policy)
                new_curr_action, new_curr_action_log_pi = self.cql_get_policy_actions(target_next_observations, self.policy)

                random_Q1, random_Q2 = self.doubleQ(formatted_obs, random_action)
                curr_Q1, curr_Q2 = self.doubleQ(formatted_obs, curr_action)
                new_curr_Q1, new_curr_Q2 = self.doubleQ(formatted_obs, new_curr_action)

                random_Q1, random_Q2 = random_Q1.view(observations.shape[0], self.cql_num_sampled_actions, 1), random_Q2.view(observations.shape[0], self.cql_num_sampled_actions, 1)
                curr_Q1, curr_Q2 = curr_Q1.view(observations.shape[0], self.cql_num_sampled_actions, 1), curr_Q2.view(observations.shape[0], self.cql_num_sampled_actions, 1)
                new_curr_Q1, new_curr_Q2 = new_curr_Q1.view(observations.shape[0], self.cql_num_sampled_actions, 1), new_curr_Q2.view(observations.shape[0], self.cql_num_sampled_actions, 1)

                cat_q1 = torch.cat([random_Q1 - random_action_log_pi, new_curr_Q1 - new_curr_action_log_pi, curr_Q1 - curr_action_log_pi], 1)
                cat_q2 = torch.cat([random_Q2 - random_action_log_pi, new_curr_Q2 - new_curr_action_log_pi, curr_Q2 - curr_action_log_pi], 1)
                
                policy_qvals1 = torch.logsumexp(cat_q1 / self.cql_lse_temp, dim=1) * self.cql_lse_temp
                policy_qvals2 = torch.logsumexp(cat_q2 / self.cql_lse_temp, dim=1) * self.cql_lse_temp
        
                dataset_q_vals1 = Qsa1
                dataset_q_vals2 = Qsa2

                cql_term1 = (policy_qvals1 - dataset_q_vals1).mean(0)
                cql_term2 = (policy_qvals2 - dataset_q_vals2).mean(0)
                
                cql_term = cql_term1 + cql_term2
                cql_loss = self.cql_alpha * cql_term

                loss += cql_loss
            
            else:
                cql_loss = torch.tensor(0.)
        
        else:
            cql_loss = torch.tensor(0.)

        # clean potentials left-over gradients and updates networks
        self.doubleQ.q1.optim.zero_grad()
        self.doubleQ.q2.optim.zero_grad()
        self.memory.optim.zero_grad()
        loss.backward()
        self.doubleQ.q1.optim.step()
        self.doubleQ.q2.optim.step()
        self.memory.optim.step()

        return loss.detach().cpu().numpy(), cql_loss.detach().cpu().numpy()

    def update_pi(self, batch):
        
        observations = self.concat_embedings_to_obs(batch, target=False)

        pi = self.policy(observations)

        Q = self.doubleQ.q1(observations, pi)

        loss_td3 = -Q.mean(0)

        if self.bc_coeff > 0.:
            loss_bc = ((pi - batch['actions'])**2).sum(1).mean(0)
            lmbda = self.bc_coeff/Q.abs().mean(0).detach()
        else: 
            loss_bc = torch.tensor(0.)
            lmbda = torch.tensor(1.)
        
        if self.omar_coeff > 0.:
            
            actions = batch['actions']
            curr_pol_out = pi
            pred_qvals = Q

            # copy pasted and adapted from https://github.com/ling-pan/OMAR/blob/0e52ad6fc23585a83eb771e9315d4966e3faa128/algorithms/maddpg.py#L246
            self.omar_mu = torch.cuda.FloatTensor(actions.shape[0], actions.shape[1]).zero_() + self.init_omar_mu
            self.omar_sigma = torch.cuda.FloatTensor(actions.shape[0], actions.shape[1]).zero_() + self.init_omar_sigma

            formatted_obs = observations.unsqueeze(1).repeat(1, self.omar_num_samples, 1).view(-1, observations.shape[1])

            last_top_k_qvals, last_elite_acs = None, None
            for iter_idx in range(self.omar_iters):
                dist = torch.distributions.Normal(loc=self.omar_mu, scale=self.omar_sigma)
                
                cem_sampled_acs = dist.sample((self.omar_num_samples,)).detach().permute(1, 0, 2).clamp(-1., 1.)

                formatted_cem_sampled_acs = cem_sampled_acs.reshape(-1, cem_sampled_acs.shape[-1])

                all_pred_qvals = self.doubleQ.q1(formatted_obs, formatted_cem_sampled_acs).view(actions.shape[0], -1, 1)

                if iter_idx > 0:
                    all_pred_qvals = torch.cat((all_pred_qvals, last_top_k_qvals), dim=1)
                    cem_sampled_acs = torch.cat((cem_sampled_acs, last_elite_acs), dim=1)

                top_k_qvals, top_k_inds = torch.topk(all_pred_qvals, self.omar_num_elites, dim=1)
                elite_ac_inds = top_k_inds.repeat(1, 1, actions.shape[1])
                elite_acs = torch.gather(cem_sampled_acs, 1, elite_ac_inds)

                last_top_k_qvals, last_elite_acs = top_k_qvals, elite_acs

                updated_mu = torch.mean(elite_acs, dim=1)
                updated_sigma = torch.std(elite_acs, dim=1)

                self.omar_mu = updated_mu
                self.omar_sigma = updated_sigma + 1e-6 # to make sure sigma is not 0

            top_qvals, top_inds = torch.topk(all_pred_qvals, 1, dim=1)
            top_ac_inds = top_inds.repeat(1, 1, actions.shape[1])
            top_acs = torch.gather(cem_sampled_acs, 1, top_ac_inds)

            cem_qvals = top_qvals
            pol_qvals = pred_qvals.unsqueeze(1)
            cem_acs = top_acs
            pol_acs = curr_pol_out.unsqueeze(1)

            candidate_qvals = torch.cat([pol_qvals, cem_qvals], 1)
            candidate_acs = torch.cat([pol_acs, cem_acs], 1)

            max_qvals, max_inds = torch.max(candidate_qvals, 1, keepdim=True)
            max_ac_inds = max_inds.repeat(1, 1, actions.shape[1])

            max_acs = torch.gather(candidate_acs, 1, max_ac_inds).squeeze(1)
            
            mimic_acs = max_acs.detach()
            
            mimic_term = F.mse_loss(curr_pol_out, mimic_acs)

            loss_omar = self.omar_coeff * mimic_term

            loss_omar += (curr_pol_out ** 2).mean() * 1e-3
            
        else: 
            loss_omar = torch.tensor(0.)

        if self.init_dict['td3_shortcut_omar']:
            loss = lmbda * loss_td3 + loss_bc
            loss_omar = torch.tensor(0.)
        else:
            loss = (1. - self.omar_coeff) * lmbda * loss_td3 + loss_bc + loss_omar

        # clean potentials left-over gradients and updates networks
        self.policy.optim.zero_grad()
        self.memory.optim.zero_grad()
        loss.backward()
        self.policy.optim.step()

        if self.memory_backprop_actor:
            self.memory.optim.step()

        return loss.detach().cpu().numpy(), loss_td3.detach().cpu().numpy(), loss_bc.detach().cpu().numpy(), loss_omar.detach().cpu().numpy(), lmbda.detach().cpu().numpy()

    def update(self, dataset, batch_size, train_device, **kwargs):
        batch = dataset.sample(batch_size)
        return self.update_from_batch(batch, train_device, **kwargs)

    def update_from_batch(self, batch, train_device, **kwargs):
        # put models on training device
        self.to(train_device)

        self.total_it += 1

        batch = batch_to_device(batch_as_tensor(batch), train_device)

        loss_q, loss_cql = self.update_q(batch)

        if self.total_it % self.policy_freq == 0:

            loss_pi, loss_td3, loss_bc, loss_omar, lmbda = self.update_pi(batch)
            
            update_target_soft(target=self.doubleQtarget, source=self.doubleQ, tau=self.target_update_coef)
            update_target_soft(target=self.memorytarget, source=self.memory, tau=self.target_update_coef)
            update_target_soft(target=self.policytarget, source=self.policy, tau=self.target_update_coef)

            return {'loss_q': loss_q, 'loss_pi': loss_pi, 'loss_td3': loss_td3, 'loss_bc': loss_bc, 'loss_omar': loss_omar, 'lmbda': lmbda, 'loss_cql': loss_cql}

        else: 

            return {'loss_q': loss_q}


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

    @property
    def device(self):
        return next(self.parameters()).device
    
    def get_state_dict(self):
        return {'doubleQ': self.doubleQ.get_state_dict(),
                'doubleQtarget': self.doubleQtarget.get_state_dict(),
                'policy': self.policy.get_state_dict(), 
                'policytarget': self.policytarget.get_state_dict(), 
                'memory': self.memory.get_state_dict(),
                'memorytarget': self.memorytarget.get_state_dict()}
    
    def do_load_state_dict(self, state_dict):
        self.doubleQ.do_load_state_dict(state_dict['doubleQ'])
        self.doubleQtarget.do_load_state_dict(state_dict['doubleQtarget'])
        self.policy.do_load_state_dict(state_dict['policy'])
        self.policytarget.do_load_state_dict(state_dict['policytarget'])
        self.memory.do_load_state_dict(state_dict['memory']) 
        self.memorytarget.do_load_state_dict(state_dict['memorytarget'])