
import torch
import torch.nn as nn
import numpy as np

from offline_marl.utils.ml import batch_as_tensor, batch_to_device, check_if_nan_in_batch

from offline_marl.single_agent.ppo import PPOLearner
from offline_marl.multi_agent.mixer_networks import MixerNetwork
from offline_marl.utils.base_algo import BaseAlgo

class MAPPO(BaseAlgo, nn.Module):
    def __init__(self, **kwargs):
        assert 'ma-ppo' in kwargs['alg_name']
        BaseAlgo.__init__(self)
        nn.Module.__init__(self)

        self.name = kwargs['alg_name']
        self.n_learner = kwargs['n_learner']
        self.grad_clip_norm = kwargs['grad_clip_norm']

        self.batch_size = kwargs['batch_size']
        self.epochs_per_update = kwargs['ppo_epochs_per_update']
        self.update_clip_param = kwargs['ppo_update_clip_param']
        self.discount = kwargs['discount_factor']
        self.lamda = kwargs['ppo_lamda']
        self.critic_loss_coeff = kwargs['ppo_critic_loss_coeff']
        self.ppo_entropy_bonus_coeff = kwargs['ppo_entropy_bonus_coeff']
        self.ppo_entropy_target = kwargs.get('ppo_entropy_target', -100)
        self.ppo_action_penalty_coeff = kwargs.get('ppo_action_penalty_coeff', 0.)

        self.entropy_alpha = nn.parameter.Parameter(torch.tensor([0.], dtype=torch.float32), requires_grad=False)

        # defines learners (i.e. agent wise actor critic with memory)
        self.learners = nn.ModuleList()
        for act_space, obs_space in zip(kwargs['act_space'], kwargs['obs_space']):
            params = kwargs.copy()
            params.update({'obs_space': obs_space, 'act_space': act_space, 'set_final_bias': True, 'action_squashing': kwargs['ppo_actor_squashing'],
                           'double_V': kwargs['ppo_double_V']})
            self.learners.append(PPOLearner(**params))

        # defines V mixer on joint obs (no memory)
        self.V_mixer_input_size = sum([obs_space.shape[0] for obs_space in kwargs['obs_space']])
        self.mixer_network_V = MixerNetwork(global_obs_size=self.V_mixer_input_size, n_learners=self.n_learner,  hidden_size=kwargs['hidden_size'], lr=kwargs['lr_v'], train_device=kwargs['train_device'])


        self.init_dict = kwargs

        self.metrics_to_record = {'loss', 'actor_loss', 'critic_weighted_loss', 'entropy_bonus', 'entropy_alpha',
                                     'ppo_rewards_mean', 'ppo_rewards_max', 'ppo_rewards_min', 'ppo_rewards_std', 'ppo_rewards_median',
                                     'ppo_actions_mean', 'ppo_actions_max', 'ppo_actions_min', 'ppo_actions_std', 'ppo_actions_median',
                                     'action_penalty', 'mean_squared_action', 'action_bound_error'}
    
    def get_v_mixer_input(self, batches, next_obs=False):
        if next_obs:
            return torch.cat([batch['next_observations'] for batch in batches], dim=1)
        else: 
            return torch.cat([batch['observations'] for batch in batches], dim=1)
    
    def subsample_batch(self, batch, idx):
        return {key: val[idx] for key, val in batch.items()}
    
    def subsample_batches(self, batches, idx):
        return [self.subsample_batch(batch, idx) for batch in batches]


    def update_from_batch(self, batches, train_device, **kwargs):
        

        # we construct the history of memories
        for batch, learner in zip(batches, self.learners):
            batch.update(learner.build_memories(batch))

        # put models on training device
        self.to(train_device)

        batches = [batch_as_tensor(batch) for batch in batches]
        batches = [batch_to_device(batch, train_device) for batch in batches]

        _ = [check_if_nan_in_batch(batch) for batch in batches]

        ## WE COMPUTE OLD VALUES

        old_values_list = []
        old_next_values_list = []
        old_policy_list = []

        for batch, learner in zip(batches, self.learners):
            with torch.no_grad():
                observations = learner.concat_embedings_to_obs(batch)

                next_observations = learner.concat_embedings_to_obs(batch, next_obs=True)

                if learner.double_V:
                    v1, v2 = learner.V(observations)
                    old_values_list.append(torch.minimum(v1,v2))

                    vp1, vp2 = learner.V(next_observations)
                    old_next_values_list.append(torch.minimum(vp1, vp2))
                else:
                    old_values_list.append(learner.V(observations))
                    old_next_values_list.append(learner.V(next_observations))

                old_policy_list.append(learner.policy.get_log_prob_from_obs_action_pairs(batch['actions'], observations))
                
            
        V_mixing_weights, V_mixing_biases = self.mixer_network_V(self.get_v_mixer_input(batches))
        next_V_mixing_weights, next_V_mixing_biases = self.mixer_network_V(self.get_v_mixer_input(batches, next_obs=True))

        old_values = MixerNetwork.mix(torch.cat(old_values_list, dim=1), V_mixing_weights, V_mixing_biases).detach()
        old_next_values = MixerNetwork.mix(torch.cat(old_next_values_list, dim=1), next_V_mixing_weights, next_V_mixing_biases).detach()
        old_policy = torch.cat(old_policy_list, dim=1).sum(1, keepdim=True)
        
        
        # fully cooperative setting
        rewards = batches[0]['rewards']
        masks = batches[0]['masks']
        time_out_masks = batches[0]['time_out_masks']

        returns, advants = self.get_gae(rewards=rewards,
                                        masks=masks,
                                        time_out_masks=time_out_masks,
                                        values=old_values,
                                        next_values=old_next_values)

        ### We update for several epochs
        criterion = torch.nn.MSELoss()
        n = len(rewards)
        loss_val_list = {'loss': [], 'actor_loss': [], 'critic_weighted_loss': [], 'entropy_bonus': [], 'entropy_alpha': [], 'action_penalty': [],
                            'mean_squared_action':[], 'action_bound_error': []}
        for it_update in range(self.epochs_per_update):
            shuffled_idxs = torch.randperm(n, device=self.device)

            for i in range(n // self.batch_size):
                batch_idxs = shuffled_idxs[self.batch_size * i: min(self.batch_size * (i + 1), len(shuffled_idxs))]

                # we do minibatches
                batch_samples = self.subsample_batches(batches, batch_idxs)
                oldvalue_samples = old_values[batch_idxs]
                old_policy_samples = old_policy[batch_idxs]
                returns_samples = returns[batch_idxs]
                advants_samples = advants[batch_idxs]

                # we recompute current quantities
                new_policy_list = []
                actions_list = []
                observations_list = []
                for batch, learner in zip(batch_samples, self.learners):
                    observations = learner.concat_embedings_to_obs(batch)
                    n_pi = learner.policy.get_log_prob_from_obs_action_pairs(batch['actions'], observations)
                    new_policy_list.append(n_pi)
                    actions_list.append(learner.policy.act(observations, sample=False, return_log_pi=False))
                    observations_list.append(observations)

                #critic loss

                def get_critic_loss(values_list):

                    V_mixing_weights, V_mixing_biases = self.mixer_network_V(self.get_v_mixer_input(batch_samples))
                    values = MixerNetwork.mix(torch.cat(values_list, dim=1), V_mixing_weights, V_mixing_biases)

                    clipped_values = oldvalue_samples + \
                                    torch.clamp(values - oldvalue_samples,
                                                -self.update_clip_param,
                                                self.update_clip_param)
                    critic_loss1 = criterion(clipped_values, returns_samples)
                    critic_loss2 = criterion(values, returns_samples)
                    critic_loss = torch.max(critic_loss1, critic_loss2)
                    return critic_loss
                
                # Deal with double V
                if self.init_dict['ppo_double_V']:
                    # with v1 for each agent
                    values_list = []
                    for observations, learner in zip(observations_list, self.learners):
                        values_list.append(learner.V.v1(observations))

                    critic_lossv1 = get_critic_loss(values_list)

                    #with v2 for each agent
                    values_list = []
                    for observations, learner in zip(observations_list, self.learners):
                        values_list.append(learner.V.v2(observations))

                    critic_lossv2 = get_critic_loss(values_list)

                    critic_loss = 0.5*(critic_lossv1 + critic_lossv2)
                else:
                    values_list = []
                    for observations, learner in zip(observations_list, self.learners):
                        values_list.append(learner.V(observations))

                    critic_loss = get_critic_loss(values_list)

                # actor loss
                new_policy = torch.cat(new_policy_list, dim=1).sum(1, keepdim=True)

                ratio = torch.exp(torch.clamp(new_policy - old_policy_samples, max=10.)) # we clamp the exponential for numerical stability
                # ratio = torch.exp(new_policy - old_policy_samples)
                surrogate_loss = ratio * advants_samples

                clipped_ratio = torch.clamp(ratio,
                                            1.0 - self.update_clip_param,
                                            1.0 + self.update_clip_param)
                clipped_loss = clipped_ratio * advants_samples

                actor_loss = -torch.min(surrogate_loss, clipped_loss).mean(0) 

                # entropy bonus
                if self.ppo_entropy_bonus_coeff > 0.:
                    
                    # dimensions are batch, act_dim, n_agents. We cannot have closed form entropy for tanhgaussian so we estimate it with E(-pi log pi) were
                    # the expectation is sampled over pi_old so we have to correct with pi_new/pi_old (which is ratio!)
                    
                    # we do two losses (clipped / not clipped like with the actor loss)
                    surrogate_entropy = - (ratio * new_policy).mean(0)
                    clipped_entropy = - (clipped_ratio * new_policy).mean(0)

                    entropy = torch.min(surrogate_entropy, clipped_entropy)

                    self.entropy_alpha += self.ppo_entropy_bonus_coeff*(self.ppo_entropy_target - entropy).detach()
                    self.entropy_alpha.data.clamp_min_(0.)

                    ## minus sign because we minimize the expressions so max(ent) = min(-ent)
                    entropy_bonus = - entropy * self.entropy_alpha
                    
                else:
                    entropy_bonus = torch.tensor(0., device=self.device)

                if self.ppo_action_penalty_coeff > 0.: 
                    cat_actions = torch.cat(actions_list, dim=1)
                    if self.init_dict['ppo_actor_squashing'] == "tanh":
                        cat_actions = cat_actions.arctanh()
                        mean_squared_action = (cat_actions**2).sum(1, keepdim=True).mean(0)
                        action_bound_error = torch.tensor(0., device=self.device)

                        action_penalty = self.ppo_action_penalty_coeff * mean_squared_action

                    elif self.init_dict['ppo_actor_squashing'] == "none":
                        delta = (1. - cat_actions.abs())
                        action_bound_error = ((delta < 0.).to(torch.float32) * delta**2).sum(1, keepdim=True)

                        surrogate_action_bound_error = (ratio * action_bound_error).mean(0)
                        clipped_action_bound_error = (clipped_ratio * action_bound_error).mean(0)

                        action_bound_error = torch.max(surrogate_action_bound_error, clipped_action_bound_error)
                        action_penalty = self.ppo_action_penalty_coeff * action_bound_error
                        
                        mean_squared_action = torch.tensor(0., device=self.device)

                        
                    else:
                        raise NotImplementedError

                    
                else:
                    action_penalty = torch.tensor(0., device=self.device)
                    mean_squared_action = torch.tensor(0., device=self.device)
                    action_bound_error = torch.tensor(0., device=self.device)

                
                # total loss
                loss = actor_loss + self.critic_loss_coeff * critic_loss + entropy_bonus + action_penalty

                # clean gradients
                for learner in self.learners:
                    learner.memory.optim.zero_grad()
                    learner.policy.optim.zero_grad()

                    if learner.double_V:
                        learner.V.v1.optim.zero_grad()
                        learner.V.v2.optim.zero_grad()

                    else:
                        learner.V.optim.zero_grad()
                self.mixer_network_V.optim.zero_grad()
            
                # backprop
                loss.backward()

                # if any([torch.isnan(p.grad).any() for p in self.parameters() if p.grad is not None]):
                #     raise ValueError("found nan in grad")

                # if any([torch.isinf(p.grad).any() for p in self.parameters() if p.grad is not None]):
                #     raise ValueError("found inf in grad")

                # clip
                nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip_norm)

                # update
                for learner in self.learners:
                    learner.memory.optim.step()
                    learner.policy.optim.step()
                    if learner.double_V:
                        learner.V.v1.optim.step()
                        learner.V.v2.optim.step()
                    else:
                        learner.V.optim.step()
                        
                self.mixer_network_V.optim.step()
                
                if torch.isinf(loss):
                    raise ValueError("found inf in loss")

                if torch.isnan(loss):
                    raise ValueError("found nan in loss")
                
                loss_val_list['loss'].append(loss.detach().cpu().numpy())
                loss_val_list['actor_loss'].append(actor_loss.detach().cpu().numpy())
                loss_val_list['critic_weighted_loss'].append(self.critic_loss_coeff * critic_loss.detach().cpu().numpy())
                loss_val_list['entropy_bonus'].append(entropy_bonus.detach().cpu().numpy())
                loss_val_list['entropy_alpha'].append(self.entropy_alpha.clone().detach().cpu().numpy())
                loss_val_list['action_penalty'].append(action_penalty.detach().cpu().numpy())
                loss_val_list['mean_squared_action'].append(mean_squared_action.detach().cpu().numpy())
                loss_val_list['action_bound_error'].append(action_bound_error.detach().cpu().numpy())

        actions = torch.stack([batch['actions'] for batch in batches])

        return_dict = {key: np.mean(val) for key, val in loss_val_list.items()}
        return_dict.update({
            'ppo_rewards_mean': rewards.mean().detach().cpu().numpy(),
            'ppo_rewards_max': rewards.max().detach().cpu().numpy(),
            'ppo_rewards_min': rewards.min().detach().cpu().numpy(),
            'ppo_rewards_std': rewards.std().detach().cpu().numpy(),
            'ppo_rewards_median': rewards.median().detach().cpu().numpy(),
        })

        return_dict.update({
            'ppo_actions_mean': actions.mean().detach().cpu().numpy(),
            'ppo_actions_max': actions.max().detach().cpu().numpy(),
            'ppo_actions_min': actions.min().detach().cpu().numpy(),
            'ppo_actions_std': actions.std().detach().cpu().numpy(),
            'ppo_actions_median': actions.median().detach().cpu().numpy(),
        })

        return return_dict

    def get_gae(self, rewards, masks, time_out_masks, values, next_values):
        rewards = torch.as_tensor(rewards, device=self.device)
        masks = torch.as_tensor(masks, device=self.device)
        returns = torch.zeros_like(rewards, device=self.device)
        advants = torch.zeros_like(rewards, device=self.device)

        # initial (end) running returns is next state value
        running_returns = next_values[-1] * masks[-1]
        
        # initial (end) advantage is 0 because no difference in value and return
        running_advants = 0

        for t in reversed(range(0, len(rewards))):

            # we are going reverse so if done, only reward because end of episode
            # if timeout, we cut the line and use value as bootstrap like done in initialization 
            running_returns = rewards[t] + self.discount * masks[t] * (running_returns * time_out_masks[t] + (1. - time_out_masks[t]) * next_values[t] * masks[t])
            returns[t] = running_returns

            ## No accumulation here and timeout doesn't influence next_state value
            running_delta = rewards[t] + (self.discount * next_values[t] * masks[t]) - values[t]

            ## if timeout running advants goes back to running_delta because we do not have extra rewards to estimate it (cf above initialization)
            running_advants = running_delta + (self.discount * self.lamda * running_advants * masks[t]) * time_out_masks[t]

            advants[t] = running_advants


        advants = (advants - advants.mean()) / (advants.std() + 1e-3)

        # for key, val in {'returns': returns, 'advants': advants}.items():
        #     if any(torch.isnan(val)):
        #         raise ValueError(f"found nan in {key} when computing gae")
        #     if any(torch.isinf(val)):
        #         raise ValueError(f"found inf in {key} when computing gae")

        return returns, advants


    def get_state_dict(self):
        self.to('cpu')
        learners_state_dict = {f'learner_{i}': learner.get_state_dict() for i, learner in enumerate(self.learners)}
        learners_state_dict.update({'mixer_network_V': self.mixer_network_V.get_state_dict()})
        learners_state_dict.update({'entropy_alpha': self.entropy_alpha})
        return learners_state_dict
    
    def do_load_state_dict(self, state_dict):
        # this loads network, optim, scheduler params
        _ = [learner.do_load_state_dict(state_dict[f'learner_{i}']) for i, learner in enumerate(self.learners)]
        self.mixer_network_V.do_load_state_dict(state_dict['mixer_network_V'])
        self.entropy_alpha = state_dict.get('entropy_alpha', nn.parameter.Parameter(torch.tensor([0.], requires_grad=False)))
    
    def load_state_dict(self, state_dict):
        # this only load network params
        _ = [learner.load_state_dict(state_dict[f'learner_{i}']) for i, learner in enumerate(self.learners)]
        self.mixer_network_V.load_state_dict(state_dict['mixer_network_V'])
        self.entropy_alpha = state_dict['entropy_alpha']

    @property
    def device(self):
        return next(self.parameters()).device

