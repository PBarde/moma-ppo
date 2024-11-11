from multiprocessing import reduction
from platform import node
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from gym.spaces import Box
from nop import NOP


from offline_marl.utils.base_algo import BaseAlgo
from offline_marl.utils import histories
from offline_marl.utils.histories import HistoryEncoder
import offline_marl.utils.ml as ml
from offline_marl.single_agent.world_models import MixtureWorldModelNNs, VAEWorldModelNNs, WorldModelNNs, DeterministicWorldModelNNs, BinaryRewardSigmoidStateWorldModelNNs
from offline_marl.single_agent.actors import epsilon

class WorldModel(BaseAlgo):
    def __init__(self, **kwargs):
        super().__init__()

        self._init(kwargs)
    
    def _init(self, kwargs):
        
        # computes joint actions and observation size since one world model predicts for all of the agents at the same time

        if isinstance(kwargs['act_space'][0], Box):
            self.action_dim_list = [act_space.shape[0] for act_space in kwargs['act_space']]
            self.discrete_actions = False
            
        else:
            self.action_dim_list = [act_space.n for act_space in kwargs['act_space']]
            self.discrete_actions = True

        joint_actions_dim = sum(self.action_dim_list)
        joint_observations_dim = sum([obs_space.shape[0] for obs_space in kwargs['obs_space']])

        self.act_spaces = kwargs['act_space']
        self.obs_spaces = kwargs['obs_space']
        self.n_agents = len(self.act_spaces)
        self.loss_weights = kwargs['loss_weights']

        
        # Memory encoder
        self.memory_len = kwargs['memory_len']
        if kwargs.get('memory_len', 0) > 0:
            self.memory, self.memorytarget, memory_embedding_vec = self.make_history_encoder(num_in=joint_observations_dim + joint_actions_dim,
                                                                                                    num_out=kwargs['memory_out_size'], 
                                                                                                    hidden_size=kwargs['hidden_size'],
                                                                                                    lr=kwargs['lr_memory'],
                                                                                                    reduce_op=kwargs['memory_op'],
                                                                                                    history_len=kwargs['memory_len'],
                                                                                                    train_device=kwargs['train_device'])

            joint_memory_dim = len(memory_embedding_vec)

        else:
            self.memory = NOP()
            self.memorytarget = NOP()
            joint_memory_dim = 0

        self.joint_memory_dim = joint_memory_dim
        self.joint_action_dim = joint_actions_dim
        self.joint_observation_dim = joint_observations_dim

        if any([name in kwargs['task_name'] for name in ['toy', 'ant', 'halfcheetah', 'walker2d', 'hopper', 'reacher']]):
            # learns next_state of each agent, reward for each agent (even if same for all the agents) and mask for each agent (even if same for all agents)
            out_mask_dim = self.n_agents  # mask for each agent
            out_legal_move_dim = 0
            self.legal_move = False

        elif 'hanabi' in kwargs['task_name']:
            # learns next_state of each agent, reward for each agent (even if same for all the agents), mask for each agent (even if same for all agents) and legal move for each agent
            out_mask_dim = self.n_agents
            out_legal_move_dim = joint_actions_dim  # mask for each agent and legal_move for each agent
            self.legal_move = True
        
        else:
            raise NotImplementedError
        
        self.k_mixture_gaussian = kwargs['k_mixture_gaussian']
        self.deterministic_wm = kwargs['deterministic_wm']
        self.binary_reward_sigmoid_state = kwargs.get('binary_reward_sigmoid_state', False)
        self.vae_wm = kwargs.get('vae_wm', False)
        self.beta_vae = kwargs.get('beta_vae', None)
        self.vae_loss = kwargs.get('vae_loss', None)
        
        if self.vae_wm:
            self._model = VAEWorldModelNNs(obs_mem_dim=joint_observations_dim + joint_memory_dim,
                                                            act_dim= joint_actions_dim,
                                                            out_state_dim=joint_observations_dim, 
                                                            out_reward_dim=self.n_agents,
                                                            out_mask_dim=out_mask_dim, 
                                                            out_legal_move_dim=out_legal_move_dim, 
                                                            hidden_size=kwargs['hidden_size'],
                                                            latent_size=kwargs['latent_size'],
                                                            state_sigmoid_output=kwargs['state_sigmoid_output'], 
                                                            lr=kwargs['lr'], 
                                                            train_device=kwargs['train_device'], 
                                                            weight_decay=kwargs['weight_decay'])
        
            if self.vae_loss == 'l1':
                def loss(pred, target):
                    return ((pred-target)**2).sum(1)
            elif self.vae_loss == 'l2':
                def loss(pred, target):
                    return torch.abs(pred-target).sum(1)
            else:
                raise NotImplementedError
            
            self.vae_loss_fct = loss
        
        else:
            if self.deterministic_wm:
                if self.binary_reward_sigmoid_state: 
                    self._model = BinaryRewardSigmoidStateWorldModelNNs(obs_mem_dim=joint_observations_dim + joint_memory_dim,
                                                            act_dim= joint_actions_dim,
                                                            out_state_dim=joint_observations_dim, 
                                                            out_reward_dim=self.n_agents,
                                                            out_mask_dim=out_mask_dim, 
                                                            out_legal_move_dim=out_legal_move_dim, 
                                                            hidden_size=kwargs['hidden_size'], 
                                                            lr=kwargs['lr'], 
                                                            train_device=kwargs['train_device'], 
                                                            weight_decay=kwargs['weight_decay'])
                else:
                    self._model = DeterministicWorldModelNNs(obs_mem_dim=joint_observations_dim + joint_memory_dim,
                                                            act_dim= joint_actions_dim,
                                                            out_state_dim=joint_observations_dim, 
                                                            out_reward_dim=self.n_agents,
                                                            out_mask_dim=out_mask_dim, 
                                                            out_legal_move_dim=out_legal_move_dim, 
                                                            hidden_size=kwargs['hidden_size'], 
                                                            lr=kwargs['lr'], 
                                                            train_device=kwargs['train_device'], 
                                                            weight_decay=kwargs['weight_decay'])
            else:
                if self.k_mixture_gaussian > 1:
                    self._model = MixtureWorldModelNNs(obs_mem_dim=joint_observations_dim + joint_memory_dim,
                                                        act_dim= joint_actions_dim,
                                                        out_state_dim=joint_observations_dim, 
                                                        out_reward_dim=self.n_agents,
                                                        out_mask_dim=out_mask_dim, 
                                                        out_legal_move_dim=out_legal_move_dim,
                                                        hidden_size=kwargs['hidden_size'], 
                                                        lr=kwargs['lr'], 
                                                        train_device=kwargs['train_device'], 
                                                        weight_decay=kwargs['weight_decay'],
                                                        k=self.k_mixture_gaussian)

                else:
                    self._model = WorldModelNNs(obs_mem_dim=joint_observations_dim + joint_memory_dim,
                                                act_dim= joint_actions_dim,
                                                out_state_dim=joint_observations_dim, 
                                                out_reward_dim=self.n_agents,
                                                out_mask_dim=out_mask_dim, 
                                                out_legal_move_dim=out_legal_move_dim, 
                                                hidden_size=kwargs['hidden_size'], 
                                                lr=kwargs['lr'], 
                                                train_device=kwargs['train_device'], 
                                                weight_decay=kwargs['weight_decay'], 
                                                spectral_norm=kwargs.get('spectral_norm', False),
                                                four_layers_wm=kwargs.get('four_layers_wm', False))
            
        
        # to save and reload
        self.init_dict = kwargs
        self.name = kwargs.get('wm_name', 'wm')
        self.grad_clip_norm = kwargs.get('grad_clip_norm', 1e8)

        # data to monitor
        self.train_metrics = {'loss_T', 'loss_R', 'loss_legal_moves', 'loss_masks', 'loss_KL_vae'}

        self.evaluation_metrics = {'T_dist_train',
                                    'T_dist_means_train',
                                    'T_dist_means_weighted_train',
                                    'R_dist_means_weighted_train',
                                    'R_dist_train', 
                                    'R_acc_greedy_train',
                                    'R_min_train',
                                    'R_max_train',
                                    'R_dist_means_train',
                                    'T_std_mean_train' , 'T_std_max_train', 'T_std_min_train', 'R_std_mean_train', 'R_std_max_train', 'R_std_min_train',
                                    'masks_acc_sample_train', 'legal_moves_acc_sample_train', 'masks_acc_greedy_train', 'legal_moves_acc_greedy_train', 'weights_T_entropy_train', 'weights_R_entropy_train',
                                    'T_dist_valid',
                                    'T_dist_means_valid',
                                    'T_dist_means_weighted_valid',
                                    'R_dist_means_weighted_valid',
                                    'R_dist_valid', 
                                    'R_acc_greedy_valid',
                                    'R_min_valid',
                                    'R_max_valid',
                                    'R_dist_means_valid',
                                    'T_std_mean_valid' , 'T_std_max_valid', 'T_std_min_valid', 'R_std_mean_valid', 'R_std_max_valid', 'R_std_min_valid',
                                    'masks_acc_sample_valid', 'legal_moves_acc_sample_valid', 'masks_acc_greedy_valid', 'legal_moves_acc_greedy_valid', 'weights_T_entropy_valid', 'weights_R_entropy_valid',
                                    'T_baseline_train', 'R_baseline_train', 'masks_baseline_train', 'legal_moves_baseline_train',
                                    'T_baseline_valid', 'R_baseline_valid', 'masks_baseline_valid', 'legal_moves_baseline_valid',}
        
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

            HistoryEncoder.update_target_hard(encodertarget, encoder)

        # we extend observation space with memory embedding size 
        embeding_vec = np.ones(num_out)
        
        return encoder, encodertarget, embeding_vec

    def concat_embedings_to_obs(self, batch, target, next_obs=False):
        if target:
            memory_encoder = self.memorytarget
        else:
            memory_encoder = self.memory
        return histories.concat_embedings_to_obs(batch, memory_encoder, self.memory_len, next_obs)
    
    def transform_actions(self, actions):
        if self.discrete_actions:
            assert len(actions.shape) == 2
            assert actions.shape[1] == self.n_agents

            batch_size = actions.shape[0]
            actions_oh_list = []

            # we do one-hot action for each agent on the whole batch (this is why we transpose)
            for action, action_dim in zip(actions.T, self.action_dim_list):
                actions_oh = torch.zeros(batch_size*action_dim, device=action.device)
                index = action.flatten() + torch.arange(batch_size, device=action.device)*action_dim
                actions_oh[index] = 1. 
                actions_oh = actions_oh.reshape(batch_size, action_dim)
                actions_oh_list.append(actions_oh)
            
            joint_actions_oh = torch.cat(actions_oh_list, dim=1)

            return joint_actions_oh
        else:
            # we ensure we are in the correct range (for example if using gaussian policy without squashing)
            actions = torch.clamp(actions, min=-1. + epsilon, max=1. - epsilon)
            return actions


    def update(self, batch):
        self.train()
        # computes logT(o'|o,a,m,c)
        observations = self.concat_embedings_to_obs(batch, target=False)
        actions = self.transform_actions(batch['actions'])

        if self.vae_wm:

            T_pred, R_pred, mask_logits, legal_move_logits, mean_latent, log_std_latent = self._model(observations, actions)

            # loss_T = ((batch.next_observations - T_pred)**2).sum(1).mean(0)
            loss_T = self.vae_loss_fct(T_pred, batch['next_observations']).mean(0)

            # loss_R = ((batch.rewards - R_pred)**2).sum(1).mean(0)
            loss_R = self.vae_loss_fct(R_pred, batch['rewards']).mean(0)

            masks_prob = self._model.mask_decoder.binary_prob(mask_logits)
            loss_masks = torch.nn.functional.binary_cross_entropy(masks_prob.flatten(), batch['masks'].flatten(), reduction="none").mean(0)

            if self.legal_moves:
                legal_move_prob = self._model.legal_move.binary_prob(legal_move_logits)
                loss_legal_moves = torch.nn.functional.binary_cross_entropy(legal_move_prob.flatten(), batch['legal_moves'].flatten(), reduction="none").mean(0)
            else:
                loss_legal_moves = torch.tensor(0.)

            # kl between to multivariate diagonal gaussian were the second is a unit gaussian from https://mr-easy.github.io/2020-04-16-kl-divergence-between-2-gaussian-distributions/
            loss_KL_vae = 0.5 * ((mean_latent**2).sum(1) + (log_std_latent.exp()**2).sum(1) - mean_latent.size(1) - 2 * log_std_latent.sum(1)).mean(0)

            # normalized beta coeff 
            beta_coeff = self.beta_vae * observations.size(1) / mean_latent.size(1)

        else:
            loss_KL_vae = torch.tensor(0.)
            beta_coeff = torch.tensor(0.)

            observations_actions = torch.cat((observations, actions), dim=1)
            mask_prob = self._model.mask.prob(inputs=observations_actions)
            loss_masks = torch.nn.functional.binary_cross_entropy(mask_prob.flatten(), batch['masks'].flatten(), reduction="none").mean(0)

            if self.deterministic_wm:

                # for the state the loss is L2 regardless if we constraint the state with a sigmoid or not
                T_pred = self._model.T(observations_actions)
                loss_T = ((batch.next_observations - T_pred)**2).sum(1).mean(0)

                # if binary rewards we use binary cross-entropy
                if self.binary_reward_sigmoid_state:
                    R_prob = self._model.R.prob(inputs=observations_actions)
                    loss_R = torch.nn.functional.binary_cross_entropy(R_prob.flatten(), batch['rewards'].flatten(), reduction="none").mean(0)

                # otherwise we just use L2
                else:
                    R_pred = self._model.R(observations_actions)
                    loss_R = ((batch['rewards'] - R_pred)**2).sum(1).mean(0)

            else:
                log_probs_T = self._model.T.log_prob(inputs=observations_actions, sample=batch['next_observations'], reduction='mean', constant=False)
                log_probs_R = self._model.R.log_prob(inputs=observations_actions, sample=batch['rewards'], reduction='mean', constant=False)
                
                loss_T = (-log_probs_T).mean(0)
                loss_R = (-log_probs_R).mean(0)
            
            if self.legal_move:
                legal_move_prob = self._model.legal_move.prob(inputs=observations)
                loss_legal_moves = torch.nn.functional.binary_cross_entropy(legal_move_prob.flatten(), batch['legal_moves'].flatten(), reduction="none").mean(0)
            else:
                loss_legal_moves = torch.tensor(0.)
        
        loss = self.loss_weights['T']*loss_T + self.loss_weights['R']*loss_R + self.loss_weights['masks']*loss_masks + self.loss_weights['legal_moves']*loss_legal_moves + beta_coeff*loss_KL_vae

        # cleans left-over gradients and updates networks
        self._model.optim.zero_grad()
        self.memory.optim.zero_grad()

        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(self._model.parameters(), self.grad_clip_norm)
        torch.nn.utils.clip_grad_norm_(self.memory.parameters(), self.grad_clip_norm)
        
        self._model.optim.step()
        self.memory.optim.step()
        
        return {'loss_T': loss_T.detach().cpu().numpy(), 'loss_R': loss_R.detach().cpu().numpy(), 'loss_masks': loss_masks.detach().cpu().numpy(), 'loss_legal_moves':loss_legal_moves.detach().cpu().numpy(), 'loss_KL_vae': loss_KL_vae.detach().cpu().numpy()
        }
         

    def update_from_batches(self, batches, train_device, **kwargs):

        # put models on training device
        self.to(train_device)

        # we convert and concat the list of batches into a batch of joint quantities
        joint_batch = ml.join_batches(batches, train_device)

        losses = self.update(joint_batch)

        return losses

    def binary_accuracy(self, predictions, targets):
        assert predictions.shape == targets.shape
        acc = torch.eq(predictions.flatten(),targets.flatten()).float().mean(0)
        return acc.detach().cpu().numpy()
    
    def mean_l2_distance(self, predictions, targets):
        assert predictions.shape == targets.shape
        return ((((predictions - targets)**2).sum(1))**0.5).mean(0).detach().cpu().numpy()

    def compute_accuraries(self, batches):
        self.eval()
        joint_batch = ml.join_batches(batches, self._model.device)

        observations = self.concat_embedings_to_obs(joint_batch, target=False)
        actions = self.transform_actions(joint_batch['actions'])

        if self.vae_wm: 

            T_pred, R_pred, mask_logits, legal_move_logits, mean_latent, log_std_latent = self._model(observations, actions)

            T_samples = T_pred
            mean_T = T_pred
            log_std_T = torch.zeros_like(mean_T)

            R_samples = R_pred
            mean_R = R_pred
            log_std_R = torch.zeros_like(mean_R)

            weights_T_entropy = 0.
            weights_R_entropy = 0.

            T_dist_means_weighted = 0.
            R_dist_means_weighted = 0.
            R_acc_greedy = 0.

            mask_probs = self._model.mask_decoder.binary_prob(mask_logits)
            mask_samples = torch.bernoulli(mask_probs)
            mask_greedy = torch.round(mask_probs)

            if self.legal_move:
                
                legal_moves_probs = self._model.legal_move.binary_prob(legal_move_logits)
                legal_moves_samples = torch.bernoulli(legal_moves_probs)
                legal_moves_greedy = torch.round(legal_moves_probs)

                legal_moves_acc_sample = self.binary_accuracy(legal_moves_samples, joint_batch['legal_moves'])
                legal_moves_acc_greedy = self.binary_accuracy(legal_moves_greedy, joint_batch['legal_moves'])
                legal_moves_baseline = self.binary_accuracy(torch.ones_like(joint_batch['legal_moves']), joint_batch['legal_moves'])

            else:
                legal_moves_acc_sample = 0.
                legal_moves_acc_greedy = 0.
                legal_moves_baseline = 0.

        else:
            observations_actions = torch.cat((observations, actions), dim=1)

            if self.deterministic_wm:
                
                # state predictions are the same regardless if we use sigmoid as activation function or not
                T_pred = self._model.T(observations_actions)
                T_samples, mean_T, log_std_T = T_pred, T_pred, torch.zeros_like(T_pred)

                weights_T_entropy = 0.
                weights_R_entropy = 0.

                T_dist_means_weighted = 0.
                R_dist_means_weighted = 0.

                if self.binary_reward_sigmoid_state:
                    R_pred = torch.round(self._model.R.prob(observations_actions))
                    R_acc_greedy = self.binary_accuracy(R_pred, joint_batch['rewards'])

                else:
                    R_pred = self._model.R(observations_actions)
                    R_acc_greedy = 0.
                
                R_samples, mean_R, log_std_R = R_pred, R_pred, torch.zeros_like(R_pred)
                    
                
            else:
                if self.k_mixture_gaussian > 1:
                    T_samples, modes_T, log_std_T, weights_T = self._model.T.sample(observations_actions, return_mode_log_std=True)
                    
                    R_samples, modes_R, log_std_R, weights_R = self._model.R.sample(observations_actions, return_mode_log_std=True)
                    weights_T_entropy = (- weights_T*weights_T.log().sum(1).mean(0)).detach().cpu().numpy()
                    weights_R_entropy = (- weights_R*weights_R.log().sum(1).mean(0)).detach().cpu().numpy()

                    mean_T = modes_T['sampled_mode']
                    T_dist_means_weighted = self.mean_l2_distance(modes_T['weighted_mode'], joint_batch['next_observations'])
                    mean_R = modes_R['sampled_mode']
                    R_dist_means_weighted = self.mean_l2_distance(modes_R['weighted_mode'], joint_batch['rewards'])

                    log_std_T = log_std_T['sampled_std']
                    log_std_R = log_std_R['sampled_std']
                else:
                    T_samples, mean_T, log_std_T = self._model.T.sample(observations_actions, return_mode_log_std=True)
                    R_samples, mean_R, log_std_R = self._model.R.sample(observations_actions, return_mode_log_std=True)
                    
                    weights_T_entropy = 0.
                    weights_R_entropy = 0.

                    T_dist_means_weighted = 0.
                    R_dist_means_weighted = 0.

                    R_acc_greedy = 0.

                
            mask_samples, mask_probs = self._model.mask.sample(observations_actions, return_prob=True)
            mask_greedy = torch.round(mask_probs)

            if self.legal_move:
            
                legal_moves_samples, legal_moves_probs = self._model.legal_move.sample(observations, return_prob=True)
                legal_moves_greedy = torch.round(legal_moves_probs)

                legal_moves_acc_sample = self.binary_accuracy(legal_moves_samples, joint_batch['legal_moves'])
                legal_moves_acc_greedy = self.binary_accuracy(legal_moves_greedy, joint_batch['legal_moves'])

                legal_moves_baseline = self.binary_accuracy(torch.ones_like(joint_batch['legal_moves']), joint_batch['legal_moves'])

            else:
                legal_moves_acc_sample = 0.
                legal_moves_acc_greedy = 0.
                legal_moves_baseline = 0.

        T_dist = self.mean_l2_distance(T_samples, joint_batch['next_observations'])
        R_dist = self.mean_l2_distance(R_samples, joint_batch['rewards'])

        T_baseline = self.mean_l2_distance(joint_batch['observations'], joint_batch['next_observations'])
        R_baseline = self.mean_l2_distance(torch.zeros_like(joint_batch['rewards']), joint_batch['rewards'])

        T_dist_means = self.mean_l2_distance(mean_T, joint_batch['next_observations'])
        R_dist_means = self.mean_l2_distance(mean_R, joint_batch['rewards'])

        T_std, R_std = log_std_T.exp(), log_std_R.exp()

        T_std_mean, T_std_max, T_std_min = T_std.mean().detach().cpu().numpy(), T_std.max().detach().cpu().numpy(), T_std.min().detach().cpu().numpy()
        R_std_mean, R_std_max, R_std_min = R_std.mean().detach().cpu().numpy(), R_std.max().detach().cpu().numpy(), R_std.min().detach().cpu().numpy()

        
        masks_acc_sample = self.binary_accuracy(mask_samples, joint_batch['masks'])
        masks_acc_greedy = self.binary_accuracy(mask_greedy, joint_batch['masks'])

        masks_baseline = self.binary_accuracy(torch.ones_like(joint_batch['masks']), joint_batch['masks'])

        R_min = R_samples.min().detach().cpu().numpy()
        R_max = R_samples.max().detach().cpu().numpy()

        return T_dist, T_dist_means, T_dist_means_weighted, R_dist, R_acc_greedy, R_min, R_max, R_dist_means, R_dist_means_weighted, T_std_mean, T_std_max, T_std_min, R_std_mean, R_std_max, R_std_min, masks_acc_sample, masks_acc_greedy, legal_moves_acc_sample, legal_moves_acc_greedy, weights_T_entropy, weights_R_entropy, T_baseline, R_baseline, masks_baseline, legal_moves_baseline

    def evaluate(self, train_batches, valid_batches):
        with torch.no_grad():
            T_dist_train, T_dist_means_train, T_dist_means_weighted_train, R_dist_train, R_acc_greedy_train, R_min_train, R_max_train, R_dist_means_train, R_dist_means_weighted_train, T_std_mean_train, T_std_max_train, T_std_min_train, R_std_mean_train, R_std_max_train, R_std_min_train, masks_acc_sample_train, masks_acc_greedy_train, legal_moves_acc_sample_train, legal_moves_acc_greedy_train, weights_T_entropy_train, weights_R_entropy_train, T_baseline_train, R_baseline_train, masks_baseline_train, legal_moves_baseline_train = self.compute_accuraries(train_batches)
            T_dist_valid, T_dist_means_valid, T_dist_means_weighted_valid, R_dist_valid, R_acc_greedy_valid, R_min_valid, R_max_valid, R_dist_means_valid, R_dist_means_weighted_valid, T_std_mean_valid, T_std_max_valid, T_std_min_valid, R_std_mean_valid, R_std_max_valid, R_std_min_valid, masks_acc_sample_valid, masks_acc_greedy_valid, legal_moves_acc_sample_valid, legal_moves_acc_greedy_valid, weights_T_entropy_valid, weights_R_entropy_valid, T_baseline_valid, R_baseline_valid, masks_baseline_valid, legal_moves_baseline_valid = self.compute_accuraries(valid_batches)
            return {'T_dist_train': T_dist_train,
                    'T_dist_means_train' : T_dist_means_train,
                    'T_dist_means_weighted_train': T_dist_means_weighted_train,
                    'R_dist_means_weighted_train': R_dist_means_weighted_train,
                    'R_dist_train': R_dist_train, 
                    'R_acc_greedy_train': R_acc_greedy_train,
                    'R_min_train': R_min_train,
                    'R_max_train': R_max_train,
                    'R_dist_means_train' : R_dist_means_train,
                    'T_std_mean_train': T_std_mean_train , 'T_std_max_train': T_std_max_train, 'T_std_min_train': T_std_min_train, 'R_std_mean_train': R_std_mean_train, 'R_std_max_train': R_std_max_train, 'R_std_min_train': R_std_min_train,
                    'masks_acc_sample_train': masks_acc_sample_train, 'legal_moves_acc_sample_train': legal_moves_acc_sample_train, 'masks_acc_greedy_train': masks_acc_greedy_train, 'legal_moves_acc_greedy_train': legal_moves_acc_greedy_train, 'weights_T_entropy_train': weights_T_entropy_train, 'weights_R_entropy_train': weights_R_entropy_train,
                    'T_dist_valid': T_dist_valid,
                    'T_dist_means_valid' : T_dist_means_valid,
                    'T_dist_means_weighted_valid': T_dist_means_weighted_valid,
                    'R_dist_means_weighted_valid': R_dist_means_weighted_valid,
                    'R_dist_valid': R_dist_valid, 
                    'R_acc_greedy_valid': R_acc_greedy_valid,
                    'R_min_valid': R_min_valid,
                    'R_max_valid': R_max_valid,
                    'R_dist_means_valid' : R_dist_means_valid,
                    'T_std_mean_valid': T_std_mean_valid , 'T_std_max_valid': T_std_max_valid, 'T_std_min_valid': T_std_min_valid, 'R_std_mean_valid': R_std_mean_valid, 'R_std_max_valid': R_std_max_valid, 'R_std_min_valid': R_std_min_valid,
                    'masks_acc_sample_valid': masks_acc_sample_valid, 'legal_moves_acc_sample_valid': legal_moves_acc_sample_valid, 'masks_acc_greedy_valid': masks_acc_greedy_valid, 'legal_moves_acc_greedy_valid': legal_moves_acc_greedy_valid, 'weights_T_entropy_valid': weights_T_entropy_valid, 'weights_R_entropy_valid': weights_R_entropy_valid,
                    'T_baseline_train': T_baseline_train, 'R_baseline_train': R_baseline_train, 'masks_baseline_train': masks_baseline_train, 'legal_moves_baseline_train': legal_moves_baseline_train,
                    'T_baseline_valid': T_baseline_valid, 'R_baseline_valid': R_baseline_valid, 'masks_baseline_valid': masks_baseline_valid, 'legal_moves_baseline_valid': legal_moves_baseline_valid,}
    
    def get_state_dict(self):
        return {'_model': self._model.get_state_dict(), 
                'memory': self.memory.get_state_dict(),
                'memorytarget': self.memorytarget.get_state_dict()}
    
    def do_load_state_dict(self, state_dict):
        self._model.do_load_state_dict(state_dict['_model'])
        self.memory.do_load_state_dict(state_dict['memory']) 
        self.memorytarget.do_load_state_dict(state_dict['memorytarget'])