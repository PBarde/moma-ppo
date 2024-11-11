import torch.nn.functional as F
import torch
from torch import nn, optim
import math
from nop import NOP
from offline_marl.utils.networks import MLPNetwork, SpectralMLPNetwork, MLPNetwork4Layers

MIN_LOG_STD = -6
MAX_LOG_STD = 2

class Gaussian(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_size, four_layers=False):
        super().__init__()

        if four_layers:
            self.network = MLPNetwork4Layers(in_dim, 2*out_dim, hidden_size)
        else:
            self.network = MLPNetwork(in_dim, 2*out_dim, hidden_size)

    def __call__(self, inputs):
        logits = self.network(inputs)
        mean, log_std = logits.split(logits.shape[1] // 2, dim=1)
        log_std = torch.clamp(log_std, min=MIN_LOG_STD, max=MAX_LOG_STD)
        return mean, log_std

    def log_prob(self, inputs, sample, reduction='sum', constant=False):
        mean, log_std = self(inputs)
        return self.gaussian_log_prob(mean, log_std, sample, reduction)

    def sample(self, inputs, return_mode_log_std=True):
        mean, log_std = self(inputs)
        if return_mode_log_std:
            return self.gaussian_sample(mean, log_std), mean, log_std
        else:
            return self.gaussian_sample(mean, log_std)

    @staticmethod
    def gaussian_log_prob(mean, log_std, sample, reduction='sum', constant=False):
        var = log_std.exp()
        # we clip the var (cf torch implementation of GaussianNLL) for stability but this doesn't affect the gradients 
        cliped_var = var + (torch.clamp(var, min=MIN_LOG_STD)-var).detach()

        z = (sample - mean) / cliped_var
        log_prob_vector = (- 0.5 * (2 * log_std + z ** 2 + math.log(2 * math.pi)*constant))
        if reduction =='sum':
            log_prob = log_prob_vector.sum(dim=1).unsqueeze(1)
        elif reduction == 'mean':
            # to match GaussianNLL implementation
            log_prob = log_prob_vector.mean(dim=1).unsqueeze(1)
        else:
            raise NotImplementedError
        return log_prob

    @staticmethod
    def gaussian_sample(mean, log_std):
        return mean + torch.randn_like(mean)*log_std.exp()

    @property
    def device(self):
        return next(self.parameters()).device

    def get_state_dict(self):
        return {'network': self.network.state_dict()}

    def do_load_state_dict(self, state_dict):
        self.network.load_state_dict(state_dict['network'])

class SpectralGaussian(Gaussian):
    def __init__(self, in_dim, out_dim, hidden_size):
        super().__init__(in_dim, out_dim, hidden_size)

        self.network = SpectralMLPNetwork(num_inputs=in_dim, 
                                            num_outputs_with_spectral_left=0, 
                                            num_outputs_without_spectral_middle=2*out_dim, 
                                            num_outputs_with_spectral_right=0, 
                                            hidden_size=hidden_size)
    

class Binary(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_size):
        super().__init__()

        self.network = MLPNetwork(in_dim, out_dim, hidden_size)

    def __call__(self, inputs):
        return self.network(inputs)

    def prob(self, inputs):
        return self.binary_prob(self(inputs))
        
    def sample(self, inputs, return_prob=True):
        logits = self(inputs)
        probs = self.binary_prob(logits)
        samples = torch.bernoulli(probs)
        if return_prob:
            return samples, probs
        else:
            return samples

    @staticmethod
    def binary_prob(logits):
        return torch.sigmoid(logits)

    @property
    def device(self):
        return next(self.parameters()).device

    def get_state_dict(self):
        return {'network': self.network.state_dict()}

    def do_load_state_dict(self, state_dict):
        self.network.load_state_dict(state_dict['network'])


class GaussianMixture(nn.Module):
    def __init__(self, k, in_dim, out_dim, hidden_size):
        super().__init__()
        
        self.k = k

        self.network = MLPNetwork(in_dim, 2*out_dim*k + k, hidden_size)

        self.out_dim = out_dim

    def __call__(self, inputs):
        logits = self.network(inputs)
        mean, log_std, weight_logits = logits.split((self.k*self.out_dim, self.k*self.out_dim, self.k), dim=1)
        log_std = torch.clamp(log_std, min=MIN_LOG_STD, max=MAX_LOG_STD)

        weights_list = torch.softmax(weight_logits, dim=1).split([1 for _ in range(self.k)], dim=1)
        mean_list = mean.split([self.out_dim for _ in range(self.k)], dim=1)
        log_std_list = log_std.split([self.out_dim for _ in range(self.k)], dim=1)
        return mean_list, log_std_list, weights_list

    def log_prob(self, inputs, sample, reduction='sum', constant=False):
        mean_list, log_std_list, weights_list  = self(inputs)
        weighted_log_likelihoods = [Gaussian.gaussian_log_prob(mean, log_std, sample, reduction, constant) + weight.log() 
                                        for weight, mean, log_std in zip(weights_list, mean_list, log_std_list)]
        
        weighted_log_likelihoods = torch.cat(weighted_log_likelihoods, dim=1)
        log_prob = torch.logsumexp(weighted_log_likelihoods, dim=1)
        return log_prob

    def sample(self, inputs, return_mode_log_std=True):
        mean_list, log_std_list, weights_list = self(inputs)
        weights = torch.cat(weights_list, dim=1)

        chosen_gaussians = torch.multinomial(weights, num_samples=1)
        
        mean_stacked = torch.stack(mean_list, dim=1)
        log_std_stacked = torch.stack(log_std_list, dim=1)
        chosen_gaussians_idx = chosen_gaussians.unsqueeze(-1).repeat(1, 1, mean_stacked.shape[2])
        means = torch.gather(mean_stacked, dim=1, index=chosen_gaussians_idx).squeeze(1)
        log_stds = torch.gather(log_std_stacked, dim=1, index=chosen_gaussians_idx).squeeze(1)

        if return_mode_log_std:
            weighted_mode = (mean_stacked*weights.unsqueeze(2)).sum(1)
            return Gaussian.gaussian_sample(means, log_stds), {'sampled_mode': means, 'weighted_mode': weighted_mode}, {'sampled_std': log_stds}, weights
        else:
            return Gaussian.gaussian_sample(means, log_stds)

    @property
    def device(self):
        return next(self.parameters()).device

    def get_state_dict(self):
        return {'network': self.network.state_dict()}

    def do_load_state_dict(self, state_dict):
        self.network.load_state_dict(state_dict['network'])

class GaussianModel(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_size, lr, train_device, weight_decay):
        super().__init__()

        self.gaussian = Gaussian(in_dim, out_dim, hidden_size)

        # we put the parameters on the right device before creating the optimizer
        self.to(train_device)

        self.optim = optim.Adam(params=self.parameters(), lr=lr, weight_decay=weight_decay)

    @property
    def device(self):
        return next(self.parameters()).device

    def get_state_dict(self):
        return {'network': self.network.state_dict(),
                'optim': self.optim.state_dict()}

    def do_load_state_dict(self, state_dict):
        self.network.load_state_dict(state_dict['network'])
        self.optim.load_state_dict(state_dict['optim'])

class WorldModelNNs(nn.Module):
    def __init__(self, obs_mem_dim, act_dim, out_state_dim, out_reward_dim, out_mask_dim, out_legal_move_dim, hidden_size, lr, train_device, weight_decay, **kwargs):
        nn.Module.__init__(self)

        if kwargs['spectral_norm']:
            self.T = SpectralGaussian(obs_mem_dim + act_dim, out_state_dim, hidden_size)
            self.R = SpectralGaussian(obs_mem_dim + act_dim, out_reward_dim, hidden_size)
        else:
            self.T = Gaussian(obs_mem_dim + act_dim, out_state_dim, hidden_size, four_layers=kwargs['four_layers_wm'])
            self.R = Gaussian(obs_mem_dim + act_dim, out_reward_dim, hidden_size, four_layers=kwargs['four_layers_wm'])

        self.mask = Binary(obs_mem_dim + act_dim, out_mask_dim, hidden_size)

        if out_legal_move_dim > 0:
            self.legal_move = Binary(obs_mem_dim, out_legal_move_dim, hidden_size)
        else:
            self.legal_move = NOP()

        # we put the parameters on the right device before creating the optimizer
        self.to(train_device)

        self.optim = optim.Adam(params=self.parameters(), lr=lr, weight_decay=weight_decay)

    def __call__(self, obs_mem, act):

        obs_mem_act = torch.cat((obs_mem, act), dim=1)

        mean_T, log_std_T = self.T(obs_mem_act)
        mean_R, log_std_R = self.R(obs_mem_act)
        mask_logits = self.mask(obs_mem_act)
        legal_move_logits = self.legal_move(obs_mem)
        return mean_T, log_std_T, mean_R, log_std_R, mask_logits, legal_move_logits

    @property
    def device(self):
        return next(self.parameters()).device

    def get_state_dict(self):
        return {'T': self.T.state_dict(),
                'R': self.R.state_dict(),
                'mask': self.mask.state_dict(),
                'legal_move': self.legal_move.state_dict(),
                'optim': self.optim.state_dict()}

    def do_load_state_dict(self, state_dict):
        self.T.load_state_dict(state_dict['T'])
        self.R.load_state_dict(state_dict['R'])
        self.mask.load_state_dict(state_dict['mask'])
        self.legal_move.load_state_dict(state_dict['legal_move'])
        self.optim.load_state_dict(state_dict['optim'])

class DeterministicWorldModelNNs(nn.Module):
    def __init__(self,  obs_mem_dim, act_dim, out_state_dim, out_reward_dim, out_mask_dim, out_legal_move_dim, hidden_size, lr, train_device, weight_decay):
        nn.Module.__init__(self)

        self.T = MLPNetwork(obs_mem_dim + act_dim, out_state_dim, hidden_size)
        self.R = MLPNetwork(obs_mem_dim + act_dim, out_reward_dim, hidden_size)
        self.mask = Binary(obs_mem_dim + act_dim, out_mask_dim, hidden_size)

        if out_legal_move_dim > 0:
            self.legal_move = Binary(obs_mem_dim, out_legal_move_dim, hidden_size)
        else:
            self.legal_move = NOP()

        # we put the parameters on the right device before creating the optimizer
        self.to(train_device)

        self.optim = optim.Adam(params=self.parameters(), lr=lr, weight_decay=weight_decay)

    def __call__(self, obs_mem, act):

        obs_mem_act = torch.cat((obs_mem, act), dim=1)

        T_pred = self.T(obs_mem_act)
        R_pred = self.R(obs_mem_act)
        mask_logits = self.mask(obs_mem_act)
        legal_move_logits = self.legal_move(obs_mem)
        return T_pred, R_pred, mask_logits, legal_move_logits

    @property
    def device(self):
        return next(self.parameters()).device

    def get_state_dict(self):
        return {'T': self.T.state_dict(),
                'R': self.R.state_dict(),
                'mask': self.mask.state_dict(),
                'legal_move': self.legal_move.state_dict(),
                'optim': self.optim.state_dict()}

    def do_load_state_dict(self, state_dict):
        self.T.load_state_dict(state_dict['T'])
        self.R.load_state_dict(state_dict['R'])
        self.mask.load_state_dict(state_dict['mask'])
        self.legal_move.load_state_dict(state_dict['legal_move'])
        self.optim.load_state_dict(state_dict['optim'])


class BinaryRewardSigmoidStateWorldModelNNs(nn.Module):
    def __init__(self,  obs_mem_dim, act_dim, out_state_dim, out_reward_dim, out_mask_dim, out_legal_move_dim, hidden_size, lr, train_device, weight_decay):
        nn.Module.__init__(self)

        self.T = MLPNetwork(obs_mem_dim + act_dim, out_state_dim, hidden_size, sigmoid_output=True)
        self.R = Binary(obs_mem_dim + act_dim, out_reward_dim, hidden_size)
        self.mask = Binary(obs_mem_dim + act_dim, out_mask_dim, hidden_size)

        if out_legal_move_dim > 0:
            self.legal_move = Binary(obs_mem_dim, out_legal_move_dim, hidden_size)
        else:
            self.legal_move = NOP()

        # we put the parameters on the right device before creating the optimizer
        self.to(train_device)

        self.optim = optim.Adam(params=self.parameters(), lr=lr, weight_decay=weight_decay)

    def __call__(self, obs_mem, act):

        obs_mem_act = torch.cat((obs_mem, act), dim=1)

        T_pred = self.T(obs_mem_act)
        R_logits = self.R(obs_mem_act)
        mask_logits = self.mask(obs_mem_act)
        legal_move_logits = self.legal_move(obs_mem)
        return T_pred, R_logits, mask_logits, legal_move_logits

    @property
    def device(self):
        return next(self.parameters()).device

    def get_state_dict(self):
        return {'T': self.T.state_dict(),
                'R': self.R.state_dict(),
                'mask': self.mask.state_dict(),
                'legal_move': self.legal_move.state_dict(),
                'optim': self.optim.state_dict()}

    def do_load_state_dict(self, state_dict):
        self.T.load_state_dict(state_dict['T'])
        self.R.load_state_dict(state_dict['R'])
        self.mask.load_state_dict(state_dict['mask'])
        self.legal_move.load_state_dict(state_dict['legal_move'])
        self.optim.load_state_dict(state_dict['optim'])

class VAEWorldModelNNs(nn.Module):
    def __init__(self, obs_mem_dim, act_dim, out_state_dim, out_reward_dim, out_mask_dim, out_legal_move_dim, hidden_size, latent_size, state_sigmoid_output, lr, train_device, weight_decay):
        nn.Module.__init__(self)

        # encodes memory and state
        self.encoder = MLPNetwork(num_inputs=obs_mem_dim, num_outputs=hidden_size, hidden_size=hidden_size)

        # produces latent gaussian from encoding and actions
        self.latent_gaussian = Gaussian(in_dim=hidden_size+act_dim, out_dim=latent_size, hidden_size=hidden_size)

        # the different decoders

        self.T_decoder = MLPNetwork(latent_size + act_dim, out_state_dim, hidden_size, sigmoid_output=state_sigmoid_output)
        self.R_decoder = MLPNetwork(latent_size + act_dim, out_reward_dim, hidden_size, sigmoid_output=False)
        self.mask_decoder = Binary(latent_size + act_dim, out_mask_dim, hidden_size)

        if out_legal_move_dim > 0:
            # legal_move is a deterministic function of the current state nothing more
            self.legal_move = Binary(obs_mem_dim, out_legal_move_dim, hidden_size)
        else:
            self.legal_move = NOP()

        # we put the parameters on the right device before creating the optimizer
        self.to(train_device)

        self.optim = optim.Adam(params=self.parameters(), lr=lr, weight_decay=weight_decay)

    def get_latent(self, obs_mem, act):
        # encodes memory and obs 
        emb = self.encoder(obs_mem)
        
        # concatenates embedding to action taken
        emb_act = torch.cat((emb, act), dim=1)

        # computes latent gaussian mean and std and samples in it (reparametrization trick obvi)
        latent, mean_latent, log_std_latent = self.latent_gaussian.sample(inputs=emb_act, return_mode_log_std=True)

        return latent, mean_latent, log_std_latent


    def __call__(self, obs_mem, act):
        
        latent, mean_latent, log_std_latent = self.get_latent(obs_mem, act)

        # concatenates latent and action for reconstruction
        latent_act = torch.cat((latent, act), dim=1)

        # decodes latent for reconstruction losses
        T_pred = self.T_decoder(latent_act)
        R_pred = self.R_decoder(latent_act)
        mask_logits = self.mask_decoder(latent_act)

        # legal move is just a deteministic function of the state
        legal_move_logits = self.legal_move(obs_mem)

        return T_pred, R_pred, mask_logits, legal_move_logits, mean_latent, log_std_latent

    @property
    def device(self):
        return next(self.parameters()).device

    def get_state_dict(self):
        return {'encoder': self.encoder.state_dict(),
                'latent_gaussian': self.latent_gaussian.state_dict(),
                'T_decoder': self.T_decoder.state_dict(),
                'R_decoder': self.R_decoder.state_dict(),
                'mask_decoder': self.mask_decoder.state_dict(),
                'legal_move': self.legal_move.state_dict(),
                'optim': self.optim.state_dict()}

    def do_load_state_dict(self, state_dict):
        self.encoder.load_state_dict(state_dict['encoder'])
        self.latent_gaussian.load_state_dict(state_dict['latent_gaussian'])
        self.T_decoder.load_state_dict(state_dict['T_decoder'])
        self.R_decoder.load_state_dict(state_dict['R_decoder'])
        self.mask_decoder.load_state_dict(state_dict['mask_decoder'])
        self.legal_move.load_state_dict(state_dict['legal_move'])
        self.optim.load_state_dict(state_dict['optim'])
class MixtureWorldModelNNs(nn.Module):
    def __init__(self, k, obs_mem_dim, act_dim, out_state_dim, out_reward_dim, out_mask_dim, out_legal_move_dim, hidden_size, lr, train_device, weight_decay):
        nn.Module.__init__(self)

        self.T = GaussianMixture(k, obs_mem_dim + act_dim, out_state_dim, hidden_size)
        self.R = GaussianMixture(k, obs_mem_dim + act_dim, out_reward_dim, hidden_size)
        self.mask = Binary(obs_mem_dim + act_dim, out_mask_dim, hidden_size)

        if out_legal_move_dim > 0:
            self.legal_move = Binary(obs_mem_dim, out_legal_move_dim, hidden_size)
        else:
            self.legal_move = NOP()

        # we put the parameters on the right device before creating the optimizer
        self.to(train_device)

        self.optim = optim.Adam(params=self.parameters(), lr=lr, weight_decay=weight_decay)

    @property
    def device(self):
        return next(self.parameters()).device

    def get_state_dict(self):
        return {'T': self.T.state_dict(),
                'R': self.R.state_dict(),
                'mask': self.mask.state_dict(),
                'legal_move': self.legal_move.state_dict(),
                'optim': self.optim.state_dict()}

    def do_load_state_dict(self, state_dict):
        self.T.load_state_dict(state_dict['T'])
        self.R.load_state_dict(state_dict['R'])
        self.mask.load_state_dict(state_dict['mask'])
        self.legal_move.load_state_dict(state_dict['legal_move'])
        self.optim.load_state_dict(state_dict['optim'])