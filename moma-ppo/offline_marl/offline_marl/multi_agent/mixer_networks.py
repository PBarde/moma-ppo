from torch import nn
from torch.optim import Adam

from offline_marl.utils.networks import MLPNetwork
from offline_marl.single_agent.critics import Q

#TODO: not sure about this double inheritance to get Q's methods
class MixerNetwork(Q, nn.Module):
    def __init__(self, global_obs_size, n_learners, hidden_size, lr, train_device):
        nn.Module.__init__(self)

        self.global_obs_size = global_obs_size
        self.n_learners = n_learners
        self.network = MLPNetwork(num_inputs=global_obs_size, num_outputs=n_learners + 1, hidden_size=hidden_size)
        # we put the parameters on the right device before creating the optimizer
        self.to(train_device)

        self.optim = Adam(params=self.parameters(), lr=lr)

    def __call__(self, input):
        mixing_coefs = self.network(input)
        mixing_weights, mixing_biaises = mixing_coefs[:,:self.n_learners], mixing_coefs[:, self.n_learners:]
        return mixing_weights, mixing_biaises

    @staticmethod
    def mix(values, weights, bias):
        return (values * weights).sum(1, keepdim=True) + bias
    
