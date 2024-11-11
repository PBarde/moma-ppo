import torch
from torch import nn

from offline_marl.utils.ml import *

class BaseAlgo(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)

    def save(self, filepath):
        save_dict = {'init_dict': self.init_dict, 'state_dict': self.get_state_dict()}
        save_checkpoint(save_dict, filename=filepath)

    @classmethod
    def init_from_save(cls, filename, device):
        save_dict = torch.load(filename, map_location=device)
        return cls.init_from_dict(save_dict)

    @classmethod
    def init_from_dict(cls, save_dict):
        instance = cls(**save_dict['init_dict'])
        instance.do_load_state_dict(save_dict['state_dict'])
        return instance

    def do_load_state_dict(self, state_dict):
        raise NotImplementedError

    def get_state_dict(self):
        raise NotImplementedError

    def prep_rollout(self, rollout_device):
        raise NotImplementedError

    def prep_training(self, train_device):
        raise NotImplementedError

    def act(self, **kwargs):
        raise NotImplementedError

    @property
    def device(self):
        raise NotImplementedError

    def save_training_graphs(self, *args, **kwargs):
        raise NotImplementedError

    def wandb_watchable(self):
        raise 