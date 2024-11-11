import numpy as np
import random
import os
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from gym import spaces
import warnings


def set_seeds(seed, env):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if env is not None:
        env.seed(seed)
    torch.backends.cudnn.deterministic = True


def get_env_dims(env):
    try:
        obs_space = {obs_name: obs_space_i.shape for obs_name, obs_space_i in env.observation_space.items()}
    except AttributeError:
        obs_space = env.observation_space
    act_space = env.action_space
    dims = {'obs_space': obs_space, 'act_space': act_space}

    if hasattr(env, 'n_agent'):
        dims.update({'n_learner':env.n_agent})

    return dims 


def get_act_properties_from_act_space(act_space):
    if isinstance(act_space, spaces.Box):
        act_size = int(np.prod(act_space.shape))
        is_discrete = int(False)
    elif isinstance(act_space, spaces.Discrete):
        act_size = act_space.n
        is_discrete = int(True)
    else:
        raise NotImplementedError

    return act_size, is_discrete


def discrete_from_one_hot(one_hot):
    if isinstance(one_hot, np.ndarray):
        return np.argmax(one_hot, -1)
    else:
        return torch.argmax(one_hot, -1, keepdim=True)


def data_to_device(data, device):
    if device is not None:
        data.to(device)


def to_device(model, device):
    if not model.device == device:
        data_to_device(data=model, device=device)


def to_numpy(data):
    return data.data.cpu().numpy()

def convert_to_numpy(val):
    if not isinstance(val, np.ndarray):
        if isinstance(val, torch.Tensor):
            val = val.detach().cpu().numpy()
        else:
            val = np.asarray(val)
    return val

def save_checkpoint(state, filename):
    torch.save(state, filename)


def onehot_from_index(index, onehot_size):
    if hasattr(index, '__len__'):
        one_hots = np.zeros((len(index), onehot_size))
        one_hots[np.arange(len(index)), index] = 1
    else:
        one_hots = np.zeros(onehot_size)
        one_hots[index] = 1
    return one_hots

def batch_to_device(batch, device):
    if not isinstance(batch, dict):
        batch = batch._asdict()
    
    for key, val in batch.items():
        batch[key] = val.to(device) if hasattr(val, 'to') else val

    return batch

def batch_as_tensor(batch):
    if not isinstance(batch, dict):
        batch = batch._asdict()
    
    for key, val in batch.items():
        batch[key] = torch.as_tensor(val)

    return batch

def join_batches(batches, device):
    # we convert and concat the list of batches into a batch of joint quantities

    joint_batch_as_dict = {}
    for batch in batches:
        batch = batch_to_device(batch_as_tensor(batch), device)

        for key, data in batch.items():

            if key in joint_batch_as_dict:
                joint_batch_as_dict[key].append(data)
            else:
                joint_batch_as_dict[key] = [data]
    
    joint_batch = {key: torch.cat(datas_list, dim=-1) if hasattr(datas_list[0], '__len__') else datas_list[0] for key, datas_list in joint_batch_as_dict.items()}

    return joint_batch

def check_if_nan_in_batch(batch):
    if not isinstance(batch, dict):
        batch = batch._asdict()

    for key, val in batch.items():
        if torch.isnan(val).any():
            raise ValueError(f"nan found in {key}")


def onehot_from_logits(logits, eps=0.0):
    """
    Given batch of logits, return one-hot sample using epsilon greedy strategy
    (based on given epsilon)
    """
    # get best (according to current policy) actions in one-hot form
    argmax_acs = (logits == logits.max(1, keepdim=True)[0]).float()
    if eps == 0.0:
        return argmax_acs
    # get random actions in one-hot form
    rand_acs = Variable(torch.eye(logits.shape[1])[[np.random.choice(
        range(logits.shape[1]), size=logits.shape[0])]], requires_grad=False)
    # chooses between best and random actions using epsilon greedy
    return torch.stack([argmax_acs[i] if r > eps else rand_acs[i] for i, r in
                        enumerate(torch.rand(logits.shape[0]))])


# modified for PyTorch from https://github.com/ericjang/gumbel-softmax/blob/master/Categorical%20VAE.ipynb
def sample_gumbel(same_as, eps=1e-20):
    """Sample from Gumbel(0, 1)"""
    U = torch.zeros_like(same_as, requires_grad=False)
    U.data.uniform_()
    return -torch.log(-torch.log(U + eps) + eps)


# modified for PyTorch from https://github.com/ericjang/gumbel-softmax/blob/master/Categorical%20VAE.ipynb
def gumbel_softmax_sample(logits, temperature):
    """ Draw a sample from the Gumbel-Softmax distribution"""
    y = logits + sample_gumbel(same_as=logits)
    return F.softmax(y / temperature, dim=-1)


# modified for PyTorch from https://github.com/ericjang/gumbel-softmax/blob/master/Categorical%20VAE.ipynb
def gumbel_softmax(logits, temperature=1.0, hard=False):
    """Sample from the Gumbel-Softmax distribution and optionally discretize.
    Args:
      logits: [batch_size, n_class] unnormalized log-probs (but divided by alpha)
      temperature: non-negative scalar
      hard: if True, take argmax, but differentiate w.r.t. soft sample y
    Returns:
      [batch_size, n_class] sample from the Gumbel-Softmax distribution.
      If hard=True, then the returned sample will be one-hot, otherwise it will
      be a probabilitiy distribution that sums to 1 across classes
    """
    y = gumbel_softmax_sample(logits, temperature)
    if hard:
        y_hard = onehot_from_logits(y)
        y = (y_hard - y).detach() + y
    return y


def log_sum_exp(value, dim=None, keepdim=False):
    """Numerically stable implementation of the operation
    value.exp().sum(dim, keepdim).log()
    """
    if dim is not None:
        m, _ = torch.max(value, dim=dim, keepdim=True)
        value0 = value - m
        if keepdim is False:
            m = m.squeeze(dim)
        return m + torch.log(torch.sum(torch.exp(value0),
                                       dim=dim, keepdim=keepdim))
    else:
        m = torch.max(value)
        sum_exp = torch.sum(torch.exp(value - m))
        return m + torch.log(sum_exp)


def save(models, directory, suffix):
    if not directory is None:
        for model in models:
            model.save(f"{directory / model.name}_{suffix}")


def wandb_watch(wandb, learners):
    to_watch = []
    for learner in learners:
        to_watch += learner.wandb_watchable()

    if len(to_watch) > 0:
        wandb.watch(to_watch, log="all")


def remove(models, directory, suffix):
    for model in models:
        os.remove(f"{directory / model.name}_{suffix}")

def mask(done):
    return 0 if done else 1

def soft_update(target, source, tau):
    """
    Perform DDPG soft update (move target params toward source based on weight factor tau)
    Inputs:
        target (torch.nn.Module): Net to copy parameters to
        source (torch.nn.Module): Net whose parameters to copy
        tau (float, 0 <= x <= 1): Weight factor for update
    """
    assert 0. <= tau and tau <= 1.
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
    
def hard_update(target, source):
    soft_update(target, source, 1.)

def get_freer_gpu():
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
    memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    try:
        return np.argmax(memory_available)
    except:
        warnings.warn(f'get_freer_gpu got a problem, just returning gpu:0')
        return 0

def get_computing_devices(use_gpu, torch, do_rollouts_on_cpu, logger=None):
    if use_gpu:
        if torch.cuda.is_available():
            gpu_idx = get_freer_gpu()

            train_device = torch.device(f'cuda:{gpu_idx}')

        else:
            train_device = torch.device('cpu')
            if logger is not None:
                logger.warning("You requested GPU usage but torch.cuda.is_available() returned False")
            else:
                warnings.warn("You requested GPU usage but torch.cuda.is_available() returned False")
    else:
        train_device = torch.device('cpu')

    if do_rollouts_on_cpu:
        rollout_device = torch.device('cpu')
    else:
        rollout_device = train_device

    return train_device, rollout_device, 