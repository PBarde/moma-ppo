from collections import OrderedDict
from torch import nn
import time
from copy import deepcopy
import torch

import torch.multiprocessing as multiprocessing
import numpy as np

from alfred.utils.recorder import Aggregator

from offline_marl.multi_agent.worldmodel import WorldModel
from offline_marl.utils.base_algo import BaseAlgo
from offline_marl.utils import ml

class WorldModelEnsemble(BaseAlgo):
    def __init__(self, wm_constructor=WorldModel, learner_name='world-model', wm_suffix='wm', **kwargs):
        BaseAlgo.__init__(self)

        self.init_manager(self, kwargs, wm_constructor, learner_name, wm_suffix)

    @staticmethod
    def init_manager(manager, kwargs, wm_constructor, learner_name, wm_suffix):
        manager.n_wm_trained = kwargs['n_wm_trained']
        manager.n_wm_used = kwargs['n_wm_used']
        manager.top_wms = [None for _ in range(manager.n_wm_used)]
        
        manager.wms = nn.ModuleList()

        for i in range(manager.n_wm_trained):
            wm_kwargs = kwargs.copy()
            wm_kwargs.update({'wm_name': wm_suffix + f'_{i}'})
            manager.wms.append(wm_constructor(**wm_kwargs))
    
        # to save and reload
        manager.init_dict = kwargs
        manager.name = learner_name

        manager.evaluation_metrics = set()
        manager.train_metrics = {'training_step'}
        for i, wm in enumerate(manager.wms):
            prefix = f'wm_{i}_'
            for metric in wm.train_metrics:
                metric = prefix + metric
                manager.train_metrics.add(metric)

            for metric in wm.evaluation_metrics:
                metric = prefix + metric
                manager.evaluation_metrics.add(metric)
        
        manager.metrics_to_record = manager.evaluation_metrics | manager.train_metrics

        # for multithreading
        manager.processes = None
        manager.batch_queues = None
        manager.stat_queues = None
        manager.model_queues = None

        manager.dataset_on_shared_memory = False

    def update(self, dataset, batch_size, train_device, train_in_parallel=False):
        self.train()
        if train_in_parallel: 
            
            # this list dims are (n_wm, n_agents, n_idx) because each wm considers all the agents as one
            batches_list = [dataset.sample(batch_size) for _ in range(self.n_wm_trained)]

            if self.processes is None:

                self.batch_queues = [multiprocessing.Queue() for _ in range(self.n_wm_trained)]
                self.stat_queues = [multiprocessing.Queue() for _ in range(self.n_wm_trained)]
                self.model_queues = [multiprocessing.Queue() for _ in range(self.n_wm_trained)]
                self.processes = [multiprocessing.Process(target=update_wm_with_queues, args=(wm, batch_queue, stat_queue, model_queue, train_device)) 
                        for wm, batch_queue, stat_queue, model_queue in zip(self.wms, self.batch_queues, self.stat_queues, self.model_queues)]
                
                for p in self.processes:
                    p.start()
            
            _ = [batch_queue.put(batches) for batch_queue, batches in zip(self.batch_queues, batches_list)]
            
            return None

            
        else:
            # this list is (n_wm, n_agents, batch_object) because each wm considers all the agents as one
            batches_list = [dataset.sample(batch_size) for _ in range(self.n_wm_trained)]

            stats = [update_wm(wm, batches, train_device) for wm, batches in zip(self.wms, batches_list)]

            wm_dict_stats = {}
            wm_dict_stats.update({f'wm_{i}_' + key: value for i, wm_stats in enumerate(stats) for key, value in wm_stats.items()})

            return wm_dict_stats

    def load_wm_models(self):
        if not self.processes is None:

            self.check_all_processes_alive()

            # we wait that all worker processed their minibatches
            while not all(q.empty() for q in self.batch_queues):
                time.sleep(0.5)
                self.check_all_processes_alive()
            
            # we get the latest model pushed by each worker
            wm_state_dicts = {}
            for i, model_queue in enumerate(self.model_queues):
                
                model = model_queue.get()
                while not model_queue.empty():
                    model = model_queue.get()
                wm_state_dicts[f'wm_{i}'] = model

            # we update main models with worker models
            self.do_load_state_dict(wm_state_dicts)

    def evaluate(self, train_dataset, valid_dataset, max_eval_episode, batch_size):
        self.eval()
        # gets and load agent models from processes
        self.load_wm_models()

        stats = {}
        mean_l2_distances_list = []
        for _ in range(max_eval_episode):
            mean_l2_distances = []

            for i, wm in enumerate(self.wms):
                train_batches = train_dataset.sample(batch_size)
                valid_batches = valid_dataset.sample(batch_size)
                
                wm_stats = wm.evaluate(train_batches, valid_batches)
                
                for key, val in wm_stats.items():
                    wm_key = f'wm_{i}_' + key 
                    if wm_key in stats: 
                        stats[wm_key].append(val)
                    else:
                        stats[wm_key] = [val]

                mean_l2_distances.append(np.mean([wm_stats['T_dist_valid'], wm_stats['R_dist_valid']]))
            
            mean_l2_distances_list.append(mean_l2_distances)
        
        mean_l2_distances_list = np.asarray(mean_l2_distances_list)
        mean_l2_distances_accross_ep = np.mean(mean_l2_distances_list, axis=0)
        sorted_by_valid_distance = np.argsort(mean_l2_distances_accross_ep)

        stats = {key:np.mean(val) for key, val in stats.items()}
        self.top_wms = sorted_by_valid_distance[:self.n_wm_used]

        return stats

    def get_legal_move(self, batches, device):
        # we only keep the n_bests
        self.top_wms = self.top_wms[:self.n_wm_used]

        with torch.no_grad():
            self.to(device)

            n_agents = len(batches)

            # contrarily to training, when generating data with the ensemble, all models in the ensemble
            # are queried with the same batches so we can join the batches only once
            joint_batch = self.wms[0].join_batches(batches, device)
            
            # we have to ensure correct range 
            raise NotImplementedError("legal_moves are for discrete actions so clipping won't do, we might not have to do anything but make sure of that")

            # we get the legal move for each member of the ensemble
            legal_moves_list = []
            for model_idx in self.top_wms:
                model = self.wms[model_idx]
                
                assert model.legal_move

                observations = model.concat_embedings_to_obs(joint_batch, target=False)

                legal_moves_prob = model._model.legal_move.prob(observations)
                legal_moves_list.append(legal_moves_prob)
            
            # we do a majority vote accross members for each component (the mean)
            legal_moves_list = torch.stack(legal_moves_list, dim=2)
            legal_moves = torch.round(legal_moves_list.mean(2))

            # we return it per agent
            split_size = legal_moves.shape[1] // n_agents
            assert legal_moves.shape[1] % n_agents == 0
            
            return torch.split(legal_moves, split_size, dim=1)

    def sample(self, batches, config, device):
        self.eval()
        
        # we only keep the n_bests
        self.top_wms = self.top_wms[:config.n_wm_used]

        with torch.no_grad():
            self.to(device)

            n_agents = len(batches)

            # contrarily to training, when generating data with the ensemble, all models in the ensemble
            # are queried with the same batches so we can join the batches only once
            joint_batch = ml.join_batches(batches, device)

            # agent_dims = {}
            # for agent_batch in batches:
            #     for key, data in agent_batch._asdict().items():
            #         agent_dims[key] = agent_dims.get(key, []) + [data.shape[1]]

            batches = None

            ## we compute samples and variance for all the members of the ensemble
            synthetic_data = OrderedDict(zip(['rewards', 'masks', 'gaussian_var', 'next_observations'], [[] for _ in range(4)]))

            for model_idx in self.top_wms:
                model = self.wms[model_idx]

                observations = model.concat_embedings_to_obs(joint_batch, target=False)
                actions = model.transform_actions(joint_batch['actions'])

                if model.vae_wm:
                    T_pred, R_pred, mask_logits, legal_move_logits, mean_latent, log_std_latent = model._model(observations, actions)
                    
                    synthetic_data['next_observations'].append(T_pred)
                    synthetic_data['rewards'].append(R_pred)
                    synthetic_data['gaussian_var'].append(torch.cat((torch.zeros_like(T_pred), torch.zeros_like(R_pred)), dim=1)) 
                    synthetic_data['masks'].append(model._model.mask_decoder.binary_prob(mask_logits))


                else:

                    observations_actions = torch.cat((observations, actions), dim=1)
                    
                    _, mask_probs = model._model.mask.sample(observations_actions, return_prob=True)

                    if model.deterministic_wm:
                        pred_T = model._model.T(observations_actions)
                        pred_R = model._model.R(observations_actions)

                        synthetic_data['next_observations'].append(pred_T)
                        synthetic_data['rewards'].append(pred_R)
                        synthetic_data['gaussian_var'].append(torch.cat((torch.zeros_like(pred_T), torch.zeros_like(pred_R)), dim=1))    

                    else:

                        if model.k_mixture_gaussian > 1:
                            _, modes_T, log_std_T, _ = model._model.T.sample(observations_actions, return_mode_log_std=True)
                            _, modes_R, log_std_R, _ = model._model.R.sample(observations_actions, return_mode_log_std=True)

                            synthetic_data['next_observations'].append(modes_T['sampled_mode'])
                            synthetic_data['rewards'].append(modes_R['sampled_mode'])
                            synthetic_data['gaussian_var'].append(torch.cat((log_std_T['sampled_std'].exp()**2, log_std_R['sampled_std'].exp()**2), dim=1))

                        else: 
                            _, mean_T, log_std_T = model._model.T.sample(observations_actions, return_mode_log_std=True)
                            _, mean_R, log_std_R = model._model.R.sample(observations_actions, return_mode_log_std=True)

                            synthetic_data['next_observations'].append(mean_T)
                            synthetic_data['rewards'].append(mean_R)
                            # updated line below because was missing a .exp()**2 on the log_std
                            synthetic_data['gaussian_var'].append(torch.cat((log_std_T.exp()**2, log_std_R.exp()**2), dim=1))

                    synthetic_data['masks'].append(mask_probs) # we take the prob because we then do a majority vote (between agents, not ensemble)
                    
            ## we convert data to tensors, ensemble dim is 2
            for key, data in synthetic_data.items():
                if len(data) > 0:
                    synthetic_data[key] = torch.stack(data, dim=2)

            ## we compute "across ensemble" quantities

            # the variance across modes is dones on (s',r)
            ensemble_cov = batch_cov(torch.cat((synthetic_data['next_observations'], synthetic_data['rewards']),dim=1))
            ensemble_cov_norm = batch_frobenius_norm(ensemble_cov)
            synthetic_data['ensemble_cov_norm'] = torch.clamp_max(ensemble_cov_norm, max=config.ensemble_cov_norm_threshold)
            # synthetic_data['ensemble_cov_norm'] = ensemble_cov_norm


            # the max gaussian variance across ensemble 
            # diagonal cov so just norm of the trace
            model_max_cov_norm = (((synthetic_data['gaussian_var']**2).sum(1))**0.5).max(1).values.unsqueeze(1)
            del synthetic_data['gaussian_var']
            synthetic_data['model_max_cov_norm'] = torch.clamp_max(model_max_cov_norm, max=config.ensemble_cov_norm_threshold)
            # synthetic_data['model_max_cov_norm'] = model_max_cov_norm


            # we also compute just the std on the reward
            mean_rewards = synthetic_data['rewards'].mean(1)
            ensemble_reward_variance = torch.var(mean_rewards, dim=1, unbiased=True, keepdim=True)
            synthetic_data['rewards_cov'] = torch.clamp_max(ensemble_reward_variance, max=config.ensemble_cov_norm_threshold)
            # synthetic_data['rewards_cov'] = ensemble_reward_variance

            ## we select the data we want to return from the ensemble
            batch_size = len(joint_batch['observations'])
            n_models = len(self.top_wms)
            return_synthetic_data = {}
            selection_indices = None

            if config.wm_use_min_rewards:
                ## min-reward and eventually the corresponding model 
                # we take the mean reward because all the agents must have the same reward (fully-cooperative)
                min_rewards_output = mean_rewards.min(1)
                min_rewards = min_rewards_output.values
                min_indices = min_rewards_output.indices

                return_synthetic_data['rewards'] = [min_rewards.unsqueeze(1) for _ in range(n_agents)]

                if config.wm_use_all_min:
                    selection_indices = min_indices
                
            if selection_indices is None:
                ## we sample a specific model in the ensemble to get its synthetic data
                selection_indices = torch.randint(high=n_models, size=(batch_size,), device=device)


            for key, data in synthetic_data.items():
                if key in ['model_max_cov_norm', 'ensemble_cov_norm', 'rewards_cov']:
                    # these are ensemble metrics so we do not sample an ensemble model 
                    # we just copy the ensemble metric for each agent
                    return_synthetic_data[key] = [synthetic_data[key] for _ in range(n_agents)]

                else:
                    
                    if len(data)>0:
                        
                        if key == 'rewards':
        
                            if config.wm_use_min_rewards:
                                continue

                            else:
                                # we have already computed the mean_rewards so it is a bit different
                                mean_reward = mean_rewards[torch.arange(mean_rewards.size(0)), selection_indices]
                                return_synthetic_data[key] = [mean_reward.unsqueeze(1) for _ in range(n_agents)]

                        else:
                            sampled_data = data.permute(0,2,1)[torch.arange(data.size(0)), selection_indices]

                            if key == 'masks':
                                split_sizes = [1 for _ in range(n_agents)]
                                agent_splitted_data = split_data_from_sizes_per_agent(sampled_data, split_sizes)
                                ## mask from sampling a model at random
                                stacked_mask = torch.stack(agent_splitted_data, dim=2)
                                # we do a majority vote
                                mask_majority_vote = torch.round(stacked_mask.mean(2))
                                return_synthetic_data[key] = [mask_majority_vote for _ in range(n_agents)]

                            elif key == 'next_observations':
                                # we split data back to per agent
                                split_sizes = [obs_space.shape[0] for obs_space in self.init_dict['obs_space']]
                                agent_splitted_data = split_data_from_sizes_per_agent(sampled_data, split_sizes)
                                return_synthetic_data[key] = agent_splitted_data
                            else:
                                # agent_splitted_data = split_data_uniformly_per_agent(sampled_data, n_agents)
                                # return_synthetic_data[key] = agent_splitted_data
                                raise NotImplementedError

            # we reorganize the data per agent and convert to numpy
            return_synthetic_data = [{key: ml.convert_to_numpy(val[i]) for key, val in return_synthetic_data.items()} for i in range(n_agents)]

            for i, data in enumerate(return_synthetic_data):
                for key, val in data.items():
                    if np.isinf(val).any():
                        raise ValueError(f'Agent {i} found inf in {key}')

                    if np.isnan(val).any():
                        raise ValueError(f'Agent {i} found nan in {key}')

            return return_synthetic_data

    def get_stats(self):
        train_aggregator = Aggregator()
        self.check_all_processes_alive()
        while not (all([q.empty() for q in self.stat_queues]) and all([q.empty() for q in self.batch_queues])):
            self.check_all_processes_alive()
            stats = [q.get() for q in self.stat_queues]
            wm_dict_stats = {}
            wm_dict_stats.update({f'wm_{i}_' + key: value for i, wm_stats in enumerate(stats) for key, value in wm_stats.items()})

            train_aggregator.update(wm_dict_stats)
        return train_aggregator.pop_all_means()

    def check_all_processes_alive(self):
        if not all([p.is_alive() for p in self.processes]):
            raise RuntimeError('Some worker process died !!!')

    def terminate_processes(self):
        if not self.processes is None:

            _ = [p.terminate() for p in self.processes]

            time.sleep(10.)

            _ = [p.close() for p in self.processes]
    
    def get_state_dict(self):
        self.to('cpu')
        wms_state_dict = {f'wm_{i}': wm.get_state_dict() for i, wm in enumerate(self.wms)}
        wms_state_dict['top_wms'] = self.top_wms
        return wms_state_dict
    
    def do_load_state_dict(self, state_dict):
        # this loads network, optim, scheduler params
        _ = [wm.do_load_state_dict(state_dict[f'wm_{i}']) for i, wm in enumerate(self.wms)]
        self.top_wms = state_dict['top_wms']
    
    def load_state_dict(self, state_dict):
        # this only load network params
        _ = [wm.load_state_dict(state_dict[f'wm_{i}']) for i, wm in enumerate(self.wms)]
        self.top_wms = state_dict['top_wms']

def update_wm(wm, batch, train_device):
    return wm.update_from_batches(batch, train_device)


def update_wm_with_queues(wm, batch_queue, stat_queue, model_queue, train_device):
    while True:
        # we update model
        batches = batch_queue.get()
        stat_queue.put(wm.update_from_batches(batches, train_device))

        # batches = None # to release memory ? 

        # if we think that we have finished updating we push a new model (main will take care of getting the latest pushed model)
        if batch_queue.empty():
            # deepcopy to avoid https://github.com/pytorch/pytorch/issues/10375 error
            model_queue.put(deepcopy(wm.get_state_dict()))


def batch_cov(batch_of_matrices):
    batch_size, vec_len, n_vec = batch_of_matrices.shape

    # random variable are vec components (in vec_len), samples are ensemble element (in n_vec)
    
    vec_means = batch_of_matrices.mean(2).unsqueeze(2)
    diff = batch_of_matrices - vec_means

    # we need all the products combination between the random variables
    prod = torch.einsum('bve,bwe->bvwe', diff, diff)
    # prod2 = diff.unsqueeze(1)*diff.unsqueeze(2)

    # unbiased estimate
    batch_cov = prod.sum(3) / (n_vec-1)

    return batch_cov

def batch_frobenius_norm(batch_of_matrices):
    return (((batch_of_matrices**2).sum((1, 2))**0.5)).unsqueeze(1)

def split_data_uniformly_per_agent(data, n_agents):
    split_size = data.shape[1] // n_agents
    assert data.shape[1] % n_agents == 0
    agent_splitted_data = torch.split(data, split_size, dim=1)
    return agent_splitted_data

def split_data_from_sizes_per_agent(data, split_sizes):
    agent_splitted_data = torch.split(data, split_sizes, dim=1)
    return agent_splitted_data