from torch import nn
import time
from copy import deepcopy
import torch

import torch.multiprocessing as multiprocessing
from queue import Empty

from tqdm import tqdm
import numpy as np

from alfred.utils.recorder import Aggregator

from offline_marl.single_agent.iql import IQLLearner, IQLLearnerDiscrete
from offline_marl.utils.base_algo import BaseAlgo

class IQLIndependentLearner(BaseAlgo):
    def __init__(self, agent_constructor=IQLLearner, learner_name='iql-independent', agents_suffix='iql', **kwargs):
        BaseAlgo.__init__(self)

        if kwargs['alg_name'] == 'bc':
            learner_name = 'bc'
            agents_suffix = 'bc'
            self.bc_only = True
        else:
            self.bc_only = False
            
        self.init_manager(self, kwargs, agent_constructor, learner_name, agents_suffix)

    @staticmethod
    def init_manager(manager, kwargs, agent_constructor, learner_name, agents_suffix):
        manager.n_learner = kwargs['n_learner']
        
        manager.learners = nn.ModuleList()

        act_spaces = kwargs['act_space']
        obs_spaces = kwargs['obs_space']

        for i in range(manager.n_learner):
            learner_kwargs = kwargs.copy()
            learner_kwargs.update({'learner_name': agents_suffix + f'_{i}', 'act_space':act_spaces[i], 'obs_space':obs_spaces[i]})
            manager.learners.append(agent_constructor(**learner_kwargs))
    
        # to save and reload
        manager.init_dict = kwargs
        manager.name = learner_name

        manager.evaluation_metrics = {'return', 'length', 'training_step'}
        manager.train_metrics = set()
        for i, learner in enumerate(manager.learners):
            prefix = f'agent_{i}_'
            for metric in learner.train_metrics:
                metric = prefix + metric
                manager.train_metrics.add(metric)
        
        manager.metrics_to_record = manager.evaluation_metrics | manager.train_metrics

        # for multithreading
        manager.processes = None
        manager.idx_queues = None
        manager.stat_queues = None
        manager.model_queues = None

    def update(self, dataset, batch_size, train_device, train_in_parallel=False):

        indexes = dataset.sample_indexes(batch_size)

        if train_in_parallel: 
            
            if self.processes is None:

                assert len(self.learners) == len(dataset.datasets)

                self.idx_queues = [multiprocessing.Queue() for _ in range(self.n_learner)]
                self.stat_queues = [multiprocessing.Queue() for _ in range(self.n_learner)]
                self.model_queues = [multiprocessing.Queue() for _ in range(self.n_learner)]
                self.processes = [multiprocessing.Process(target=update_learner_with_queues, args=(learner, dataset, idx_queue, stat_queue, model_queue, train_device)) 
                        for learner, dataset, idx_queue, stat_queue, model_queue in zip(self.learners, dataset.datasets, self.idx_queues, self.stat_queues, self.model_queues)]
                
                for p in self.processes:
                    p.start()

                # # whats below is useless as it doesn't change nvidia-smi reports
                # # once the learner have been pickled and copied to sub-processes we can bring the models of the main process on cpu to avoid wasting gpu memory
                # self.to(torch.device('cpu'))

                # # similarly we can "delete" the datasets from the main process
                # del dataset.datasets
                # torch.cuda.empty_cache()
            
            
            _ = [idx_queue.put(idx) for idx_queue, idx in zip(self.idx_queues, indexes)]
            
            return None

            
        else:

            assert len(self.learners) == len(dataset.datasets)

            batches = [dataset.sample_from_idx(idx) for dataset, idx in zip(dataset.datasets, indexes)]

            stats = [update_learner(learner, batch, train_device) for learner, batch in zip(self.learners, batches)]

            agent_dict_stats = {}
            agent_dict_stats.update({f'agent_{i}_' + key: value for i, agent_stats in enumerate(stats) for key, value in agent_stats.items()})

            return agent_dict_stats

    def load_agents_models(self):
        if not self.processes is None:

            self.check_all_processes_alive()

            # we wait that all worker processed their minibatches
            while not all(q.empty() for q in self.idx_queues):
                time.sleep(0.5)
                self.check_all_processes_alive()
            
            # we get the latest model pushed by each worker
            learners_state_dicts = {}
            for i, model_queue in enumerate(self.model_queues):
                model = model_queue.get()
                while not model_queue.empty():
                    model = model_queue.get()
                learners_state_dicts[f'learner_{i}'] = model

            # we update main models with worker models
            self.do_load_state_dict(learners_state_dicts)

    def reset_agents_history(self):
        [learner.reset_histories() for learner in self.learners]
    
    def append_history_agents(self, observations, actions):
        [learner.append_histories(observation, action) for learner, observation, action in zip(self.learners, observations, actions)]

    def evaluate(self, env, max_eval_episode, rollout_device='cpu'):

        # gets and load agent models from processes
        self.load_agents_models()

        # puts model on rollout device
        self.to(rollout_device)

        stats = {'return': [], 'length': [], 'frames': []}


        for ep in tqdm(range(max_eval_episode), desc="evaluation"):
            
            observations, _ = env.reset()
            done = False

            self.reset_agents_history()

            # env has a timelimit wrapper of 1000 max-steps
            # it also has a wrapper that computes episode return and length
            while not done:
                actions = []
                for observation, learner in zip(observations, self.learners):

                    torch_observation = torch.as_tensor(observation, device=rollout_device).unsqueeze(0)
                    
                    # add memory embedding
                    torch_observation = learner.process_current_histories(torch_observation)

                    action = learner.policy.act(torch_observation, sample=False, return_log_pi=False).squeeze(0).detach().cpu().numpy()

                    actions.append(action)

                next_observations, reward, done, info = env.step(actions)

                self.append_history_agents(observations, actions)

                observations = next_observations

                
            for k in stats.keys():
                if k == 'frames':
                    if len(stats[k]) >= env.n_ep_per_gifs:
                        continue
                stats[k].append(info['episode'][k])

        for k, v in stats.items():
            if k == 'frames':
                stats[k] = np.concatenate(np.asarray(v), axis=0)
            else:
                stats[k] = np.mean(v)
        
        return stats

    def get_stats(self):
        train_aggregator = Aggregator()
        self.check_all_processes_alive()
        while not (all([q.empty() for q in self.stat_queues]) and all([q.empty() for q in self.idx_queues])):
            self.check_all_processes_alive()
            stats = [q.get() for q in self.stat_queues]
            agent_dict_stats = {}
            agent_dict_stats.update({f'agent_{i}_' + key: value for i, agent_stats in enumerate(stats) for key, value in agent_stats.items()})

            train_aggregator.update(agent_dict_stats)
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
        return {f'learner_{i}': learner.get_state_dict() for i, learner in enumerate(self.learners)}
    
    def do_load_state_dict(self, state_dict):
        # this loads network, optim, scheduler params
        _ = [learner.do_load_state_dict(state_dict[f'learner_{i}']) for i, learner in enumerate(self.learners)]
    
    def load_state_dict(self, state_dict):
        # this only load network params
        _ = [learner.load_state_dict(state_dict[f'learner_{i}']) for i, learner in enumerate(self.learners)]


class IQLIndependentLearnerDiscrete(IQLIndependentLearner):
    def __init__(self, agent_constructor=IQLLearnerDiscrete, learner_name='iql-independent', agents_suffix='iql', **kwargs):
        IQLIndependentLearner.__init__(self, agent_constructor, learner_name, agents_suffix, **kwargs)

    def evaluate(self, env, max_eval_episode, rollout_device='cpu'):

        # gets and load agent models from processes
        self.load_agents_models()

        # puts model on rollout device
        self.to(rollout_device)

        stats = {'return': [], 'length': []}          

        for ep in tqdm(range(max_eval_episode), desc="evaluation"):

            observations, infos = env.reset()
            done = False

            self.reset_agents_history()

            # env has a timelimit wrapper of 1000 max-steps
            # it also has a wrapper that computes episode return and length
            while not done:
                actions = []
                for observation, legal_move, learner in zip(observations, infos['legal_moves'], self.learners):

                    torch_observation = env.to_torch(observation, device=rollout_device).unsqueeze(0)
                    torch_legal_move = env.to_torch(legal_move, device=rollout_device).unsqueeze(0)

                    # add memory embedding
                    torch_observation = learner.process_current_histories(torch_observation)

                    action = learner.policy.act(torch_observation, legal_move=torch_legal_move, sample=False, return_log_pi=False).squeeze(0).detach().cpu().numpy()

                    actions.append(action)

                next_observations, reward, done, infos = env.step(actions)

                self.append_history_agents(observations, actions)

                observations = next_observations

            for k in stats.keys():
                    stats[k].append(infos['episode'][k])

        for k, v in stats.items():
                stats[k] = np.mean(v)
        
        return stats

def update_learner(learner, batch, train_device):
    return learner.update_from_batch(batch, train_device)


def update_learner_with_queues(learner, dataset, index_queue, stat_queue, model_queue, train_device):
    while True:
        # we update model
        idx = index_queue.get()
        batch = dataset.sample_from_idx(idx)
        stat_queue.put(learner.update_from_batch(batch, train_device))

        # if we think that we have finished updating we push a new model (main will take care of getting the latest pushed model)
        if index_queue.empty():
            # deepcopy to avoid https://github.com/pytorch/pytorch/issues/10375 error
            model_queue.put(deepcopy(learner.get_state_dict()))



