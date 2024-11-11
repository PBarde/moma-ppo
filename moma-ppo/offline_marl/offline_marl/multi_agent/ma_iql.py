from alfred.utils.misc import Bunch
import torch
from torch import nn

from offline_marl.multi_agent.iql_independent import IQLIndependentLearner, IQLIndependentLearnerDiscrete
from offline_marl.multi_agent.mixer_networks import MixerNetwork
from offline_marl.single_agent.iql import IQLLearnerDiscrete
from offline_marl.utils.ml import batch_as_tensor, batch_to_device, check_if_nan_in_batch
from offline_marl.utils.ml import soft_update as update_target_soft

class MaIQL(IQLIndependentLearner):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._init(**kwargs)

    def _init(self, **kwargs):
        self.name = kwargs['alg_name']
        self.independent_awr = kwargs['independent_awr']
        self.memory_backprop_actor = kwargs['memory_backprop_actor']
        self.action_aware_mixer = kwargs['action_aware_mixer']

        self.grad_clip_norm = kwargs['grad_clip_norm']

        # the global state is obtained by concatenating the observations of all of the agents
        self.Q_mixer_input_size, self.V_mixer_input_size = self.get_mixer_input_size(kwargs)

        # note that mixer networks do not use memory (because centralized) and use the same lr as corresponding critic
        self.mixer_network_Q = MixerNetwork(global_obs_size=self.Q_mixer_input_size, n_learners=self.n_learner,  hidden_size=kwargs['hidden_size'], lr=kwargs['lr_q'], train_device=kwargs['train_device'])

        self.mixer_network_V = MixerNetwork(global_obs_size=self.V_mixer_input_size, n_learners=self.n_learner,  hidden_size=kwargs['hidden_size'], lr=kwargs['lr_v'], train_device=kwargs['train_device'])
        
        with torch.no_grad():
            self.target_mixer_network_Q = MixerNetwork(global_obs_size=self.Q_mixer_input_size, n_learners=self.n_learner,  hidden_size=kwargs['hidden_size'], lr=kwargs['lr_q'], train_device=kwargs['train_device'])
            MixerNetwork.update_target_hard(self.target_mixer_network_Q, self.mixer_network_Q)


        self.evaluation_metrics = self.learners[0].evaluation_metrics
        self.train_metrics = self.learners[0].train_metrics | {'weights'}
        self.metrics_to_record = self.train_metrics | self.evaluation_metrics

    def get_mixer_input_size(self, kwargs):
        V_mixer = sum([obs_space.shape[0] for obs_space in kwargs['obs_space']])
        if self.action_aware_mixer:
            Q_mixer = sum([obs_space.shape[0] for obs_space in kwargs['obs_space']] + [act_space.shape[0] for act_space in kwargs['act_space']])
        else:    
            Q_mixer = V_mixer

        return Q_mixer, V_mixer 

    def get_q_mixer_input(self, batches):
        if self.action_aware_mixer:
            return torch.cat([batch['observations'] for batch in batches] + [batch['actions'] for batch in batches], dim=1)
        else:
            return torch.cat([batch['observations'] for batch in batches], dim=1)

    def get_v_mixer_input(self, batches, next_obs=False):
        if next_obs:
            return torch.cat([batch['next_observations'] for batch in batches], dim=1)
        else: 
            return torch.cat([batch['observations'] for batch in batches], dim=1)


    def update_v(self, batches):

        ## Mixing weights
        # gets global-state to compute mixing weights
        v_mixer_input = self.get_v_mixer_input(batches)
        q_mixer_input = self.get_q_mixer_input(batches)
        
        # mixing weights
        with torch.no_grad():
            target_q_mixing_weights, target_q_mixing_biases = self.target_mixer_network_Q(q_mixer_input)
        v_mixing_weights, v_mixing_biases = self.mixer_network_V(v_mixer_input)

        ## Individual values
        # get individual values for each agent
        learners_vs = []
        learners_target_q_val = []
        for batch, learner in zip(batches, self.learners):
            # comuptes target Q(s,a) with double clipped Q
            with torch.no_grad():
                target_observations = learner.concat_embedings_to_obs(batch, target=True)
                target_q_val1, target_q_val2 = learner.doubleQtarget(target_observations, batch['actions'])
                target_q_val = torch.minimum(target_q_val1, target_q_val2).detach()

            observations = learner.concat_embedings_to_obs(batch, target=False)

            # Computes V(s)
            vs = learner.V(observations)

            learners_vs.append(vs)
            learners_target_q_val.append(target_q_val)

        ## Mixed loss
        # we concatenate individual values to be able to operate on whole batches
        learners_vs = torch.cat(learners_vs, dim=1)
        learners_target_q_val = torch.cat(learners_target_q_val, dim=1)    

        with torch.no_grad():
            mixed_target_q = MixerNetwork.mix(learners_target_q_val, target_q_mixing_weights, target_q_mixing_biases)
        
        mixed_vs = MixerNetwork.mix(learners_vs, v_mixing_weights, v_mixing_biases)

        # Computes loss |tau - I(Q(s,a) - V(s)<0)|(Q(s,a)-V(s))^2
        # we assume that all learners have same expectile (we can because we are centralized at training)
        loss = (torch.abs(self.learners[0].expectile - (mixed_target_q < mixed_vs).float())*(mixed_target_q - mixed_vs)**2).mean(0)

        ## Updates each learner as well as the mixing network
        # cleans potentials left-over gradients and updates networks
        _ = [learner.V.optim.zero_grad() for learner in self.learners]
        _ = [learner.memory.optim.zero_grad() for learner in self.learners]
        self.mixer_network_V.optim.zero_grad()
        
        loss.backward()

        # clips the gradients
        _ = [torch.nn.utils.clip_grad_norm_(learner.V.parameters(), self.grad_clip_norm) for learner in self.learners]
        _ = [torch.nn.utils.clip_grad_norm_(learner.memory.parameters(), self.grad_clip_norm) for learner in self.learners]
        torch.nn.utils.clip_grad_norm_(self.mixer_network_V.parameters(), self.grad_clip_norm)

        # takes a gradient step
        _ = [learner.V.optim.step() for learner in self.learners]
        _ = [learner.memory.optim.step() for learner in self.learners]
        self.mixer_network_V.optim.step()

        if torch.isnan(loss).any():
            raise ValueError("found nan in v_loss")

        return loss.detach().cpu().numpy(), mixed_vs.mean(0).detach().cpu().numpy(), mixed_vs.min(0).values.detach().cpu().numpy(), mixed_vs.max(0).values.detach().cpu().numpy()

    def update_q(self, batches):

         ## Mixing weights
        # gets global-state to compute mixing weights
        q_mixer_inputs = self.get_q_mixer_input(batches)
        next_v_mixer_inputs = self.get_v_mixer_input(batches, next_obs=True)

        with torch.no_grad():
            next_v_mixing_weights, next_v_mixing_biases = self.mixer_network_V(next_v_mixer_inputs)
        
        q_mixing_weights, q_mixing_biases = self.mixer_network_Q(q_mixer_inputs)

        learners_next_v_target = []
        learners_q1 = []
        learners_q2 = []

        for batch, learner in zip(batches, self.learners):
            observations = learner.concat_embedings_to_obs(batch, target=False)
            
            # next state value for temporal difference learning V(s')
            with torch.no_grad():
                target_next_observations = learner.concat_embedings_to_obs(batch, next_obs=True, target=True)
                target_next_value = learner.V(target_next_observations)


            # computes Q(s,a) with double clipped Q
            Qsa1, Qsa2 = learner.doubleQ(observations, batch['actions'])

            learners_next_v_target.append(target_next_value)
            learners_q1.append(Qsa1)
            learners_q2.append(Qsa2)

        ## Computes mixed values
        learners_next_v_target = torch.cat(learners_next_v_target, dim=1)
        learners_q1 = torch.cat(learners_q1, dim=1)
        learners_q2 = torch.cat(learners_q2, dim=1)

        with torch.no_grad():
            mixed_next_v_target = MixerNetwork.mix(learners_next_v_target, next_v_mixing_weights, next_v_mixing_biases)
            
            # centralized training in cooperative setting so all learners have same reward, mask and discount_factor
            target = batches[0]['rewards'] + self.learners[0].discount_factor*batches[0]['masks']*mixed_next_v_target
        
        mixed_q1 = MixerNetwork.mix(learners_q1, q_mixing_weights, q_mixing_biases)
        mixed_q2 = MixerNetwork.mix(learners_q2, q_mixing_weights, q_mixing_biases)
        
        # loss is computed for both Q: (r(s,a) + gamma*V(s') - Q(s,a))**2
        loss = ((target - mixed_q1)**2 + (target - mixed_q2)**2).mean(0)

        ## Updates all the learners and the Q mixing network

        # clean potentials left-over gradients and updates networks
        _ = [learner.doubleQ.q1.optim.zero_grad() for learner in self.learners]
        _ = [learner.doubleQ.q2.optim.zero_grad() for learner in self.learners]
        _ = [learner.memory.optim.zero_grad() for learner in self.learners]
        self.mixer_network_Q.optim.zero_grad()

        loss.backward()

        # clips the gradients
        _ = [torch.nn.utils.clip_grad_norm_(learner.doubleQ.q1.parameters(), self.grad_clip_norm) for learner in self.learners]
        _ = [torch.nn.utils.clip_grad_norm_(learner.doubleQ.q2.parameters(), self.grad_clip_norm) for learner in self.learners]
        _ = [torch.nn.utils.clip_grad_norm_(learner.memory.parameters(), self.grad_clip_norm) for learner in self.learners]
        torch.nn.utils.clip_grad_norm_(self.mixer_network_Q.parameters(), self.grad_clip_norm)

        # takes a gradient step
        _ = [learner.doubleQ.q1.optim.step() for learner in self.learners]
        _ = [learner.doubleQ.q2.optim.step() for learner in self.learners]
        _ = [learner.memory.optim.step() for learner in self.learners]
        self.mixer_network_Q.optim.step()

        mixed_q = torch.cat((mixed_q1, mixed_q2), dim=0)

        if torch.isnan(loss).any():
            raise ValueError("found nan in q_loss")

        return loss.detach().cpu().numpy(), mixed_q.mean(0).detach().cpu().numpy(), mixed_q.min(0).values.detach().cpu().numpy(), mixed_q.max(0).values.detach().cpu().numpy()


    def update_pi(self, batches):

        ## Computes advantage weights 
        with torch.no_grad():
            ## computes mixing weights
            q_mixer_inputs = self.get_q_mixer_input(batches)
            v_mixer_inputs = self.get_v_mixer_input(batches)

            v_mixing_weights, v_mixing_biases = self.mixer_network_V(v_mixer_inputs)
            target_q_mixing_weights, target_q_mixing_biases = self.target_mixer_network_Q(q_mixer_inputs)
            
            learners_v = []
            learners_target_q = []
            for batch, learner in zip(batches, self.learners):
                target_observations = learner.concat_embedings_to_obs(batch, target=True)
                target_q1, target_q2 = learner.doubleQtarget(target_observations, batch['actions'])
                target_q = torch.minimum(target_q1, target_q2)
                vs = learner.V(target_observations)

                learners_v.append(vs)
                learners_target_q.append(target_q)
            
            learners_v = torch.cat(learners_v, dim=1)
            learners_target_q = torch.cat(learners_target_q, dim=1)

            mixed_v = MixerNetwork.mix(learners_v, v_mixing_weights, v_mixing_biases)
            mixed_target_q = MixerNetwork.mix(learners_target_q, target_q_mixing_weights, target_q_mixing_biases)

            if self.independent_awr:
                # we compute the weight independently for each agent
                individual_weight = -torch.clamp(torch.exp(self.learners[0].awr_temperature*(
                    learners_target_q*target_q_mixing_weights + target_q_mixing_biases - learners_v*v_mixing_weights - v_mixing_biases)), max=100.0)

            else:    
                # we compute a team weight directly with the mixed values
                weight = -torch.clamp(torch.exp(self.learners[0].awr_temperature*(mixed_target_q-mixed_v)), max=100.0)
        
        ## Computes learners' action likelihood   
        learners_log_pi = []
        for batch, learner in zip(batches, self.learners):
            observations = learner.concat_embedings_to_obs(batch, target=False)

            log_pi = learner.policy.get_log_prob_from_obs_action_pairs(batch['actions'], observations)

            learners_log_pi.append(log_pi)
        
        ## loss
        learners_log_pi = torch.cat(learners_log_pi, dim=1)
        
        if self.independent_awr:
            # compute the loss for each agent on the batch
            individual_losses = (individual_weight*learners_log_pi).mean(0, keepdim=True)
            # sum all the agents loss to get a team loss
            loss = individual_losses.sum(1)
            # just for recordings
            weight = individual_weight.sum(1, keepdim=True)

        else:
            summed_log_pi = learners_log_pi.sum(1, keepdim=True)
            loss = (weight * summed_log_pi).mean(0)
        

        # clean potentials left-over gradients and updates networks
        _ = [learner.policy.optim.zero_grad() for learner in self.learners]
        _ = [learner.memory.optim.zero_grad() for learner in self.learners]

        loss.backward()

        #clips the gradients
        _ = [torch.nn.utils.clip_grad_norm_(learner.policy.parameters(), self.grad_clip_norm) for learner in self.learners]

        # takes a gradient step
        _ = [learner.policy.optim.step() for learner in self.learners]

        if self.memory_backprop_actor:
            # clips the gradients
            _ = [torch.nn.utils.clip_grad_norm_(learner.memory.parameters(), self.grad_clip_norm) for learner in self.learners]
            # takes a gradient step
            _ = [learner.memory.optim.step() for learner in self.learners]

        # updates cosine lr schedule
        _ = [learner.policy.scheduler.step() for learner in self.learners]

        if torch.isnan(weight).any():
            raise ValueError("found nan in weights of pi_loss")
        
        if torch.isnan(loss).any():
            raise ValueError("found nan in pi_loss")

        return loss.detach().cpu().numpy(), weight.cpu().numpy()

    def update(self, dataset, batch_size, train_device, **kwargs):

        batches = dataset.sample(batch_size)

        return self.update_from_batch(batches, train_device, **kwargs)


    def update_from_batch(self, batches, train_device, **kwargs):
        # from https://github.com/ikostrikov/implicit_q_learning/blob/master/learner.py
        # with same batch the update goes
        # 1. update V-network with L_V
        # 2. use new V-network to update policy with AWR
        # 3. update Q-network with new V-network and L_Q
        # 4. update target Q-network

        # put models on training device
        self.to(train_device)

        batches = [batch_to_device(batch_as_tensor(batch), train_device) for batch in batches]

        _ = [check_if_nan_in_batch(batch) for batch in batches]

        # 1. update v_network with L_V
        loss_v, mean_v, min_v, max_v = self.update_v(batches)

        # 2. use new V-network to update policy with AWR
        loss_pi, weight = self.update_pi(batches)

        # 3. update Q-network with new V-network and L_Q
        loss_q, mean_q, min_q, max_q = self.update_q(batches)

        # 4. update target networks
        for learner in self.learners:
            update_target_soft(target=learner.doubleQtarget, source=learner.doubleQ, tau=learner.target_update_coef)
            update_target_soft(target=learner.memorytarget, source=learner.memory, tau=learner.target_update_coef)

        update_target_soft(target=self.target_mixer_network_Q, source=self.mixer_network_Q, tau=self.learners[0].target_update_coef)
        
        return {'loss_v': loss_v, 'loss_pi': loss_pi, 'loss_q': loss_q, 'weights': weight,
                'mean_v': mean_v, 'min_v': min_v, 'max_v': max_v, 
                'mean_q': mean_q, 'min_q': min_q, 'max_q': max_q}

    def get_state_dict(self):
        learners_state_dict = super().get_state_dict()
        learners_state_dict.update({'mixer_network_Q': self.mixer_network_Q.get_state_dict(),
                                    'target_mixer_network_Q': self.target_mixer_network_Q.get_state_dict(), 
                                    'mixer_network_V': self.mixer_network_V.get_state_dict()})
        return learners_state_dict
    
    def do_load_state_dict(self, state_dict):
        # this loads network, optim, scheduler params
        _ = [learner.do_load_state_dict(state_dict[f'learner_{i}']) for i, learner in enumerate(self.learners)]
        self.mixer_network_Q.do_load_state_dict(state_dict['mixer_network_Q'])
        self.target_mixer_network_Q.do_load_state_dict(state_dict['target_mixer_network_Q'])
        self.mixer_network_V.do_load_state_dict(state_dict['mixer_network_V'])
    
    def load_state_dict(self, state_dict):
        # this only load network params
        _ = [learner.load_state_dict(state_dict[f'learner, reward_{i}']) for i, learner in enumerate(self.learners)]
        self.mixer_network_Q.load_state_dict(state_dict['mixer_network_Q'])
        self.target_mixer_network_Q.load_state_dict(state_dict['target_mixer_network_Q'])
        self.mixer_network_V.load_state_dict(state_dict['mixer_network_V'])

class MaSAC(IQLIndependentLearner):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._init(**kwargs)

    def _init(self, **kwargs):
        self.name = kwargs['alg_name']
        self.independent_awr = kwargs['independent_awr']
        self.memory_backprop_actor = kwargs['memory_backprop_actor']
        self.action_aware_mixer = kwargs['action_aware_mixer']

        self.grad_clip_norm = kwargs['grad_clip_norm']

        # the global state is obtained by concatenating the observations of all of the agents
        self.Q_mixer_input_size = self.get_mixer_input_size(kwargs)

        # note that mixer networks do not use memory (because centralized) and use the same lr as corresponding critic
        self.mixer_network_Q = MixerNetwork(global_obs_size=self.Q_mixer_input_size, n_learners=self.n_learner,  hidden_size=kwargs['hidden_size'], lr=kwargs['lr_q'], train_device=kwargs['train_device'])
        
        with torch.no_grad():
            self.target_mixer_network_Q = MixerNetwork(global_obs_size=self.Q_mixer_input_size, n_learners=self.n_learner,  hidden_size=kwargs['hidden_size'], lr=kwargs['lr_q'], train_device=kwargs['train_device'])
            MixerNetwork.update_target_hard(self.target_mixer_network_Q, self.mixer_network_Q)

        self.target_entropy = -torch.tensor(sum([act_space.shape[0] for act_space in kwargs['act_space']]), dtype=torch.float32).to(kwargs['train_device'])
        self.log_alpha = nn.parameter.Parameter(torch.zeros(1, device=kwargs['train_device']))
        self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=kwargs['lr_pi'])


        self.evaluation_metrics = self.learners[0].evaluation_metrics
        self.train_metrics = self.learners[0].train_metrics | {'loss_alpha'}
        self.metrics_to_record = self.train_metrics | self.evaluation_metrics

    def get_mixer_input_size(self, kwargs):
        if self.action_aware_mixer:
            Q_mixer = sum([obs_space.shape[0] for obs_space in kwargs['obs_space']] + [act_space.shape[0] for act_space in kwargs['act_space']])
        else:    
            Q_mixer = sum([obs_space.shape[0] for obs_space in kwargs['obs_space']])

        return Q_mixer

    def get_q_mixer_input(self, batches, next_obs=False, next_actions=None):

        if next_obs:
            if self.action_aware_mixer:
                return torch.cat([batch.next_observations for batch in batches] + next_actions, dim=1)
            else:
                return torch.cat([batch.next_observations for batch in batches], dim=1)
        else:
            if self.action_aware_mixer:
                return torch.cat([batch.observations for batch in batches] + [batch.actions for batch in batches], dim=1)
            else:
                return torch.cat([batch.observations for batch in batches], dim=1)

    def update_q(self, batches):

         ## Mixing weights
        # gets global-state to compute mixing weights
        q_mixer_inputs = self.get_q_mixer_input(batches)
        q_mixing_weights, q_mixing_biases = self.mixer_network_Q(q_mixer_inputs)

        learners_next_q_target = []
        learners_q1 = []
        learners_q2 = []
        learners_next_actions = []

        for batch, learner in zip(batches, self.learners):
            observations = learner.concat_embedings_to_obs(batch, target=False)

            # computes Q(s,a) with double clipped Q
            Qsa1, Qsa2 = learner.doubleQ(observations, batch.actions)

                        # next state value for temporal difference learning V(s')
            with torch.no_grad():
                next_observations = learner.concat_embedings_to_obs(batch, next_obs=True, target=True)
                next_actions, next_log_pi = learner.policy.act(next_observations, sample=True, return_log_pi=True)
                target_next_q1, target_next_q2 = learner.doubleQtarget(next_observations, next_actions) 
                target_next_q = torch.minimum(target_next_q1, target_next_q2) - self.log_alpha.exp()*next_log_pi

            learners_next_q_target.append(target_next_q)
            learners_q1.append(Qsa1)
            learners_q2.append(Qsa2)
            learners_next_actions.append(next_actions)

        ## Computes mixed values
        learners_next_q_target = torch.cat(learners_next_q_target, dim=1)
        learners_q1 = torch.cat(learners_q1, dim=1)
        learners_q2 = torch.cat(learners_q2, dim=1)

        with torch.no_grad():
            next_q_mixer_inputs = self.get_q_mixer_input(batches, next_obs=True, next_actions=learners_next_actions)
            next_q_mixing_weights, next_q_mixing_biases = self.target_mixer_network_Q(next_q_mixer_inputs)
            mixed_next_q_target = MixerNetwork.mix(learners_next_q_target, next_q_mixing_weights, next_q_mixing_biases)
            
            # centralized training in cooperative setting so all learners have same reward, mask and discount_factor
            target = batches[0].rewards + self.learners[0].discount_factor*batches[0].masks*mixed_next_q_target
        
        mixed_q1 = MixerNetwork.mix(learners_q1, q_mixing_weights, q_mixing_biases)
        mixed_q2 = MixerNetwork.mix(learners_q2, q_mixing_weights, q_mixing_biases)
        
        # loss is computed for both Q: (r(s,a) + gamma*V(s') - Q(s,a))**2
        loss = ((target - mixed_q1)**2 + (target - mixed_q2)**2).mean(0)

        ## Updates all the learners and the Q mixing network

        # clean potentials left-over gradients and updates networks
        _ = [learner.doubleQ.q1.optim.zero_grad() for learner in self.learners]
        _ = [learner.doubleQ.q2.optim.zero_grad() for learner in self.learners]
        _ = [learner.memory.optim.zero_grad() for learner in self.learners]
        self.mixer_network_Q.optim.zero_grad()

        loss.backward()

        # clips the gradients
        _ = [torch.nn.utils.clip_grad_norm_(learner.doubleQ.q1.parameters(), self.grad_clip_norm) for learner in self.learners]
        _ = [torch.nn.utils.clip_grad_norm_(learner.doubleQ.q2.parameters(), self.grad_clip_norm) for learner in self.learners]
        _ = [torch.nn.utils.clip_grad_norm_(learner.memory.parameters(), self.grad_clip_norm) for learner in self.learners]
        torch.nn.utils.clip_grad_norm_(self.mixer_network_Q.parameters(), self.grad_clip_norm)

        # takes a gradient step
        _ = [learner.doubleQ.q1.optim.step() for learner in self.learners]
        _ = [learner.doubleQ.q2.optim.step() for learner in self.learners]
        _ = [learner.memory.optim.step() for learner in self.learners]
        self.mixer_network_Q.optim.step()

        mixed_q = torch.cat((mixed_q1, mixed_q2), dim=0)

        return loss.detach().cpu().numpy(), mixed_q.mean(0).detach().cpu().numpy(), mixed_q.min(0).values.detach().cpu().numpy(), mixed_q.max(0).values.detach().cpu().numpy()


    def update_pi(self, batches):

        ## computes mixing weights
        q_mixer_inputs = self.get_q_mixer_input(batches)
        q_mixing_weights, q_mixing_biases = self.mixer_network_Q(q_mixer_inputs)
        
        learners_q = []
        learners_log_pi = []
        for batch, learner in zip(batches, self.learners):
            observations = learner.concat_embedings_to_obs(batch, target=False)
            actions, log_pi = learner.policy.act(observations, sample=True, return_log_pi=True)

            q1, q2 = learner.doubleQ(observations, actions)
            q = torch.minimum(q1, q2)

            learners_q.append(q)
            learners_log_pi.append(log_pi)
            
        learners_q = torch.cat(learners_q, dim=1)
        learners_log_pi = torch.cat(learners_log_pi, dim=1)

        mixed_q = MixerNetwork.mix(learners_q, q_mixing_weights, q_mixing_biases)
        log_pi = learners_log_pi.sum(1, keepdim=True)

        loss = ((self.log_alpha.exp() * log_pi) - mixed_q).mean(0)

        # clean potentials left-over gradients and updates networks
        _ = [learner.policy.optim.zero_grad() for learner in self.learners]
        _ = [learner.memory.optim.zero_grad() for learner in self.learners]

        loss.backward()

        #clips the gradients
        _ = [torch.nn.utils.clip_grad_norm_(learner.policy.parameters(), self.grad_clip_norm) for learner in self.learners]

        # takes a gradient step
        _ = [learner.policy.optim.step() for learner in self.learners]

        if self.memory_backprop_actor:
            # clips the gradients
            _ = [torch.nn.utils.clip_grad_norm_(learner.memory.parameters(), self.grad_clip_norm) for learner in self.learners]
            # takes a gradient step
            _ = [learner.memory.optim.step() for learner in self.learners]

        # updates cosine lr schedule
        _ = [learner.policy.scheduler.step() for learner in self.learners]


        ## temperature loss
        alpha_loss = - (self.log_alpha * ((log_pi + self.target_entropy).detach())).mean(0)
    
        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()

        return loss.detach().cpu().numpy(), alpha_loss.detach().cpu().numpy()

    def update(self, dataset, batch_size, train_device, **kwargs):

        batches = dataset.sample(batch_size)

        return self.update_from_batch(batches, train_device, **kwargs)

    def batch_to_device(self, batch, device):
        if not all([data.device == device for data in batch if hasattr(data, 'to')]):
            datas = {k: data.to(device) if hasattr(data, 'to') else data for k, data in batch._asdict().items()}
            batch = batch.__class__(**datas)
        
        return batch


    def update_from_batch(self, batches, train_device, **kwargs):
        # from https://github.com/ikostrikov/implicit_q_learning/blob/master/learner.py
        # with same batch the update goes
        # 1. update V-network with L_V
        # 2. use new V-network to update policy with AWR
        # 3. update Q-network with new V-network and L_Q
        # 4. update target Q-network

        # put models on training device
        self.to(train_device)

        batches = [self.batch_to_device(batch, train_device) for batch in batches]

        # 1. update v_network with L_V
        loss_v, mean_v, min_v, max_v = 0., 0., 0., 0., 

        # 2. use new V-network to update policy with AWR
        loss_pi, loss_alpha = self.update_pi(batches)

        # 3. update Q-network with new V-network and L_Q
        loss_q, mean_q, min_q, max_q = self.update_q(batches)

        # 4. update target networks
        for learner in self.learners:
            learner.doubleQ.update_target_soft(target=learner.doubleQtarget, source=learner.doubleQ, tau=learner.target_update_coef)
            learner.memory.update_target_soft(target=learner.memorytarget, source=learner.memory, tau=learner.target_update_coef)

        self.mixer_network_Q.update_target_soft(target=self.target_mixer_network_Q, source=self.mixer_network_Q, tau=self.learners[0].target_update_coef)
        
        return {'loss_v': loss_v, 'loss_pi': loss_pi, 'loss_q': loss_q, 'loss_alpha': loss_alpha,
                'mean_v': mean_v, 'min_v': min_v, 'max_v': max_v, 
                'mean_q': mean_q, 'min_q': min_q, 'max_q': max_q}

    def get_state_dict(self):
        learners_state_dict = super().get_state_dict()
        learners_state_dict.update({'mixer_network_Q': self.mixer_network_Q.get_state_dict(),
                                    'target_mixer_network_Q': self.target_mixer_network_Q.get_state_dict(),
                                    'log_alpha': self.state_dict()['log_alpha'],
                                    'alpha_optim': self.alpha_optim.state_dict()})
        return learners_state_dict
    
    def do_load_state_dict(self, state_dict):
        # this loads network, optim, scheduler params
        _ = [learner.do_load_state_dict(state_dict[f'learner_{i}']) for i, learner in enumerate(self.learners)]
        self.mixer_network_Q.do_load_state_dict(state_dict['mixer_network_Q'])
        self.target_mixer_network_Q.do_load_state_dict(state_dict['target_mixer_network_Q'])
        self.log_alpha.data = state_dict['log_alpha'].to(self.log_alpha.device)
        self.alpha_optim.load_state_dict(state_dict['alpha_optim'])
    
    def load_state_dict(self, state_dict):
        # this only load network params
        _ = [learner.load_state_dict(state_dict[f'learner_{i}']) for i, learner in enumerate(self.learners)]
        self.mixer_network_Q.load_state_dict(state_dict['mixer_network_Q'])
        self.target_mixer_network_Q.load_state_dict(state_dict['target_mixer_network_Q'])
        self.log_alpha.data = state_dict['log_alpha'].to(self.log_alpha.device)
        self.alpha_optim.load_state_dict(state_dict['alpha_optim'])

class MaIQLDiscrete(MaIQL):
    def __init__(self, **kwargs):
        agent_c = kwargs.pop('agent_constructor', IQLLearnerDiscrete)
        super().__init__(agent_constructor=agent_c, **kwargs)

    def get_mixer_input_size(self, kwargs):
        V_mixer = sum([obs_space.shape[0] for obs_space in kwargs['obs_space']])
        if self.action_aware_mixer:
            Q_mixer = sum([obs_space.shape[0] for obs_space in kwargs['obs_space']] + [act_space.n for act_space in kwargs['act_space']])
            self.act_n = [act_space.n for act_space in kwargs['act_space']]
        else:    
            Q_mixer = V_mixer

        return Q_mixer, V_mixer 
    
    def make_one_hot(self, actions, act_dim):
        assert len(actions.shape) == 2
        assert actions.shape[1] == 1

        batch_size = actions.shape[0]

        actions_oh = torch.zeros(batch_size*act_dim, device=actions.device)
        index = actions.flatten() + torch.arange(batch_size, device=actions.device)*act_dim
        actions_oh[index] = 1.
        actions_oh = actions_oh.reshape(batch_size, act_dim)

        return actions_oh

    def get_q_mixer_input(self, batches):
        if self.action_aware_mixer:
            return torch.cat([batch['observations'] for batch in batches] + [self.make_one_hot(batch['actions'], act_dim) for batch, act_dim in zip(batches, self.act_n)], dim=1)
        else:
            return torch.cat([batch['observations'] for batch in batches], dim=1)


    def evaluate(self, *args, **kwargs):
        return IQLIndependentLearnerDiscrete.evaluate(self, *args, **kwargs)

    def update_pi(self, batches):

        ## Computes advantage weights 
        with torch.no_grad():
            ## computes mixing weights
            q_mixer_inputs = self.get_q_mixer_input(batches)
            v_mixer_inputs = self.get_v_mixer_input(batches)

            v_mixing_weights, v_mixing_biases = self.mixer_network_V(v_mixer_inputs)
            target_q_mixing_weights, target_q_mixing_biases = self.target_mixer_network_Q(q_mixer_inputs)
            
            learners_v = []
            learners_target_q = []
            for batch, learner in zip(batches, self.learners):
                target_observations = learner.concat_embedings_to_obs(batch, target=True)
                target_q1, target_q2 = learner.doubleQtarget(target_observations, batch['actions'])
                target_q = torch.minimum(target_q1, target_q2)
                vs = learner.V(target_observations)

                learners_v.append(vs)
                learners_target_q.append(target_q)
            
            learners_v = torch.cat(learners_v, dim=1)
            learners_target_q = torch.cat(learners_target_q, dim=1)

            mixed_v = MixerNetwork.mix(learners_v, v_mixing_weights, v_mixing_biases)
            mixed_target_q = MixerNetwork.mix(learners_target_q, target_q_mixing_weights, target_q_mixing_biases)

            if self.independent_awr:
                # we compute the weight independently for each agent
                individual_weight = -torch.clamp(torch.exp(self.learners[0].awr_temperature*(
                    learners_target_q*target_q_mixing_weights + target_q_mixing_biases - learners_v*v_mixing_weights - v_mixing_biases)), max=100.0)

            else:    
                # we compute a team weight directly with the mixed values
                weight = -torch.clamp(torch.exp(self.learners[0].awr_temperature*(mixed_target_q-mixed_v)), max=100.0)
        
        ## Computes learners' action likelihood   
        learners_log_pi = []
        for batch, learner in zip(batches, self.learners):
            observations = learner.concat_embedings_to_obs(batch, target=False)

            logits = learner.policy(observations)

            # we make non-legal moves very unlikely
            legal_moves = batch['legal_moves']
            assert logits.size() == legal_moves.size()
            logits = logits - (1 - legal_moves) * 1e10

            log_pi = torch.nn.functional.log_softmax(logits, dim=1).gather(dim=1, index=batch['actions'])

            learners_log_pi.append(log_pi)
        
        ## loss
        learners_log_pi = torch.cat(learners_log_pi, dim=1)
        
        if self.independent_awr:
            # compute the loss for each agent on the batch
            individual_losses = (individual_weight*learners_log_pi).mean(0, keepdim=True)
            # sum all the agents loss to get a team loss
            loss = individual_losses.sum(1)
            # just for recordings
            weight = individual_weight.sum(1, keepdim=True)

        else:
            summed_log_pi = learners_log_pi.sum(1, keepdim=True)
            loss = (weight * summed_log_pi).mean(0)
        

        # clean potentials left-over gradients and updates networks
        _ = [learner.policy.optim.zero_grad() for learner in self.learners]
        _ = [learner.memory.optim.zero_grad() for learner in self.learners]

        loss.backward()

         #clips the gradients
        _ = [torch.nn.utils.clip_grad_norm_(learner.policy.parameters(), self.grad_clip_norm) for learner in self.learners]

        # takes a gradient step
        _ = [learner.policy.optim.step() for learner in self.learners]

        if self.memory_backprop_actor:
            # clips the gradients
            _ = [torch.nn.utils.clip_grad_norm_(learner.memory.parameters(), self.grad_clip_norm) for learner in self.learners]
            # takes a gradient step
            _ = [learner.memory.optim.step() for learner in self.learners]

        # updates cosine lr schedule
        _ = [learner.policy.scheduler.step() for learner in self.learners]

        return loss.detach().cpu().numpy(), weight.cpu().numpy()