from offline_marl.multi_agent.iql_independent import IQLIndependentLearner
from offline_marl.single_agent.td3 import TD3Learner

class ITD3Learner(IQLIndependentLearner):
    def __init__(self, agent_constructor=TD3Learner, learner_name='itd3', agents_suffix='td3', **kwargs):
        super().__init__(agent_constructor=agent_constructor, learner_name=learner_name, agents_suffix=agents_suffix, **kwargs)