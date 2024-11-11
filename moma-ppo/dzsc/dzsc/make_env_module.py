
import dzsc.ma_d4rl.utils.constants as sa_d4rl_constants
import dzsc.ma_d4rl.constants as ma_d4rl_constants

SINGLE_AGENT_TASKS = sa_d4rl_constants.SINGLE_AGENT_TASKS
MULTI_AGENT_TASKS = ma_d4rl_constants.MULTI_AGENT_TASKS

def get_make_env_module(task_name):
    import dzsc.ma_d4rl.utils.env_and_dataset as single_agent_env_and_dataset
    import dzsc.ma_d4rl.multiagent_env_and_dataset as multiagent_env_and_dataset
        
    if task_name in SINGLE_AGENT_TASKS:
        return single_agent_env_and_dataset
    elif task_name in MULTI_AGENT_TASKS:
        return multiagent_env_and_dataset
    else:
        raise NotImplementedError(f"unknown task_name {task_name}")