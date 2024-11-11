import argparse


def get_argparser():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--scenario", type=str, default='HalfCheetah-v2', help="Mujoco env")
    parser.add_argument("--agent_conf", type=str, default='6x1', help="how to split single agent env into multiple-agent: n_agents X action_dim")
    parser.add_argument("--agent_obsk", type=int, default=0, help="multi-agent mujoco observation distance. 0 = observe just the correspinding agent, None = observe global state")
    parser.add_argument("--max_episode", type=int, default=10, help="number of episodes to collect")

    return parser

def get_obs_indexes(configuration, agent_conf, agent_obsk):
    obs_indxs = []

    if agent_obsk == 'full':
        # Fully observable, we take all measurement for all the parts for each agents
        for agent in configuration.CONFIGS['obs'][agent_conf]:
            obs_indxs.append(configuration.FULL_OBSERVATION_INDEXES)
        
        return obs_indxs

    else:
        # partially observable

        # parts that we need to observe in addition to the controlled ones 
        additional_parts = []

        # if the environment contains a goal we need to observe it
        if 'goal' in configuration.OBSERVATION_INDEXES:
            additional_parts.append('goal')

        # additional observed parts due to obsk (observe parts at distance k)
        if agent_obsk == '0t':
            #observes self and torso
            additional_parts.append('torso_no_vel')
            additional_parts.append('torso_xvel')
            additional_parts.append('torso_yvel')
            additional_parts.append('torso_zvel')
        
        elif agent_obsk == '0tv':
            additional_parts.append('torso_xvel')
            additional_parts.append('torso_yvel')
            additional_parts.append('torso_zvel')
        
        elif agent_obsk == '0txv':
            additional_parts.append('torso_xvel')

        elif agent_obsk == '0tyv':
            additional_parts.append('torso_yvel')

        elif agent_obsk == '0txyv':
            additional_parts.append('torso_xvel')
            additional_parts.append('torso_yvel')

        elif agent_obsk == '0tnv':
            #observes self and torso
            additional_parts.append('torso_no_vel')

        elif agent_obsk == '0':
            # no additional parts
            pass

        else:
            raise NotImplementedError


        for agent in configuration.CONFIGS['obs'][agent_conf]:
            agent_obs_indxs = []
            for part in agent + additional_parts:
                part_measurements = [indx for indx in configuration.OBSERVATION_INDEXES[part].values()]
                agent_obs_indxs += part_measurements

            obs_indxs.append(agent_obs_indxs)


        return obs_indxs

def get_action_indexes(configuration, agent_conf):
    action_indxs = []
    
    for agent in configuration.CONFIGS['act'][agent_conf]:
        agent_action_indxs = [configuration.ACTION_INDEXES[part] for part in agent if part in configuration.ACTION_INDEXES]
        action_indxs.append(agent_action_indxs)

    return action_indxs

def get_mappings(args):

    if 'ant' in args.scenario.lower():
        import dzsc.ma_d4rl.obs_mapping.ant as configuration
    elif 'halfcheetah' in args.scenario.lower():
        import dzsc.ma_d4rl.obs_mapping.halfcheetah as configuration
    elif 'walker2d' in args.scenario.lower():
        import dzsc.ma_d4rl.obs_mapping.walker2d as configuration
    elif 'hopper' in args.scenario.lower():
        import dzsc.ma_d4rl.obs_mapping.hopper as configuration
    elif 'antmaze' in args.scenario.lower():
        import dzsc.ma_d4rl.obs_mapping.antmaze as configuration
    elif 'reacher' in args.scenario.lower():
        import dzsc.ma_d4rl.obs_mapping.reacher as configuration
    else:
        raise NotImplementedError

    
    return {'obs_indexes': get_obs_indexes(configuration, args.agent_conf, args.agent_obsk),
            'action_indexes': get_action_indexes(configuration, args.agent_conf),
            'q_info': configuration.Q_INFO}

if __name__ == "__main__":

    args = get_argparser().parse_args()

    mappings = get_mappings(args)

    print(mappings)


