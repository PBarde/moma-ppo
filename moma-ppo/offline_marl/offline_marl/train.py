from pathlib import Path
import sys

from torch.multiprocessing import set_start_method
try:
     set_start_method('spawn')
except RuntimeError:
    pass


def set_up_import():
    alfred_folder = (Path(__file__).resolve().parents[2] / 'alfred_omarl').resolve()
    alg_folder = (Path(__file__).resolve().parents[2] / 'offline_marl').resolve()
    env_folder = (Path(__file__).resolve().parents[2] / 'dzsc').resolve()
    
    for p in [alg_folder, env_folder, alfred_folder]:
        if p not in sys.path:
            sys.path.append(str(p))


set_up_import()

import argparse

def get_training_argsparser():
    parser = argparse.ArgumentParser()

    # Alfred args
    parser.add_argument('--alg_name', type=str, default='mb-ma-iql', help="name of the algo")
    main_args, extra_args = parser.parse_known_args()

    return main_args, extra_args

def make_trainer(alg_name):
    if alg_name == 'world-model':
        from offline_marl.train_worldmodel import WorldTraining
        trainer = WorldTraining()
    elif 'mb-' in alg_name:
        if 'ppo' in alg_name:
            from offline_marl.train_rl_model_based import MBRLAgentTraining
            trainer = MBRLAgentTraining()
        else:
            from offline_marl.train_model_based_agent import ModelBasedAgentTraining
            trainer = ModelBasedAgentTraining()
    elif 'ppo' in alg_name:
        from offline_marl.train_rl import RLAgentTraining
        trainer = RLAgentTraining()
    else:
        from offline_marl.train_agents import AgentTraining
        trainer = AgentTraining()

    return trainer

if __name__ == '__main__':

    main_args, extra_args = get_training_argsparser()

    trainer = make_trainer(main_args.alg_name)

    # gets script arguments
    config = trainer.get_training_argsparser().parse_args()

    trainer.train(config)