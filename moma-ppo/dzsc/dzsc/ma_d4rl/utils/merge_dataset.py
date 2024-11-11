import sys
from pathlib import Path
import numpy as np

def set_up_import():
    alfred_folder = (Path(__file__).resolve().parents[4] / 'alfred_omarl').resolve()
    alg_folder = (Path(__file__).resolve().parents[4] / 'offline_marl').resolve()
    env_folder = (Path(__file__).resolve().parents[4] / 'dzsc').resolve()
    
    for p in [alg_folder, env_folder, alfred_folder]:
        if p not in sys.path:
            sys.path.append(str(p))
set_up_import()


from dzsc.ma_d4rl.generated_datasets import dataset_path
import pickle

if __name__ == "__main__":

    def load_dataset(env_name):
        data_path = dataset_path.datasets[env_name]

        with open(data_path, 'rb') as fh:
            dataset = pickle.load(fh)
        
        return dataset

    env_name_1 = 'reacher-expert-C-v0'
    env_name_2 = 'reacher-expert-D-v0'

    dataset_1 = load_dataset(env_name_1)
    dataset_2 = load_dataset(env_name_2)

    n_1 = len(dataset_1['observations'])
    n_2 = len(dataset_2['observations'])

    assert n_1 == n_2

    idx = int(n_1/2)

    mixed_dataset = {}

    keys1 = set(list(dataset_1.keys()))
    keys2 = set(list(dataset_2.keys()))

    assert keys1 == keys2

    for k in keys1: 
        mixed_dataset[k] = np.concatenate((dataset_1[k][:idx], dataset_2[k][:idx]), axis=0)
    
    # we might have truncated trajectory when extracting the sub-datasets so we set time-out-mask to end of trajectory
    mixed_dataset['time_out_masks'][idx-1] = 0
    mixed_dataset['time_out_masks'][-1] = 0
    
    file_name = Path(__file__).parent.resolve() / f'mix_{env_name_1}_{env_name_2}.pt'

    with open(str(file_name), 'wb') as fh: 
        pickle.dump(mixed_dataset, fh)
    
    print("done mixing")


