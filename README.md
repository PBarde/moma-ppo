
## Disclaimer
This repository is **not** meant to be a clean, off-the-shelf, codebase. Rather we share the (rough) code we used for the paper [A Model-Based Solution to the Offline Multi-Agent Reinforcement Learning Coordination Problem](https://arxiv.org/pdf/2305.17198) in which we present the MOMA-PPO algorithm. The aim is transparency and reproducability, the code is provided as is and will not be supported/extended/cleaned by us. The goal is to allow others to have a look at our implementations and to possibly reuse part of it in their own codebase. 
If you still want to try and install the code and associated libraries follow the instructions under [the installation section](#installation).

## Get the datasets
1. Download the reacher datasets from https://huggingface.co/datasets/pbarde1/moma-ppo/tree/main
2. The other datasets will be automatically downloaded from D4RL servers. 
   
## Get pretrained world-models
1. You can download the world-models we trained and used in the paper at https://huggingface.co/pbarde1/moma-ppo/tree/main

## Organise the directory structure
```
offline_marl/
├── alfred_omarl
│   ├── alfred
│   └── alfred.egg-info
├── moma-ppo
│   ├── dzsc
│   └── offline_marl
└── scratch
    ├── ma_d4rl_generated_datasets
    └── world_models
```

### Update pointers to correct paths
In `dzsc/ma_d4rl/generated_datasets/dataset_path.py`
```
ROOT = "path_to_scratch/ma_d4rl_generated_datasets"
```
In `offline_marl/load_worldmodel.py l.39`
```
code_root = "path_to_scratch"
```

## Run experiments
1. Go under `moma-ppo/offline_marl/offline_marl` to run experiments.
2. `train_worldmodel.py` to train world models. 
3. `train_agents.py` to train IQL, MAIQL, TD3+BC, CQL, OMAR, for instance for MAIQL on Two-Agent Reacher leader-only task: 
```
python train_agents.py --alg_name ma-iql --task_name reacher-expert-mix-v0_2x1first_0
```
4. `train_rl_model_based.py` to train MOMA-PPO: 
```
python train_rl_model_based.py --task_name reacher-expert-mix-v0_2x1first_0
```
5. `train_rl.py` to train purely RL algos like MA-PPO (to generate expert datasets for instance).
6. `train_model_based_agent.py` like `train_agent.py` but uses the world-models to generate additionnal data and trains the agents on a mixture of true and synthetic data. 

### Licence
https://creativecommons.org/licenses/by-nc/4.0/

## Installation

As you are about to find out that the installation is a real headache (due to dependencies on `mujoco, d4rl, dm_control`, etc.)

If you still want to try and go through it you can find the outputs from the following commands: 

```
pip freeze > pip_freeze.txt
conda list -e > conda_list.txt
conda env export --no-builds | grep -v "prefix" > environment.yml
```

you can try and see if you are lucky (I doubt it) with: 
```
conda env create -n omarl -f environment.yml
conda activate omarl
```

#### Manual installation
Below are the steps I followed to recreate a conda env that was able to run the code.

1. Set up initial environment 
```
conda create -y -n omarl python=3.8.12 pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -c conda-forge

conda activate omarl

conda install numpy==1.22.1
conda install gym==0.21.0

conda install -c conda-forge glew
conda install -c conda-forge mesalib
conda install -c menpo glfw3

pip install pip==24.0
pip install mujoco-py==2.1.2.14
```


2. install Mujoco
    * Download the MuJoCo version 2.1 binaries for [Linux](https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz) or [OSX](https://mujoco.org/download/mujoco210-macos-x86_64.tar.gz).
    *    Extract the downloaded mujoco210 directory into `~/.mujoco/mujoco210.`
    If you want to specify a nonstandard location for the package, use the env variable `MUJOCO_PY_MUJOCO_PATH`.
    * add `export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:path_to_mujoco/mujoco210/bin` to `.bashrc`


3. install D4RL
    * `git clone git@github.com:rail-berkeley/d4rl.git`
    * `cd d4rl`
    *  **edit `setup.py` file**: from `dm_control >= 1.0.3` to `dm_control @ git+https://github.com/deepmind/dm_control@4f1a9944bf74066b1ffe982632f20e6c687d45f1`
    *  **edit `setup.py` file**: from `mjrl @ git+git://github.com/aravindr93/mjrl@master#egg=mjrl` to `mjrl @ git+https://github.com/aravindr93/mjrl@master#egg=mjrl`
    ```
    pip install -e .
    pip install dm_control
    pip install -e .
    pip install "Cython<3"
    export CPATH=$CONDA_PREFIX/include
    pip install patchelf
    ```
    Then building mujoco_py by importing it in python should go through

1. install `wandb`
    ```
    pip install wandb
    wandb login
    ```
    You will have to modify the `wandb.init` line accordingly for your project and entity, look for
    ```
    wandb.init(id=self.config.wandb_run_id, project='mila_omarl', reinit=True, resume="allow", entity='paul-b-barde', config=self.config)
    ```
2. install stuff so `wandb` can record videos

     `pip install moviepy imageio`


3. install `alfred_omarl` (an open souce library to monitor experiments that we modified) by following its `README`

    we recommend installing it in editable: `pip install -e .`

4. install submitit `pip install submitit`

5. Misc
    ```
    pip install nop
    pip install readchar
    ```

#### Note 
Note that for me after this install `env.render(‘rgb_array’)` is not working (probably a problem with `mujoco` installation) so set `--render_gif` to `False` when training. 
