ROOT = "/home/mila/b/bardepau/code/offline_marl/scratch/ma_d4rl_generated_datasets"

datasets = {
    "reacher-expert-D-v0": (
        f"{ROOT}/reacher-expert-D-v0/mujoco_ppo_4/No1_eb3c07e-285dd16_ma-ppo_reacher-dataset-v0_2x1_full_grid_mujoco_ppo_4/experiment1/seed1/generated_datasets/eval_seed_10101_n_1000000_ma-ppo_model.pt/dataset.pkl"),
    
    "reacher-expert-C-v0": (
        f"{ROOT}/reacher-expert-C-v0/mujoco_ppo_4/No1_eb3c07e-285dd16_ma-ppo_reacher-dataset-v0_2x1_full_grid_mujoco_ppo_4/experiment1/seed2/generated_datasets/eval_seed_10101_n_1000000_ma-ppo_model.pt/dataset.pkl"),
        
    "reacher-expert-mix-v0": (
        f"{ROOT}/reacher-expert-mix-v0/mix_reacher-expert-C-v0_reacher-expert-D-v0.pt"
    )}