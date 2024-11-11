SINGLE_AGENT_TASKS = [

    ## MUJOCO GYM 
    "hopper-random-v1",
    "hopper-random-v2",
    "hopper-medium-v1",
    "hopper-medium-v2",
    "hopper-expert-v1",
    "hopper-expert-v2",
    "hopper-medium-expert-v1",
    "hopper-medium-expert-v2",
    "hopper-medium-replay-v1",
    "hopper-medium-replay-v2",
    "hopper-full-replay-v1",
    "hopper-full-replay-v2",
    "halfcheetah-random-v1",
    "halfcheetah-random-v2",
    "halfcheetah-medium-v1",
    "halfcheetah-medium-v2",
    "halfcheetah-expert-v1",
    "halfcheetah-expert-v2",
    "halfcheetah-medium-expert-v1",
    "halfcheetah-medium-expert-v2",
    "halfcheetah-medium-replay-v1",
    "halfcheetah-medium-replay-v2",
    "halfcheetah-full-replay-v1",
    "halfcheetah-full-replay-v2",
    "ant-random-v1",

    "ant-random-v2",
    
    "ant-random-v2",
    "ant-random-v2_full",
    "ant-random-v2_0t",
    "ant-random-v2_0tnv",
    "ant-random-v2_0tv",
    "ant-random-v2_0t",
    
    "ant-medium-v1",

    "ant-medium-v2",
    "ant-medium-v2_full",
    "ant-medium-v2_0t",

    "ant-medium-v2_0tnv",
    "ant-medium-v2_0tv",
    "ant-medium-v2_0txv",
    "ant-medium-v2_0txyv",

    "ant-medium-v2_0",

    "ant-expert-v1",
    "ant-expert-v2",
    "ant-expert-v2_0t",
    "ant-expert-v2_full",

    "ant-medium-expert-v1",
    "ant-medium-expert-v2",
    "ant-medium-replay-v1",
    "ant-medium-replay-v2",

    "ant-medium-replay-v2_0tv",
    "ant-medium-replay-v2_0txv",
    "ant-medium-replay-v2_0txyv",

    "ant-full-replay-v1",
    "ant-full-replay-v2",

    "ant-full-replay-v2_0t",
    "ant-full-replay-v2_full",

    "ant-full-replay-v2_0tv",
    "ant-full-replay-v2_0txv",
    "ant-full-replay-v2_0txyv",

    "walker2d-random-v1",
    "walker2d-random-v2",
    "walker2d-medium-v1",
    "walker2d-medium-v2",
    "walker2d-expert-v1",
    "walker2d-expert-v2",
    "walker2d-medium-expert-v1",
    "walker2d-medium-expert-v2",
    "walker2d-medium-replay-v1",
    "walker2d-medium-replay-v2",
    "walker2d-full-replay-v1",
    "walker2d-full-replay-v2",

    ## LOCOMOTION
    "antmaze-medium-diverse-v2",
    "antmaze-medium-diverse-v0",
    "antmaze-umaze-v2"
]

ADDED_TASKS = ["reacher-dataset-v0",
                "reacher-expert-D-v0",
                "reacher-expert-C-v0",
                "reacher-expert-mix-v0"]

SINGLE_AGENT_TASKS += ADDED_TASKS

if __name__ == "__main__":
    task_names = []
    for agent in ["hopper", "halfcheetah", "ant", "walker2d"]:
        for dataset in [
            "random",
            "medium",
            "expert",
            "medium-expert",
            "medium-replay",
            "full-replay",
        ]:
            for version in ["v1", "v2"]:
                env_name = "%s-%s-%s" % (agent, dataset, version)
                task_names.append(env_name)

    print(task_names)
