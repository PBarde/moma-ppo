from alfred.utils.config import *
from alfred.utils.directory_tree import *
from alfred.utils.misc import create_logger, select_storage_dirs
from alfred.prepare_schedule import create_experiment_dir
from importlib import import_module


def my_type_func(add_arg):
    name, val_type = add_arg.split("=", 1)
    val, typ = val_type.split(",", 1)
    if typ == 'float':
        val = float(val)
    elif val == "None":
        val = None
    elif val == "False":
        val = False
    elif val == "True":
        val = True
    elif typ == 'str':
        val = str(val)
    elif typ == 'int':
        val = int(val)
    else:
        raise NotImplementedError
    return name, val


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--from_file', type=str, default=None,
                        help="Path containing all the storage_names for which to create retrainBests")
    parser.add_argument('--storage_name', type=str, default=None)
    parser.add_argument('--experiment_num', type=int, default=None, help="If None, copy all experiments", nargs='+')
    parser.add_argument('--new_desc', type=str, default=None)
    parser.add_argument('--append_new_desc', type=parse_bool, default=True)
    parser.add_argument("--additional_param", action='append',
                        type=my_type_func, dest='additional_params',
                        help='To add two params p1 and p2 with values v1 and v2 of type t1 and t2 do : --additional_param p1=v1,t1 '
                             '--additional_param p2=v2,t2')
    parser.add_argument('--n_seeds', type=int, default=None)
    parser.add_argument('--train_time_factor', type=float, default=2.,
                        help="Factor by which training time should be increased / decreased")
    parser.add_argument('--train_time_key', type=str,
                        help="Config key setting the training duration, e.g. max_episodes, "
                             "max_steps, etc.")
    parser.add_argument("--root_dir", default=None, type=str)
    return parser.parse_args()


def copy_configs(from_file, storage_name, experiment_num, new_desc, append_new_desc, additional_params,
                 train_time_factor, train_time_key, n_seeds, root_dir):
    logger = create_logger(name="COPY CONFIG", loglevel=logging.INFO)
    logger.info("\nCOPYING Config")

    # Select storage_dirs to run over

    storage_dirs = select_storage_dirs(from_file, storage_name, root_dir)

    # Sanity-check that storages exist

    storage_dirs = [storage_dir for storage_dir in storage_dirs if sanity_check_exists(storage_dir, logger)]

    # Imports schedule file to have same settings for DirectoryTree.git_repos_to_track

    if from_file:
        schedule_file = str([path for path in Path(from_file).parent.iterdir() if
                             'schedule' in path.name and path.name.endswith('.py')][0])
        schedule_module = ".".join(schedule_file.split('/')).strip('.py')
        schedule = import_module(schedule_module)

    for storage_to_copy in storage_dirs:

        # extract storage name info

        _, _, _, _, old_desc = \
            DirectoryTree.extract_info_from_storage_name(storage_to_copy.name)

        # overwrites it

        tmp_dir_tree = DirectoryTree(alg_name="nope", task_name="nap", desc="nip", seed=1, root=root_dir)
        storage_name_id, git_hashes, _, _, _ = \
            DirectoryTree.extract_info_from_storage_name(str(tmp_dir_tree.storage_dir.name))

        if new_desc is None:
            desc = old_desc
        elif new_desc is not None and append_new_desc:
            desc = f"{old_desc}_{new_desc}"
        else:
            desc = new_desc

        if experiment_num is None:
            expe_to_copy = DirectoryTree.get_all_experiments(storage_to_copy)
        else:
            expe_to_copy = [storage_to_copy / f'experiment{num}' for num in experiment_num]

        for expe in expe_to_copy:

            to_copy = DirectoryTree.get_all_seeds(expe)[0]
            # find the path to the configs files

            config_path = to_copy / 'config.json'
            config_unique_path = to_copy / 'config_unique.json'

            # load the configs to copy
            config = load_config_from_json(str(config_path))
            config.desc = desc

            config_unique_dict = load_dict_from_json(str(config_unique_path))

            # Adds the additional params
            if additional_params is not None:

                for (key, value) in additional_params:
                    config.__dict__[key] = value
                    config_unique_dict[key] = value

                    # Creates the new storage_dir for retrainBest

            dir_tree = create_experiment_dir(storage_name_id=storage_name_id,
                                             config=config,
                                             config_unique_dict=config_unique_dict,
                                             SEEDS=[i * 10 for i in range(n_seeds)],
                                             root_dir=root_dir,
                                             git_hashes=DirectoryTree.get_git_hashes())

        open(str(dir_tree.storage_dir / f'config_copied_from_{str(storage_to_copy.name)}'), 'w+').close()


if __name__ == "__main__":
    args = vars(get_args())
    print(args)
    copy_configs(**args)
