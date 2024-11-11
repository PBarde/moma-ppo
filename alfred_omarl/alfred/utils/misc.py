import logging
import sys
from math import floor, log10
import re
from tqdm import tqdm
from pathlib import Path

LOGGER = logging.getLogger(__name__)

from alfred.utils.config import config_to_str
from alfred.utils.directory_tree import DirectoryTree, get_root

COMMENTING_CHAR_LIST = ['#']


def create_logger(name, loglevel, logfile=None, streamHandle=True):
    logger = logging.getLogger(name)
    logger.setLevel(loglevel)
    formatter = logging.Formatter(fmt='%(asctime)s - %(levelname)s - {} - %(message)s'.format(name),
                                  datefmt='%d/%m/%Y %H:%M:%S', )

    handlers = []
    if logfile is not None:
        handlers.append(logging.FileHandler(logfile, mode='a'))
    if streamHandle:
        handlers.append(logging.StreamHandler(stream=sys.stdout))

    for handler in handlers:
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


def create_new_filehandler(logger_name, logfile):
    formatter = logging.Formatter(fmt='%(asctime)s - %(levelname)s - {} - %(message)s'.format(logger_name),
                                  datefmt='%d/%m/%Y %H:%M:%S', )

    file_handler = logging.FileHandler(logfile, mode='a')
    file_handler.setFormatter(formatter)

    return file_handler


def keep_two_signif_digits(x):
    try:
        if x == 0.:
            return x
        else:
            return round(x, -int(floor(log10(abs(x))) - 1))
    except:
        return x


def sorted_nicely(l):
    """ Sort the given iterable in the way that humans expect."""
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)


def create_management_objects(dir_tree, logger, pbar, config, pbar_total):
    # Creates directory tres

    if dir_tree is None:
        dir_tree = DirectoryTree(alg_name=config.alg_name,
                                 task_name=config.task_name,
                                 desc=config.desc,
                                 seed=config.seed,
                                 root=config.root_dir)

    dir_tree.create_directories()

    # Creates logger and prints config

    if logger is None:
        logger = create_logger('MASTER', config.log_level, dir_tree.seed_dir / 'logger.out')
    logger.debug(config_to_str(config))

    # Creates a progress-bar

    if pbar == "default_pbar":
        pbar = tqdm(range(*pbar_total), desc=f'{dir_tree.storage_dir.name}/{dir_tree.experiment_dir.name}/{dir_tree.seed_dir.name}')

    elif pbar is not None:
        pbar.n = 0
        pbar.desc += f'{dir_tree.storage_dir.name}/{dir_tree.experiment_dir.name}/{dir_tree.seed_dir.name}'
        pbar.total = pbar_total

    elif pbar is None:
        pbar = range(*pbar_total)

    return dir_tree, logger, pbar


def check_params_defined_twice(keys):
    counted_keys = {key: keys.count(key) for key in keys}
    for key in counted_keys.keys():
        if counted_keys[key] > 1:
            raise ValueError(f'Parameter "{key}" appears {counted_keys[key]} times in the schedule.')


def is_commented(str_line, commenting_char_list):
    return str_line[0] in commenting_char_list


def remove_commented_at_end_of_line(str_line, commenting_char_list):
    '''
    Careful this function only works if commenting_char_list is ['#'] as provided at the top of this file
    '''
    if not len(COMMENTING_CHAR_LIST) == 1:
        raise NotImplementedError("commenting_char_list must be equal to ['#']")
    if commenting_char_list[0] in str_line:
        splitted = str_line.split(commenting_char_list[0], 1)[0]
        return splitted.replace(' ', '')
    else:
        return str_line


def select_storage_dirs(from_file, storage_name, root_dir):
    if from_file is not None:
        assert storage_name is None, "If launching --from_file, no storage_name should be provided"
        assert Path(from_file).suffix == '.txt', f"The provided --from_file should be a text file listing " \
                                                 f"storage_name's of directories to act on. " \
                                                 f"You provided '--from_file={from_file}'"

    if storage_name is not None:
        assert from_file is None, "Cannot launch --from_file if --storage_name"

    if from_file is not None:
        with open(from_file, "r") as f:
            storage_names = f.readlines()
        storage_names = [sto_name.strip('\n') for sto_name in storage_names]

        # drop the commented lignes in the .txt
        storage_names = [sto_name for sto_name in storage_names if not sto_name == '']  # remove empty lines
        storage_names = [sto_name for sto_name in storage_names if
                         not is_commented(sto_name, COMMENTING_CHAR_LIST)]  # Removing commented lines
        storage_names = [remove_commented_at_end_of_line(sto_name, COMMENTING_CHAR_LIST) for sto_name in
                         storage_names]  # Removing comments at the end of line

        storage_dirs = [get_root(root_dir) / sto_name for sto_name in storage_names]

    elif storage_name is not None:
        if isinstance(storage_name, str):
            storage_name = [storage_name]
            
        storage_dirs = [get_root(root_dir) / s for s in storage_name]

    else:
        raise NotImplementedError(
            "storage_dirs to operate over must be specified either by --from_file or --storage_name")

    return storage_dirs


def formatted_time_diff(total_time_seconds):
    n_hours = int(total_time_seconds // 3600)
    n_minutes = int((total_time_seconds - n_hours * 3600) // 60)
    n_seconds = int(total_time_seconds - n_hours * 3600 - n_minutes * 60)
    return f"{n_hours}h{str(n_minutes).zfill(2)}m{str(n_seconds).zfill(2)}s"


def uniquify(newfilepath):
    """
    Appends a number ID to newfilepath if files with same name (but different ID) already exist
    :param newfilepath (pathlib.Path): Full path to new file
    """
    max_num = -1
    for existing_file in newfilepath.parent.iterdir():
        if newfilepath.stem in existing_file.stem and newfilepath.suffix == existing_file.suffix:
            str_end = str(existing_file.stem).split('_')[-1]
            if str_end.isdigit():
                num = int(str_end)
                if num > max_num:
                    max_num = num

    return newfilepath.parent / (newfilepath.stem + f"_{max_num + 1}" + newfilepath.suffix)


def robust_seed_aggregate(list_of_list_of_values, aggregator, casting_op=lambda x: x):
    ## this function works for seeds-values lists that have different lengths by aggregating only on seeds
    ## that do actually have values for this timestep. Assumes that the lists are dense, meaning they have the
    ## same time-steps except that some terminate earlier

    n_seeds = len(list_of_list_of_values)
    list_lens = [len(l) for l in list_of_list_of_values]
    max_len = max(list_lens)
    robust_aggregate = []
    for t in range(max_len):
        values_t = [list_of_list_of_values[s][t] for s in range(n_seeds) if t < list_lens[s]]
        if aggregator == 'strictly_equal':
            assert len(set(values_t)) == 1
            robust_aggregate.append(values_t[0])
        else:
            robust_aggregate.append(aggregator(values_t))

    return casting_op(robust_aggregate)


class logging_tqdm(tqdm):
    def __init__(
            self,
            *args,
            logger: logging.Logger = None,
            mininterval: float = 1,
            bar_format: str = '{desc}{percentage:3.0f}%{r_bar}',
            desc: str = 'progress: ',
            **kwargs):
        self._logger = logger
        super().__init__(
            *args,
            mininterval=mininterval,
            bar_format=bar_format,
            desc=desc,
            **kwargs
        )

    @property
    def logger(self):
        if self._logger is not None:
            return self._logger
        return LOGGER

    def display(self, msg=None, pos=None):
        if not self.n:
            # skip progress bar before having processed anything
            return
        if not msg:
            msg = self.__str__()
        self.logger.info('%s', msg)

class Bunch(object):
    def __init__(self, *args, **kwargs):
        # we either get a dict of kwargs
        if len(args)>0:
            assert len(args) == 1
            assert len(kwargs) == 0
            arg = args[0]
            assert isinstance(arg, dict)
            adict = arg
        else:
            adict = kwargs
        
        self.__dict__.update(adict)

    def _asdict(self):
        return self.__dict__

    def __iter__(self):
        # be mindfull that Bunch is unordered but the NamedTuples are
        return iter(self.__dict__.values())