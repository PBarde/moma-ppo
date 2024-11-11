from ssl import ALERT_DESCRIPTION_BAD_CERTIFICATE_STATUS_RESPONSE
import sys
from pathlib import Path
sys.path.append(str((Path(__file__).resolve().parents[4] / 'dzsc').resolve()))
from copy import deepcopy

import dzsc.ma_d4rl.obs_mapping.ant as ant_mapping


# The decomposition here is the same as ant except that the xy_direction of the goal is appended to the observation vector
ACTION_INDEXES = deepcopy(ant_mapping.ACTION_INDEXES)

OBSERVATION_INDEXES = deepcopy(ant_mapping.OBSERVATION_INDEXES)
OBSERVATION_INDEXES['goal'] = {'x_pos': 27,
                                'y_pos': 28}

CONFIGS = deepcopy(ant_mapping.CONFIGS)