# Here the decomposition is done accordingly to https://www.gymlibrary.dev/environments/mujoco/half_cheetah/
from collections import OrderedDict

# Remember that here idxs are shifted by n_append_left
Q_INFO = OrderedDict({'qpos_idxs': [0, 1, 2, 3, 4, 5],
                        'qvel_idxs': [6, 7, 8, 9, 10, 11],
                        'n_append_left': 1}) # because x,y pos not present in state

ACTION_INDEXES = OrderedDict({'thigh_joint': 0, 
                    'leg_joint': 1, 
                    'foot_joint': 2})

OBSERVATION_INDEXES = OrderedDict({'torso_no_vel': OrderedDict({'z_pos': 0, 
                                    'y_theta': 1,
                                    'y_omega': 7}),
                        'torso_vel': OrderedDict({'x_vel': 5, 
                                        'z_vel': 6}), 
                        'thigh_joint': OrderedDict({'theta': 2, 
                                        'omega': 8}),
                        'leg_joint': OrderedDict({'theta': 3, 
                                        'omega': 9}),
                        'foot_joint': OrderedDict({'theta': 4, 
                                        'omega':10})
                        })

####
CONFIGS = {'obs':OrderedDict({'3x1': [['thigh_joint'], 
                    ['leg_joint'], 
                    ['foot_joint']]}),
            'act': OrderedDict({'3x1': [['thigh_joint'], 
                    ['leg_joint'], 
                    ['foot_joint']]})}