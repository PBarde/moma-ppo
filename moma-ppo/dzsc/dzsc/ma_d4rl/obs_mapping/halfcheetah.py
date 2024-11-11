# Here the decomposition is accordingly to https://www.gymlibrary.dev/environments/mujoco/half_cheetah/
from collections import OrderedDict

# here all indexes shifted to the left by 1 because of the n_append_left (there is one more qpos-idx)
Q_INFO = OrderedDict({'qpos_idxs': [0, 1, 2, 3, 4, 5, 6, 7, 8],
                        'qvel_idxs': [9, 10, 11, 12, 13, 14, 15, 16, 17],
                        'n_append_left': 1}) 


ACTION_INDEXES = OrderedDict({'bthigh': 0,
                    'bshin': 1,
                    'bfoot': 2,
                    'fthigh': 3,
                    'fshin': 4,
                    'ffoot': 5})


FULL_OBSERVATION_INDEXES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

OBSERVATION_INDEXES = OrderedDict({ 'torso_no_vel': OrderedDict({'z_pos': 0,
                                    'y_theta': 1,
                                    'y_omega': 10}), 
                        'torso_vel': OrderedDict({'x_vel': 8,
                                    'y_vel': 9}), 
                        'bthigh': OrderedDict({'theta': 2,
                                    'omega': 11}), 
                        'bshin' : OrderedDict({'theta': 3,
                                    'omega': 12}),
                        'bfoot' : OrderedDict({'theta': 4, 
                                    'omega': 13}),
                        'fthigh': OrderedDict({'theta': 5, 
                                    'omega': 14}),
                        'fshin' : OrderedDict({'theta': 6,
                                    'omega': 15}),
                        'ffoot' : OrderedDict({'theta': 7, 
                                    'omega': 16})
                        })

####
CONFIGS = {'obs': OrderedDict({'2x3': [['bthigh', 'bshin', 'bfoot'], 
                                        ['fthigh', 'fshin', 'ffoot']], 

                                '6x1': [['bthigh'], 
                                        ['bshin'], 
                                        ['bfoot'], 
                                        ['fthigh'], 
                                        ['fshin'], 
                                        ['ffoot']]}),
        'act': OrderedDict({'2x3': [['bthigh', 'bshin', 'bfoot'], 
                                        ['fthigh', 'fshin', 'ffoot']], 

                                '6x1': [['bthigh'], 
                                        ['bshin'], 
                                        ['bfoot'], 
                                        ['fthigh'], 
                                        ['fshin'], 
                                        ['ffoot']]}),}
        