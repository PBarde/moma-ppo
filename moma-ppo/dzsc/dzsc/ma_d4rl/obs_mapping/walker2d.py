# Here the decomposition is done accordingly to https://www.gymlibrary.dev/environments/mujoco/half_cheetah/
from collections import OrderedDict

ACTION_INDEXES = OrderedDict({'thigh_joint':0,
                    'leg_joint': 1, 
                    'foot_joint': 2, 
                    'thigh_left_joint': 3, 
                    'leg_left_joint': 4, 
                    'foot_left_joint': 5})

OBSERVATION_INDEXES = OrderedDict({'torso_no_vel': OrderedDict({'z_pos': 0, 
                                    'y_theta': 1,
                                    'y_omega': 10}),
                        'torso_vel': OrderedDict({'x_vel': 8, 
                                    'z_vel': 9}),
                        'thigh_joint': OrderedDict({'theta': 2, 
                                        'omega': 11}), 
                        'leg_joint': OrderedDict({'theta': 3, 
                                        'omega': 12}), 
                        'foot_joint': OrderedDict({'theta': 4, 
                                        'omega': 13}),
                        'thigh_left_joint':OrderedDict({'theta': 5, 
                                                'omega': 14}), 
                        'leg_left_joint': OrderedDict({'theta': 6,
                                            'omega': 15}), 
                        'foot_left_joint': OrderedDict({'theta': 7, 
                                        'omega': 16})
                    })

####
CONFIGS = OrderedDict({'2x3': [['thigh_joint', 'leg_joint', 'foot_joint'], 
                    ['thigh_left_joint', 'leg_left_joint', 'foot_left_joint']],
                    
            '6x1': [['thigh_joint'], 
                    ['leg_joint'], 
                    ['foot_joint'], 
                    ['thigh_left_joint'], 
                    ['leg_left_joint'], 
                    ['foot_left_joint']]
        })