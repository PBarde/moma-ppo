# Here the decomposition is done accordingly to https://www.gymlibrary.dev/environments/mujoco/
from collections import OrderedDict
import numpy as np

class Q_extractor_function(object):
    def __init__(self):
        self
    
    def q_pos(self, obs):
        # q pos is joint0, joint1, xtarget, ytarget
        sin_0 = obs[2]
        cos_0 = obs[0]

        sin_1 = obs[3]
        cos_1 = obs[1]

        joint0 = np.arctan2(sin_0, cos_0)
        joint1 = np.arctan2(sin_1, cos_1)

        x_target = obs[4]
        y_target = obs[5]

        return np.array([joint0, joint1, x_target, y_target])
    
    def q_vel(self, obs):
        # q vel is joint0_vel, joint1_vel, xtarget_vel=0, ytarget_vel=0
        return np.array([obs[6], obs[7], 0., 0.])


Q_INFO = Q_extractor_function() #Note that the following is complicated because of gym modifying the obs for reacher, cf https://www.gymlibrary.dev/environments/mujoco/reacher/
# we would need to replace Q_INFO by a function to do the mapping

ACTION_INDEXES = OrderedDict({'first_arm': 0, 
                                'second_arm': 1})

FULL_OBSERVATION_INDEXES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

OBSERVATION_INDEXES = OrderedDict({'first_arm': OrderedDict({'cos': 0, 
                                                            'sin': 2,
                                                            'omega': 6}),

                                 'second_arm': OrderedDict({'cos': 1, 
                                                            'sin': 3,
                                                            'omega': 7}), 

                                 'target': OrderedDict({'x_pos': 4, 
                                                        'y_pos': 5}),

                                 'tip_diff': OrderedDict({'x_diff': 8,
                                                          'y_diff': 9,
                                                          'z_diff': 10})
                                })

####
CONFIGS = {'obs': OrderedDict({'2x1notip': [['first_arm', 'target'], ['second_arm', 'target']],
                                '2x1': [['first_arm', 'target', 'tip_diff'], ['second_arm', 'target', 'tip_diff']],
                                '2x1first': [['first_arm', 'second_arm', 'target'], ['first_arm', 'second_arm']],
                                '2x1second': [['first_arm', 'second_arm'], ['first_arm', 'second_arm', 'target']]}),

            'act': OrderedDict({'2x1': [['first_arm'], ['second_arm']],
                            '2x1notip': [['first_arm'], ['second_arm']],
                            '2x1first': [['first_arm'], ['second_arm']],
                            '2x1second': [['first_arm'], ['second_arm']]})}


def color_reacher_env(env):

      white = np.array([1.,1.,1.,1.])*1000.
      red = np.array([1.,0.,0.,1.000])*2
      blue = np.array([0.,0.,1.,1.])*2
      green = np.array([0.,1.,0.,1.])*2
      black = np.array([0.,0.,0.,1.])*2

      set_color(env, 'ground', white)
      set_color(env, 'link0', red)
      set_color(env, 'root', red)
      set_color(env, 'link1', blue)
      set_color(env, 'fingertip', green)
      set_color(env, 'target', black)


def set_color(env, name, rgba_vec):
      env.sim.model.geom_rgba[np.array(env.sim.model.geom_names) == name] = rgba_vec
