# Here the decomposition is done accordingly to https://www.gymlibrary.dev/environments/mujoco/ant/
# NOTE: these mappings are used by antmaze, modifying them will modify antmaze's, see dszc/ma_d4rl/obs_mapping/antmaze.py

from collections import OrderedDict
import numpy as np

# Remember that here idxs are shifted by n_append_left
Q_INFO = OrderedDict({'qpos_idxs': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
                        'qvel_idxs': [15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28],
                        'n_append_left': 2}) # because x,y pos not present in state

ACTION_INDEXES = OrderedDict({'front_left_hip': 0,
                  'front_left_ankle': 1, 
                  'front_right_hip': 2,
                  'front_right_ankle': 3,
                  'back_left_hip': 4,
                  'back_left_ankle': 5, 
                  'back_right_hip': 6, 
                  'back_right_ankle': 7})

FULL_OBSERVATION_INDEXES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]

OBSERVATION_INDEXES = OrderedDict({'torso_no_vel': OrderedDict({'z_pos': 0, 
                                                'x_heading': 1, 
                                                'y_heading': 2, 
                                                'z_heading': 3, 
                                                'w_heading': 4,
                                                'x_omega': 16,
                                                'y_omega': 17, 
                                                'z_omega': 18}),

                        'torso_xvel': OrderedDict({'x_vel': 13}),
                        'torso_yvel': OrderedDict({'y_vel': 14}),
                        'torso_zvel': OrderedDict({'z_vel': 15}),
                        
                        'front_left_hip':OrderedDict({'theta': 5, 
                                          'omega': 19}),
                        'front_left_ankle':OrderedDict({'theta': 6,
                                            'omega': 20}),
                        'front_right_hip':OrderedDict({'theta': 7,
                                            'omega': 21}),
                        'front_right_ankle':OrderedDict({'theta': 8, 
                                              'omega': 22}),
                        'back_left_hip':OrderedDict({'theta': 9, 
                                        'omega': 23}),
                        'back_left_ankle':OrderedDict({'theta':10,
                                            'omega':24}), 
                        'back_right_hip':OrderedDict({'theta':11, 
                                            'omega':25}),
                        'back_right_ankle':OrderedDict({'theta':12,
                                            'omega':26})})
#### 
CONFIGS = {'obs': OrderedDict({'4x2':[['front_left_hip', 'front_left_ankle'],
                  ['front_right_hip', 'front_right_ankle'], 
                  ['back_left_hip', 'back_left_ankle'],
                  ['back_right_hip', 'back_right_ankle']],

                  't4x2':[['torso_no_vel', 'torso_xvel', 'torso_yvel', 'torso_zvel', 'front_left_hip', 'front_left_ankle'],
                        ['front_right_hip', 'front_right_ankle'], 
                        ['back_left_hip', 'back_left_ankle'],
                        ['back_right_hip', 'back_right_ankle']],

                  '2x4':[['front_left_hip', 'front_left_ankle', 'front_right_hip', 'front_right_ankle'], 
                        ['back_left_hip', 'back_left_ankle', 'back_right_hip', 'back_right_ankle']],

                  '2x4d':[['front_left_hip', 'front_left_ankle', 'back_right_hip', 'back_right_ankle'], 
                        ['back_left_hip', 'back_left_ankle', 'front_right_hip', 'front_right_ankle']], 

                  # single-agent config
                  'sa': [['front_left_hip', 'front_left_ankle', 'front_right_hip', 'front_right_ankle', 
                        'back_left_hip', 'back_left_ankle', 'back_right_hip', 'back_right_ankle']]}),
                        
            'act': OrderedDict({'4x2':[['front_left_hip', 'front_left_ankle'],
                  ['front_right_hip', 'front_right_ankle'], 
                  ['back_left_hip', 'back_left_ankle'],
                  ['back_right_hip', 'back_right_ankle']],

                  't4x2':[['torso_no_vel', 'torso_xvel', 'torso_yvel', 'torso_zvel', 'front_left_hip', 'front_left_ankle'],
                        ['front_right_hip', 'front_right_ankle'], 
                        ['back_left_hip', 'back_left_ankle'],
                        ['back_right_hip', 'back_right_ankle']],

                  '2x4':[['front_left_hip', 'front_left_ankle', 'front_right_hip', 'front_right_ankle'], 
                        ['back_left_hip', 'back_left_ankle', 'back_right_hip', 'back_right_ankle']],

                  '2x4d':[['front_left_hip', 'front_left_ankle', 'back_right_hip', 'back_right_ankle'], 
                        ['back_left_hip', 'back_left_ankle', 'front_right_hip', 'front_right_ankle']], 

                  # single-agent config
                  'sa': [['front_left_hip', 'front_left_ankle', 'front_right_hip', 'front_right_ankle', 
                        'back_left_hip', 'back_left_ankle', 'back_right_hip', 'back_right_ankle']]}),}


def color_ant_env(env):

      white = np.array([1,1,1,1])
      red = np.array([1,0,0,1])
      green = np.array([0,1,0,1])
      blue = np.array([0,0,1,1])
      yellow = np.array([1,1,0,1])

      set_color(env, 'torso_geom', white)

      set_color(env, 'aux_1_geom', yellow)
      set_color(env, 'left_leg_geom', yellow)
      set_color(env, 'left_ankle_geom', yellow)

      # set_color(env, 'aux_1_geom', red)
      # set_color(env, 'left_leg_geom', red)
      # set_color(env, 'left_ankle_geom', red)

      set_color(env, 'aux_2_geom', red)
      set_color(env, 'right_leg_geom', red)
      set_color(env, 'right_ankle_geom',red)

      # set_color(env, 'aux_2_geom', green)
      # set_color(env, 'right_leg_geom', green)
      # set_color(env, 'right_ankle_geom', green)

      set_color(env, 'aux_3_geom', green)
      set_color(env, 'back_leg_geom', green)
      set_color(env, 'third_ankle_geom', green)

      # set_color(env, 'aux_3_geom', blue)
      # set_color(env, 'back_leg_geom', blue)
      # set_color(env, 'third_ankle_geom', blue)

      set_color(env, 'aux_4_geom', blue)
      set_color(env, 'rightback_leg_geom', blue)
      set_color(env, 'fourth_ankle_geom', blue)

      # set_color(env, 'aux_4_geom', yellow)
      # set_color(env, 'rightback_leg_geom', yellow)
      # set_color(env, 'fourth_ankle_geom', yellow)


def set_color(env, name, rgba_vec):
      env.sim.model.geom_rgba[np.array(env.sim.model.geom_names) == name] = rgba_vec

      # env.set_state(np.concatenate((np.array([0]*2), [100], np.array([0.01, 0.005, 0.0, 0.0]), np.array([0., 1., 0., -1., 0., -1., 0., 1.]))), np.zeros(env.model.nv))