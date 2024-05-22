""" This module defines the base environment for AEC experiments with MIMo.

The abstract base class is :class:`~mimoAEC.envs.aec.MIMoAECEnv`.
"""
import os
import numpy as np
import mujoco
from mujoco import MjData, MjModel
import copy
from typing import Dict, Type
from sklearn.metrics import mean_squared_error

from gymnasium import spaces, utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer

from mimoVision.vision import SimpleVision
import mimoEnv.utils as mimo_utils
import mimoAEC.utils as aec_utils
from mimoAEC.envs.saccade_vergence_env import MIMoSaccadeVergenceEnv, DEFAULT_SIZE, SCENE_DIRECTORY


FLAT_TEXTURES_SCENE = os.path.join(SCENE_DIRECTORY, "flat_textures_scene.xml")

TEXTURES = ["texture"+str(idx) for idx in range(25)]    # TODO: replace with 300 textures

TARGET_IDS = [
    "03",
    "12",
    "13",
    "14",
    "21",
    "22",
    "23",
    "24",
    "25",
    "30",
    "31",
    "32",
    "33",
    "34",
    "35",
    "36",
    "41",
    "42",
    "43",
    "44",
    "45",
    "52",
    "53",
    "54",
    "63",
]

TARGET_MATERIALS = {
    "03":"target03",
    "12":"target12",
    "13":"target13",
    "14":"target14",
    "21":"target21",
    "22":"target22",
    "23":"target23",
    "24":"target24",
    "25":"target25",
    "30":"target30",
    "31":"target31",
    "32":"target32",
    "33":"target33",
    "34":"target34",
    "35":"target35",
    "36":"target36",
    "41":"target41",
    "42":"target42",
    "43":"target43",
    "44":"target44",
    "45":"target45",
    "52":"target52",
    "53":"target53",
    "54":"target54",
    "63":"target63"
}

TARGET_POSITIONS = {
    "03" : (-3,  0),
    "12" : (-2, -1),
    "13" : (-2,  0),
    "14" : (-2,  1),
    "21" : (-1, -2),
    "22" : (-1, -1),
    "23" : (-1,  0),
    "24" : (-1,  1),
    "25" : (-1,  2),
    "30" : ( 0, -3),
    "31" : ( 0, -2),
    "32" : ( 0, -1),
    "33" : ( 0,  0),
    "34" : ( 0,  1),
    "35" : ( 0,  2),
    "36" : ( 0,  3),
    "41" : ( 1, -2),
    "42" : ( 1, -1),
    "43" : ( 1,  0),
    "44" : ( 1,  1),
    "45" : ( 1,  2),
    "52" : ( 2, -1),
    "53" : ( 2,  0),
    "54" : ( 2,  1),
    "63" : ( 3,  0),
}

TARGET_POSSIBLE_INITS = TARGET_IDS

TARGET_POSSIBLE_SACCADES = {
    "03" : ["03", "12", "13", "14"],
    "12" : ["03", "12", "13", "21", "22", "23"], 
    "13" : ["03", "12", "13", "14", "22", "23", "24"],
    "14" : ["03", "13", "14", "23", "24", "25"],
    "21" : ["12", "21", "22", "30", "31", "32"],
    "22" : ["12", "13", "21", "22", "23", "31", "32", "33"],
    "23" : ["12", "13", "14", "22", "23", "24", "32", "33", "34"],
    "24" : ["13", "14", "23", "24", "25", "33", "34", "35"],
    "25" : ["14", "24", "25", "34", "35", "36"],
    "30" : ["21", "30", "31", "41"],
    "31" : ["21", "22", "30", "31", "32", "41", "42"],
    "32" : ["21", "22", "24", "31", "32", "33", "41", "42", "43"],
    "33" : ["22", "23", "24", "32", "33", "34", "42", "43", "44"],
    "34" : ["23", "24", "25", "33", "34", "35", "43", "44", "45"],
    "35" : ["24", "25", "34", "35", "36", "44", "45"],
    "36" : ["25", "35", "36", "45"],
    "41" : ["30", "31", "32", "41", "42", "52"],
    "42" : ["31", "32", "33", "41", "42", "43", "52", "53"],
    "43" : ["32", "33", "34", "42", "43", "44", "52", "53", "54"],
    "44" : ["33", "34", "35", "43", "44", "45", "53", "54"],
    "45" : ["30", "31", "40", "41"],
    "52" : ["41", "42", "43", "52", "53", "63"],
    "53" : ["42", "43", "44", "52", "53", "54", "63"],
    "54" : ["43", "44", "45", "53", "54", "63"],
    "63" : ["52", "53", "54", "63"],
}


class MIMoFlatTargetsEnv(MIMoSaccadeVergenceEnv, utils.EzPickle):
    """ 
    
    """

    def __init__(self,
                 model_path=FLAT_TEXTURES_SCENE,
                 width= DEFAULT_SIZE,
                 height=DEFAULT_SIZE,
                 vision_pixels=64,
                 textures_list=TEXTURES,
                 target_ids = TARGET_IDS,
                 target_materials=TARGET_MATERIALS,
                 target_positions=TARGET_POSITIONS,
                 target_possible_inits=TARGET_POSSIBLE_INITS,
                 target_possible_saccades=TARGET_POSSIBLE_SACCADES,
                 target_position_scale=30,   # from position to degrees 
                 target_size_min=25,         # target sizes in degrees of field of view
                 target_size_max=25,          
                 min_target_dist=0.5,
                 max_target_dist=1.0,
                 min_vergence_angle=-2,
                 max_vergence_angle=10,
                 max_vergence_action=10,
                 min_pan_angle=-24,
                 max_pan_angle=24,
                 min_tilt_angle=-24,
                 max_tilt_angle=24,
                 max_saccade_action=24,
                 extrinsic_reward=False,
                 ):

        super().__init__(
            model_path=model_path,
            width=width,
            height=height,
            vision_pixels=vision_pixels,
            min_vergence_angle=min_vergence_angle,
            max_vergence_angle=max_vergence_angle,
            max_vergence_action=max_vergence_action,
            min_pan_angle=min_pan_angle,
            max_pan_angle=max_pan_angle,
            min_tilt_angle=min_tilt_angle,
            max_tilt_angle=max_tilt_angle,
            max_saccade_action=max_saccade_action,
            extrinsic_reward=extrinsic_reward,
        )

        # EXPERIMENT PARAMETERS
        self.target_textures_list = textures_list
        self.target_materials = target_materials
        self.target_ids = target_ids
        self.target_positions = target_positions
        self.target_possible_inits = target_possible_inits
        self.target_possible_saccades = target_possible_saccades
        self.target_position_scale = target_position_scale
        self.target_size_min = target_size_min
        self.target_size_max = target_size_max
        self.min_target_dist = min_target_dist
        self.max_target_dist = max_target_dist        

        # Preload target textures
        if self.target_textures_list is not None:
            self.target_textures = {}
            for texture in self.target_textures_list:
                self.target_textures[texture] = self.model.texture(texture).id
            self._target_materials_ids = {}
            for target in self.target_materials:
                self._target_materials_ids[target] = self.model.material(self.target_materials[target]).id

   
    def reset_model(self, initial_vergence_angle=None, initial_target=None,
              initial_target_textures=None, initial_target_dists=None,):
        """ Attempt to reset the simulator and sample a new goal.

        Resets the simulation state, samples a new goal and collects an initial set of observations.
        This function calls :meth:`._reset_sim` until it returns `True`. This is useful if your resetting function has
        a randomized component that can end up in an illegal state. In this case this function will try again until a
        valid state is reached.

        Returns:
            dict: The observations after reset.
        """
        self.steps = 0

        # Reset target textures

        if initial_target_textures is None:
            initial_target_textures = np.random.choice(
                self.target_textures_list,
                size=len(self.target_ids),
                replace=False,
                )
            self.current_target_textures = dict(zip(self.target_ids, initial_target_textures))
        else:
            self.current_target_textures = initial_target_textures
            
        if initial_target_dists is None:
            initial_target_dists = np.random.uniform(
                low=self.min_target_dist,
                high=self.max_target_dist,
                size=len(self.target_ids),
                )
            self.target_dists = dict(zip(self.target_ids, initial_target_dists))
        else:
            self.target_dists = initial_target_dists
        
        for target_num, target in enumerate(self.target_ids):
            self.swap_target_texture(target=target, texture=self.current_target_textures[target])
            self._reset_target(target_num=target_num, target_id=target, target_dist=self.target_dists[target])

        # Reset eyes

        if initial_vergence_angle is None:
            initial_vergence_angle = np.random.uniform(low=self.min_vergence_angle, high=self.max_vergence_angle)

        if initial_target is not None:
            self.current_target = initial_target
        else:
            self.current_target = np.random.choice(self.target_possible_inits)
        
        initial_target_position = self.target_positions[self.current_target]
        initial_pan_angle = initial_target_position[0] * self.target_position_scale
        initial_tilt_angle = initial_target_position[1] * self.target_position_scale

        self._reset_eyes(
            initial_vergence_angle=initial_vergence_angle,
            initial_pan_angle=initial_pan_angle,
            initial_tilt_angle=initial_tilt_angle
            )

        obs = self._get_obs()
        return obs

    def _reset_target(self, target_num, target_id, target_dist=1):
        """
        Resets the texture position. The initial distance (depth) can be specified (in meters).
        """
        target_angle_positions = self.target_positions[target_id]
        target_pan = -target_angle_positions[0] * self.target_position_scale
        target_tilt = target_angle_positions[1] * self.target_position_scale

        # resize mocap image
        size_in_degrees = np.random.uniform(low=self.target_size_min, high=self.target_size_max)
        size = target_dist * np.tan(size_in_degrees/2 * np.pi/180)
        self.model.geom("target"+target_id).size = np.array([size, size, 0.001])

        # set mocap position using cartesian coordinates
        target_pos_cart = aec_utils.spher_to_cart((target_dist, target_tilt, target_pan))
        self.data.mocap_pos[target_num] = target_pos_cart

        # set mocap orientation using spherical coordinates
        polar = -(target_tilt) * np.pi/180
        azim = -(target_pan) * np.pi/180
        quat_azim = np.array([np.cos(azim/2), 0, 0, np.sin(azim/2)])
        quat_polar = np.array([np.cos(polar/2), np.sin(polar/2)*np.sin(azim/2), np.sin(polar/2)*np.cos(azim/2), 0])
        quat = aec_utils.quaternion_multiply(quat_polar, quat_azim)
        #self.data.mocap_quat[target_num] = quat
    
    def swap_target_texture(self, target, texture):
        """ Changes target texture. Valid texture names are in self.target_textures, which links readable
        texture names to their associated texture ids """
        assert texture in self.target_textures, "{} is not a valid texture!".format(texture)
        new_tex_id = self.target_textures[texture]
        self.model.mat_texid[self._target_materials_ids[target]] = new_tex_id

    def saccade_angles(self, new_target=None):  # NOT USED
        if new_target is None:
            new_target = np.random.choice(self.target_possible_saccades[self.current_target])
        current_target_position = self.target_positions[self.current_target]
        new_target_position = self.target_positions[new_target]
        pan = (new_target_position[0] - current_target_position[0]) * self.target_position_scale
        tilt = (new_target_position[1] - current_target_position[1]) * self.target_position_scale
        self.current_target = new_target
        return pan, tilt