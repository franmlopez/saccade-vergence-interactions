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


FLAT_TEXTURES_SCENE = os.path.join(SCENE_DIRECTORY, "flat_target_one_scene.xml")
FLAT_TEXTURES_WITH_BACKGROUND_SCENE = os.path.join(SCENE_DIRECTORY, "flat_target_one_with_background_scene.xml")

TEXTURES = ["texture"+str(idx) for idx in range(25)]

class MIMoFlatTargetsOneEnv(MIMoSaccadeVergenceEnv, utils.EzPickle):
    """ 
    
    """

    def __init__(self,
                 model_path=FLAT_TEXTURES_SCENE,
                 width= DEFAULT_SIZE,
                 height=DEFAULT_SIZE,
                 vision_pixels=64,
                 textures_list=TEXTURES,
                 min_vergence_angle=0,
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

        # Preload target textures
        if self.target_textures_list is not None:
            self.target_textures = {}
            for texture in self.target_textures_list:
                self.target_textures[texture] = self.model.texture(texture).id
            self._target_materials_ids = {'target': self.model.material('target').id}

   
    def reset_model(self,
                    vergence_angle=None, 
                    target_texture=0,
                    target_size=0.2,
                    target_dist=0.5):
        """ Attempt to reset the simulator and sample a new goal.

        Resets the simulation state, samples a new goal and collects an initial set of observations.
        This function calls :meth:`._reset_sim` until it returns `True`. This is useful if your resetting function has
        a randomized component that can end up in an illegal state. In this case this function will try again until a
        valid state is reached.

        Returns:
            dict: The observations after reset.
        """
        self.steps = 0

        # Reset target
        self.swap_target_texture(target='target', texture=self.target_textures_list[target_texture])
        self._reset_target(target_size=target_size, target_dist=target_dist)

        # Reset eyes

        if vergence_angle is None:
            vergence_angle = np.random.uniform(
                low=self.min_vergence_angle, 
                high=self.max_vergence_angle
                )
            
        self._reset_eyes(
            initial_vergence_angle=vergence_angle,
            initial_pan_angle=0,
            initial_tilt_angle=0,
            )

        obs = self._get_obs()
        return obs

    def _reset_target(self, target_size, target_dist):
        """
        Resets the texture position. The initial distance (depth) can be specified (in meters).
        """
        
        # resize target
        self.model.geom("target").size = np.array([target_size, target_size, 0.001])

        # set mocap position using cartesian coordinates
        self.data.mocap_pos[0] = np.array((target_dist, 0, 0))

    
    def swap_target_texture(self, target, texture):
        """ Changes target texture. Valid texture names are in self.target_textures, which links readable
        texture names to their associated texture ids """
        assert texture in self.target_textures, "{} is not a valid texture!".format(texture)
        new_tex_id = self.target_textures[texture]
        self.model.mat_texid[self._target_materials_ids[target]] = new_tex_id