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
from scipy.spatial.transform import Rotation

from gymnasium import spaces, utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer

from mimoVision.vision import SimpleVision
import mimoEnv.utils as mimo_utils
import mimoAEC.utils as aec_utils
from mimoAEC.envs.saccade_vergence_env import MIMoSaccadeVergenceEnv, DEFAULT_SIZE, SCENE_DIRECTORY

PLAY_ROOM_SCENE = os.path.join(SCENE_DIRECTORY, "play-room/play_room_scene.xml")

VALID_POSITIONS = {
    'x_min': -1.4,
    'x_max': 1.0,
    'y_min': -1.8,
    'y_max': 1.3,
}

class MIMoPlayRoomEnv(MIMoSaccadeVergenceEnv, utils.EzPickle):
    """ MIMo reaches for an object.

        Class to demonstrate saccades with binocular saliency map by Marcel Raabe
    """
    def __init__(self,
                 valid_positions=VALID_POSITIONS,
                 model_path=PLAY_ROOM_SCENE,
                 width= DEFAULT_SIZE,
                 height=DEFAULT_SIZE,
                 vision_pixels=64,
                 min_vergence_angle=-2,
                 max_vergence_angle=10.0,
                 max_vergence_action=10,
                 min_pan_angle=-24,
                 max_pan_angle=24,
                 min_tilt_angle=-24,
                 max_tilt_angle=24,
                 max_saccade_action=12,
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

        self.valid_positions = valid_positions


    def reset_model(self, initial_vergence_angle=None,
                    initial_position=None, initial_rotation=None, initial_head_tilt=None,):
        """
        Resets MIMo in the room. The initial position, orientation, and head
        tilt of MIMo, as well as his vergence, are random.
        """
        self.steps = 0

        # Initial position
        if initial_position is None:
            initial_position = np.array([
                np.random.uniform(low=self.valid_positions['x_min'], high=self.valid_positions['x_max']),
                np.random.uniform(low=self.valid_positions['y_min'], high=self.valid_positions['y_max']),
            ])

        # Initial rotation
        if initial_rotation is None:
            initial_rotation = np.random.uniform(low=0, high=2*np.pi)
        

        # Initial head tilt
        if initial_head_tilt is None:
            # Randomly sample tilt angle with laplace destribution
            # Set max_tilt for cut of tilt angle, currently 30°
            # laplace: mean=-3.9°, sigma=4.33° => scale = sigma / sqrt(2)
            # For numbers see: Kretch et.al.(2014), Crawling and Walking Infants See the World Differently
            max_tilt=0.5235987755982988
            angle=np.inf
            while(np.absolute(angle) > max_tilt):
                angle = np.random.laplace(loc = -0.06863941064129744, scale=0.05343800867284924)
            initial_head_tilt = angle

        self._reset_body(
            initial_position=initial_position,
            initial_rotation=initial_rotation,
            initial_head_tilt=initial_head_tilt,
        )

        # Initial vergence
        if initial_vergence_angle is None:
            initial_vergence_angle = np.random.uniform(low=self.min_vergence_angle, high=self.max_vergence_angle)

        self._reset_eyes(
            initial_vergence_angle=initial_vergence_angle,
            initial_pan_angle=0,
            initial_tilt_angle=0,
            )
        
        obs = self._get_obs()
        return obs

    def _reset_body(self, initial_position, initial_rotation, initial_head_tilt, include_target=True):

        qpos = self.data.qpos.copy()
        qvel = self.data.qvel.copy()
        
        # Position
        qpos[0] = initial_position[0]
        qpos[1] = initial_position[1]

        # Rotation
        rot = Rotation.from_euler('xyz', [initial_rotation, 0, 0])
        rot_quat = rot.as_quat()
        qpos[3:7] = rot_quat

        # Head tilt
        qpos[14] = -initial_head_tilt

        self.set_state(qpos, np.zeros(qvel.shape))