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

DEFAULT_SIZE = 500
""" Default window size for gym rendering functions.

:meta hide-value:
"""

SCENE_DIRECTORY = os.path.abspath(os.path.join(__file__, "..", "..", "assets"))
""" Path to the scene directory.

:meta hide-value:
"""

FLAT_TEXTURES_SCENE = os.path.join(SCENE_DIRECTORY, "flat_textures_scene.xml")

VISION_PARAMS = {
    "eye_left_coarse":{"width": 64, "height": 64},
    "eye_right_coarse":{"width": 64, "height": 64},
}
""" Default vision parameters.

:meta hide-value:
"""

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


class MIMoSaccadeVergenceEnv(MujocoEnv, utils.EzPickle):
    """ 
    
    """

    def __init__(self,
                 model_path=FLAT_TEXTURES_SCENE,
                 initial_qpos={},
                 frame_skip=1,
                 n_actions=1,
                 render_mode=None,
                 camera_id=None,
                 camera_name=None,
                 width= DEFAULT_SIZE,
                 height=DEFAULT_SIZE,
                 default_camera_config=None,
                 vision_params=VISION_PARAMS,
                 textures_list=TEXTURES,
                 target_ids = TARGET_IDS,
                 target_materials=TARGET_MATERIALS,
                 target_positions=TARGET_POSITIONS,
                 target_possible_inits=TARGET_POSSIBLE_INITS,
                 target_possible_saccades=TARGET_POSSIBLE_SACCADES,
                 target_position_scale=8,   # from position to degrees 
                 target_size_min=4,         # target sizes in degrees of field of view
                 target_size_max=6,         # 
                 min_target_dist=0.4,
                 max_target_dist=1.0,
                 min_vergence_angle=0.5,
                 max_vergence_angle=5,
                 max_vergence_action=2,
                 min_pan_angle=-16,
                 max_pan_angle=16,
                 min_tilt_angle=-16,
                 max_tilt_angle=16,
                 max_saccade_action=8,
                 extrinsic_reward=False,
                 ):

        utils.EzPickle.__init__(**locals())

        # EXPERIMENT PARAMETERS
        self.frame_skip = frame_skip
        self.n_actions = n_actions
        self.vision_params = vision_params
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
        self.min_vergence_angle = min_vergence_angle
        self.max_vergence_angle = max_vergence_angle
        self.max_vergence_action = max_vergence_action
        self.min_pan_angle = min_pan_angle
        self.max_pan_angle = max_pan_angle
        self.min_tilt_angle = min_tilt_angle
        self.max_tilt_angle = max_tilt_angle
        self.max_saccade_action = max_saccade_action
        self.extrinsic_reward = extrinsic_reward
        
        self.vision = None

        fullpath = os.path.abspath(model_path)
        if not os.path.exists(fullpath):
            raise IOError("File {} does not exist".format(fullpath))

        self.viewer = None
        self._viewers = {}

        # Load XML and initialize everything
        super().__init__(model_path,
                         frame_skip,
                         observation_space=None,
                         render_mode=render_mode,
                         width=width,
                         height=height,
                         camera_id=camera_id,
                         camera_name=camera_name,
                         default_camera_config=default_camera_config)

        self._env_setup(initial_qpos=initial_qpos)

        # Action space
        self.action_space = spaces.Box(
            np.array([-max_vergence_action, -max_saccade_action, -max_saccade_action]),
            np.array([max_vergence_action, max_saccade_action, max_saccade_action])
            )
        
        # Observation space
        obs = self._get_obs()
        obs_space_dict = {}
        for idx in ['left_coarse','right_coarse']:
            obs_space_dict[idx] = spaces.Box(0, 1, shape=obs[idx].shape, dtype="float32")
        self.observation_space = spaces.Dict(obs_space_dict)

        # Preload target textures
        if self.target_textures_list is not None:
            self.target_textures = {}
            for texture in self.target_textures_list:
                self.target_textures[texture] = self.model.texture(texture).id
            self._target_materials_ids = {}
            for target in self.target_materials:
                self._target_materials_ids[target] = self.model.material(self.target_materials[target]).id


    def _initialize_simulation(self,):
        super()._initialize_simulation()

        fps = int(np.round(1 / self.dt))
        self.metadata = {
            "render_modes": [
                "human",
                "rgb_array",
                "depth_array",
            ],
            "render_fps": fps,
        }

    def _env_setup(self, initial_qpos):
        """ This function initializes all the sensory components of the model.

        Calls the setup functions for all the sensory components and sets the initial positions of the joints according
        to the qpos dictionary.

        Args:
            initial_qpos (dict): A dictionary with the intial joint position for each joint. Keys are the joint names.

        """
        # Our init goes here. At this stage the mujoco model is already loaded, but most of the gym attributes, such as
        # observation space and goals are not set yet

        # Vision setup
        if self.vision_params is not None:
            self._vision_setup(self.vision_params)

        # Initialize experiment parameters
        self.steps = 0
        self.vergence_angle = 0
        self.pan_angle = 0
        self.tilt_angle = 0
        self.cyclo_angle = 0
        self.current_target = None

    def _vision_setup(self, vision_params):
        """ Perform the setup and initialization of the vision system.

        This should be overridden if you want to use another implementation!

        Args:
            vision_params (dict): The parameter dictionary.
        """
        self.vision = SimpleVision(self, vision_params)

    def _reset_simulation(self):
        """ Resets MuJoCo and actuation simulation data and samples a new goal."""
        super()._reset_simulation()
        # Gym mujoco renderer breaks when MjModel and MjData are reset, so re-initialize here.
        default_camera_config = self.mujoco_renderer.default_cam_config
        self.mujoco_renderer.close()
        self.mujoco_renderer = MujocoRenderer(
            self.model, self.data, default_camera_config
        )
    
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

    def _reset_eyes(self,
                   initial_vergence_angle=0,
                   initial_pan_angle=0,
                   initial_tilt_angle=0,):
        """
        Resets the eye positions when resetting the experiment. Initial angles can be specified (in degrees).
        """
        self.vergence_angle = initial_vergence_angle
        self.pan_angle = initial_pan_angle
        self.tilt_angle = initial_tilt_angle

        qpos = self.data.qpos.copy()
        qvel = self.data.qvel.copy()

        qpos[16] = -(self.vergence_angle - self.pan_angle) * (np.pi/180)   # left eye - horizontal
        qpos[17] =  self.tilt_angle * (np.pi/180)  # left eye - vertical
        qpos[19] = -(self.vergence_angle + self.pan_angle) * (np.pi/180)   # right eye - horizontal
        qpos[20] =  self.tilt_angle * (np.pi/180)   # right eye - vertical
        
        self.set_state(qpos, np.zeros(qvel.shape))

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

    def _get_vision_obs(self):
        """ Collects and returns the outputs of the vision system.

        Override this function if you want to make some simple post-processing!

        Returns:
            dict: A dictionary with one entry for each separate image. In the default implementation each eye renders
            one image, so each eye gets one entry.
        """
        vision_obs = self.vision.get_vision_obs()

        # Crop vision images to fine and coarse scales
        vision_left_coarse = aec_utils.to_grayscale(vision_obs['eye_left_coarse'])
        vision_right_coarse = aec_utils.to_grayscale(vision_obs['eye_right_coarse'])

        vision_obs_dict = {
            'left_coarse': vision_left_coarse,
            'right_coarse': vision_right_coarse,
        }

        return vision_obs_dict

    def _get_obs(self):
        """Returns the observation.

        This function should return all simulation outputs relevant to whatever learning algorithm you wish to use. We
        always return proprioceptive information in the 'observation' entry, and this information always includes
        relative joint positions. Other sensory modalities get their own entries, if they are enabled. If
        :attr:`.goals_in_observation` is set to `True`, the achieved and desired goal are also included.

        Returns:
            dict: A dictionary containing simulation outputs with separate entries for each sensor modality.
        """
        observation = self._get_vision_obs()
        return observation

    def _single_mujoco_step(self):
        mujoco.mj_step(self.model, self.data)
        mujoco.mj_rnePostConstraint(self.model, self.data)

    def do_simulation(self, action, n_frames):
        """ Step simulation forward for `n_frames` number of steps.

        Args:
            action (np.ndarray): The control input for the actuators.
            n_frames (int): The number of physics steps to perform.
        """
        self._set_action(action)

        #DON'T STEP IN MUJOCO SIMULATION
        for _ in range(n_frames):
            pass
            #self._single_mujoco_step()


    def step(self, action):
        """ The step function for the simulation.

        This function takes a simulation step, collects the observations, computes the reward and finally determines if
        we are done with this episode or not. :meth:`._get_obs` collects the observations, :meth:`.compute_reward`
        calculates the reward.`:meth:`._is_done` is called to determine if we are done with the episode and
        :meth:`._step_callback` can be used for extra functions each step, such as incrementing a step counter.

        Args:
            action (numpy.ndarray): A numpy array with the control inputs for this step. The shape must match the action
                space!

        Returns:
            A tuple `(observations, reward, done, info)` as described above, with info containing extra information,
            such as whether we reached a success state specifically.
        """

        # compute score of action selected
        best_action = aec_utils.dist_to_angle(texture_dist=self.target_dists[self.current_target])
        score = 1 - np.abs(best_action - action[0]) / (np.abs(best_action) + np.abs(action[0]) + 1e-6)

        # perform action and step on environment
        self.do_simulation(action, self.frame_skip)
        self._step_callback()
        obs = self._get_obs()

        info = {
            'score': score,
            'disparity': mean_squared_error(obs['left_coarse'], obs['right_coarse'])
        }

        reward = self.compute_reward(info)

        done = False
        trunc = False
        
        return obs, reward, done, trunc, info

    def _step_callback(self):
        """
        Manually set unused torsional eye movements to 0
        """
        self.steps += 1
        qpos = self.data.qpos.copy()
        qvel = self.data.qvel.copy()
        qpos[16] = -(self.vergence_angle - self.pan_angle) * (np.pi/180)   # left eye - pan
        qpos[19] = -(self.vergence_angle + self.pan_angle) * (np.pi/180)   # right eye - pan
        qpos[17] =  self.tilt_angle * (np.pi/180)  # left eye - tilt
        qpos[20] =  self.tilt_angle * (np.pi/180)  # right eye - tilt
        qpos[18] = 0  # left eye - torsional
        qpos[21] = 0  # right eye - torsional
        self.set_state(qpos, np.zeros(qvel.shape))
    
    def _set_action(self, action):
        self.vergence(action[0])
        self.saccade(pan_angle=action[1], tilt_angle=action[2])

    def saccade_angles(self, new_target=None):
        if new_target is None:
            new_target = np.random.choice(self.target_possible_saccades[self.current_target])
        current_target_position = self.target_positions[self.current_target]
        new_target_position = self.target_positions[new_target]
        pan = (new_target_position[0] - current_target_position[0]) * self.target_position_scale
        tilt = (new_target_position[1] - current_target_position[1]) * self.target_position_scale
        self.current_target = new_target
        return pan, tilt
    
    def saccade(self, pan_angle, tilt_angle):
        # pan
        pan_angle = np.clip(pan_angle, -self.max_saccade_action, self.max_saccade_action)
        new_pan_angle = self.pan_angle + pan_angle
        if (new_pan_angle < self.min_pan_angle):
            new_pan_angle = self.min_pan_angle
        if (new_pan_angle > self.max_pan_angle):
            new_pan_angle = self.max_pan_angle
        self.pan_angle = new_pan_angle
        # tilt
        tilt_angle = np.clip(tilt_angle, -self.max_saccade_action, self.max_saccade_action)
        new_tilt_angle = self.tilt_angle + tilt_angle
        if (new_tilt_angle < self.min_tilt_angle):
            new_tilt_angle = self.min_tilt_angle
        if (new_tilt_angle > self.max_tilt_angle):
            new_tilt_angle = self.max_tilt_angle
        self.tilt_angle = new_tilt_angle

        """
        qpos = self.data.qpos.copy()
        qvel = self.data.qvel.copy()
        qpos[16] = -(self.vergence_angle - self.pan_angle) * (np.pi/180)   # left eye - pan
        qpos[19] = -(self.vergence_angle + self.pan_angle) * (np.pi/180)   # right eye - pan
        qpos[17] =  self.tilt_angle * (np.pi/180)  # left eye - tilt
        qpos[20] =  self.tilt_angle * (np.pi/180)  # right eye - tilt
        
        self.set_state(qpos, np.zeros(qvel.shape))
        """

    def vergence(self, angle=0):
        angle = np.clip(angle, -self.max_vergence_action, self.max_vergence_action)
        new_vergence_angle = self.vergence_angle + angle
        if (new_vergence_angle < self.min_vergence_angle):
            new_vergence_angle = self.min_vergence_angle
        if (new_vergence_angle > self.max_vergence_angle):
            new_vergence_angle = self.max_vergence_angle
        self.vergence_angle = new_vergence_angle
        
        """
        qpos = self.data.qpos.copy()
        qvel = self.data.qvel.copy()
        qpos[16] = -(self.vergence_angle - self.pan_angle) * (np.pi/180)   # left eye - pan
        qpos[19] = -(self.vergence_angle + self.pan_angle) * (np.pi/180)   # right eye - pan
        qpos[17] =  self.tilt_angle * (np.pi/180)  # left eye - tilt
        qpos[20] =  self.tilt_angle * (np.pi/180)  # right eye - tilt
        
        self.set_state(qpos, np.zeros(qvel.shape))
        """

    def compute_reward(self, info):
        """Compute the step reward.

        This externalizes the reward function and makes it dependent on a desired goal and the one that was achieved.
        If you wish to include additional rewards that are independent of the goal, you can include the necessary values
        to derive it in 'info' and compute it accordingly.

        Args:
            info (dict): an info dictionary with additional information

        Returns:
            float: The reward that corresponds to the provided achieved goal w.r.t. to the desired
            goal.
        """
        
        if self.extrinsic_reward:
            reward = -info['disparity']
        else:
            reward = 0
        return reward