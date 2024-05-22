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

class MIMoSaccadeVergenceEnv(MujocoEnv, utils.EzPickle):
    """ 
    
    """

    def __init__(self,
                 model_path=None,
                 initial_qpos={},
                 frame_skip=1,
                 n_actions=1,
                 render_mode=None,
                 camera_id=None,
                 camera_name=None,
                 width= DEFAULT_SIZE,
                 height=DEFAULT_SIZE,
                 default_camera_config=None,
                 vision_pixels=32,
                 min_vergence_angle=-2,
                 max_vergence_angle=10.0,
                 max_vergence_action=10,
                 min_pan_angle=-27,
                 max_pan_angle=27,
                 min_tilt_angle=-27,
                 max_tilt_angle=27,
                 max_saccade_action=27,
                 extrinsic_reward=False,
                 ):

        utils.EzPickle.__init__(**locals())

        # EXPERIMENT PARAMETERS
        self.frame_skip = frame_skip
        self.n_actions = n_actions
        self.vision_pixels = vision_pixels
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
        for idx in ['left_fine','right_fine','left_coarse','right_coarse']:
            obs_space_dict[idx] = spaces.Box(0, 1, shape=obs[idx].shape, dtype="float32")
        self.observation_space = spaces.Dict(obs_space_dict)

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
        self._vision_setup({
            "eye_left_fine":{"width": self.vision_pixels, "height": self.vision_pixels},
            "eye_right_fine":{"width": self.vision_pixels, "height": self.vision_pixels},
            "eye_left_coarse":{"width": self.vision_pixels, "height": self.vision_pixels},
            "eye_right_coarse":{"width": self.vision_pixels, "height": self.vision_pixels},
        })

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

        self._reset_eyes()

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

    def _get_vision_obs(self):
        """ Collects and returns the outputs of the vision system.

        Override this function if you want to make some simple post-processing!

        Returns:
            dict: A dictionary with one entry for each separate image. In the default implementation each eye renders
            one image, so each eye gets one entry.
        """
        vision_obs = self.vision.get_vision_obs()

        # Crop vision images to fine and coarse scales
        vision_left_fine = aec_utils.to_grayscale(vision_obs['eye_left_fine'])
        vision_right_fine = aec_utils.to_grayscale(vision_obs['eye_right_fine'])
        vision_left_coarse = aec_utils.to_grayscale(vision_obs['eye_left_coarse'])
        vision_right_coarse = aec_utils.to_grayscale(vision_obs['eye_right_coarse'])

        vision_obs_dict = {
            'left_fine': vision_left_fine,
            'right_fine': vision_right_fine,
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

        # perform action and step on environment
        self.do_simulation(action, self.frame_skip)
        self._step_callback()
        obs = self._get_obs()

        info = {
            'score': 0,
            'disparity': 0,
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

    def vergence(self, angle=0):
        angle = np.clip(angle, -self.max_vergence_action, self.max_vergence_action)
        new_vergence_angle = self.vergence_angle + angle
        if (new_vergence_angle < self.min_vergence_angle):
            new_vergence_angle = self.min_vergence_angle
        if (new_vergence_angle > self.max_vergence_angle):
            new_vergence_angle = self.max_vergence_angle
        self.vergence_angle = new_vergence_angle

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
        return 0