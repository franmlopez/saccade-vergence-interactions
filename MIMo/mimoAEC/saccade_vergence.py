""" Training script for the reach scenario.

This script allows simple training and testing of RL algorithms in the Reach environment with a command line interface.
A selection of RL algorithms from the Stable Baselines3 library can be selected.
Interactive rendering is disabled during training to speed up computation, but enabled during testing, so the behaviour
of the model can be observed directly.

Trained models are saved automatically into the `models` directory and prefixed with `reach`, i.e. if you name your
model `my_model`, it will be saved as `models/reach_my_model`.

To train a given algorithm for some number of timesteps::

    python reach.py --train_for=200000 --test_for=1000 --algorithm=PPO --save_name=<model_suffix>

To review a trained model::

    python reach.py --test_for=1000 --load_name=<your_model_suffix>

The available algorithms are `PPO`, `SAC`, `TD3`, `DDPG` and `A2C`.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import time
import argparse
import mujoco
import gymnasium as gym
from stable_baselines3.common.env_checker import check_env

#import mimoEnv
import mimoAEC
import mimoAEC.utils as aec_utils
from mimoAEC.envs.saccade_vergence_env import TEST_SCENE

def get_vision(env):
        """ Collects and returns the outputs of the vision system.

        Override this function if you want to make some simple post-processing!

        Returns:
            dict: A dictionary with one entry for each separate image. In the default implementation each eye renders
            one image, so each eye gets one entry.
        """
        vision_obs = env.vision.get_vision_obs()

        # Crop vision images to fine and coarse scales
        vision_left_fine, vision_left_coarse = aec_utils.get_fine_and_coarse(vision_obs['eye_left'])
        vision_right_fine, vision_right_coarse = aec_utils.get_fine_and_coarse(vision_obs['eye_right'])

        vision_obs_dict = {
            'left_fine': vision_left_fine,
            'right_fine': vision_right_fine,
            'left_coarse': vision_left_coarse,
            'right_coarse': vision_right_coarse,
        }

        return vision_obs_dict

def test(env, test_for=1000, model=None, render=True):
    """ Testing function to view the behaviour of a model.

    Args:
        env (gym.Env): The environment on which the model should be tested. This does not have to be the same training
            environment, but action and observation spaces must match.
        test_for (int): The number of timesteps the testing runs in total. This will be broken into multiple episodes
            if necessary.
        model:  The stable baselines model object. If ``None`` we take random actions instead.
    """
    
    obs, _ = env.reset()    

    plt.ion()
    fig, [[ax00,ax01,ax02],[ax10,ax11,ax12]] = plt.subplots(ncols=3, nrows=2, figsize=(24,16))
    for ax in [ax00, ax01, ax02, ax10, ax11, ax12]:
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)

    if render:
        observer_img = env.mujoco_renderer.render(render_mode="rgb_array", camera_name='observer_side')    
        axim00 = ax00.imshow(observer_img)
        observer_img = env.mujoco_renderer.render(render_mode="rgb_array", camera_name='observer')
        axim10 = ax10.imshow(observer_img)
        observer_img = env.mujoco_renderer.render(render_mode="rgb_array", camera_name='eye_left_coarse')
        axim01 = ax01.imshow(observer_img)
        observer_img = env.mujoco_renderer.render(render_mode="rgb_array", camera_name='eye_right_coarse')
        axim11 = ax11.imshow(observer_img)
    img_left = obs['left_coarse']
    img_right = obs['right_coarse']
    img_left_3d = np.reshape(img_left, img_left.shape + (1,))
    img_right_3d = np.reshape(img_right, img_right.shape + (1,))
    img_center_3d = (img_left_3d + img_right_3d) / 2.0
    img_stereo = np.concatenate((img_left_3d,img_center_3d,img_right_3d), axis=2)
    axim02 = ax02.imshow(img_stereo)
    img_left = obs['left_fine']
    img_right = obs['right_fine']
    img_left_3d = np.reshape(img_left, img_left.shape + (1,))
    img_right_3d = np.reshape(img_right, img_right.shape + (1,))
    img_center_3d = (img_left_3d + img_right_3d) / 2.0
    img_stereo = np.concatenate((img_left_3d,img_center_3d,img_right_3d), axis=2)
    axim12 = ax12.imshow(img_stereo)
    
    t_since_reset = 0   # TODO remove

    for idx in range(test_for):

        if model is None:
            action = env.action_space.sample()
        else:
            action, _ = model.predict(obs)
        #TODO remove
        current_vergence = env.vergence_angle
        #best_vergence = aec_utils.dist_to_angle(texture_dist=env.target_dists[env.current_target])
        eyes_target_distance = np.linalg.norm(
             env.data.body("target"+env.current_target).xpos - 
             (env.data.body('left_eye').xpos + env.data.body('right_eye').xpos) / 2
             )
        best_vergence = aec_utils.dist_to_angle(texture_dist=eyes_target_distance)
        action = np.array([best_vergence - current_vergence])
        obs, rew, done, trunc, info = env.step(action)

        if render:
            observer_img = env.mujoco_renderer.render(render_mode="rgb_array", camera_name='observer_side')    
            axim00.set_data(observer_img)
            observer_img = env.mujoco_renderer.render(render_mode="rgb_array", camera_name='observer')
            axim10.set_data(observer_img)
            observer_img = env.mujoco_renderer.render(render_mode="rgb_array", camera_name='eye_left_coarse')
            axim01.set_data(observer_img)
            observer_img = env.mujoco_renderer.render(render_mode="rgb_array", camera_name='eye_right_coarse')
            axim11.set_data(observer_img)
        img_left = obs['left_coarse']
        img_right = obs['right_coarse']
        img_left_3d = np.reshape(img_left, img_left.shape + (1,))
        img_right_3d = np.reshape(img_right, img_right.shape + (1,))
        img_center_3d = (img_left_3d + img_right_3d) / 2.0
        img_stereo = np.concatenate((img_left_3d,img_center_3d,img_right_3d), axis=2)
        axim02.set_data(img_stereo)
        img_left = obs['left_fine']
        img_right = obs['right_fine']
        img_left_3d = np.reshape(img_left, img_left.shape + (1,))
        img_right_3d = np.reshape(img_right, img_right.shape + (1,))
        img_center_3d = (img_left_3d + img_right_3d) / 2.0
        img_stereo = np.concatenate((img_left_3d,img_center_3d,img_right_3d), axis=2)
        axim12.set_data(img_stereo)

        fig.canvas.flush_events()
        time.sleep(0.1)

        # TODO remove!
        t_since_reset += 1
        """
        if t_since_reset % 6 == 0:
            pan, tilt = env.saccade_angles()
            env.saccade(pan, tilt)
        """

        if done or (t_since_reset % 18 == 0):

            obs, _ = env.reset()
            #obs = reset_model(
            #    initial_target="13",
            #    initial_target_dists={'00': 0.75, '01': 0.75, '02': 0.75, '03': 0.75, '04': 0.75, '10': 0.75, '11': 0.75, '12': 0.75, '13': 0.25, '14': 0.75, '20': 0.75, '21': 0.75, '22': 0.75, '23': 0.75, '24': 0.75, '30': 0.75, '31': 0.75, '32': 0.75, '33': 0.75, '34': 0.75, '40': 0.75, '41': 0.75, '42': 0.75, '43': 0.75, '44': 0.75}
            #    )
            
            if render:
                observer_img = env.mujoco_renderer.render(render_mode="rgb_array", camera_name='observer_side')    
                axim00.set_data(observer_img)
                observer_img = env.mujoco_renderer.render(render_mode="rgb_array", camera_name='observer')
                axim10.set_data(observer_img)
                observer_img = env.mujoco_renderer.render(render_mode="rgb_array", camera_name='eye_left_coarse')
                axim01.set_data(observer_img)
                observer_img = env.mujoco_renderer.render(render_mode="rgb_array", camera_name='eye_right_coarse')
                axim11.set_data(observer_img)
            img_left = obs['left_coarse']
            img_right = obs['right_coarse']
            img_left_3d = np.reshape(img_left, img_left.shape + (1,))
            img_right_3d = np.reshape(img_right, img_right.shape + (1,))
            img_center_3d = (img_left_3d + img_right_3d) / 2.0
            img_stereo = np.concatenate((img_left_3d,img_center_3d,img_right_3d), axis=2)
            axim02.set_data(img_stereo)
            img_left = obs['left_fine']
            img_right = obs['right_fine']
            img_left_3d = np.reshape(img_left, img_left.shape + (1,))
            img_right_3d = np.reshape(img_right, img_right.shape + (1,))
            img_center_3d = (img_left_3d + img_right_3d) / 2.0
            img_stereo = np.concatenate((img_left_3d,img_center_3d,img_right_3d), axis=2)
            axim12.set_data(img_stereo)
            fig.canvas.flush_events()
            
            time.sleep(1)
            t_since_reset = 0   #TODO remove

    env.reset()


def main():
    """ CLI for this scenario.

    Command line interface that can train and load models for the reach scenario. Possible parameters are:

    - ``--train_for``: The number of time steps to train. No training takes place if this is 0. Default 0.
    - ``--test_for``: The number of time steps to test. Testing renders the environment to an interactive window, so
      the trained behaviour can be observed. Default 1000.
    - ``--save_every``: The number of time steps between model saves. This can be larger than the total training time,
      in which case we save once when training completes. Default 100000.
    - ``--algorithm``: The algorithm to train. This argument must be provided if you train. Must be one of
      ``PPO, SAC, TD3, DDPG, A2C, HER``.
    - ``--load_name``: The model to load. Note that this only takes suffixes, i.e. an input of `my_model` tries to
      load `models/reach_my_model`.
    - ``--save_name``: The name under which we save. Like above this is a suffix.
    """

    env = gym.make('MIMoSaccadeVergence-v0', model_path=TEST_SCENE)
    _ = env.reset()
    #check_env(env)

    parser = argparse.ArgumentParser()
    parser.add_argument('--train_for', default=0, type=int,
                        help='Total timesteps of training')
    parser.add_argument('--test_for', default=0, type=int,
                        help='Total timesteps of testing of trained policy')               
    parser.add_argument('--save_every', default=100000, type=int,
                        help='Number of timesteps between model saves')
    parser.add_argument('--algorithm', default=None, type=str, 
                        choices=['PPO', 'SAC', 'TD3', 'DDPG', 'A2C', 'HER'],
                        help='RL algorithm from Stable Baselines3')
    parser.add_argument('--load_name', default=False, type=str,
                        help='Name of model to load')
    parser.add_argument('--save_name', default='', type=str,
                        help='Name of model to save')
    parser.add_argument('--render', default=True, type=bool,
                        help='Whether to render cameras from environment')

    args = parser.parse_args()
    algorithm = args.algorithm
    load_name = args.load_name
    save_name = args.save_name
    save_every = args.save_every
    train_for = args.train_for
    test_for = args.test_for
    render = args.render

    if algorithm == 'PPO':
        from stable_baselines3 import PPO as RL
    elif algorithm == 'SAC':
        from stable_baselines3 import SAC as RL
    elif algorithm == 'TD3':
        from stable_baselines3 import TD3 as RL
    elif algorithm == 'DDPG':
        from stable_baselines3 import DDPG as RL
    elif algorithm == 'A2C':
        from stable_baselines3 import A2C as RL

    # load pretrained model or create new one
    if algorithm is None:
        model = None
    elif load_name:
        load_file = "models/vergence_" + load_name
        model = RL.load(load_file, env, gamma=0, buffer_size=1000)
        env.autoencoder.load(load_file + '_ae')
    else:
        model = RL("MlpPolicy", env, tensorboard_log="models/tensorboard_logs/vergence_" + save_name, verbose=1)

    # train model
    counter = 0
    while train_for > 0:
        counter += 1
        train_for_iter = min(train_for, save_every)
        train_for = train_for - train_for_iter
        model.learn(total_timesteps=train_for_iter, reset_num_timesteps=False)
        save_file = "models/vergence_" + save_name + "_" + str(counter)
        model.save(save_file)
        env.autoencoder.save(save_file + '_ae')
    
    if test_for > 0:
        test(env, model=model, test_for=test_for, render=render)


if __name__ == '__main__':
    main()
