import time
import random
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torchsummary import summary
import mujoco
import gymnasium as gym
import argparse
from sklearn.metrics import mean_squared_error

import os, sys
file_path = os.path.abspath(os.path.dirname(__file__))
main_path = os.path.abspath(os.path.join(file_path,'..'))
if main_path != sys.path[0]:
    sys.path.insert(0, main_path)
    os.chdir(main_path)
    del main_path, file_path

import MIMo.mimoAEC
from src.agent import Agent
from src.buffer import AutoencoderBuffer, VergenceBuffer, SaccadeBuffer
from src.utils import img_to_obs, imgs_from_dict, dist_to_angle, obs_to_imgs, join_imgs, \
                        compute_reconstruction_map, compute_saliency_map, \
                        choose_saccade_action, compute_vergence_reward

ENVIRONMENTS = {
    'playroom' : 'MIMoPlayRoom-v0',
    'flat' : 'MIMoFlatTargetsOne-v0',
}

class Embodiment():
    
    def __init__(self,
                 env='MIMoPlayRoom-v0',
                 fov=54,    # for coarse vision
                 pixels=32,
                 n_episodes=1,
                 n_timesteps=15,
                 saccades_every=5,
                 sac_window=5,  # Total size of window around target to compute saccade target (in pixels)
                 sac_vrg_int=True,
                 vrg_reward_method='lopez',
                 checkpoint_name=None, checkpoint_num=None,):
        
        self.env = gym.make(env, 
                            vision_pixels=pixels,
                            min_pan_angle=-fov, 
                            max_pan_angle=fov,  
                            min_tilt_angle=-fov,    
                            max_tilt_angle=fov, 
                            max_saccade_action=fov/2,)
        self.n_episodes = n_episodes
        self.n_timesteps = n_timesteps
        self.saccades_every = saccades_every
        self.ae_buffer = AutoencoderBuffer()
        self.vrg_buffer = VergenceBuffer()
        self.sac_buffer = SaccadeBuffer()
        self.agent = Agent(
            ae_input_size=pixels,
            vrg_input_size=7,
            sac_input_size=7,
            vrg_max_action=3,  # in degrees
            sac_max_action=np.floor((pixels-sac_window)/2), # in pixels: saccade action converted to degrees when passed to MIMo
            )
        self.sac_window = sac_window
        self.angle_to_pixel = pixels / fov
        self.sac_vrg_int = sac_vrg_int
        self.vrg_reward_method = vrg_reward_method
        
        # training metrics
        self.ep_vergence_reward = None
        self.ep_disparity = None

        # load pre-trained agent
        if (checkpoint_name is not None) and (checkpoint_num is not None):
            self.load_agent(checkpoint_name=checkpoint_name, checkpoint_num=checkpoint_num)

        
    def load_agent(self, checkpoint_name, checkpoint_num):
        self.agent.autoencoder.load('results/'+checkpoint_name+'/models/model'+str(checkpoint_num))
        self.agent.vergence.load('results/'+checkpoint_name+'/models/model'+str(checkpoint_num))
        self.agent.saccade.load('results/'+checkpoint_name+'/models/model'+str(checkpoint_num))

    def save_agent(self, checkpoint_name, checkpoint_num):  
        self.agent.autoencoder.save(checkpoint_name+'/models/model'+str(checkpoint_num))
        self.agent.vergence.save(checkpoint_name+'/models/model'+str(checkpoint_num))
        self.agent.saccade.save(checkpoint_name+'/models/model'+str(checkpoint_num))

    def run(self, deterministic=False):
        '''
        Runs simulation in the embodiment. Returns buffer and score.
        '''
        running_vergence_reward = 0

        for episode_idx in range(self.n_episodes):

            # Reset environment and get initial observations
            obs, _ = self.env.reset() # TODO sample random parameters and pass as arguments for new episode
            observation_fine = img_to_obs(obs['left_fine'], obs['right_fine'])
            observation_coarse = img_to_obs(obs['left_coarse'], obs['right_coarse'])

            # Get encodings and reconstruction error
            encoding_fine, reconstruction_fine = self.agent.autoencoder.get_encoded(observation_fine)
            encoding_coarse, reconstruction_coarse = self.agent.autoencoder.get_encoded(observation_coarse)
            reconstruction_map = compute_reconstruction_map(observation_coarse, reconstruction_coarse)
            
            for idx in range(self.n_timesteps):
                
                # Saccade 
                if (idx % self.saccades_every == 0) and (idx+1 < self.n_timesteps):
                    sac_action = choose_saccade_action(reconstruction_map,
                                                       window_size=self.sac_window)
                else:
                    sac_action = np.array([0,0])

                if self.sac_vrg_int:
                    sac_action_for_vrg = sac_action
                else:
                    sac_action_for_vrg = np.array([0,0])

                # Vergence
                vrg_action = self.agent.vergence.choose_action(encoding_fine, encoding_coarse,
                                                               torch.from_numpy(sac_action_for_vrg).float().to(self.agent.device),
                                                               deterministic=deterministic)
                
                # Perform step in environment
                obs, _,_,_, info = self.env.step(np.concatenate([vrg_action, sac_action/self.angle_to_pixel]))
                
                # Get new observations, encodings, and reconstruction error
                new_observation_fine = img_to_obs(obs['left_fine'], obs['right_fine'])
                new_observation_coarse = img_to_obs(obs['left_coarse'], obs['right_coarse'])
                new_encoding_fine, new_reconstruction_fine = self.agent.autoencoder.get_encoded(new_observation_fine)
                new_encoding_coarse, new_reconstruction_coarse = self.agent.autoencoder.get_encoded(new_observation_coarse)
                new_reconstruction_map = compute_reconstruction_map(new_observation_coarse, new_reconstruction_coarse)

                # Check done flag used for training the critic
                done = (idx == self.n_timesteps-1)

                # Store observations in autoencoder buffer 
                self.ae_buffer.store(observation=observation_fine)
                self.ae_buffer.store(observation=observation_coarse)

                # Vergence: compute reward and store in buffer
                vrg_reward, disparity = compute_vergence_reward(
                    observation_fine=observation_fine,
                    reconstruction_fine=reconstruction_fine,
                    observation_coarse=observation_coarse,
                    reconstruction_coarse=reconstruction_coarse,
                    new_observation_fine=new_observation_fine,
                    new_reconstruction_fine=new_reconstruction_fine,
                    new_observation_coarse=new_observation_coarse,
                    new_reconstruction_coarse=new_reconstruction_coarse,
                    method=self.vrg_reward_method,
                )
                self.vrg_buffer.store(
                    encoding_fine=encoding_fine,
                    encoding_coarse=encoding_coarse,
                    sac_action=sac_action_for_vrg,
                    vrg_action=vrg_action,
                    done=done,
                    reward=vrg_reward,
                )

                # Update for next iteration
                if not done:
                    observation_fine = new_observation_fine
                    observation_coarse = new_observation_coarse
                    reconstruction_fine = new_reconstruction_fine
                    reconstruction_coarse = new_reconstruction_coarse
                    encoding_fine = new_encoding_fine
                    encoding_coarse = new_encoding_coarse
                    reconstruction_map = new_reconstruction_map

                running_vergence_reward += vrg_reward

        self.ep_vergence_reward = running_vergence_reward / (self.n_episodes*self.n_timesteps)
        self.ep_disparity = disparity

    def animate(self, timesteps=1000, reset_every=20, deterministic=True):
        """
        Animates simulation in embodied environment.
        """

        plt.ion()
        fig, [[ax00,ax01,ax02],[ax10,ax11,ax12]] = plt.subplots(ncols=3, nrows=2, figsize=(16,16))
        for ax in [ax00, ax01, ax02, ax10, ax11, ax12]:
            ax.axes.xaxis.set_visible(False)
            ax.axes.yaxis.set_visible(False)
            
        ax00.set_title("Side view")
        ax10.set_title("MIMo")
        ax01.set_title("Fine scale")
        ax11.set_title("Coarse scale")
        ax02.set_title("Reconstruction")
        ax12.set_title("Saliency map")

        obs, _ = self.env.reset()
        
        observation_fine = img_to_obs(obs['left_fine'], obs['right_fine'])
        encoding_fine, _ = self.agent.autoencoder.get_encoded(observation_fine)

        observation_coarse = img_to_obs(obs['left_coarse'], obs['right_coarse'])
        encoding_coarse, reconstruction_coarse = self.agent.autoencoder.get_encoded(observation_coarse)
        reconstruction_map = compute_reconstruction_map(observation_coarse, reconstruction_coarse)
        saliency_map = compute_saliency_map(reconstruction_map, self.sac_window)
        
        observer_img = self.env.mujoco_renderer.render(render_mode="rgb_array", camera_name='observer_side')    
        axim00 = ax00.imshow(observer_img)
        observer_img = self.env.mujoco_renderer.render(render_mode="rgb_array", camera_name='face')
        axim10 = ax10.imshow(observer_img)
        
        img_left, img_right = obs_to_imgs(observation_fine)
        img_left_3d = np.reshape(img_left, img_left.shape + (1,))
        img_right_3d = np.reshape(img_right, img_right.shape + (1,))
        img_center_3d = (img_left_3d + img_right_3d) / 2.0
        img_stereo = np.concatenate((img_left_3d,img_center_3d,img_right_3d), axis=2)
        axim01 = ax01.imshow(img_stereo)

        img_left, img_right = obs_to_imgs(observation_coarse)
        img_left_3d = np.reshape(img_left, img_left.shape + (1,))
        img_right_3d = np.reshape(img_right, img_right.shape + (1,))
        img_center_3d = (img_left_3d + img_right_3d) / 2.0
        img_stereo = np.concatenate((img_left_3d,img_center_3d,img_right_3d), axis=2)
        axim11 = ax11.imshow(img_stereo)

        axim02 = ax02.imshow(reconstruction_map)
        axim12 = ax12.imshow(saliency_map)

        fig.tight_layout()
        
        time.sleep(0.5)
        sac_action = np.array([0,0])

        for timestep in range(timesteps):

            # Vergence
            vrg_action = self.agent.vergence.choose_action(encoding_fine, encoding_coarse,
                                                           torch.from_numpy(sac_action).float().to(self.agent.device),
                                                           deterministic=deterministic)
            
            # Perform step in environment
            obs, _,_,_, info = self.env.step(np.concatenate([vrg_action, sac_action/self.angle_to_pixel]))
            
            observation_fine = img_to_obs(obs['left_fine'], obs['right_fine'])
            encoding_fine, _ = self.agent.autoencoder.get_encoded(observation_fine)

            observation_coarse = img_to_obs(obs['left_coarse'], obs['right_coarse'])
            encoding_coarse, reconstruction_coarse = self.agent.autoencoder.get_encoded(observation_coarse)
            reconstruction_map = compute_reconstruction_map(observation_coarse, reconstruction_coarse)
            saliency_map = compute_saliency_map(reconstruction_map, self.sac_window)
        
            observer_img = self.env.mujoco_renderer.render(render_mode="rgb_array", camera_name='observer_side')    
            axim00.set_data(observer_img)
            observer_img = self.env.mujoco_renderer.render(render_mode="rgb_array", camera_name='face')
            axim10.set_data(observer_img)
            
            img_left, img_right = obs_to_imgs(observation_fine)
            img_left_3d = np.reshape(img_left, img_left.shape + (1,))
            img_right_3d = np.reshape(img_right, img_right.shape + (1,))
            img_center_3d = (img_left_3d + img_right_3d) / 2.0
            img_stereo = np.concatenate((img_left_3d,img_center_3d,img_right_3d), axis=2)
            axim01.set_data(img_stereo)

            img_left, img_right = obs_to_imgs(observation_coarse)
            img_left_3d = np.reshape(img_left, img_left.shape + (1,))
            img_right_3d = np.reshape(img_right, img_right.shape + (1,))
            img_center_3d = (img_left_3d + img_right_3d) / 2.0
            img_stereo = np.concatenate((img_left_3d,img_center_3d,img_right_3d), axis=2)
            axim11.set_data(img_stereo)

            axim02.set_data(reconstruction_map)
            axim12.set_data(saliency_map)

            if (timestep % self.saccades_every == 0) and ((timestep+1) % reset_every != 0):
                sac_action = choose_saccade_action(reconstruction_map, window_size=self.sac_window,
                                                   deterministic=True)
                marker = ax12.scatter(16 - sac_action[0], 16 - sac_action[1], marker='X', color='r', s=100)
                fig.canvas.flush_events()
                time.sleep(0.5)
            else:
                sac_action = np.array([0,0])
                try:
                    marker.remove()
                except:
                    pass
                fig.canvas.flush_events()
                time.sleep(0.1)

            if (timestep+1) % reset_every == 0:
                obs, _ = self.env.reset()
                
                observation_fine = img_to_obs(obs['left_fine'], obs['right_fine'])
                encoding_fine, _ = self.agent.autoencoder.get_encoded(observation_fine)

                observation_coarse = img_to_obs(obs['left_coarse'], obs['right_coarse'])
                encoding_coarse, reconstruction_coarse = self.agent.autoencoder.get_encoded(observation_coarse)
                reconstruction_map = compute_reconstruction_map(observation_coarse, reconstruction_coarse)
                saliency_map = compute_saliency_map(reconstruction_map, self.sac_window)

                observer_img = self.env.mujoco_renderer.render(render_mode="rgb_array", camera_name='observer_side')    
                axim00.set_data(observer_img)
                observer_img = self.env.mujoco_renderer.render(render_mode="rgb_array", camera_name='face')
                axim10.set_data(observer_img)

                img_left, img_right = obs_to_imgs(observation_fine)
                img_left_3d = np.reshape(img_left, img_left.shape + (1,))
                img_right_3d = np.reshape(img_right, img_right.shape + (1,))
                img_center_3d = (img_left_3d + img_right_3d) / 2.0
                img_stereo = np.concatenate((img_left_3d,img_center_3d,img_right_3d), axis=2)
                axim01.set_data(img_stereo)

                img_left, img_right = obs_to_imgs(observation_coarse)
                img_left_3d = np.reshape(img_left, img_left.shape + (1,))
                img_right_3d = np.reshape(img_right, img_right.shape + (1,))
                img_center_3d = (img_left_3d + img_right_3d) / 2.0
                img_stereo = np.concatenate((img_left_3d,img_center_3d,img_right_3d), axis=2)
                axim11.set_data(img_stereo)

                axim02.set_data(reconstruction_map)
                axim12.set_data(saliency_map)

                fig.canvas.flush_events()
                time.sleep(0.5)


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--runtype', default='run', type=str,
        choices=['run','animate','evaluate'])
    parser.add_argument('--env', default='playroom', type=str,
        choices=['playroom', 'flat',])
    parser.add_argument('--checkpoint_num', default=None, type=int)
    parser.add_argument('--checkpoint_name', default=None, type=str)
    
    args = parser.parse_args()

    embodiment = Embodiment(checkpoint_name=args.checkpoint_name, checkpoint_num=args.checkpoint_num,
                            env=ENVIRONMENTS[args.env])

    if args.runtype == 'run':
        embodiment.run()
        print(f"Embodiment run successful. Vergen reward for this episode: {embodiment.ep_vergence_reward}")

    elif args.runtype == 'animate':
        embodiment.animate()