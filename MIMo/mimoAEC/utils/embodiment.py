import time
import random
import torch
import numpy as np
from sklearn.metrics import mean_squared_error
from multiprocessing import Pool

import os, sys
file_path = os.path.abspath(os.path.dirname(__file__))
main_path = os.path.abspath(os.path.join(file_path,'..'))
if main_path != sys.path[0]:
    sys.path.insert(0, main_path)
    os.chdir(main_path)
    del main_path, file_path

from utils.agent import Agent
from utils.environment import Environment
from utils.buffer import Buffer
from utils.auxiliary import img_to_obs, dist_to_angle, get_device
from src.config import max_vergence_action, \
                       vergence, min_initial_vergence_angle, max_initial_vergence_angle, \
                       cyclo, min_initial_cyclo_angle, max_initial_cyclo_angle, \
                       version, min_initial_version_angle, max_initial_version_angle, max_version_action, \
                       min_texture_dist, max_texture_dist, \
                       reward_c 

def run_embodiment(name, textures, autoencoder_params, actor_critic_params, epsilon, n_episodes):
    '''
    Function passed as an argument for multiprocessing. Creates and runs one embodiment, returns buffer and score.
    '''
    embodiment = Embodiment(name, textures, autoencoder_params, actor_critic_params, epsilon=epsilon,n_episodes=n_episodes)
    buffer, score = embodiment.run()
    return buffer, score


class Embodiment():
    def __init__(self, name, textures, autoencoder_params=None, vergence_actor_critic_params=None, agent=None,
                n_episodes=1, n_timesteps=10, ):
        super().__init__()

        self.device = get_device()
        self.name = 'proc-%02d' % name
        self.min_texture_dist = min_texture_dist
        self.max_texture_dist = max_texture_dist
        self.n_timesteps = n_timesteps
        self.n_episodes = n_episodes
        self.environment = Environment(min_texture_dist=self.min_texture_dist, max_texture_dist=self.max_texture_dist)
        self.buffer = Buffer()
        self.agent = Agent()
        self.textures = random.sample(textures, len(textures))
        self.reward_c = 1
        self.max_version_action = max_version_action
        # load parameters from global agent
        if agent:
            self.agent = agent
        if autoencoder_params:
            self.agent.autoencoder.load_state_dict(autoencoder_params)
        if vergence_actor_critic_params:
            self.agent.vergence_actor.load_state_dict(vergence_actor_critic_params[0])
            self.agent.vergence_critic_1.load_state_dict(vergence_actor_critic_params[1])
            self.agent.vergence_critic_2.load_state_dict(vergence_actor_critic_params[2])        
        
    def update_agent(self, agent):
        self.agent = agent

    def empty_buffer(self):
        self.buffer = Buffer()

    def shuffle_textures(self):
        self.textures = random.sample(self.textures, len(self.textures))

    def run(self, deterministic=False):
        '''
        Runs simulation in the embodiment. Returns buffer and score.
        '''
        score = 0

        for episode_idx in range(self.n_episodes):
            start = time.time()
            running_reward = 0

            # TODO ADD HYPERPARAMS
            texture_velocity = np.array([
                np.random.uniform(low=-self.max_version_action, high=self.max_version_action),
                np.random.uniform(low=-self.max_version_action, high=self.max_version_action),
                0])
            
            # texture sampled from random list of textures
            texture_file = self.textures[episode_idx%len(self.textures)]
            # texture distance in random position between min and max distances
            texture_dist = self.min_texture_dist + (self.max_texture_dist-self.min_texture_dist)*np.random.random()

            # initial angles set to random values
            if vergence:
                initial_vergence_angle = min_initial_vergence_angle + (
                                         max_initial_vergence_angle-min_initial_vergence_angle)*np.random.random()
            else:
                initial_vergence_angle = 0
            if cyclo:
                initial_cyclo_angle = min_initial_cyclo_angle + (
                                         max_initial_cyclo_angle-min_initial_cyclo_angle)*np.random.random()
            else:
                initial_cyclo_angle = 0

            self.environment.reset_eyes(
                initial_vergence_angle=initial_vergence_angle,
                initial_cyclo_angle=initial_cyclo_angle,
                )
            self.environment.new_episode(texture_dist=texture_dist, texture_file=texture_file)

            # Get observation before moving texture (only coarse scale for magno pathway)
            img_left_fine, img_left_coarse, img_right_fine, img_right_coarse = self.environment.get_observations()
            old_observation_fine = img_to_obs(img_left_fine, img_right_fine)
            old_observation_coarse = img_to_obs(img_left_coarse, img_right_coarse)

            for idx in range(self.n_timesteps):

                # Move texture and perform zero acceleration eye movements
                self.environment.move_texture(velocity=texture_velocity)
                if version:
                    self.environment.perform_action(eye_movement='baseline-version')

                # Get observations after moving texture
                img_left_fine, img_left_coarse, img_right_fine, img_right_coarse = self.environment.get_observations()
                observation_fine = img_to_obs(img_left_fine, img_right_fine)
                observation_coarse = img_to_obs(img_left_coarse, img_right_coarse)
                encoding_fine, reconstruction_loss_fine = self.agent.get_parvo_encoding(observation_fine)
                encoding_coarse, reconstruction_loss_coarse = self.agent.get_parvo_encoding(observation_coarse)
                magno_encoding_fine, magno_reconstruction_loss_fine = self.agent.get_magno_encoding(observation_fine, old_observation_fine)
                magno_encoding_coarse, magno_reconstruction_loss_coarse = self.agent.get_magno_encoding(observation_coarse, old_observation_coarse)
                reconstruction_loss = (reconstruction_loss_fine + reconstruction_loss_coarse)/2
                magno_reconstruction_loss = (magno_reconstruction_loss_fine + magno_reconstruction_loss_coarse)/2

                # Perform corrective eye movements
                if vergence:
                    vergence_action = self.agent.choose_action(encoding_fine, encoding_coarse, eye_movement='vergence', deterministic=deterministic)
                    self.environment.perform_action(vergence_action[0], eye_movement='vergence')
                else:
                    vergence_action = 0
                if cyclo:
                    cyclo_action = self.agent.choose_action(encoding_fine, encoding_coarse, eye_movement='cyclo', deterministic=deterministic)
                    self.environment.perform_action(cyclo_action[0], eye_movement='cyclo')
                else:
                    cyclo_action = 0
                if version:
                    version_action = self.agent.choose_action(magno_encoding_fine, magno_encoding_coarse, eye_movement='version', deterministic=deterministic)
                    self.environment.perform_action(version_action[[0,1]], eye_movement='version')
                    
                else:
                    version_action = 0
                action = np.concatenate([vergence_action,version_action])
                
                img_left_fine, img_left_coarse, img_right_fine, img_right_coarse = self.environment.get_observations()
                new_observation_fine = img_to_obs(img_left_fine, img_right_fine)
                new_observation_coarse = img_to_obs(img_left_coarse, img_right_coarse)
                _, new_reconstruction_loss_fine = self.agent.get_parvo_encoding(new_observation_fine)
                _, new_reconstruction_loss_coarse = self.agent.get_parvo_encoding(new_observation_coarse)
                _, new_magno_reconstruction_loss_fine = self.agent.get_magno_encoding(new_observation_fine, old_observation_fine)
                _, new_magno_reconstruction_loss_coarse = self.agent.get_magno_encoding(new_observation_coarse, old_observation_coarse)
                new_reconstruction_loss = (new_reconstruction_loss_fine + new_reconstruction_loss_coarse)/2
                new_magno_reconstruction_loss = (new_magno_reconstruction_loss_fine + new_magno_reconstruction_loss_coarse)/2

                # done flags used for training the critic
                done = (idx == self.n_timesteps-1)
                
                vergence_reward = reward_c * (reconstruction_loss - new_reconstruction_loss) if vergence else 0
                cyclo_reward = reward_c * (reconstruction_loss - new_reconstruction_loss) if cyclo else 0
                version_reward = reward_c * (magno_reconstruction_loss - new_magno_reconstruction_loss) if version else 0
                accommodation_reward = 0   # Not implemented

                # store everything in buffer 
                self.buffer.store(observation_fine, observation_coarse, old_observation_fine, old_observation_coarse, 
                                  encoding_fine, encoding_coarse, magno_encoding_fine, magno_encoding_coarse,
                                  action, done, vergence_reward, cyclo_reward, version_reward, accommodation_reward)
                                
                running_reward += vergence_reward+cyclo_reward+version_reward+accommodation_reward

                # Update old observation and move texture
                old_observation_fine = new_observation_fine
                old_observation_coarse = new_observation_coarse

            score += running_reward / self.n_timesteps

            end = time.time()
            #print('Episode', episode_idx,'of', self.name, 'completed in %.1f seconds' %(end-start),
            #        '– AE loss: %.2e, Score: %.2e' % (new_reconstruction_loss_coarse, running_reward/self.n_timesteps))

        return self.buffer, score / self.n_episodes

if __name__ == '__main__':
    
    texture = ['texture1']
    embodiment = Embodiment(name=0,
                            textures=texture,
                            n_episodes=5,
                            n_timesteps=10,
                            )
    embodiment.run()

