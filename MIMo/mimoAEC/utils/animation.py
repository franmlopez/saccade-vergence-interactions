import gym
import time
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt

import os, sys
file_path = os.path.abspath(os.path.dirname(__file__))
main_path = os.path.abspath(os.path.join(file_path,'..'))
if main_path != sys.path[0]:
    sys.path.insert(0, main_path)
    os.chdir(main_path)
    del main_path, file_path

from utils.auxiliary import img_to_obs, dist_to_angle
from utils.evaluation import Environment
from utils.agent import Agent
from src.config import vergence, cyclo, version, \
                        texture_files, \
                        min_texture_dist, max_texture_dist, \
                        min_initial_vergence_angle, max_initial_vergence_angle, \
                        max_version_action

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--test_for', default=1000, type=int,
                        help='Total timesteps of testing of trained policy')
    parser.add_argument('--episode_length', default=20, type=int,
                        help='Timesteps of one episode')
    parser.add_argument('--checkpoint', default=None, type=int,
                        help='Number of checkpoit to load')
    parser.add_argument('--name', default='', type=str,
                        help='Results folder name')
    
    args = parser.parse_args()
    checkpoint_id = args.checkpoint
    name = args.name
    test_for = args.test_for
    episode_length = args.episode_length
    
    # load agent and create environment
    environment = Environment()
    agent = Agent()

    if checkpoint_id:
        checkpoint = torch.load('results/' + name + '/models/model' + str(checkpoint_id) + '.pt')
        autoencoder_params = checkpoint['ae_state_dict']
        magno_autoencoder_params = checkpoint['magno_ae_state_dict']
        vergence_actor_critic_params = checkpoint['vergence_state_dict'] if vergence else None
        cyclo_actor_critic_params = checkpoint['cyclo_state_dict'] if cyclo else None
        version_actor_critic_params = checkpoint['version_state_dict'] if version else None

        if autoencoder_params:
            agent.autoencoder.load_state_dict(autoencoder_params)
        if magno_autoencoder_params:
            agent.magno_autoencoder.load_state_dict(magno_autoencoder_params)
        if vergence_actor_critic_params:
            agent.vergence_actor.load_state_dict(vergence_actor_critic_params[0])
            agent.vergence_critic_1.load_state_dict(vergence_actor_critic_params[1])
            agent.vergence_critic_2.load_state_dict(vergence_actor_critic_params[2])
        if cyclo_actor_critic_params:
            agent.cyclo_actor.load_state_dict(cyclo_actor_critic_params[0])
            agent.cyclo_critic_1.load_state_dict(cyclo_actor_critic_params[1])
            agent.cyclo_critic_2.load_state_dict(cyclo_actor_critic_params[2])
        if version_actor_critic_params:
            agent.version_actor.load_state_dict(version_actor_critic_params[0])
            agent.version_critic_1.load_state_dict(version_actor_critic_params[1])
            agent.version_critic_2.load_state_dict(version_actor_critic_params[2])
    
    plt.ion()
    fig, [ax1,ax2] = plt.subplots(ncols=2, figsize=(16,8))
    ax1.axes.xaxis.set_visible(False)
    ax1.axes.yaxis.set_visible(False)
    ax2.axes.xaxis.set_visible(False)
    ax2.axes.yaxis.set_visible(False)
    _, img_left_coarse, _, img_right_coarse = environment.get_observations()
    img_left_3d = np.reshape(img_left_coarse, img_left_coarse.shape + (1,))
    img_right_3d = np.reshape(img_right_coarse, img_right_coarse.shape + (1,))
    img_center_3d = (img_left_3d + img_right_3d) / 2.0
    img_stereo = np.concatenate((img_left_3d,img_center_3d,img_right_3d), axis=2)
    img_stereo = (img_stereo+1)/2
    axim1 = ax1.imshow(img_stereo)
    observer_img = environment.interface.env.sim.render(width=200, height=200, camera_name='observer', depth=False)
    axim2 = ax2.imshow(np.flip(observer_img, axis=1))
    time.sleep(0.5)

    texture_dist = np.random.uniform(low=min_texture_dist, high=max_texture_dist)
    texture_velocity = [np.random.uniform(low=-max_version_action/2, high=max_version_action/2),
                        np.random.uniform(low=-max_version_action/2, high=max_version_action/2),
                        0]
    initial_vergence_angle = np.random.uniform(low=min_initial_vergence_angle, high=max_initial_vergence_angle)
    environment.reset_eyes(initial_vergence_angle=initial_vergence_angle)
    environment.new_episode(texture_dist=texture_dist)

    timestep = 0
    for idx in range(test_for):
        timestep += 1

        _, img_left_coarse, _, img_right_coarse = environment.get_observations()
        old_observation_coarse = img_to_obs(img_left_coarse, img_right_coarse)
        
        # Move texture
        environment.move_texture(velocity=texture_velocity)
        environment.perform_action(eye_movement='baseline-version')        

        img_left_fine, img_left_coarse, img_right_fine, img_right_coarse = environment.get_observations()
        observation_fine = img_to_obs(img_left_fine, img_right_fine)
        observation_coarse = img_to_obs(img_left_coarse, img_right_coarse)
        encoding_fine, _ = agent.get_parvo_encoding(observation_fine)
        encoding_coarse, _ = agent.get_parvo_encoding(observation_coarse)
        magno_encoding_left, _ = agent.get_magno_encoding(observation_coarse[0,:,:], old_observation_coarse[0,:,:])
        magno_encoding_right, _ = agent.get_magno_encoding(observation_coarse[1,:,:], old_observation_coarse[1,:,:])

        if vergence:
            action = agent.choose_action(encoding_fine, encoding_coarse, 
                                            eye_movement='vergence',deterministic=True)
            environment.perform_action(action, eye_movement='vergence')
        if version:
            action = agent.choose_action(magno_encoding_left, magno_encoding_right, 
                                            eye_movement='version',deterministic=True)  
            #environment.perform_action(action, eye_movement='version')
            ### REMOVE!
            if timestep == 1:
                environment.perform_action(texture_velocity+
                                            np.random.uniform(low=-0.02,high=0.02,size=3)+
                                            np.random.uniform(low=-0.05, high=0.05, size=3)*(np.random.uniform(size=1)<0.2),
                                           eye_movement='version')
            
                    
        _, img_left_coarse, _, img_right_coarse = environment.get_observations()
        img_left_3d = np.reshape(img_left_coarse, img_left_coarse.shape + (1,))
        img_right_3d = np.reshape(img_right_coarse, img_right_coarse.shape + (1,))
        img_center_3d = (img_left_3d + img_right_3d) / 2.0
        img_stereo = np.concatenate((img_left_3d,img_center_3d,img_right_3d), axis=2)
        img_stereo = (img_stereo+1)/2
        axim1.set_data(img_stereo)
        observer_img = environment.interface.env.sim.render(width=200, height=200, camera_name='observer', depth=False)
        axim2.set_data(np.flip(observer_img, axis=1))
        fig.canvas.flush_events()
        time.sleep(0.1)

        if (timestep%episode_length==0):
            timestep = 0
            # new episode
            texture_file = np.random.choice(texture_files)
            texture_velocity = [np.random.uniform(low=-max_version_action, high=max_version_action),
                                np.random.uniform(low=-max_version_action, high=max_version_action),
                                0]
            texture_dist = np.random.uniform(low=min_texture_dist, high=max_texture_dist)
            initial_vergence_angle = np.random.uniform(low=min_initial_vergence_angle, high=max_initial_vergence_angle)
            environment.reset_eyes(initial_vergence_angle=initial_vergence_angle)
            environment.new_episode(texture_dist=texture_dist, texture_file=texture_file)
            _, img_left_coarse, _, img_right_coarse = environment.get_observations()
            img_left_3d = np.reshape(img_left_coarse, img_left_coarse.shape + (1,))
            img_right_3d = np.reshape(img_right_coarse, img_right_coarse.shape + (1,))
            img_center_3d = (img_left_3d + img_right_3d) / 2.0
            img_stereo = np.concatenate((img_left_3d,img_center_3d,img_right_3d), axis=2)
            img_stereo = (img_stereo+1)/2
            axim1.set_data(img_stereo)
            observer_img = environment.interface.env.sim.render(width=200, height=200, camera_name='observer', depth=False)
            axim2.set_data(np.flip(observer_img, axis=1))
            fig.canvas.flush_events()
            
            time.sleep(1)
            
if __name__ == '__main__':
    main()
