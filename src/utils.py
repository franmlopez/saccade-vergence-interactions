import numpy as np
import os
import time
import torch
from math import tan, atan, pi
from scipy import ndimage

# get the computation device
def get_device(print_to_screen=True):
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    if print_to_screen:
        print('Device selected:', device)
    return device

# make the `images` directory
def make_dir(folder_name):
    image_dir = folder_name+'/images'
    model_dir = folder_name+'/models'
    plots_dir = folder_name+'/plots'
    values_dir = folder_name+'/values'
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
    if not os.path.exists(values_dir):
        os.makedirs(values_dir)

def write_values(array, name, folder_name):
    np.savetxt(folder_name+'/values/'+name+'.csv', array, delimiter=',')

def imgs_from_dict(img_dict):
    return img_dict['left_fine'], img_dict['right_fine'], img_dict['left_coarse'], img_dict['right_coarse']

def img_to_obs(img_left, img_right):
    width = img_left.shape[1]
    observation = np.zeros([2,width,width])
    observation[0,:,:] = img_left
    observation[1,:,:] = img_right
    return observation

def obs_to_imgs(observation):
    img_left = observation[0,:,:]
    img_right = observation[1,:,:]
    return img_left, img_right

def dist_to_angle(target_dist, distance_view=0.049):     # distance_view = dist between eyes 
    return atan((distance_view/2) /target_dist)*180/pi

def angle_to_dist(angle, distance_view=0.049):
    if angle < 0.1:     # this catches zero-divisions
        return 10
    else:
        return 1/tan(angle*pi/180) * distance_view/2

def vel_angle_to_cm_version(angle, target_dist):
    return target_dist * tan(angle*pi/180)

def vel_angle_to_cm_vergence(angle, target_dist, distance_view=0.049):
    return (distance_view/2) / (tan(atan(distance_view/2/target_dist) - angle)) - target_dist

def join_imgs(img_fine, img_coarse):
    img = ndimage.zoom(img_coarse, 2, order=0)
    img_size = img.shape[0]
    img[int(img_size/4):int(img_size*3/4),int(img_size/4):int(img_size*3/4)] = img_fine
    return img

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def compute_reconstruction_map(observation, reconstruction):
    difference = (observation - reconstruction)**2
    reconstruction_map = np.mean(difference, axis=0)
    return reconstruction_map

def gauss(size, sigma=1):
    m = np.linspace(-1, 1, size)  # points between -1 and 1
    (x, y) = np.meshgrid(m, m)
    return 1 / (2 * np.pi * sigma ** 2) * np.exp(-.5 * (x ** 2 + y ** 2) / sigma ** 2)

def compute_saliency_map(reconstruction_map, reward_window=9):
    filter = gauss(reward_window, sigma=reward_window)
    saliency_map = ndimage.convolve(reconstruction_map, filter, mode='constant', cval=-np.min(reconstruction_map))
    saliency_map = softmax(saliency_map)
    return saliency_map

def choose_saccade_action(reconstruction_map, window_size=9, deterministic=False, foveation_weights=None):
    saliency_map = compute_saliency_map(reconstruction_map, window_size, foveation_weights)
    flat_saliency_map = saliency_map.flatten()

    if deterministic:
        target_index = np.argmax(flat_saliency_map)
    else:
        target_index = np.random.choice(np.arange(len(flat_saliency_map)), p=flat_saliency_map)

    target_coords = np.unravel_index(target_index, shape=saliency_map.shape)
    action = -np.array([
        target_coords[1] - (saliency_map.shape[0]-1)/2,
        target_coords[0] - (saliency_map.shape[1]-1)/2
        ])
    return action

def mse(a, b):
    return np.square(a - b).mean() 

def compute_vergence_reward(observation_fine, reconstruction_fine,
                            observation_coarse, reconstruction_coarse,
                            new_observation_fine, new_reconstruction_fine,
                            new_observation_coarse, new_reconstruction_coarse,
                            method='lopez'):
    # MSEs between observation and reconstructions
    mse_fine = mse(observation_fine, reconstruction_fine)
    new_mse_fine = mse(new_observation_fine, new_reconstruction_fine)

    if method == 'lopez':
        reward = 10 * (mse_fine - new_mse_fine) / (mse_fine + new_mse_fine)
    elif method == 'wilmot':
        reward = 600 * (mse_fine - new_mse_fine)
    elif method == 'zhu':
        reward = 600 * (-new_mse_fine)

    # Disparity used for learning curves or extrinsic reward
    disparity = mse(observation_fine[0,:,:], observation_fine[1,:,:])

    return reward, -disparity