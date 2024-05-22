import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.normal import Normal
import random
from collections import namedtuple, deque
from PIL import Image
from sklearn.preprocessing import minmax_scale
from torch.utils.data import DataLoader
from math import tan, atan, pi


def get_fine_and_coarse(img, input_size=64, output_size=32):
    fine_size, coarse_size = input_size/2, input_size
    resize_scale = output_size

    img = to_grayscale(img)
    img = minmax_scale(img, feature_range=(-1,1))
    pil_img = Image.fromarray(img)
    width, height = pil_img.size
    
    area_fine = (width/2-(fine_size/2-1), height/2-(fine_size/2-1), width/2+(fine_size/2+1), height/2+(fine_size/2+1))
    crop_img_fine = pil_img.crop(area_fine)
    resize_img_fine = crop_img_fine.resize([resize_scale, resize_scale])
    #img_fine = minmax_scale(resize_img_fine, feature_range=(-1,1))
    img_fine = np.asarray(resize_img_fine)
    
    area_coarse = (width/2-(coarse_size/2-1), height/2-(coarse_size/2-1), width/2+(coarse_size/2+1), height/2+(coarse_size/2+1))
    crop_img_coarse = pil_img.crop(area_coarse)
    resize_img_coarse = crop_img_coarse.resize([resize_scale, resize_scale])
    #img_coarse = minmax_scale(resize_img_coarse, feature_range=(-1,1))
    img_coarse = np.asarray(resize_img_coarse)

    return img_fine, img_coarse

def to_grayscale(rgb):
    #rgb = (rgb-np.min(rgb))/(np.max(rgb)-np.min(rgb))
    gray = 0.2989*rgb[:,:,0] + 0.5870*rgb[:,:,1] + 0.1140*rgb[:,:,2]
    gray = np.clip(minmax_scale(gray, feature_range=(0,1)), a_min=0, a_max=1)
    return gray

def stack_images(img_left, img_right):
    image = np.zeros([2,32,32])
    image[0,:,:] = img_left
    image[1,:,:] = img_right
    return image

def dist_to_angle(texture_dist, distance_view=0.049):     # distance_view = dist between eyes 
    return atan((distance_view/2) / texture_dist)*180/pi

def angle_to_dist(angle, distance_view=0.049):
    if angle==0:
        return 999
    return 1/tan(angle*pi/180) * distance_view/2

def cart_to_spher(coords):
    x,y,z = coords[0], coords[1], coords[2]
    dist = np.sqrt(x**2 + y**2 + z**2)
    polar = np.arccos(z / dist) * 180/np.pi
    azim = np.arctan2(y, x) * 180/np.pi
    return dist, polar, azim

def spher_to_cart(coords):
    dist, polar, azim = coords[0], coords[1], coords[2]
    polar = (90 - polar) * np.pi/180
    azim = (90 + azim) * np.pi/180
    x = dist * np.sin(polar) * np.sin(azim)
    y = dist * np.sin(polar) * np.cos(azim)
    z = dist * np.cos(polar)
    return x, y, z

def quaternion_multiply(quaternion1, quaternion0):
    w0, x0, y0, z0 = quaternion0
    w1, x1, y1, z1 = quaternion1
    return np.array([-x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
                     x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
                     -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
                     x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0], dtype=np.float64)