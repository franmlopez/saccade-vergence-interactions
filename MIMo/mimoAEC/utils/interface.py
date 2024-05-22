from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import trimesh
import time
import gym
import mujoco_py

import os, sys
file_path = os.path.abspath(os.path.dirname(__file__))
main_path = os.path.abspath(os.path.join(file_path,'..'))
if main_path != sys.path[0]:
    sys.path.insert(0, main_path)
    os.chdir(main_path)
    del main_path, file_path

from utils.MIMo import mimoEnv
from utils.auxiliary import vel_angle_to_cm_version, vel_angle_to_cm_vergence, dist_to_angle
from src.config import vergence, cyclo, \
                        min_vergence_angle, max_vergence_angle, min_cyclo_angle, max_cyclo_angle, \
                        min_version_angle, max_version_angle, \
                        max_texture_displacement, min_texture_dist, max_texture_dist

class Interface():
    def __init__(self, width=32, height=32, fov=0.3,
                 initial_vergence_angle=0, initial_cyclo_angle=0):
        self.distance_view = 4.9    # Distance between eyes
        self.texture = "texture1"
        self.texture_position = np.array([0, 0, 50/100])
        self.vergence_angle = initial_vergence_angle
        self.pan_angle = 0
        self.tilt_angle = 0
        self.pan_velocity = 0
        self.tilt_velocity = 0
        self.env = gym.make('mimoEnv:MIMoBinocular-v0')
        self.env.reset()

    def vergence(self, angle=0, 
                 min_vergence_angle=min_vergence_angle,
                 max_vergence_angle=max_vergence_angle):
        new_vergence_angle = self.vergence_angle + angle
        if (new_vergence_angle < min_vergence_angle):
            new_vergence_angle = min_vergence_angle
        if (new_vergence_angle > max_vergence_angle):
            new_vergence_angle = max_vergence_angle
        self.vergence_angle = new_vergence_angle
        old_state = self.env.sim.get_state()
        qpos = self.env.sim.data.qpos
        qpos[16] = -(self.vergence_angle - self.pan_angle) * (np.pi/180)  # MIMo angles in radians
        qpos[19] = -(self.vergence_angle + self.pan_angle) * (np.pi/180)  
        qvel = np.zeros(old_state.qvel.shape)
        new_state = mujoco_py.MjSimState(
            old_state.time, qpos, qvel, old_state.act, old_state.udd_state
        )
        self.env.sim.set_state(new_state)
        self.env.sim.forward()
        

    def version(self, pan_action=None, tilt_action=None,  # version actions are acceleration changes
                 min_version_angle=min_version_angle,
                 max_version_angle=max_version_angle):
        # pan
        new_pan_angle = (self.pan_angle + pan_action if pan_action else self.pan_angle + self.pan_velocity)
        if (new_pan_angle < min_version_angle):
            new_pan_angle = min_version_angle
            self.pan_velocity = 0
        elif (new_pan_angle > max_version_angle):
            new_pan_angle = max_version_angle
            self.pan_velocity = 0
        else:
            self.pan_velocity += (pan_action if pan_action else 0)
        
        # tilt 
        new_tilt_angle = (self.tilt_angle + tilt_action if tilt_action else self.tilt_angle + self.tilt_velocity)
        if (new_tilt_angle < min_version_angle):
            new_tilt_angle = min_version_angle
            self.tilt_velocity = 0
        elif (new_tilt_angle > max_version_angle):
            new_tilt_angle = max_version_angle
            self.tilt_velocity = 0
        else:
            self.tilt_velocity += (tilt_action if tilt_action else 0)

        self.pan_angle = new_pan_angle
        self.tilt_angle = new_tilt_angle            
        old_state = self.env.sim.get_state()
        qpos = self.env.sim.data.qpos
        qpos[16] = -(self.vergence_angle - self.pan_angle) * (np.pi/180)   # left eye - pan
        qpos[17] =  self.tilt_angle * (np.pi/180)  # left eye - tilt
        qpos[19] = -(self.vergence_angle + self.pan_angle) * (np.pi/180)   # right eye - pan
        qpos[20] =  self.tilt_angle * (np.pi/180)  # right eye - tilt
        qvel = np.zeros(old_state.qvel.shape)
        new_state = mujoco_py.MjSimState(
            old_state.time, qpos, qvel, old_state.act, old_state.udd_state
        )
        self.env.sim.set_state(new_state)
        self.env.sim.forward()

    def reset_eyes(self, initial_vergence_angle=0, initial_cyclo_angle=0):
        self.vergence_angle = initial_vergence_angle
        self.pan_angle = 0
        self.tilt_angle = 0
        self.pan_velocity = 0
        self.tilt_velocity = 0
        old_state = self.env.sim.get_state()
        qpos = self.env.sim.data.qpos
        qpos[16] = -(self.vergence_angle - self.pan_angle) * (np.pi/180)   # left eye - pan
        qpos[17] =  self.tilt_angle * (np.pi/180)  # left eye - tilt
        qpos[19] = -(self.vergence_angle + self.pan_angle) * (np.pi/180)   # right eye - pan
        qpos[20] =  self.tilt_angle * (np.pi/180)  # right eye - tilt
        qvel = np.zeros(old_state.qvel.shape)
        new_state = mujoco_py.MjSimState(
            old_state.time, qpos, qvel, old_state.act, old_state.udd_state
        )
        self.env.sim.set_state(new_state)
        self.env.sim.forward()
        
    def new_background(self, background_file):
        pass

    def change_texture(self, texture_file):
        self.env.swap_target_texture(texture_file)

    def add_texture(self, dist=50):
        self.texture_position = np.array([0, 0, dist/100])  #in cm
        old_state = self.env.sim.get_state()
        qpos = self.env.sim.data.qpos
        qpos[[-7,-6,-5]] = np.array([
                self.texture_position[2],
                self.texture_position[0],
                self.texture_position[1]
            ]) 
        qvel = np.zeros(old_state.qvel.shape)
        new_state = mujoco_py.MjSimState(
            old_state.time, qpos, qvel, old_state.act, old_state.udd_state
        )
        self.env.sim.set_state(new_state)
        self.env.sim.forward()
        
    def remove_texture(self):
        pass

    def move_texture(self, velocity=[0, 0, 0]):
        # Convert velocity in angles to velocity in cm
        movement = np.array([
            vel_angle_to_cm_version(velocity[0], self.texture_position[2]),
            vel_angle_to_cm_version(velocity[1], self.texture_position[2]),
            vel_angle_to_cm_vergence(velocity[2], self.texture_position[2], self.distance_view)
        ])
        new_texture_position = self.texture_position + movement
        """
        new_texture_position = np.array([
            min(max(new_texture_position[0], -max_texture_displacement/100), max_texture_displacement/100),
            min(max(new_texture_position[1], -max_texture_displacement/100), max_texture_displacement/100),
            min(max(new_texture_position[2], min_texture_dist/100), max_texture_dist/100)
        ])
        """
        self.texture_position = new_texture_position
        
        old_state = self.env.sim.get_state()
        qpos = self.env.sim.data.qpos
        qpos[[-7,-6,-5]] = np.array([
            self.texture_position[2],
            self.texture_position[0],
            self.texture_position[1]
        ]) 
        qvel = np.zeros(old_state.qvel.shape)
        new_state = mujoco_py.MjSimState(
            old_state.time, qpos, qvel, old_state.act, old_state.udd_state
        )
        self.env.sim.set_state(new_state)
        self.env.sim.forward()

    def render_scene(self):
        obs = self.env.return_obs()
        img_left = obs['eye_left']
        img_right = obs['eye_right']
        return img_left, img_right


if __name__ == '__main__':
    
    t0 = time.time()
    interface = Interface(width=320, height=320)
    interface.add_texture(300)
    img_left, img_right = interface.render_scene()
    plt.imshow(img_left)
    plt.show()

    interface.move_texture([2, 0, 0])
    interface.version(pan_action=0)
    img_left, img_right = interface.render_scene()
    plt.imshow(img_left)
    plt.show()
    
    interface.add_texture(300)
    interface.reset_eyes(initial_vergence_angle=dist_to_angle(300, interface.distance_view))
    img_left, img_right = interface.render_scene()
    plt.imshow(img_left)
    plt.show()

    interface.move_texture([2, 0, 0])
    interface.version(pan_action=2)
    img_left, img_right = interface.render_scene()
    plt.imshow(img_left)
    plt.show()

    for _ in range(4):
        interface.move_texture([2, 0, 0])
        interface.version(pan_action=0)
    img_left, img_right = interface.render_scene()
    plt.imshow(img_left)
    plt.show()