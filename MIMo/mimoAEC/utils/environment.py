import matplotlib.pyplot as plt
import numpy as np
import argparse
from sklearn.metrics import mean_squared_error

import os, sys
file_path = os.path.abspath(os.path.dirname(__file__))
main_path = os.path.abspath(os.path.join(file_path,'..'))
if main_path != sys.path[0]:
    sys.path.insert(0, main_path)
    os.chdir(main_path)
    del main_path, file_path

from utils.auxiliary import get_cropped_image, dist_to_angle
from utils.plots import plot_observations
from src.config import interface, vergence_test_angles, min_vergence_angle, max_vergence_angle, \
                       min_cyclo_angle, max_cyclo_angle, interface

if interface == 'pyrender':
    from utils.interface_pyrender import Interface
elif interface == 'MIMo':
    from utils.interface_MIMo import Interface
elif interface is None:
    from utils.interface_MIMo import Interface

class Environment():
    
    def __init__(self, min_texture_dist=50, max_texture_dist=500, 
                    min_vergence_angle=min_vergence_angle, max_vergence_angle=max_vergence_angle,
                    min_cyclo_angle=min_cyclo_angle, max_cyclo_angle=max_cyclo_angle):
        self.interface = Interface(width=256, height=256, fov=0.10)
        self.min_texture_dist = min_texture_dist
        self.max_texture_dist = max_texture_dist
        self.min_vergence_angle = min_vergence_angle
        self.max_vergence_angle = max_vergence_angle
        self.min_cyclo_angle = min_cyclo_angle
        self.max_cyclo_angle = max_cyclo_angle
        self.action_to_angle = vergence_test_angles

    def reset_eyes(self, initial_vergence_angle=0, initial_cyclo_angle=0):
        self.interface.reset_eyes(
            initial_vergence_angle=initial_vergence_angle,
            initial_cyclo_angle=initial_cyclo_angle,
        )

    def new_background(self, background_file):
        self.interface.new_background(background_file)

    def new_episode(self, texture_dist=None, texture_file=None):
        if texture_dist is None:
            texture_dist = self.min_texture_dist + (self.max_texture_dist-self.min_texture_dist)*np.random.random()
        if interface == 'pyrender':
            if self.interface.texture_node:
                self.interface.remove_texture()
        if texture_file:
            self.interface.change_texture(texture_file)
        self.interface.add_texture(texture_dist)

    def move_texture(self, velocity):
        self.interface.move_texture(velocity)

    def perform_action(self, action=None, eye_movement=None):
        if eye_movement=='vergence':
            self.interface.vergence(action, 
                                    min_vergence_angle=self.min_vergence_angle,
                                    max_vergence_angle=self.max_vergence_angle)
        elif eye_movement=='cyclo':
            self.interface.cyclo(action, 
                                min_cyclo_angle=self.min_cyclo_angle,
                                max_cyclo_angle=self.max_cyclo_angle)
        elif eye_movement=='version':
            self.interface.version(action[0], action[1])
        elif eye_movement=='baseline-version':
            self.interface.version()


    def get_observations(self):
        img_left, img_right = self.interface.render_scene()
        img_left_fine, img_left_coarse = get_cropped_image(img_left)
        img_right_fine, img_right_coarse = get_cropped_image(img_right)
        return img_left_fine, img_left_coarse, img_right_fine, img_right_coarse


if __name__ == '__main__':

    texture_dist = 300
    environment = Environment()
    environment.new_episode(texture_dist=texture_dist, texture_file='texture1')

    environment.interface.vergence(3)
    _, img_left_coarse, _, img_right_coarse = environment.get_observations()
    mse = mean_squared_error(img_left_coarse, img_right_coarse)
    plot_observations(img_left_coarse, img_right_coarse, texture_dist=texture_dist,
                eyes_angle=environment.interface.vergence_angle, mse=mse)

    """
    environment.interface.move_texture([10, 10, 0])
    _, img_left_coarse, _, img_right_coarse = environment.get_observations()
    mse = mean_squared_error(img_left_coarse, img_right_coarse)
    plot_observations(img_left_coarse, img_right_coarse, texture_dist=texture_dist,
                    eyes_angle=environment.interface.vergence_angle, mse=mse)

    environment.interface.version(-10, -10)
    _, img_left_coarse, _, img_right_coarse = environment.get_observations()
    mse = mean_squared_error(img_left_coarse, img_right_coarse)
    plot_observations(img_left_coarse, img_right_coarse, texture_dist=texture_dist,
                    eyes_angle=environment.interface.vergence_angle, mse=mse)
    """