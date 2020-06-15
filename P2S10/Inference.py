# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 21:46:16 2020

@author: AtulHome
"""
# This file is used for the testing of the model. The model i trained in Colab
# but tested locally with this file

# Import the required packages
import torch
#import torch.nn as nn
#import torch.nn.functional as F
import cv2 as cv
import numpy as np
import os

import ai, map

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Take the image of Car, City and City Mask
car_file = './images/car.jpg'
city_file = './images/citymap.png'
city_map_file = "./images/MASK1.png"
car_img = cv.imread(car_file)
car1 = map.car(0,0,0)
city1 = map.city(city_file)
citymap1 = map.city(city_map_file)

# Create environment object
env = map.env(car1, city1, citymap1, car_img) # Instantiate the environment
seed = 8 # Random seed number
save_models = True # Boolean checker whether or not to save the pre-trained model

#Define the file name for the model to be tested
file_name = "%s_%s" % ("TD3_car", str(seed))
file_name = file_name + "_31673"
print ("---------------------------------------")
print ("Settings: %s" % (file_name))
print ("---------------------------------------")
if not os.path.exists("./results"):
  os.makedirs("./results")
if save_models and not os.path.exists("./pytorch_models"):
  os.makedirs("./pytorch_models")

torch.manual_seed(seed)
np.random.seed(seed)
state_dim = env.state_dim
action_dim = env.action_dim
max_action = env.max_action

torch.manual_seed(seed)
np.random.seed(seed)

# Create Policy
policy = ai.TD3(state_dim, action_dim, max_action)
replay_buffer = ai.ReplayBuffer()

#Load the model
policy.load(file_name, './pytorch_models/')

obs = env.reset()

# we randomly step through the environment and add 1000 transitions to replay_buffer
for i in range(50000):
      
    action = policy.select_action(obs)
    new_obs,reward,done = env.step(action)
    new_city = env.show_image()
    cv.namedWindow('EndGame', cv.WINDOW_AUTOSIZE) #WINDOW_AUTOSIZE
    cv.imshow("EndGame",new_city.city_img)
    if done: 
        obs =env.reset()
    else:
        obs = new_obs
        
    if cv.waitKey(25) == 27:    # delay of 5 ms or exit loop on 'esc' key press
        break

cv.destroyAllWindows()
