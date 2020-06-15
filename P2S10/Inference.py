# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 21:46:16 2020

@author: AtulHome
"""

import torch
#import torch.nn as nn
#import torch.nn.functional as F
import cv2 as cv
import numpy as np
import os

import ai, map

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

car_file = './images/car.jpg'
city_file = './images/citymap.png'
city_map_file = "./images/MASK1.png"
car_img = cv.imread(car_file)
car1 = map.car(0,0,0)
city1 = map.city(city_file)
citymap1 = map.city(city_map_file)

env = map.env(car1, city1, citymap1, car_img) # Instantiate the environment
seed = 8 # Random seed number
save_models = True # Boolean checker whether or not to save the pre-trained model

file_name = "%s_%s" % ("TD3_car", str(seed))
file_name = file_name + "_80556"
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

max_episode_steps = env._max_episode_steps


torch.manual_seed(seed)
np.random.seed(seed)

policy = ai.TD3(state_dim, action_dim, max_action)
replay_buffer = ai.ReplayBuffer()
#evaluations = [evaluate_policy(policy)]

policy.load(file_name, './pytorch_models/')

obs = env.reset()

# we randomly step through the environment and add 1000 transitions to replay_buffer
for i in range(50000):
      
    action = policy.select_action(obs)
    new_obs,reward,done = env.step(action)
    #print('action: ',action,type(action))
    #print('Reward: ', reward)
    #print('stateValue: ',new_obs[1:])
    new_city = env.show_image()
    cv.namedWindow('EndGame', cv.WINDOW_AUTOSIZE) #WINDOW_AUTOSIZE
    cv.imshow("EndGame",new_city.city_img)
    #cv.namedWindow('state0') #WINDOW_AUTOSIZE
    #cv.imshow("state0",obs[0].squeeze())
    if done: 
        obs =env.reset()
    else:
        obs = new_obs
        
    if cv.waitKey(5) == 27:    # delay of 5 ms or exit loop on 'esc' key press
        break

cv.destroyAllWindows()
