#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
Created on Wed May 27 21:40:40 2020

@author: AtulHome
"""

import cv2 as cv
#from google.colab.patches import cv2_imshow
import copy
import numpy as np
import math

def rotate_image(image,rotation):
    height,width = image.shape[:2]
    
    (centerX, centerY) = (width / 2, height / 2)
    
    rot_mat = cv.getRotationMatrix2D((centerX, centerY), rotation, 1.0)
    cos = np.abs(rot_mat[0, 0])
    sin = np.abs(rot_mat[0, 1])

    new_Width = int((height * sin) + (width * cos))
    new_Height = int((height * cos) + (width * sin))

    rot_mat[0, 2] += (new_Width / 2) - centerX
    rot_mat[1, 2] += (new_Height / 2) - centerY

    return cv.warpAffine(image, rot_mat, (new_Width, new_Height))

class car(object):

  # x and y are center points of the car

    def __init__(
        self,
        x,
        y,
        angle,
        ):
        self.x = x
        self.y = y
        (self.length, self.width) = (int(20), int(10))
        self.angle = angle

    def move(
        self,
        velocity_x,
        velocity_y,
        rotation,
        ):
        self.x = self.x + velocity_x
        self.y = self.y + velocity_y
        self.angle = self.angle + rotation

        if self.angle > 360:
            self.angle = self.angle % 360
        elif self.angle < -360:
            self.angle = self.angle % 360


class city(object):

    def __init__(self, city_file):
        self.city_file = city_file
        self.city_img = cv.imread(self.city_file)
        (self.width, self.length, _) = self.city_img.shape

    def draw_car(
        self,
        x,
        y,
        width,
        height,
        angle,
        img,
        ):

        _angle = (180 - angle) * math.pi / 180.0
        b = math.cos(_angle) * 0.5
        a = math.sin(_angle) * 0.5
        pt0 = (int(x - a * height - b * width), int(y + b * height - a
               * width))
        pt1 = (int(x + a * height - b * width), int(y - b * height - a
               * width))
        pt2 = (int(2 * x - pt0[0]), int(2 * y - pt0[1]))
        pt3 = (int(2 * x - pt1[0]), int(2 * y - pt1[1]))
        pt4 = (int((pt0[0] + pt1[0]) / 2), int((pt0[1] + pt1[1]) / 2))
        pt5 = (int((pt0[0] + pt3[0]) / 2), int((pt0[1] + pt3[1]) / 2))
        pt6 = (int((pt1[0] + pt2[0]) / 2), int((pt1[1] + pt2[1]) / 2))

        line_color = (200, 200, 200)
        line_thickness = 4

    # print(pt0, pt1, pt2, pt3, pt4)

        cv.line(img, pt2, pt3, line_color, line_thickness)
        cv.line(img, pt3, pt5, line_color, line_thickness)
        cv.line(img, pt6, pt2, line_color, line_thickness)
        cv.line(img, pt5, pt4, line_color, line_thickness)
        cv.line(img, pt6, pt4, line_color, line_thickness)

    def get_current_loc_map(
        self,
        x,
        y,
        size,
        angle=0,
        state=False,
        ):

        newcity_img = copy.deepcopy(self.city_img)
        
        if x - size / 2 < 0 or y - size / 2 < 0 or x + size / 2 \
            > self.length - 1 or y + size / 2 > self.width - 1:
            print("This logic should not hit now")
            return np.ones((size, size, 3)), np.ones((1, size, size))
        else:
            y = self.width - y
       
        if state == True:
            img_crop = self.draw_car(
                x,
                y,
                20,
                10,
                angle,
                newcity_img,
                )
        

        img_crop = newcity_img[int(y - size / 2):int(y) + int(size / 2), int(x
                               - size / 2):int(x) + int(size / 2)]
        img_state = np.average(img_crop, axis=2) / 255
        img_state = img_state.reshape(-1, size, size)
        
        return img_crop, img_state


class env(object):

    def __init__(
        self,
        car,
        city,
        city_map,
        car_img,
        ):
        self.car = car
        self.city = city
        self.city_map = city_map
        self.car_img = car_img
        self.car_img = cv.resize(self.car_img, (self.car.length,
                                 self.car.width))
        self.size = 40
        self._max_episode_steps = 500
        self.last_distance = 0
        self.last_boundary_distance = 0
        self.goal_x = 940
        self.goal_y = 580
        self.swap = 0
        self.max_action = 5
        self.refVelX = 1.0
        self.refVelY = 0
        self.step_limit = 5000
        self.distance_normalize_factor = 1570
        self.state_dim = 4
        self.action_dim = 1

    # car_x and car_y are center points of the car

    def show_image(self):
        newcity = copy.deepcopy(self.city)
        
        car_rotated = rotate_image(self.car_img, self.car.angle)
        
        (car_wid, car_len, _) = car_rotated.shape
        pos_x = self.car.x - car_len // 2
        pos_y = newcity.width - (self.car.y + car_wid // 2)

        if pos_x < 10:
            pos_x = 10
        elif pos_x > newcity.length:
            pos_x = newcity.length - car_len

        if pos_y > newcity.width:
            pos_y = newcity.width - car_wid
        elif pos_y < 10:
            pos_y = 10

        car_rotated = cv.addWeighted(newcity.city_img[int(pos_y):int(pos_y + car_wid), int(pos_x):int(pos_x + car_len)], 0.5, car_rotated, 1, 0)
        newcity.city_img[int(pos_y):int(pos_y + car_wid), int(pos_x):int(pos_x + car_len)] = car_rotated
        newcity.city_img = cv.circle(newcity.city_img, (self.goal_x, newcity.width - self.goal_y), 5, (0,0,255), -1)
        
        return newcity

    def step(self, action):
        self.reward = 0
        done = False
        distance = 0
        wall_penalty = 12

        #new_action = self.car.angle + action
        
        angle = math.radians(self.car.angle)
        self.velocity_x = (self.refVelX * math.cos(angle)) - (self.refVelY * math.sin(angle))
        self.velocity_y = (self.refVelY * math.cos(angle)) + (self.refVelX * math.sin(angle))
        self.car.move(self.velocity_x, self.velocity_y, action)
        
        #if self.car.x - self.car.length / 2 < 25 or self.car.y \
        #    - self.car.width / 2 < 10 or self.car.x \
        #    + self.car.length / 2 > self.city_map.length - 10 \
        #    or self.car.y + (self.car.width / 2) > self.city_map.width \
        #    - 10:
        #    self.boundary_hit_count = self.boundary_hit_count + 1
        #    self.reward = self.reward - 10.0
            
        if self.car.x < 25:
            self.car.x = 25
            self.boundary_hit_count = self.boundary_hit_count + 1
            self.reward = self.reward - wall_penalty
        if self.car.x > self.city_map.length - 25:
            self.car.x = self.city_map.length - 25
            self.boundary_hit_count = self.boundary_hit_count + 1
            self.reward = self.reward - wall_penalty
        if self.car.y < 25:
            self.car.y = 25
            self.boundary_hit_count = self.boundary_hit_count + 1
            self.reward = self.reward - wall_penalty
        if self.car.y > self.city_map.width - 25:
            self.car.y = self.city_map.width - 25
            self.boundary_hit_count = self.boundary_hit_count + 1
            self.reward = self.reward - wall_penalty
    
        
        xx = self.goal_x - self.car.x
        yy = self.goal_y - self.car.y
        
        orientation = -(180 / math.pi) * math.atan2(
            self.velocity_x * yy - self.velocity_y * xx,
            self.velocity_x * xx + self.velocity_y * yy)
        
        orientation /= 180

        distance = np.sqrt((self.car.x - self.goal_x) ** 2 + (self.car.y - self.goal_y) ** 2)
        
        car_loc, img_state = self.city_map.get_current_loc_map(self.car.x, self.car.y, self.size, self.car.angle, state=True)
        
        sand_check = np.sum(self.city_map.city_img[int(self.city.width
                            - self.car.y), int(self.car.x)]) / (255 * 3)
        
        if sand_check > 0:  # **** Check whether coords are correct
            self.reward = self.reward - 5
            self.off_road_count = self.off_road_count + 1
        
        else: # moving on the road
            if distance < self.last_distance:
                self.reward = self.reward + 0.5
            else:
                self.reward = self.reward - 1.5
            self.on_road_count = self.on_road_count + 1
            self.episode_onroad = self.episode_onroad + 1
    
        if distance < 25:
            self.reward = self.reward + 50
            self.goal_hit_count += 1
            print ('*************Inside Step Function :: Hit the Goal {}: **************'.format(self.goal_hit_count))
            print("Goal: {:.2f}::{:.2f} Steps Taken {}:: On Road Steps Percentage: {:.2f}".format(self.goal_x, self.goal_y, self.current_step, (self.on_road_count*100/(self.current_step+1))))
            print ('******************************************************************************')
            self.reset_goal()
            self.current_step = 0
            self.on_road_count = 0
            self.off_road_count = 0
            
     
        if (self.boundary_hit_count == 50):
            done = True
            print ("Step Function :: Boundary hit {} times at location::{:.2f}::{:.2f} On Road Steps Percentage: {:.2f}".format(self.boundary_hit_count, self.car.x, self.car.y, (self.on_road_count*100/(self.current_step+1))))
            #temp_distance = (distance - self.last_boundary_distance)**2 
            #if ( temp_distance > 0 and temp_distance < 625):
            #self.reset_goal()
            
            self.last_boundary_distance = distance
            
            
        if (self.current_step > self.step_limit) :
            done = True
            #self.reset_goal()
            print ("Step Function :: Completed steps limit at location::{:.2f}::{:.2f} On Road Steps Percentage: {:.2f}".format(self.car.x, self.car.y, (self.on_road_count*100/(self.current_step+1))))
        
        if (self.goal_hit_count >= 3):
            print("*********** Hurray  {} Goals Hit *************".format(self.goal_hit_count))
            done = True
            self.last_boundary_distance = 0
            
        
        self.current_step += 1
        self.last_action = action
        self.last_reward = self.reward
        self.last_distance = distance
        distance = distance/self.distance_normalize_factor
        
    
        state = [img_state, distance, -orientation, orientation]
        
        return state, self.reward, done

    def reset(self):

        self.on_road_count = 0
        self.off_road_count = 0
        self.boundary_hit_count = 0
        self.goal_hit_count = 0
        self.reward = 0
        self.last_reward = 0
        self.last_action = 0
        self.current_step = 0
        self.episode_onroad = 0
        self.last_distance = 0

        self.car.x = 190
        self.car.y = 460

        self.car.angle = self.sample_action()
        
        state, _, _ = self.step(self.car.angle)
        
        return state
    
    def reset_goal(self):
        
        #goal_loc = np.array([[1070, 600], [875,480],[950, 100], [600, 100],[460, 595],[280,100]])
        goal_loc = np.array([[940, 580], [1100, 200], [380, 220]])
        next_goal_pos = int(np.random.randint(0,(len(goal_loc))))
        self.goal_x, self.goal_y = goal_loc[next_goal_pos, :]
        while (self.swap == next_goal_pos):
            next_goal_pos = int(np.random.randint(0,(len(goal_loc)-1)))
            self.goal_x, self.goal_y = goal_loc[next_goal_pos, :]
        
        self.swap = next_goal_pos
        print("####Goal {} Reset to {}::{}####".format(self.goal_hit_count, self.goal_x, self.goal_y))
        
    
    def sample_action(self):
        return int(np.random.randint(-self.max_action, self.max_action))
        

