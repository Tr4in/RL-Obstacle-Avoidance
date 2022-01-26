'''
Implementation inspired by:

https://github.com/xie9187/Monocular-Obstacle-Avoidance, last visited: 26.01.2022

'''


from collections import deque
import numpy as np
import os
from pydnet_network import Pydnet
import tensorflow as tf
from PIL import Image
import cv2

class Environment:
    def __init__(self, ue_communicator, consecutive_images, time_steps, num_laser):
        self.__load_pydnet_model()
        self.counter = 1
        self.ue_communicator = ue_communicator
        self.consequtive_images = consecutive_images
        self.image_buffer = deque(maxlen = consecutive_images)
        self.time_steps = time_steps
        self.num_laser = num_laser

    def __load_pydnet_model(self):
        network_params = {"height": 192, "width": 320, "is_training": False}

        pydnet_path = 'ckpt/pydnet'

        model = Pydnet(network_params)
        self.__tensor_image = tf.placeholder(tf.float32, shape=(192, 320, 3))
        batch_img = tf.expand_dims(self.__tensor_image, 0)
        self.__tensor_depth = model.forward(batch_img)
        self.__tensor_depth = tf.nn.relu(self.__tensor_depth)

        # restore graph
        saver = tf.train.Saver()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        saver.restore(self.sess, pydnet_path)

    def step(self, action):
        self.ue_communicator.execute_action_on_unreal_agent(action)
        collision_happend, torque, steering, laser_distances = self.ue_communicator.get_data(self.num_laser)

        print('Laser Distances: {}'.format(laser_distances))
        print('Collision: {}'.format(collision_happend))
        print('Steering: {}'.format(steering))
        print('Torque: {}'.format(torque))
        done = False

        distance_smaller_than_threshold_list = []

        for i in range(int(self.num_laser / 2)):
            distance_smaller_than_threshold_list.append(laser_distances[i] < 6)
            distance_smaller_than_threshold_list.append(laser_distances[self.num_laser -  1 - i] < 6)

        #print(distance_smaller_than_threshold_list)

        if collision_happend or any(distance_smaller_than_threshold_list) or laser_distances[int(self.num_laser / 2)] < 10:
            reward = -10
            done = True
            #print('Collision')
        else:
            reward = ((torque / 20) * np.cos(np.deg2rad(steering)) - 0.36) / 20

        print('Reward {}'.format(reward))

        next_state = np.zeros((4,192,320))
        if not done:
            filename = self.ue_communicator.request_next_filename()
            predicted_depth_image = self.get_normalized_predicted_depth_image(filename)
            self.save_predicted_depth_image_to_buffer(predicted_depth_image)
            next_state = self.get_state()
        
        return reward, next_state, done

    def reset(self):
        self.ue_communicator.reset_environment()
        self.clear_image_buffer()

        filename = self.ue_communicator.request_next_filename()
        pred_depth_image = self.get_normalized_predicted_depth_image(filename)
        
        for _ in range(self.consequtive_images):
            self.save_predicted_depth_image_to_buffer(pred_depth_image)

        return self.get_state()
        
    def save_predicted_depth_image_to_buffer(self, predicted_depth_image):
        self.image_buffer.appendleft(predicted_depth_image) 

    def clear_image_buffer(self):
        self.image_buffer.clear()
    
    def get_normalized_predicted_depth_image(self, filename):
        img = np.array(Image.open(filename + '.png').convert('RGB')) / 255.0

        depth_image = self.sess.run(self.__tensor_depth, feed_dict={self.__tensor_image: img}).squeeze(axis = (0, 3))

        squeezed_depth_image = np.squeeze(depth_image)
        min_depth = squeezed_depth_image.min()
        max_depth = squeezed_depth_image.max()
        depth_image = (depth_image - min_depth) / (max_depth - min_depth)

        os.remove(filename + '.png')

        self.__show_depth_image(depth_image)
        return depth_image


    def __show_depth_image(self, depth_image):
        cv2.imshow('Depth Vision', depth_image)
        cv2.waitKey(1)

    def get_state(self):
        state = np.stack(self.image_buffer, axis = 0)
        return state
