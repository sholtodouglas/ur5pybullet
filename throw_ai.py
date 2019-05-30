
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import tensorflow as tf
from tensorflow.python import tf2
if not tf2.enabled():
  import tensorflow.compat.v2 as tf
  tf.enable_v2_behavior()
  assert tf2.enabled()

# print(tf2.enabled())

import tensorflow_probability as tfp

import tensorflow_datasets as tfds


tfk = tf.keras
tfkl = tf.keras.layers
tfpl = tfp.layers
tfd = tfp.distributions
tfb = tfp.bijectors

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


from tensorflow.keras.layers import Dense, Flatten, Conv2D,Bidirectional, LSTM, Dropout
from tensorflow.compat.v1.keras.layers import CuDNNLSTM
from tensorflow.keras import Model
from tensorflow.keras.models import  Sequential
import matplotlib.pyplot as plt
import datetime
import os
from tqdm import tqdm, tqdm_notebook
import random
from natsort import natsorted, ns
import imageio
from IPython import display
from PIL import Image
from IPython.display import clear_output
import IPython
import time
import seaborn as sns
import matplotlib.patheffects as PathEffects
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
import pandas as pd
import traceback



print('Importing viz libraries...')
# Load the TensorBoard notebook extension

print('Tensorflow version (should be >= 2.0): '+tf.__version__)
if tf.test.gpu_device_name() != '/device:GPU:0':
  print('WARNING: GPU device not found.')

else:
  print('SUCCESS: Found GPU: {}'.format(tf.test.gpu_device_name()))




throw_num = 0
import os, inspect
import time

from tqdm import tqdm
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
print("current_dir=" + currentdir)
os.sys.path.insert(0, currentdir)

import click
import math 
import gym
import sys
from gym import spaces
from gym.utils import seeding 
import numpy as np
import time
import pybullet as p
from itertools import chain

import random
import pybullet_data

from kuka import kuka
from ur5 import ur5
import sys
from scenes import * # where our loading stuff in functions are held
import imageio
import cv2
from model import *

viewMatrix = p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition = [0,0,0], distance = 0.3, yaw = 90, pitch = -90, roll = 0, upAxisIndex = 2) 
projectionMatrix = p.computeProjectionMatrixFOV(fov = 120,aspect = 1,nearVal = 0.01,farVal = 10)

image_renderer = p.ER_BULLET_HARDWARE_OPENGL # if the rendering throws errors, use ER_TINY_RENDERER, but its hella slow cause its cpu not gpu.
cam_dims = 200

params = {'EPOCHS':100, 'BETA':0.035, 'ALPHA':0.0, 
                      'MIN_SEQ_LEN':16, 'MAX_SEQ_LEN':47, 'MAX_EVER_SEQ_LEN':47, 
                      'LATENT_DIM':20, 'LAYER_SIZE':2048,
                      'ACTION_DIM':8,
                      'NUM_DISTRIBUTIONS':1, 'NUM_QUANTISATIONS':256,
                      'DISCRETIZED' : False, 'RELATIVE' : False,
                      'P_DROPOUT' : 0.0, 'BETA_ANNEAL':1.0, 'THRESHOLD' : -999,'TEST':True, 'ARM_IN_GOAL':False, 
                      'OBS_DIM':28}

model_trainer = Model_Trainer(None, **params)


extension = 'Yeetbot_v3_4000ep_overfit_beta50_3'
model_trainer.load_weights(extension)
encoder = model_trainer.encoder
actor = model_trainer.actor


global file_ix
file_ix=0

def collect_past_trajs():
    file_goal_pairs = []
    path = '../robots/play_data/throwing/'
    for i in next(os.walk(path))[1]:
        
        final_moment = [x[0]for x in os.walk(path+i)][-1]
        final_obs = np.load(final_moment+'/obs.npy')
        
        
        print(path+i,final_obs[23:25])
        
        moments = [x[0]for x in os.walk(path+i)][1:]
        obs = []
        acts = []

        for m in moments:
            o = np.load(m+'/obs.npy')

            a = np.load(m+'/act.npy')
            obs.append(o[16:])
            acts.append(a)

        obs = np.array(obs)
        acts = np.array(acts)
        file_goal_pairs.append((path+i, final_obs[23:25], obs, acts))
    return file_goal_pairs

file_goal_pairs = collect_past_trajs()



print(file_goal_pairs[0][3])
def gripper_camera(obs):
	# Center of mass position and orientation (of link-7)
	pos = obs[-7:-4] 
	ori = obs[-4:] # last 4
	# rotation = list(p.getEulerFromQuaternion(ori))
	# rotation[2] = 0
	# ori = p.getQuaternionFromEuler(rotation)

	rot_matrix = p.getMatrixFromQuaternion(ori)
	rot_matrix = np.array(rot_matrix).reshape(3, 3)
	# Initial vectors
	init_camera_vector = (1, 0, 0) # z-axis
	init_up_vector = (0, 1, 0) # y-axis
	# Rotated vectors
	camera_vector = rot_matrix.dot(init_camera_vector)
	up_vector = rot_matrix.dot(init_up_vector)
	view_matrix_gripper = p.computeViewMatrix(pos, pos + 0.1 * camera_vector, up_vector)
	img = p.getCameraImage(cam_dims, cam_dims, view_matrix_gripper, projectionMatrix,shadow=0, flags = p.ER_NO_SEGMENTATION_MASK, renderer=image_renderer)
	w = img[0]
	h = img[1]
	rgb = img[2]
	dep = img[3]
	np_img_arr = np.reshape(rgb, (h,w,4))

	return img, np_img_arr[:,:,:3]
class graspingEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self,
                 urdfRoot=pybullet_data.getDataPath(),
                 actionRepeat=1,
                 isEnableSelfCollision=True,
                 renders=True,
                 arm = 'rbx1',
                 vr = False):
        print("init")
        
        self._timeStep = 1. / 240.
        self._urdfRoot = urdfRoot
        self._actionRepeat = actionRepeat
        self._isEnableSelfCollision = isEnableSelfCollision
        self._observation = []
        self._envStepCounter = 0
        self._renders = renders
        self._vr = vr
        self.terminated = 0
        self._p = p

        
        if self._renders:
            cid = p.connect(p.SHARED_MEMORY)
            if (cid < 0):
                cid = p.connect(p.GUI)
            if self._vr:
                p.resetSimulation()
                                #disable rendering during loading makes it much faster
                p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)

        else:
            p.connect(p.DIRECT)
        self._seed()
        self._arm_str = arm
        self._reset()
        if self._vr:
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
            p.setRealTimeSimulation(1)
        observationDim = len(self.getSceneObservation())
        observation_high = np.array([np.finfo(np.float32).max] * observationDim)
        self.action_space = spaces.Discrete(7)
        self.observation_space = spaces.Box(-observation_high, observation_high)
        self.viewer = None
        
        

    def _reset(self):
        print("reset")
        self.terminated = 0
        p.resetSimulation()
        p.setPhysicsEngineParameter(numSolverIterations=150)
        if not self._vr:
            p.setTimeStep(self._timeStep)
        self.objects = throwing_scene()

        p.setGravity(0, 0, -10)
        if self._arm_str == 'rbx1':
            self._arm = rbx1(urdfRootPath=self._urdfRoot, timeStep=self._timeStep)
        elif self._arm_str == 'kuka':
            self._arm = kuka(urdfRootPath=self._urdfRoot, timeStep=self._timeStep, vr = self._vr)
        else:
            self._arm = load_arm_dim_up('ur5',dim='Z')
            
        self._envStepCounter = 0
        p.stepSimulation()
        self._observation = self.getSceneObservation()
        p.setRealTimeSimulation(1)
        return np.array(self._observation)

    def __del__(self):
        p.disconnect()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def getSceneObservation(self):
        self._observation = self._arm.getObservation()

        # some block to be moved's location and oreintation.
        scene_obs = get_scene_observation(self.objects)

        # and also a block for it to be placed on.

        # the vector between end effector location and block to be moved. and vector between
        # end effector and goal block.

        # print(endEffectorPos, p.getEulerFromQuaternion(endEffectorOrn), blockPos, p.getEulerFromQuaternion(blockOrn))
        # invEEPos,invEEOrn = p.invertTransform(endEffectorPos,endEffectorOrn)
        # blockPosInEE,blockOrnInEE = p.multiplyTransforms(invEEPos,invEEOrn,blockPos,blockOrn)
        # blockEulerInEE = p.getEulerFromQuaternion(blockOrnInEE)
        # self._observation.extend(list(blockPosInEE))
        # self._observation.extend(list(blockEulerInEE))


        # this gives a list which is 8 joint positons, 8 joint velocities, gripper xyz and orientation(quaternion), and
        # the position and orientation of the objects in the environment.
  
        # select which image you want!


        #top_down_img = p.getCameraImage(500, 500, viewMatrix,projectionMatrix, shadow=0,renderer=image_renderer)
        #grip_img = gripper_camera(self._observation)
        obs = [self._observation, scene_obs]
        return obs

    #moves motors to desired pos
    def step(self, action):
        self._arm.action(action)

        for i in range(self._actionRepeat):
            p.stepSimulation()
            if self._renders:
                time.sleep(self._timeStep)
            self._observation = self._arm.getObservation() #self.getSceneObservation()
            self._envStepCounter += 1
        done = self._termination()
        reward = self._reward()

        return np.array(self._observation), reward, done, {}

    def _termination(self):
        if (self.terminated or self._envStepCounter > 10000):
            return True

    def _render(self, mode='human', close=False):
        return

    def _reward(self):


        # use as a RL style reward if you like #################################
        reward = 0
        # block_one = self.objects[0]
        # blockPos, blockOrn = p.getBasePositionAndOrientation(block_one)
        # closestPoints = p.getClosestPoints(block_one, self._arm.uid, 1000)

        # reward = -1000
        # numPt = len(closestPoints)

        # if (numPt > 0):
        #     # print("reward:")
        #     reward = -closestPoints[0][8] * 10

        # if (blockPos[2] > 0.2):
        #     print("grasped a block!!!")
        #     print("self._envStepCounter")
        #     print(self._envStepCounter)
        #     reward = reward + 1000

        return reward

    

##############################################################################################################


    def step_to(self, action, abs_rel = 'abs', noise = False, clip = False, repeat = None):
        motor_poses = self._arm.move_to(action, abs_rel, noise, clip)
        #print(motor_poses) # these are the angles of the joints. 
        if repeat !=None:
            rep = repeat
        else:
            rep = self._actionRepeat
        for i in range(rep):
            p.stepSimulation()
            if self._renders:
                time.sleep(self._timeStep)
            self._observation = self._arm.getObservation()
            
            self._envStepCounter += 1


        done = False #self._termination()
        reward = 0#self._reward()


        
        return np.array(self._observation), motor_poses, reward, done, {}


def get_block_orn(img):

	loc = np.where(img > 0) # find red pixels

	if len(loc[0]) == 0: # if there are no red pixels in view
		gradient = 0

	else: # seen red pixels, go to them

		y_pixels = loc[1]
		x_pixels = loc[0]

		line = np.poly1d(np.polyfit(x_pixels, y_pixels, 1))# fit a line to determine block angle
		gradient = line[1] # angle of the block from gripper

	return gradient

def throw(environment, arm, action, xyz, ori, grip, cube_pos, grasped, lifted, throw_timeout, img, throw_range):
	
	
	
	if time.time() < throw_timeout + 1: # let it rest open after throwing
		motor_poses = [0.0, -2.4473683834075928, -0.7105262279510498, 0.4210524559020996, 1.5, 0.0, -0.1, 0.1, 0.0, 0.0]
		environment._arm.action(motor_poses)
	else:
		if lifted > 10:
			# thus this shouldo only occur once
			
			action, grasped, lifted = throw_primitive(environment, arm, action, xyz, grasped, lifted, throw_range)
			throw_timeout = time.time()
		elif grasped > 10:
			print('lift primitive')
			action, grasped, lifted = lift_primitive(environment, arm, action, xyz, ori, grasped, lifted, img)
			throw_timeout = 0
		else:
			action, grasped, lifted = grasp_primitive(environment, arm, action, xyz, ori, grip, cube_pos, grasped, lifted, img)
			throw_timeout = 0
	return action , grasped, lifted, throw_timeout 

def grasp_primitive(environment, arm, action, xyz, ori, grip, cube_pos, grasped, lifted, img):

	
	## move above

	loc = np.where(img > 0) # find red pixels

	if len(loc[0]) == 0: # if there are no red pixels in view
		action[0] = 0.0
		action[1] = 0.0
		action[2] = 0.4

	else: # seen red pixels, go to them
		y_relative_dir = np.mean(loc[0]) - cam_dims/2 
		x_relative_dir = np.mean(loc[1]) - cam_dims/2 

		
		
		action[0] = xyz[0] + x_relative_dir*0.0005#cube_pos[0]
		action[1] = xyz[1] - y_relative_dir*0.0005

		y_pixels = loc[1]
		x_pixels = loc[0]
		line = np.poly1d(np.polyfit(x_pixels, y_pixels, 1))# fit a line to determine block angle
		gradient = line[1] # angle of the block from gripper

		print(xyz[2], cube_pos[2])
		
		if  (  (xyz[0] - cube_pos[0]) < 0.01) and ( (xyz[1] - cube_pos[1]) <0.01) and ((xyz[2] - cube_pos[2]) < 0.13) and (abs(gradient) < 0.015):
			if xyz[2] > cube_pos[2]+0.05:
				action[2] = xyz[2] - 0.003
			else:
				action[2] = xyz[2] + 0.003

			action[7] = grip*25 +0.1 # grip is in 0-0.04 scale
			print('grasping')
			
			
			grasped += 0.5
			action[3:7] = p.getQuaternionFromEuler([ori[0] - gradient*0.5, ori[1], ori[2]]) # adjust gripper angle to suit block
			

		elif ((xyz[0] - cube_pos[0]) < 0.01) and ((xyz[1] - cube_pos[1]) <0.01):
			#we are close enough to move down
			
			print('moving to grasp', grasped, lifted)
			
			if xyz[2] > cube_pos[2]+0.2:
				action[2] = xyz[2] - (xyz[2]-0.005)**2
			else:
				action[2] = xyz[2] -0.003
			# if xyz[2] > cube_pos[2]+0.05:
			# 	action[2] = xyz[2] - 0.003
			# else:
			# 	action[2] = xyz[2] + 0.003

			

				
			action[3:7] = p.getQuaternionFromEuler([ori[0] - gradient*0.5, ori[1], ori[2]]) # adjust gripper angle to suit block
		else:
			# go above it
			print('moving above')

			action[2] = cube_pos[2]+0.2
			 # adjust gripper angle to suit block

	return action, grasped, lifted


def drop_primitive(environment, arm, action, xyz, grasped, lifted, img):
	for i in range(0,50):
		action[7] = 0.0
		action[2] = 0.2
		environment.step_to(action)

def lift_primitive(environment, arm, action, xyz, ori, grasped, lifted, img):
	print('lifting')

	gradient = get_block_orn(img)
	if abs(gradient) > 0.1:
		print("misaligned, dropping")
		grasped = 0
		lifted = 0
		drop_primitive(environment, arm, action, xyz, grasped, lifted, img)
		action[7] = 0.0
		action[2] = 0.3
	else:
		action[7] = 0.8
		action[2] = 0.25


	# des_eul = p.getEulerFromQuaternion(action[3:7] )
	# action[3:7] = p.getQuaternionFromEuler([ori[0]+0.05*(des_eul[0] - ori[0]), ori[1], ori[2]]) # adjust gripper angle to suit block

	action[0] = 0.0
	action[1] = 0.0
	
	

	if xyz[2] > 0.21:
		lifted +=1
	return action, grasped, lifted


def move_to_throw_pos_with_speed(environment, arm, motorCommands, throw_range):
    poses = []
    indexes = []
    forces = []



    for i, name in enumerate(environment._arm.controlJoints):
        joint = environment._arm.joints[name]

        poses.append(motorCommands[i])
        indexes.append(joint.id)
        forces.append(joint.maxForce)
    l = len(poses)

    targetVelocities = [0]*l
    targetVelocities[1] = 2*throw_range
    targetVelocities[2] = 10*throw_range
    targetVelocities[3] = 0.3

    p.setJointMotorControlArray(environment._arm.uid, indexes, p.POSITION_CONTROL, targetPositions=poses, targetVelocities =targetVelocities, positionGains = [0.03]*l, forces = forces)


def throw_primitive(environment, arm, action, xyz, grasped, lifted, throw_range):
    print('throwing')
    print('throwrange-----',throw_range)
    motor_poses = [0.0, 0.4473683834075928*throw_range, -0.7105262279510498*throw_range, 0.2210524559020996, 1.5, 0.0, 0.1, -0.1, 0.0, 0.0]#list(p.calculateInverseKinematics(environment._arm.uid, environment._arm.endEffectorIndex, action[0:3], action[3:7]))


    observation = environment._arm.getObservation()
    xyz = observation[-7:-4] 


    print(motor_poses)
    do_past_throw(environment)

    # while ((xyz[0] < 0.3) and (xyz[2] < 0.3)):
    # 	move_to_throw_pos_with_speed(environment, arm, motor_poses, throw_range)
    # 	p.stepSimulation()
    	

    # 	observation = environment._arm.getObservation()
    # 	xyz = observation[-7:-4] 
    # 	print(xyz)


    # now release
    action[7] = 0.0
    grasped = 0
    lifted = 0




    return action, grasped, lifted# once done with the throw, just chill in extended position




def process_image(rgb):
	# hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
	# lower_red = np.array([30,150,50])
	# upper_red = np.array([255,255,180])

	# mask = cv2.inRange(hsv, lower_red, upper_red)
	# res = cv2.bitwise_and(rgb,rgb, mask= mask)
	# print(res.shape)

	# return res

	hsv = cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV)

	#lower red
	lower_red = np.array([0,50,50])
	upper_red = np.array([10,255,255])


	#upper red
	lower_red2 = np.array([170,50,50])
	upper_red2 = np.array([180,255,255])

	mask = cv2.inRange(hsv, lower_red, upper_red)
	res = cv2.bitwise_and(rgb,rgb, mask= mask)

	mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
	res2 = cv2.bitwise_and(rgb,rgb, mask= mask2)

	img3 = res+res2 #res1 is doing all the work here
	
	np.save("img",img3)
	return img3
	

def save_image(rgb):
	
	imageio.imwrite('rgb.jpg', rgb)

def do_past_throw(environment):
    global file_ix
    # throw = random.choice(file_goal_pairs)
    throw = file_goal_pairs[file_ix]
    file_ix+=1
    #throw = file_goal_pairs[15]
        
    obs = tf.expand_dims(throw[2].astype('float32'),0)
    actions = tf.expand_dims(throw[3].astype('float32'),0)
    obs = tf.constant(obs)
    actions = tf.constant(actions)

    mu_enc, s_enc = encoder(obs, actions)
    encoder_normal = tfd.Normal(mu_enc,s_enc)
    z = encoder_normal.sample()

    if model_trainer.ARM_IN_GOAL:
        s_g = obs[-1][:-6]
    else:
        s_g = obs[-1][8:-6]

    print(actions)
	  


    s_g =   tf.expand_dims(tf.constant(s_g), axis = 0)
    z = tf.tile(z, [1, obs.shape[1]])
    z = tf.reshape(z, [-1, obs.shape[1], model_trainer.LATENT_DIM]) # so that both end up as BATCH, SEQ, DIM
    mu, scale, prob_weight, pdf, obs_pred, gripper= actor(obs,z,s_g)
    ai_actions = np.squeeze(pdf.sample())
    print(ai_actions)

    p.addUserDebugLine([0,0,0], list(throw[1])+[0.0], lifeTime = 5.0)
    
    p.addUserDebugText('o',list(throw[1])+[0.0], lifeTime = 5.0)

    
    actions = actions.numpy()
    actions= np.squeeze(actions)
    print(gripper.shape)
    print(np.expand_dims(np.squeeze(gripper),1).shape)
    print(ai_actions.shape)
    actions = np.concatenate((ai_actions, np.expand_dims(np.squeeze(gripper),1)), axis=1)
    for i in range(0,100): # go to starting point
        state, motor_action, reward, done, info = environment.step_to(actions[0,:])
    for a in range(0,len(actions)):
        print(actions[a,:])
        state, motor_action, reward, done, info = environment.step_to(actions[a,:], repeat = 2)
        
    cube_pos = get_scene_observation(environment.objects)[0:3]
    print(cube_pos)
    while  cube_pos[2] > 0.1:
        cube_pos = get_scene_observation(environment.objects)[0:3]
        time.sleep(environment._timeStep)
        # save obs, acts, cubepos0,1 into a tuple in another folder
        print(file_ix)
        try:
            play_sequence = 'sim_labelled_data/'
            file = throw[0].split('/')[-1]
            os.mkdir(play_sequence+file)
            np.save(play_sequence+file+'/obs',obs)
            np.save(play_sequence+file+'/act',actions)
            np.save(play_sequence+file+'/xyfinal',np.array([cube_pos[0], cube_pos[1]]))
            p.stepSimulation()
        except:
            print('already exists')

    if cube_pos[0] > 0.2:
        p.addUserDebugText('o',list(cube_pos), lifeTime = 3.0, textColorRGB =[1,0,0])
    

    #imageio.imwrite('rgb.jpg', rgb)
   





def move_in_xyz(environment, arm, abs_rel = 'abs'):

    motorsIds = []

    dv = 0.01
    abs_distance =  1.0

    #
    environment._arm.resetJointPoses()
    observation = environment._arm.getObservation()
    xyz = observation[-7:-4] 
    ori = p.getEulerFromQuaternion(observation[-4:])

    if abs_rel == 'abs': 

        if arm == 'ur5':
            xin = xyz[0]
            yin = xyz[1]
            zin = xyz[2]
            rin = -0.6943#ori[0]

            pitchin = 1.587#ori[1]
            yawin = -0.694# ori[2]
            #rin = 1.57#
            #pitchin = -0.78 #
            #yawin = -0.78

        print(pitchin)
        print(rin)
        print(yawin)



        motorsIds.append(environment._p.addUserDebugParameter("X", -abs_distance, abs_distance, xin))
        motorsIds.append(environment._p.addUserDebugParameter("Y", -abs_distance, abs_distance, yin))
        motorsIds.append(environment._p.addUserDebugParameter("Z", -abs_distance, abs_distance, zin))
        motorsIds.append(environment._p.addUserDebugParameter("roll", -math.pi, math.pi, rin,))
        motorsIds.append(environment._p.addUserDebugParameter("pitch", -math.pi, math.pi, pitchin))
        motorsIds.append(environment._p.addUserDebugParameter("yaw", -math.pi, math.pi, yawin))

    motorsIds.append(environment._p.addUserDebugParameter("fingerAngle", 0, 1.5, .3))
    grasp_switch = environment._p.addUserDebugParameter("grasp", 0,2, 0)
    throw_switch = environment._p.addUserDebugParameter("throw", 0,2, 0)
    throw_range = environment._p.addUserDebugParameter("range", 0,5, 1)
    done = False


    grasped = 0
    lifted = 0
    throw_timeout  = 0


    while (not done):

        action = []

        # get state
        observation = environment._arm.getObservation()
        grip_img, rgb = gripper_camera(observation)
        
        img = process_image(rgb)
        save_image(img)
        xyz = observation[-7:-4] 
        ori = p.getEulerFromQuaternion(observation[-4:])
        grip = observation[6]
        print(grip)
        

        cube_pos = get_scene_observation(environment.objects)[0:3]
        if cube_pos[2] < 0.0: #reset cube if it falls below table.
            p.resetBasePositionAndOrientation(environment.objects[0][0], [0,0,0], [0.0,0.0,0.000000,1.0])
            

        print(grasped, lifted)
        if xyz[2] < 0.15:
        	print('reset lift')
        	lifted = 0
        if not ( (  (xyz[0] - cube_pos[0]) < 0.02) or ( (xyz[1] - cube_pos[1]) <0.02) or ((xyz[2] - cube_pos[2]) < 0.15)):
            grasped = 0


        for motorId in motorsIds:
            # print(environment._p.readUserDebugParameter(motorId))
            action.append(environment._p.readUserDebugParameter(motorId))


        update_camera(environment)
        
        #environment._p.addUserDebugLine(environment._arm.endEffectorPos, [0, 0, 0], [1, 0, 0], 5)
        
        # action is xyz positon, orietnation quaternion, gripper closedness. 
        action = action[0:3] + list(p.getQuaternionFromEuler(action[3:6])) + [action[6]]

        if environment._p.readUserDebugParameter(throw_switch) > 1:
       		
       		action , grasped, lifted, throw_timeout = throw(environment, arm, action, xyz, ori, grip, cube_pos, grasped, lifted, throw_timeout, img, environment._p.readUserDebugParameter(throw_range))

       	elif environment._p.readUserDebugParameter(grasp_switch) > 1:
       		action , grasped, lifted = grasp_primitive(environment, arm, action, xyz, ori, grip, cube_pos, grasped, lifted, img)
       	

        state, motor_action, reward, done, info = environment.step_to(action, abs_rel)
        obs = environment.getSceneObservation()

##############################################################################################################

def setup_controllable_camera(environment):
    environment._p.addUserDebugParameter("Camera Zoom", -15, 15, 1.674)
    environment._p.addUserDebugParameter("Camera Pan", -360, 360, 70)
    environment._p.addUserDebugParameter("Camera Tilt", -360, 360, -50.8)
    environment._p.addUserDebugParameter("Camera X", -10, 10,0)
    environment._p.addUserDebugParameter("Camera Y", -10, 10,0)
    environment._p.addUserDebugParameter("Camera Z", -10, 10,0)


def update_camera(environment):
    if environment._renders:
        #Lets reserve the first 6 user debug params for the camera
        p.resetDebugVisualizerCamera(environment._p.readUserDebugParameter(0),
                                     environment._p.readUserDebugParameter(1),
                                     environment._p.readUserDebugParameter(2),
                                     [environment._p.readUserDebugParameter(3),
                                      environment._p.readUserDebugParameter(4),
                                      environment._p.readUserDebugParameter(5)])


def str_to_bool(string):
    if str(string).lower() == "true":
            string = True
    elif str(string).lower() == "false":
            string = False

    return string


def launch(render):
    arm = 'ur5'
    
    environment = graspingEnv(renders=str_to_bool(render), arm = arm)

    if environment._renders:
    	setup_controllable_camera(environment)

    
    move_in_xyz(environment, arm)


@click.command()
@click.option('--render', type=bool, default=True, help='rendering')

def main(**kwargs):
    launch(**kwargs)

if __name__ == "__main__":
    main()