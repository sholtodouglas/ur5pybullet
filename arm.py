# #
# physicsClient = p.connect(p.GUI) #p.direct for non GUI version
# p.setAdditionalSearchPath(pybullet_data.getDataPath()) #used by loadURDF
# p.setGravity(0,0,-10)
# planeId = p.loadURDF("plane.urdf")
# cubeStartPos = [0,0,4]
# cubeStartOrientation = p.getQuaternionFromEuler([0,0,0])
# boxId = p.loadURDF("r2d2.urdf",cubeStartPos, cubeStartOrientation)
# p.stepSimulation()
# cubePos, cubeOrn = p.getBasePositionAndOrientation(boxId)

# while cubePos[2] > 2:
# 	p.stepSimulation()
# 	cubePos, cubeOrn = p.getBasePositionAndOrientation(boxId)
# print(cubePos,cubeOrn)
# p.disconnect()

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


viewMatrix = p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition = [0,0,0], distance = 0.3, yaw = 90, pitch = -90, roll = 0, upAxisIndex = 2) 
projectionMatrix = p.computeProjectionMatrixFOV(fov = 120,aspect = 1,nearVal = 0.01,farVal = 10)

image_renderer = p.ER_BULLET_HARDWARE_OPENGL # if the rendering throws errors, use ER_TINY_RENDERER, but its hella slow cause its cpu not gpu.



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
    img = p.getCameraImage(200, 200, view_matrix_gripper, projectionMatrix,shadow=0, flags = p.ER_NO_SEGMENTATION_MASK, renderer=image_renderer)


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
        grip_img = gripper_camera(self._observation)
        obs = [self._observation, scene_obs]
        return obs, grip_img

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


    def step_to(self, action, abs_rel = 'abs', noise = False, clip = False):
        motor_poses = self._arm.move_to(action, abs_rel, noise, clip)
        #print(motor_poses) # these are the angles of the joints. 
        for i in range(self._actionRepeat):
            p.stepSimulation()
            if self._renders:
                time.sleep(self._timeStep)
            self._observation = self._arm.getObservation()
            
            self._envStepCounter += 1


        done = False #self._termination()
        reward = 0#self._reward()


        
        return np.array(self._observation), motor_poses, reward, done, {}


def move_in_xyz(environment, arm, abs_rel):

    motorsIds = []

    dv = 0.01
    abs_distance =  1.0

    #
    environment._arm.resetJointPoses()
    observation = environment._arm.getObservation()
    xyz = observation[-7:-4] 
    ori = p.getEulerFromQuaternion(observation[-4:])
    if abs_rel == 'abs': 
        print(arm)

        if arm == 'ur5':
            xin = xyz[0]
            yin = xyz[1]
            zin = xyz[2]
            rin = ori[0]
            pitchin = ori[1]
            yawin = ori[2]
        else:
            xin = 0.537
            yin = 0.0
            zin = 0.5
            rin = math.pi/2
            pitchin = -math.pi/2
            yawin = 0

        motorsIds.append(environment._p.addUserDebugParameter("X", -abs_distance, abs_distance, xin))
        motorsIds.append(environment._p.addUserDebugParameter("Y", -abs_distance, abs_distance, yin))
        motorsIds.append(environment._p.addUserDebugParameter("Z", -abs_distance, abs_distance, zin))
        motorsIds.append(environment._p.addUserDebugParameter("roll", -math.pi, math.pi, rin,))
        motorsIds.append(environment._p.addUserDebugParameter("pitch", -math.pi, math.pi, pitchin))
        motorsIds.append(environment._p.addUserDebugParameter("yaw", -math.pi, math.pi, yawin))

    else:
        motorsIds.append(environment._p.addUserDebugParameter("dX", -dv, dv, 0))
        motorsIds.append(environment._p.addUserDebugParameter("dY", -dv, dv, 0))
        motorsIds.append(environment._p.addUserDebugParameter("dZ", -dv, dv, 0))
        if arm == 'rbx1':
            motorsIds.append(environment._p.addUserDebugParameter("wrist_rotation", -0.1, 0.1, 0))
            motorsIds.append(environment._p.addUserDebugParameter("wrist_flexsion", -0.1, 0.1, 0))
        else:
            motorsIds.append(environment._p.addUserDebugParameter("roll", -dv, dv, 0))
            motorsIds.append(environment._p.addUserDebugParameter("pitch", -dv, dv, 0))
            motorsIds.append(environment._p.addUserDebugParameter("yaw", -dv, dv, 0))
    motorsIds.append(environment._p.addUserDebugParameter("fingerAngle", 0, 1.5, .3))

    done = False
    while (not done):

        action = []

        for motorId in motorsIds:
            # print(environment._p.readUserDebugParameter(motorId))
            action.append(environment._p.readUserDebugParameter(motorId))


        update_camera(environment)
        
        #environment._p.addUserDebugLine(environment._arm.endEffectorPos, [0, 0, 0], [1, 0, 0], 5)
        
        # action is xyz positon, orietnation quaternion, gripper closedness. 
        action = action[0:3] + list(p.getQuaternionFromEuler(action[3:6])) + [action[6]]
       
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





def setup_controllable_motors(environment, arm):
    

    possible_range = 3.2  # some seem to go to 3, 2.5 is a good rule of thumb to limit range.
    motorsIds = []

    for tests in range(0, environment._arm.numJoints):  # motors

        jointInfo = p.getJointInfo(environment._arm.uid, tests)
        #print(jointInfo)
        qIndex = jointInfo[3]

        if arm == 'kuka':
            if qIndex > -1 and jointInfo[0] != 7:
        
                motorsIds.append(environment._p.addUserDebugParameter("Motor" + str(tests),
                                                              -possible_range,
                                                              possible_range,
                                                              0.0))
        else:
            motorsIds.append(environment._p.addUserDebugParameter("Motor" + str(tests),
                                                              -possible_range,
                                                              possible_range,
                                                              0.0))

    return motorsIds

def update_camera(environment):
    if environment._renders:
        #Lets reserve the first 6 user debug params for the camera
        p.resetDebugVisualizerCamera(environment._p.readUserDebugParameter(0),
                                     environment._p.readUserDebugParameter(1),
                                     environment._p.readUserDebugParameter(2),
                                     [environment._p.readUserDebugParameter(3),
                                      environment._p.readUserDebugParameter(4),
                                      environment._p.readUserDebugParameter(5)])



def send_commands_to_motor(environment, motorIds):

    done = False


    while (not done):
        action = []

        for motorId in motorIds:
            action.append(environment._p.readUserDebugParameter(motorId))
        print(action)
        state, reward, done, info = environment.step(action)
        obs = environment.getSceneObservation()
        update_camera(environment)


    environment.terminated = 1

def control_individual_motors(environment, arm):
    motorIds = setup_controllable_motors(environment, arm)
    send_commands_to_motor(environment, motorIds)



###################################################################################################
def make_dir(string):
    try:
        os.makedirs(string)
    except FileExistsError:
        pass # directory already exists



#####################################################################################

def str_to_bool(string):
    if str(string).lower() == "true":
            string = True
    elif str(string).lower() == "false":
            string = False

    return string



def launch(mode, arm, abs_rel, render):
    print(arm)
    
    environment = graspingEnv(renders=str_to_bool(render), arm = arm)

    if environment._renders:
        setup_controllable_camera(environment)

    print(mode)
    if mode == 'xyz':
            move_in_xyz(environment, arm, abs_rel)
    else:
        environment._arm.active = True

        control_individual_motors(environment, arm)



@click.command()
@click.option('--mode', type=str, default='xyz', help='motor: control individual motors, xyz: control xyz/rpw of gripper, demos: collect automated demos')
@click.option('--abs_rel', type=str, default='abs', help='absolute or relative positioning, abs doesnt really work with rbx1 yet')
@click.option('--arm', type=str, default='ur5', help='rbx1 or kuka')
@click.option('--render', type=bool, default=True, help='rendering')



def main(**kwargs):
    launch(**kwargs)

if __name__ == "__main__":
    main()

