import os, inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
print("current_dir=" + currentdir)
os.sys.path.insert(0, currentdir)

import math
import gym
import sys
from gym import spaces
from gym.utils import seeding
import numpy as np
import time
import pybullet as p
from itertools import chain
from collections import deque

import random
import pybullet_data
from collections import namedtuple
from attrdict import AttrDict
import functools
import itertools

def setup_sisbot(p, uid):
    # controlJoints = ["shoulder_pan_joint","shoulder_lift_joint",
    #                  "elbow_joint", "wrist_1_joint",
    #                  "wrist_2_joint", "wrist_3_joint",
    #                  "robotiq_85_left_knuckle_joint"]
    controlJoints = ["shoulder_pan_joint","shoulder_lift_joint",
                     "elbow_joint", "wrist_1_joint",
                     "wrist_2_joint", "wrist_3_joint", 'left_gripper_motor', 'right_gripper_motor']

    jointTypeList = ["REVOLUTE", "PRISMATIC", "SPHERICAL", "PLANAR", "FIXED"]
    numJoints = p.getNumJoints(uid)
    jointInfo = namedtuple("jointInfo", 
                           ["id","name","type","lowerLimit","upperLimit","maxForce","maxVelocity","controllable"])
    joints = AttrDict()
    for i in range(numJoints):
        info = p.getJointInfo(uid, i)
        jointID = info[0]
        jointName = info[1].decode("utf-8")
        jointType = jointTypeList[info[2]]
        jointLowerLimit = info[8]
        jointUpperLimit = info[9]
        jointMaxForce = info[10]
        jointMaxVelocity = info[11]
        controllable = True if jointName in controlJoints else False
        info = jointInfo(jointID,jointName,jointType,jointLowerLimit,
                         jointUpperLimit,jointMaxForce,jointMaxVelocity,controllable)
        if info.type=="REVOLUTE": # set revolute joint to static
            p.setJointMotorControl2(uid, info.id, p.VELOCITY_CONTROL, targetVelocity=0, force=0)
        joints[info.name] = info
    controlRobotiqC2 = False
    mimicParentName = False
    return joints, controlRobotiqC2, controlJoints, mimicParentName



class ur5:

    def __init__(self, urdfRootPath=pybullet_data.getDataPath(), timeStep=0.01, vr = False):



        self.robotUrdfPath = "./urdf/ur5.urdf"
        self.robotStartPos = [0.0,0.0,0.3]
        self.robotStartOrn = p.getQuaternionFromEuler([1.885,1.786,0.132])

        self.xin = self.robotStartPos[0]
        self.yin = self.robotStartPos[1]
        self.zin = self.robotStartPos[2]
        
        self.reset()

    def reset(self):
        
        print("----------------------------------------")
        print("Loading robot from {}".format(self.robotUrdfPath))
        self.uid = p.loadURDF(os.path.join(os.getcwd(),self.robotUrdfPath), self.robotStartPos, self.robotStartOrn, 
                             flags=p.URDF_USE_INERTIA_FROM_FILE)
        self.joints, self.controlRobotiqC2, self.controlJoints, self.mimicParentName = setup_sisbot(p, self.uid)
        self.endEffectorIndex = 7 # ee_link
        self.numJoints = p.getNumJoints(self.uid)
        self.active_joint_ids = []
        for i, name in enumerate(self.controlJoints):
            joint = self.joints[name]
            self.active_joint_ids.append(joint.id)



    def getActionDimension(self):
        # if (self.useInverseKinematics):
        #     return len(self.motorIndices)
        return 8  # position x,y,z and ori quat and finger angle
    def getObservationDimension(self):
        return len(self.getObservation())

    def setPosition(self, pos, quat):

        p.resetBasePositionAndOrientation(self.uid,pos,
                                          quat)

    def resetJointPoses(self):
                # move to this ideal init point
        for i in range(0,1000):
            self.action([0.0, 1.2105262279510498, -1.9736841917037964, -0.9210526943206787, 1.526315689086914, 0.0, -0.10526323318481445, 0.0, 0.0, 0.0])





    def getObservation(self):
        observation = []
        state = p.getLinkState(self.uid, self.endEffectorIndex, computeLinkVelocity = 1)
       #print('state',state)
        pos = state[0]
        orn = state[1]
        


        observation.extend(list(pos))
        observation.extend(list(orn))

        joint_states = p.getJointStates(self.uid, self.active_joint_ids)
        
        joint_positions = list()
        joint_velocities = list()
        

        for joint in joint_states:
            
            joint_positions.append(joint[0])
            joint_velocities.append(joint[1])
            

  
        return joint_positions + joint_velocities + observation


    def action(self, motorCommands):
        print(motorCommands)
        
        poses = []
        indexes = []
        forces = []

        for i, name in enumerate(self.controlJoints):
            joint = self.joints[name]

            poses.append(motorCommands[i])
            indexes.append(joint.id)
            forces.append(joint.maxForce)
        l = len(poses)


        p.setJointMotorControlArray(self.uid, indexes, p.POSITION_CONTROL, targetPositions=poses, targetVelocities =[0]*l, positionGains = [0.03]*l, forces = forces)
        #holy shit this is so much faster in arrayform!


    

    def move_to(self, position_delta, mode = 'abs', noise = False, clip = False):

        #at the moment UR5 only absolute

        x = position_delta[0]
        y = position_delta[1]
        z = position_delta[2]
        
        orn = position_delta[3:7]
        finger_angle = position_delta[7]
        jointPose = list(p.calculateInverseKinematics(self.uid, self.endEffectorIndex, [x,y,z], orn))

        # print(jointPose)
        # print(self.getObservation()[:len(self.controlJoints)]) ## get the current joint positions
        jointPose[-1] = -finger_angle/25 
        jointPose[-2] = finger_angle/25
        
        self.action(jointPose)
        print(jointPose)
        return jointPose

