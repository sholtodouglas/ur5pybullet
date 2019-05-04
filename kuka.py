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

import random
import pybullet_data


class kuka:

    def __init__(self, urdfRootPath=pybullet_data.getDataPath(), timeStep=0.01, vr = False):
        self.urdfRootPath = urdfRootPath
        self.timeStep = timeStep
        self.vr = vr

        self.maxForce = 200.
        self.fingerAForce = 6
        self.fingerBForce = 5.5
        self.fingerTipForce = 6
        self.useInverseKinematics = 1
        self.useSimulation = 1
        self.useNullSpace = 1
        self.useOrientation = 1
        self.endEffectorIndex = 6
        # lower limits for null space
        self.ll = [-.967, -2, -2.96, 0.19, -2.96, -2.09, -3.05]
        # upper limits for null space
        self.ul = [.967, 2, 2.96, 2.29, 2.96, 2.09, 3.05]
        # joint ranges for null space
        self.jr = [5.8, 4, 5.8, 4, 5.8, 4, 6]
        # restposes for null space
        self.rp = [0, 0, 0, 0.5 * math.pi, 0, -math.pi * 0.5 * 0.66, 0]
        # joint damping coefficents
        self.jd = [0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001,
                   0.00001, 0.00001, 0.00001]
        self.reset()

    def reset(self):
        
        #
        if self.vr:
            objects = [p.loadURDF("kuka_iiwa/model_vr_limits.urdf")]
        else:
            objects = p.loadSDF(os.path.join(self.urdfRootPath, "kuka_iiwa/kuka_with_gripper2.sdf"))
        self.uid = objects[0]
        # for i in range (p.getNumJoints(self.uid)):
        #  print(p.getJointInfo(self.uid,i))
        p.resetBasePositionAndOrientation(self.uid, [-0.100000, 0.000000, -0.130000],
                                          [0.000000, 0.000000, 0.000000, 1.000000])
        self.jointPositions = [0.006418, 0.413184, -0.011401, -1.589317, 0.005379, 1.137684, -0.006539, 0.000048,
                               -0.299912, 0.000000, -0.000043, 0.299960, 0.000000, -0.000200]
        self.numJoints = p.getNumJoints(self.uid)

        for jointIndex in range(self.numJoints):
            p.resetJointState(self.uid, jointIndex, self.jointPositions[jointIndex])
            p.setJointMotorControl2(self.uid, jointIndex, p.POSITION_CONTROL,
                                    targetPosition=self.jointPositions[jointIndex], force= 0)#self.maxForce)

        # self.trayUid = p.loadURDF(os.path.join(self.urdfRootPath,"tray/tray.urdf"), 0.640000,0.075000,-0.190000,0.000000,0.000000,1.000000,0.000000)
        self.endEffectorPos = [0.537, 0.5, 0.5]
        self.endEffectorOrn = [ math.pi/2, -math.pi, 0]
        self.endEffectorAngle = 0

        self.motorNames = []
        self.motorIndices = []

        for i in range(self.numJoints):
            jointInfo = p.getJointInfo(self.uid, i)
            # print(jointInfo)
            qIndex = jointInfo[3]
            if qIndex > -1 and jointInfo[0] != 7:  # 7 and 6 rotate on eachother, but 7 stops us from observing orientation well.
                # print("motorname")
                # print(jointInfo[1])
                self.motorNames.append(str(jointInfo[1]))
                self.motorIndices.append(i)

    def getActionDimension(self):
        if (self.useInverseKinematics):
            return len(self.motorIndices)
        return 6  # position x,y,z and roll/pitch/yaw euler angles of end effector

    def getObservationDimension(self):
        return len(self.getObservation())

    def setPosition(self, pos, quat):

        p.resetBasePositionAndOrientation(self.uid,pos,
                                          quat)

    def addGripper(self):
        objects = p.loadSDF("gripper/wsg50_one_motor_gripper_new_free_base.sdf")
        kuka_gripper = objects[0]
        print ("kuka gripper=")
        print(kuka_gripper)
        self._gripper = kuka_gripper

        p.resetBasePositionAndOrientation(kuka_gripper,[0.923103,-0.200000,1.250036],[-0.000000,0.964531,-0.000002,-0.263970])
        jointPositions=[ 0.000000, -0.011130, -0.206421, 0.205143, -0.009999, 0.000000, -0.010055, 0.000000 ]
        for jointIndex in range (p.getNumJoints(kuka_gripper)):
            p.resetJointState(kuka_gripper,jointIndex,jointPositions[jointIndex])
            p.setJointMotorControl2(kuka_gripper,jointIndex,p.POSITION_CONTROL,jointPositions[jointIndex],0)


        kuka_cid = p.createConstraint(self.uid,   6,  kuka_gripper,0,p.JOINT_FIXED, [0,0,0], [0,0,0.05],[0,0,0])

        pr2_cid2 = p.createConstraint(kuka_gripper,4,kuka_gripper,6,jointType=p.JOINT_GEAR,jointAxis =[1,1,1],parentFramePosition=[0,0,0],childFramePosition=[0,0,0])
        p.changeConstraint(pr2_cid2,gearRatio=-1, erp=0.5, relativePositionTarget=0, maxForce=100)

    def getObservation(self):
        observation = []
        state = p.getLinkState(self.uid, self.endEffectorIndex)
        pos = state[0]
        orn = state[1]
        euler = p.getEulerFromQuaternion(orn)

        observation.extend(list(pos))
        observation.extend(list(orn))

        joint_positions = list()
        joint_velocities = list()
        applied_torques = list()
        for jointIndex in range(self.numJoints):
            state = p.getJointState(self.uid, jointIndex)
            angle = state[0]
            dv = state[1]
            applied_torque = state[3]
            
            joint_positions.append(angle)
            joint_velocities.append(dv)
            applied_torques.append(applied_torque)


        # print(joint_positions_and_velocities)
        #print(np.round(np.array(applied_torques),2)[:7])

        #print(len(joint_positions), len(joint_velocities), len(observation))

        #print(np.array(joint_positions).shape, np.array(joint_velocities).shape, np.array(observation).shape)
        return joint_positions + joint_velocities + observation

    def action(self, motorCommands):
        # for the kuka, motor commands should be like this, 12 long
        # [ 0.006418, 0.413184, -0.011401, -1.589317, 0.005379, 1.137684, -0.006539, 0.000048, -0.299912, 0.000000, -0.000043, 0.299960]
        # there are 14 joints but only 12 motors?
        # zero indexed -
        # index 7 is end effector angle
        # index 8 is finger angle A
        # index 11 is finger angle B
        motorCommands = np.clip(motorCommands, self.ll, self.ul)

        #print(len(motorCommands))
        for action in range(len(motorCommands)):
            motor = self.motorIndices[action]
            #print(motor)


            p.setJointMotorControl2(self.uid, motor, p.POSITION_CONTROL, targetPosition=motorCommands[action],
                                    force=self.maxForce)

    def move_to(self, position_delta, mode = 'abs', noise = False, clip = False):

        #mode is either absolute or relative

        # print ("self.numJoints")
        # print (self.numJoints)

        

        if (self.useInverseKinematics):

            if mode == 'abs': #absolute positioning

                self.endEffectorPos  = [position_delta[0],position_delta[1],position_delta[2]]
                self.endEffectorOrn = [position_delta[3],position_delta[4],position_delta[5], position_delta[6]]
                pos = self.endEffectorPos 
                orn = self.endEffectorOrn
                fingerAngle = position_delta[7]

            else: #mode is relative

                ## this where where how much we move is extracted
                dx = position_delta[0]
                dy = position_delta[1]
                dz = position_delta[2]
                # da = position_delta[3]

                dr,dp,dyaw = p.getEulerFromQuaternion([position_delta[3], position_delta[4], position_delta[5], position_delta[6]])
                
                fingerAngle = position_delta[6]

                self.endEffectorOrn[0] = self.endEffectorOrn[0] + dr
                self.endEffectorOrn[1] = self.endEffectorOrn[1] + dp
                self.endEffectorOrn[2] = self.endEffectorOrn[2] + dyaw

                state = p.getLinkState(self.uid, self.endEffectorIndex)
                actualEndEffectorPos = state[0]
                actualEndEffectorOrn = state[1]
                
                self.endEffectorPos[0] = self.endEffectorPos[0] + dx
                self.endEffectorPos[1] = self.endEffectorPos[1] + dy
                if (dz > 0 or actualEndEffectorPos[2] > 0.10):
                    self.endEffectorPos[2] = self.endEffectorPos[2] + dz
                if (actualEndEffectorPos[2] < 0.10):
                    self.endEffectorPos[2] = self.endEffectorPos[2] + 0.0001

                # self.endEffectorAngle = self.endEffectorAngle + da
                pos = self.endEffectorPos
                # orn = p.getQuaternionFromEuler([0,-math.pi,0]) # -math.pi,yaw])
                orn = p.getQuaternionFromEuler(self.endEffectorOrn)  # -math.pi,yaw])
            # orn = actualEndEffectorOrn

            if (self.useNullSpace == 1):
                if (self.useOrientation == 1):
                    jointPoses = p.calculateInverseKinematics(self.uid, self.endEffectorIndex, pos, orn,
                                                              self.ll, self.ul, self.jr, self.rp)
                else:
                    jointPoses = p.calculateInverseKinematics(self.uid, self.endEffectorIndex, pos,
                                                              lowerLimits=self.ll, upperLimits=self.ul,
                                                              jointRanges=self.jr, restPoses=self.rp)
            else:
                if (self.useOrientation == 1):
                    jointPoses = p.calculateInverseKinematics(self.uid, self.endEffectorIndex, pos, orn,
                                                              jointDamping=self.jd)
                else:
                    jointPoses = p.calculateInverseKinematics(self.uid, self.endEffectorIndex, pos)

            #print("jointPoses")
            #print(jointPoses)
            #print(len(jointPoses))
            # print("self.endEffectorIndex")
            # print(self.endEffectorIndex)

            jointPoses = np.array(jointPoses)

            if clip == True:
                for i in range(self.endEffectorIndex + 1):
                    state = p.getJointState(self.uid,i )
                    current_angle = state[0]
                    difference = jointPoses[i] - current_angle
                    clipped_diff = np.clip(difference, -0.1, 0.1)
                    jointPoses[i] = current_angle + clipped_diff

            true_desired_positions = jointPoses

            if noise == True:

                noise_factor  = random.random()*0.5 #noise amount 
                #print(noise_factor)
                #print(jointPoses)
                # cause its range 0-1, centered, scaled.
                jointPoses = jointPoses + (np.random.rand(len(jointPoses))-0.5)*noise_factor
                #print(jointPoses)




            if (self.useSimulation):
                for i in range(self.endEffectorIndex + 1):
                    #print(i)
                    p.setJointMotorControl2(bodyIndex=self.uid, jointIndex=i, controlMode=p.POSITION_CONTROL,
                                            targetPosition=jointPoses[i], targetVelocity=0, force=self.maxForce,
                                            positionGain=0.03, velocityGain=1)
            else:
                # reset the joint state (ignoring all dynamics, not recommended to use during simulation)
                for i in range(self.numJoints):
                    p.resetJointState(self.uid, i, jointPoses[i])
            # fingers
            if not self.vr:

                # p.setJointMotorControl2(self.uid,7,p.POSITION_CONTROL,targetPosition=self.endEffectorAngle,force=self.maxForce)
                p.setJointMotorControl2(self.uid, 8, p.POSITION_CONTROL, targetPosition=-fingerAngle,
                                        force=self.fingerAForce)
                p.setJointMotorControl2(self.uid, 11, p.POSITION_CONTROL, targetPosition=fingerAngle,
                                        force=self.fingerBForce)

                p.setJointMotorControl2(self.uid, 10, p.POSITION_CONTROL, targetPosition=0, force=self.fingerTipForce)
                p.setJointMotorControl2(self.uid, 13, p.POSITION_CONTROL, targetPosition=0, force=self.fingerTipForce)


            #joint poses are the actions to be sent to the motors, i.e, what the action of the neural net will be.
            return true_desired_positions[:7] # Only return the true joint poses desired, not noise injected.
            #TODO LATER ALSO RETURN FINGER ANGLE
           


    