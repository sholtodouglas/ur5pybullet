import pybullet as p
from kuka import kuka
from ur5 import ur5
import pybullet_data
import math 
import os
urdfRoot=pybullet_data.getDataPath()
meshPath = os.getcwd()+"/meshes/objects/"
print(meshPath)
up_rot = p.getQuaternionFromEuler([-math.pi/2, math.pi,0]) # the transform from Z-Y axis up. Most meshes are Z up.

def load_arm_dim_up(arm, dim = 'Z'):
	if arm == 'ur5':
		_arm = ur5()
	elif arm == 'kuka':
		_arm = kuka(urdfRootPath=urdfRoot, timeStep=1/240, vr = True)
	elif arm == 'rbx1':
		_arm = rbx1(urdfRootPath=self._urdfRoot, timeStep=self._timeStep)
	else:
		raise Exception('Not a valid arm') 
	if dim == 'Y':
		arm_rot = p.getQuaternionFromEuler([-math.pi/2, (1/2)*math.pi,0])
		_arm.setPosition([0,-0.1,0.5], [arm_rot[0],arm_rot[1],arm_rot[2],arm_rot[3]])
	else:
		arm_rot =p.getQuaternionFromEuler([0,0.0,0])
		_arm.setPosition([-0.5,0.0,-0.1], [arm_rot[0],arm_rot[1],arm_rot[2],arm_rot[3]])
	return _arm


def load_play_Z_up():

	lego0 = [p.loadURDF((os.path.join(urdfRoot, "lego/lego.urdf")), 0.700000,-0.200000,0.700000,0.000000,0.000000,0.000000,1.000000)]
	lego1 = [p.loadURDF((os.path.join(urdfRoot, "lego/lego.urdf")), 0.700000,-0.200000,0.800000,0.000000,0.000000,0.000000,1.000000)]
	lego2 = [p.loadURDF((os.path.join(urdfRoot, "lego/lego.urdf")), 0.700000,-0.200000,0.900000,0.000000,0.000000,0.000000,1.000000)]
	lego3 = [p.loadURDF((os.path.join(urdfRoot, "lego/lego.urdf")),0.800000,-0.200000,0.600000,0.000000,0.000000,0.000000,1.000000)]
	lego4 = [p.loadURDF((os.path.join(urdfRoot, "lego/lego.urdf")), 0.800000,-0.200000,0.500000,0.000000,0.000000,0.000000,1.000000)]
	lego5 = [p.loadURDF((os.path.join(urdfRoot, "lego/lego.urdf")), 0.800000,-0.200000,0.7500000,0.000000,0.000000,0.000000,1.000000)]
	lego6 = [p.loadURDF((os.path.join(urdfRoot, "lego/lego.urdf")), 0.800000,-0.30000,0.8500000,0.000000,0.000000,0.000000,1.000000)]
	objects = [p.loadURDF((os.path.join(urdfRoot, "lego/lego.urdf")), 0.800000,-0.300000,0.800000,0.000000,0.000000,0.000000,1.000000)]
	objects = [p.loadURDF((os.path.join(urdfRoot, "lego/lego.urdf")), 0.800000,-0.300000,0.900000,0.000000,0.000000,0.000000,1.000000)]
	objects = [p.loadURDF((os.path.join(urdfRoot,"teddy_vhacd.urdf")), -0.100000,0.600000,0.850000,0.000000,0.000000,0.000000,1.000000)]
	objects = [p.loadURDF((os.path.join(urdfRoot,"sphere_small.urdf")), -0.100000,0.955006,1.169706,0.633232,-0.000000,-0.000000,0.773962)]
	objects = [p.loadURDF((os.path.join(urdfRoot,"cube_small.urdf")), 0.300000,0.600000,0.850000,0.000000,0.000000,0.000000,1.000000)]
	jenga1 = [p.loadURDF((os.path.join(urdfRoot,"jenga/jenga.urdf")), 1.300000,-0.700000,0.750000,0.000000,0.707107,0.000000,0.707107)]
	jenga2 = [p.loadURDF((os.path.join(urdfRoot,"jenga/jenga.urdf")), 1.200000,-0.700000,0.750000,0.000000,0.707107,0.000000,0.707107)]
	jenga3 = [p.loadURDF((os.path.join(urdfRoot,"jenga/jenga.urdf")), 1.100000,-0.700000,0.7510000,0.000000,0.707107,0.000000,0.707107)]
	jenga4 = [p.loadURDF((os.path.join(urdfRoot,"jenga/jenga.urdf")), 0.800000,-0.700000,0.750000,0.000000,0.707107,0.000000,0.707107)]
	jenga5 = [p.loadURDF((os.path.join(urdfRoot,"jenga/jenga.urdf")), 0.700000,-0.700000,0.750000,0.000000,0.707107,0.000000,0.707107)]
	jenga6 = [p.loadURDF((os.path.join(urdfRoot,"jenga/jenga.urdf")), 0.500000,-0.700000,0.750000,0.000000,0.707107,0.000000,0.707107)] 
	objects = [p.loadURDF((os.path.join(urdfRoot,"table/table.urdf")), 0.4000000,-0.200000,0.000000,0.000000,0.000000,0.707107,0.707107)]
	objects = [p.loadURDF((os.path.join(urdfRoot, "plane.urdf")), 0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,1.000000)]
	jenga7 = [p.loadURDF((os.path.join(urdfRoot, "jenga/jenga.urdf")), 1.300000,-0.700000,0.750000,0.000000,0.707107,0.000000,0.707107)]
	jenga8 = [p.loadURDF((os.path.join(urdfRoot, "jenga/jenga.urdf")), 1.200000,-0.700000,0.850000,0.000000,0.707107,0.000000,0.707107)]
	jenga9 = [p.loadURDF((os.path.join(urdfRoot, "jenga/jenga.urdf")), 1.100000,-0.700000,0.950000,0.000000,0.707107,0.000000,0.707107)]
	jenga10 = [p.loadURDF((os.path.join(urdfRoot, "jenga/jenga.urdf")), 1.000000,-0.700000,0.750000,0.000000,0.707107,0.000000,0.707107)]
	objects = [p.loadURDF((os.path.join(urdfRoot, "jenga/jenga.urdf")), 0.900000,-0.700000,0.750000,0.000000,0.707107,0.000000,0.707107)]


def load_lab_Y_up():
	jenga = [p.loadURDF((os.path.join(urdfRoot,"jenga/jenga.urdf")), 0,0.4,0,0.000000,0.707107,0.000000,0.707107)]
	table = [p.loadURDF((os.path.join(urdfRoot,"table/table.urdf")), 0.0,-0.70000,0.000000,up_rot[0],up_rot[1],up_rot[2],up_rot[3])]
	teddy_rot =  p.getQuaternionFromEuler([-math.pi/2, math.pi,-math.pi/4])
	
	block_red = [p.loadURDF((os.path.join(meshPath,"block.urdf")), -0.150000,0.100000,0.10000,0,0,0,0)]
	plate = [p.loadURDF((os.path.join(meshPath,"plate.urdf")), 0.100000,0.200000,0.10000,0,0,0,0)]
	#cup = [p.loadURDF((os.path.join(meshPath,"cup/cup_small.urdf")), 0.100000,0.200000,0.10000,0,0,0,0)]
	block_blue = [p.loadURDF((os.path.join(meshPath,"block_blue.urdf")), -0.000000,0.400000,0.10000,0,0,0,0)]
	return [jenga, block_red, block_blue, plate]

def load_lab_Z_up():
	jenga = [p.loadURDF((os.path.join(urdfRoot,"jenga/jenga.urdf")), 0,0.0,0,0.400000,0.707107,0.000000,0.707107)]
	table = [p.loadURDF((os.path.join(urdfRoot,"table/table.urdf")), 0.0,0.0,-0.70000,0.000000,0.000000,0.707107,0.707107)]
	block_red = [p.loadURDF((os.path.join(meshPath,"block.urdf")), 0.05,0.0,0,0.400000,0.707107,0.000000,0.707107)]
	plate = [p.loadURDF((os.path.join(meshPath,"plate.urdf")), 0.05,0.1,0,0.800000,0.0,0.700000,0.7)]
	#cup = [p.loadURDF((os.path.join(meshPath,"cup/cup_small.urdf")), 0.100000,0.200000,0.10000,0,0,0,0)]
	block_blue = [p.loadURDF((os.path.join(meshPath,"block_blue.urdf")), -0.05,0.0,0,0.400000,0.707107,0.000000,0.707107)]
	return [jenga, block_red, block_blue, plate]

def throwing_scene():
	block_red = [p.loadURDF((os.path.join(meshPath,"cube_small.urdf")), 0.0,0.0,0.05,0.400000,0.0,0.000000,1.0)]
	table = [p.loadURDF((os.path.join(urdfRoot,"table/table.urdf")), 0.0,0.0,-0.6300,0.000000,0.000000,0.0,1.0)]
	return [block_red]


def load_gripper():
	objects = [p.loadURDF((os.path.join(urdfRoot,"pr2_gripper.urdf")), 0.500000,0.300006,0.700000,-0.000000,-0.000000,-0.000031,1.000000)]
	pr2_gripper = objects[0]
	print ("pr2_gripper=")
	print (pr2_gripper)

	jointPositions=[ 0.550569, 0.000000, 0.549657, 0.000000 ]
	for jointIndex in range (p.getNumJoints(pr2_gripper)):
	    p.resetJointState(pr2_gripper,jointIndex,jointPositions[jointIndex])

	pr2_cid = p.createConstraint(pr2_gripper,-1,-1,-1,p.JOINT_FIXED,[0,0,0],[0.2,0,0],[0.500000,0.300006,0.700000])
	print ("pr2_cid")
	print (pr2_cid)
	return pr2_gripper, pr2_cid


def disable_scene_render():
	p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)

def enable_scene_render():
	p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)


def get_scene_observation(objects):
	observation = []
	for o in objects:
		pos, orn = p.getBasePositionAndOrientation(o[0])
		observation.extend(list(pos))
		observation.extend(list(orn))
	return observation



