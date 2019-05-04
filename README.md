# ur5pybullet
UR5 sim in pybullet, with control via xyz and ori of the head, or individual motor control. Uses elements of a bunch of other sims and the kuka default example but has a very simple gripper / images rendered through GPU and array form motor actions so its a fair bit faster and thus works really well in VR. The GUI/your code interacts with the arm via the step and action functions, and you can get back the state of the world through the observation function - where you can choose whether to use a camera above the table, in the gripper or both.

Usage
xyz/ori - python arm.py --mode xyz
motors  - python arm.py --mode motors
