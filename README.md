# ur5pybullet
UR5 sim in pybullet, with control via xyz and ori of the head, or individual motor control. Uses elements of a bunch of other sims and the kuka default example but has a very simple gripper / images rendered through GPU and array form motor actions so its a fair bit faster and thus works really well in VR.

Usage

xyz/ori: python arm.py --mode xyz

motors: python arm.py --mode motors
