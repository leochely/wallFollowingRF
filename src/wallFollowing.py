#!/usr/bin/env python
import rospy
import numpy as np
import random 
import time
import sys
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Pose2D
from std_srvs.srv import Empty
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import GetModelState, SetModelState

pub_ = rospy.Publisher('/triton_lidar/vel_cmd', Pose2D, queue_size=2)
reset_sim = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)
get_pose = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
set_pose = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
reset_sim()
qTable = np.ones((4, 2, 5, 2, 3))
epsilon = 0.9
d = 0.985
action = 0
episode = 1
states = [(1,1,1,1,0)]
positions = []
last_time = time.time()
time_difference = 0.2
steps = 0

def check_pos():
	global episode
	global positions
	global states
	global action	
	global steps
	msg = get_pose("triton_lidar", "").pose.position
	if msg.z > 0.1:
		print("terminated because started to fly")
		episode += 1
		print("New episode: ", episode) 
		reset_sim()
		s = ModelState()
		s.model_name = 'triton_lidar'
		s.pose.position.x = random.randrange(0,7)/2 - 2.5
		s.pose.position.y = random.randrange(0,7)/2 - 2.5
		set_pose(s)
	positions += [msg]
	try:
		if abs(positions[-1].x-positions[-2].x) < 0.01 and abs(positions[-1].x-msg.x) < 0.01 and abs(positions[-1].y-positions[-2].y) < 0.01 and abs(positions[-1].y-msg.y) < 0.01:
			episode += 1
			reset_sim()
			s = ModelState()
			s.model_name = 'triton_lidar'
			s.pose.position.x = random.randrange(0,7)/2 - 2.5
			s.pose.position.y = random.randrange(0,7)/2 - 2.5
			set_pose(s)
			states = [(1,1,1,1,0)]
			action = 0
			steps = 0
			print("New episode: ", episode) 
			print(qTable)
			np.save('table.npy', qTable)
	except Exception:
		pass

def clbk_laser(msg):
	global last_time
	if time.time() - last_time < time_difference:
		return
	last_time = time.time()
	frontValues = msg.ranges[60:120]
	dminF = min(frontValues)

	rightFrontValues = msg.ranges[30:60]
	dminRF = min(rightFrontValues)

	rightValues = msg.ranges[0:60]
	dminR = min(rightValues)

	leftValues = msg.ranges[120:180] 
	dminL = min(leftValues)	

	if dminF < 0.5:
		front = 0
	elif dminF < 0.6:
		front = 1
	elif dminF < 1.2:
		front = 2
	else:
		front = 3

	if dminRF < 1.2:
		rightFront = 0
	else:
		rightFront = 1	

	if dminR < 0.5:
		right = 0
	elif dminR < 0.6:
		right = 1
	elif dminR < 0.8:
		right = 2
	elif dminR < 1.2:
		right = 3
	else:
		right = 4

	if dminL < 0.5:
		left = 0
	else:
		left = 1
	
	print("Front:{} Right-Front:{} Right:{} Left:{}".format(dminF, dminRF, dminR, dminL))

	choose_action(front, rightFront, right, left)

def choose_action(front, rightFront, right, left):
	action = qTable[(front, rightFront, right, left)].argmax()
	execute_action(action)

def clbk_laser_traning(msg):
	global last_time
	global states
	global action
	global steps
	if time.time() - last_time < time_difference:
		return
	check_pos()
	if steps == 10000:
		print("This robot is now trained")
		exit() 
	steps += 1
	last_time = time.time()
	frontValues = msg.ranges[60:120]
	dminF = min(frontValues)

	rightFrontValues = msg.ranges[30:60]
	dminRF = min(rightFrontValues)

	rightValues = msg.ranges[0:60]
	dminR = min(rightValues)

	leftValues = msg.ranges[120:180] 
	dminL = min(leftValues)	

	if dminF < 0.5:
		front = 0
	elif dminF < 0.6:
		front = 1
	elif dminF < 1.2:
		front = 2
	else:
		front = 3

	if dminRF < 1.2:
		rightFront = 0
	else:
		rightFront = 1	

	if dminR < 0.5:
		right = 0
	elif dminR < 0.6:
		right = 1
	elif dminR < 0.8:
		right = 2
	elif dminR < 1.2:
		right = 3
	else:
		right = 4

	if dminL < 0.5:
		left = 0
	else:
		left = 1
	
	if left == 0 or right == 0 or right==3 or right == 4 or front == 0:
		reward = -1
	else:
		reward = 0
	print("Front:{} Right-Front:{} Right:{} Left:{} for reward: {}".format(dminF, dminRF, dminR, dminL, reward))
	try: 
		old = qTable[states[-1]]
		alpha = 0.2
		gamma = 0.8
		qTable[states[-1]] = old + alpha * (reward + gamma * max(qTable[(front, rightFront, right, left)]) - old)
	except Exception:
		pass

	choose_action_training(front, rightFront, right, left)
	

def choose_action_training(front, rightFront, right, left):
	global action
	global states
	e = epsilon * pow(d, episode)
	print(e)
	proba = random.uniform(0,1)
	if proba < e:
		action = random.randint(0,2)
		print("Random decision")
	else:
		action = qTable[(front, rightFront, right, left)].argmax()
		print("Chosen decision")
	states += [(front, rightFront, right, left, action)]
	execute_action(action)

def execute_action(a):
	if a == 0:
		turn_left()
	elif a == 1:
		turn_right()
	else:
		forward()

def turn_left():
	print("left")
	msg = Pose2D()
	msg.x = 0	
	msg.y = 0.3
	msg.theta = 0.78
	pub_.publish(msg)
	return msg


def turn_right():
	print("right")
	msg = Pose2D()
	msg.x = 0
	msg.y = 0.3
	msg.theta = -0.78
	pub_.publish(msg)
	return msg

def forward():
	print("forward")
	msg = Pose2D()
	msg.x = 0
	msg.y = 0.3
	msg.theta = 0
	pub_.publish(msg)
	return msg


def main(mode):
	rospy.init_node('wall_following')
	global qTable
	if mode == "training":
		rospy.Subscriber('/scan', LaserScan, clbk_laser_traning)
	elif mode == "trained":
		qTable = np.load('./table.npy')
		rospy.Subscriber('/scan', LaserScan, clbk_laser)
	else:
		print("Invalid argument: please use trained or training")
	rospy.spin()


if __name__ == "__main__":
	mode = sys.argv[1]
	main(mode)

