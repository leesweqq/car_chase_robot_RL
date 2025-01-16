import gymnasium as gym
import numpy as np
import math
import pybullet as p
import os
import pybullet_data
import matplotlib.pyplot as plt

# Car class
class Car:
    def __init__(self, client):
        # Initialize vehicle
        self.client = client
        # Load vehicle model
        f_name = os.path.join(os.path.dirname(__file__), 'car.urdf')
        self.car = p.loadURDF(fileName=f_name,
                              basePosition=[-9, 0, 0.1],
                              physicsClientId=client)

        # Connect vehicle's joints
        self.steering_joints = [0, 2]  # Steering joints
        self.drive_joints = [1, 3, 4, 5]  # Wheel drive joints
        # Joint speed
        self.joint_speed = 0
        # Rolling friction and air resistance
        self.c_rolling = 0.2
        self.c_drag = 0.01
        # Throttle constant
        self.c_throttle = 20

    def get_ids(self):
        return self.car, self.client

    def apply_action(self, action):
        # Accept action, which includes throttle and steering angle
        throttle, steering_angle = action

        # Limit throttle and steering angle range
        throttle = min(max(throttle, 0), 1)
        steering_angle = max(min(steering_angle, 0.6), -0.6)

        # Set the target angle for steering joints
        p.setJointMotorControlArray(self.car, self.steering_joints,
                                    controlMode=p.POSITION_CONTROL,
                                    targetPositions=[steering_angle] * 2,
                                    physicsClientId=self.client)

        # Calculate vehicle's friction and mechanical resistance
        friction = -self.joint_speed * (self.joint_speed * self.c_drag +
                                        self.c_rolling)
        acceleration = self.c_throttle * throttle + friction
        # Each time step is 1/30 seconds
        self.joint_speed = self.joint_speed + 1/30 * acceleration
        if self.joint_speed < 0:
            self.joint_speed = 0

        # Set target speed for wheels
        p.setJointMotorControlArray(
            bodyUniqueId=self.car,
            jointIndices=self.drive_joints,
            controlMode=p.VELOCITY_CONTROL,
            targetVelocities=[self.joint_speed] * 4,
            forces=[1.2] * 4,
            physicsClientId=self.client)

    def get_observation(self):
        # Get the vehicle's position and orientation
        pos, ang = p.getBasePositionAndOrientation(self.car, self.client)
        ang = p.getEulerFromQuaternion(ang)
        ori = (math.cos(ang[2]), math.sin(ang[2]))
        pos = pos[:2]  # Only use x, y coordinates
        # Get the vehicle's velocity
        vel = p.getBaseVelocity(self.car, self.client)[0][0:2]
        # Combine position, orientation, and velocity as the observation
        observation = (pos + ori + vel)

        return observation

# Goal class
class Goal:
    def __init__(self, client, base):
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.loadURDF(fileName="r2d2.urdf",
                   basePosition=[base[0], base[1], 0.2],
                   physicsClientId=client)

# Simple driving environment
class CarRobotEnv(gym.Env):
    def __init__(self):
        # Initialize environment
        super(CarRobotEnv, self).__init__()
        # Define action space: throttle and steering
        self.action_space = gym.spaces.Box(
            low=np.array([0, -.6], dtype=np.float32),
            high=np.array([1, .6], dtype=np.float32)
        )
        # Define observation space: vehicle's position, velocity, etc.
        self.observation_space = gym.spaces.Box(
            low=np.array([-10, -10, -1, -1, -5, -5, -10, -10], dtype=np.float32),
            high=np.array([10, 10, 1, 1, 5, 5, 10, 10], dtype=np.float32)
        )
        # Set random number generator
        self.np_random, _ = gym.utils.seeding.np_random()

        # Connect to physics engine
        self.client = p.connect(p.GUI)
        
        # Set simulation time step
        p.setTimeStep(1/30, self.client)

        # Initialize car, goal, done state, etc.
        self.car = None
        self.goal = None
        self.done = False
        self.prev_dist_to_goal = None
        self.rendered_img = None
        self.render_rot_matrix = None
        self.reset()  # Reset environment

    def step(self, action):
        # Apply action to the car and update its state
        self.car.apply_action(action)
        p.stepSimulation()  # Execute a physics simulation step
        car_ob = self.car.get_observation()  # Get car's current observation
        
        # Calculate reward using the calculate_reward function
        reward, dist_to_goal = self.calculate_reward(car_ob)
        self.prev_dist_to_goal = dist_to_goal  # Update previous distance

        # Print the reward
        print(f"Reward: {reward}")

        # Assume no time limit, so `truncated` is set to False
        truncated = False

        # Combine vehicle observation and goal location as new observation
        ob = np.array(car_ob + self.goal, dtype=np.float32)

        # Return the necessary five values: observation, reward, done, truncated, info
        info = {}  # Additional information can be expanded if needed
        return ob, reward, self.done, truncated, info

    def calculate_reward(self, car_ob):
        # Calculate distance to the goal
        dist_to_goal = math.sqrt(((car_ob[0] - self.goal[0]) ** 2 +
                                (car_ob[1] - self.goal[1]) ** 2))
        
        # Calculate reward based on distance
        reward = abs(self.prev_dist_to_goal - dist_to_goal)  # Reward based on distance

        # Check if done: whether the car is out of bounds
        if (car_ob[0] >= 10 or car_ob[0] <= -10 or
            car_ob[1] >= 10 or car_ob[1] <= -10):
            self.done = True
            reward = 0  # Penalize the car if it goes out of bounds

        # Check if done: whether the car reaches the goal
        elif dist_to_goal < 1:
            self.done = True
            reward = 10  # High reward if the car reaches the goal
            print(f"Goal reached! Reward: {reward}")

        # Return the calculated reward
        return reward, dist_to_goal

    def seed(self, seed=None):
        # Set random seed
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def draw_boundary(self):
        # Set the size of the boundary
        x_range = 10  # x-axis range
        y_range = 10  # y-axis range
        z_height = 1  # Line height, set to 1 to show lines above the ground

        # Draw the boundary
        p.addUserDebugLine([x_range, y_range, z_height], [x_range, -y_range, z_height], lineColorRGB=[1, 0, 0], physicsClientId=self.client)
        p.addUserDebugLine([x_range, -y_range, z_height], [-x_range, -y_range, z_height], lineColorRGB=[1, 0, 0], physicsClientId=self.client)
        p.addUserDebugLine([-x_range, -y_range, z_height], [-x_range, y_range, z_height], lineColorRGB=[1, 0, 0], physicsClientId=self.client)
        p.addUserDebugLine([-x_range, y_range, z_height], [x_range, y_range, z_height], lineColorRGB=[1, 0, 0], physicsClientId=self.client)

    def reset(self, seed=None, options=None):
        # Reset simulation
        p.resetSimulation(self.client)
        p.setGravity(0, 0, -10)  # Set gravity

        # Reload plane and car
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.loadURDF(fileName="plane.urdf")
        self.car = Car(self.client)
        
        # Create boundaries
        self.draw_boundary()

        # Randomly set the goal position
        x = self.np_random.choice([i for i in range(0, 10)])
        y = self.np_random.choice([i for i in range(-9, 10)])

        self.goal = (x, y)
        self.done = False

        # Display the goal position in the environment
        Goal(self.client, self.goal)

        # Get the initial observation of the car
        car_ob = self.car.get_observation()

        # Calculate the distance from the car to the goal
        self.prev_dist_to_goal = math.sqrt(((car_ob[0] - self.goal[0]) ** 2 +
                                             (car_ob[1] - self.goal[1]) ** 2))

        # Return observation and other info
        obs = np.array(car_ob + self.goal, dtype=np.float32)
        info = {}  # Additional information
        return obs, info

    def close(self):
        # Close connection to the PyBullet simulation client
        p.disconnect(self.client)
