import os
import sys
import time

import imageio
import numpy as np
import torch
import gymnasium as gym
import gymnasium_robotics
import mujoco


def get_franka_env_params(cfg):
    # TODO: implement this
    return {}


def get_franka_env(cfg):
    # get the franka kitchen environment from the OpenAI gym
    env_params = get_franka_env_params(cfg)
    # things to initialize:
    # - initial robot position
    # - initial object position

    env = gym.make(
        "FrankaKitchen-v1", tasks_to_complete=["microwave", "kettle"], render_mode="human"
    )  # TODO: fill in the params
    env.reset()
    return env

def perform_franka_action(env, action):
    # use ik to get the next xpos of the robot
    # update the observation

    # action is 7dof end effector deltas (x, y, z, roll, pitch, yaw, gripper)

    pass


def refresh_obs(obs, env):
    pass

# ---------- Mujoco utils
# Load the model and create a data object
def get_model():
    xml = "experiments/robot/franka_kitchen/panda.xml"
    return mujoco.MjModel.from_xml_path(xml)
def get_data(model):
    return mujoco.MjData(model)

if __name__ == "__main__":
    # validate the inverse kinematics function
    get_franka_env({})

