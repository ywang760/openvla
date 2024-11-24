import os
import sys
import time

import robosuite as suite
import robosuite.macros as macros

macros.IMAGE_CONVENTION = "opencv"


camera_name = "sideview"


def get_robosuite_env(cfg):
    controller_configs = suite.load_controller_config(default_controller="OSC_POSE")
    env = suite.make(
        env_name="Stack",
        robots=["Panda"],
        gripper_types="default",
        has_renderer=True,
        use_camera_obs=True,
        camera_names=camera_name,
        camera_heights=512,
        camera_widths=512,
        controller_configs=controller_configs, 
    )

    return env

def refresh_obs(obs, env):
    obs["full_image"] = obs[f"{camera_name}_image"] # TODO: make this a config option
    obs["robot_state"] = obs["robot0_proprio-state"]
    return obs