import argparse

import imageio
import os
import numpy as np
import robosuite as suite
import robosuite.macros as macros

# Set the image convention to opencv so that the images are automatically rendered "right side up" when using imageio
# (which uses opencv convention)
macros.IMAGE_CONVENTION = "opencv"

if __name__ == "__main__":

    # create environment instance

    controller_configs = suite.load_controller_config(default_controller="OSC_POSE")
    env = suite.make(
        env_name="Lift",
        robots=["Panda"],             # load a Sawyer robot and a Panda robot
        gripper_types="default",                # use default grippers per robot arm
        has_renderer=True,                     
        use_camera_obs=True,                   # provide image observations to agent
        camera_names="sideview",               # use "agentview" camera for observations
        camera_heights=512,                     # set camera height
        camera_widths=512,                      # set camera width
        reward_shaping=True,                    # use a dense reward signal for learning
        controller_configs=controller_configs,  # use OSC controller for control
    )
    env.viewer.set_camera(camera_id=1)

    # reset the environment
    env.reset()

    for i in range(1000):
        action = np.random.randn(*env.action_spec[0].shape) * 2
        print(action)
        obs, reward, done, info = env.step(action)  # take action in the environment
        image = obs['sideview_image']  # retrieve the agentview image observation
        if i % 10 == 0:
            import imageio
            dir = "experiments/robot/robosuite/temp"
            if not os.path.exists(dir):
                os.makedirs(dir)
            imageio.imwrite(f"{dir}/image_{i}.png", image)
        env.render()  # render on display