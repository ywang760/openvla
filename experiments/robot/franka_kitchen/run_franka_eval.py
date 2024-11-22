import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Union
from spatialmath import SE3
import draccus
import roboticstoolbox as rtb

sys.path.append(".")
from experiments.robot.franka_kitchen.kitchenv1_utils import (
    get_franka_env,
    perform_franka_action
)
from experiments.robot.bridge.bridgev2_utils import (
    get_preprocessed_image,
    save_rollout_data,
    save_rollout_video,
)
from experiments.robot.openvla_utils import get_processor
from experiments.robot.robot_utils import (
    get_action,
    get_image_resize_size,
    get_model,
)
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Assign the VLM to GPU 1


@dataclass
class GenerateConfig:

    # Model-specific parameters
    model_family: str = "openvla"
    pretrained_checkpoint: Union[str, Path] = "openvla/openvla-7b"

    # Precision: default is bf16
    load_in_8bit: bool = False
    load_in_4bit: bool = True

    center_crop: bool = False

    # Environment-specific parameters
    # TODO: Add more environment-specific parameters
    max_episodes: int = 50
    max_steps: int = 60
    control_frequency: float = 5

    # Utils
    save_data: bool = True


@draccus.wrap()
def main(cfg: GenerateConfig) -> None:
    assert cfg.pretrained_checkpoint, "cfg.pretrained_checkpoint must be set."
    assert not cfg.center_crop, "`center_crop` should be disabled."

    # [OpenVLA] Set action un-normalization key
    cfg.unnorm_key = "nyu_franka_play_dataset_converted_externally_to_rlds"

    # Load model and processor
    model = get_model(cfg)
    processor = None
    if cfg.model_family == "openvla":
        processor = get_processor(cfg)

    # Get environment
    env = get_franka_env(cfg)
    resize_size = get_image_resize_size(cfg)

    # Start evaluation
    task_label = "Open the microwave"
    episode_idx = 0
    while episode_idx < cfg.max_episodes:
        # task_label = get_next_task_label(task_label)

        # Reset environment
        obs, _ = env.reset()

        # Setup
        t = 0
        step_duration = 1.0 / cfg.control_frequency
        replay_images = []
        if cfg.save_data:
            rollout_images = []
            rollout_states = []
            rollout_actions = []

        # Start episode
        input(f"Press Enter to start episode {episode_idx+1}...")
        print("Starting episode... Press Ctrl-C to terminate episode early!")
        last_tstamp = time.time()
        while t < cfg.max_steps:
            try:
                curr_tstamp = time.time()
                if curr_tstamp > last_tstamp + step_duration:
                    print(f"t: {t}")
                    print(f"Previous step elapsed time (sec): {curr_tstamp - last_tstamp:.2f}")
                    last_tstamp = time.time()

                    # Refresh the camera image and proprioceptive state
                    # obs = refresh_obs(obs, env)
                    image = env.unwrapped.robot_env.mujoco_renderer.render("rgb_array")
                    robot_state = env.unwrapped.robot_env._get_obs()  # should be array (18, )
                    robot_state = robot_state[:9] # only keep the joint angles and gripper state
                    obs["full_image"] = image
                    obs["robot_state"] = robot_state

                    # Save full (not preprocessed) image for replay video
                    replay_images.append(obs["full_image"])

                    # Get preprocessed image
                    obs["full_image"] = get_preprocessed_image(obs, resize_size)

                    # Query model to get action
                    action = get_action(
                        cfg,
                        model,
                        obs,
                        task_label,
                        processor=processor,
                    )

                    # [If saving rollout data] Save preprocessed image, robot state, and action
                    if cfg.save_data:
                        rollout_images.append(obs["full_image"])
                        rollout_states.append(obs["robot_state"])
                        rollout_actions.append(action)

                    # Execute action
                    print("action:", action)
                    # Since the action is the 7dof joint angles, directly modify the observation of the environment

                    # FIXME: ATTENTION! This section below is not working due to a mismatch between action space, will give up with Franka Kitchen
                    # roadmap: given env -> get robot_state -> perform ik to solve for q-vel -> update env with q-vel
                    robot_joint_state = robot_state[:7] # joint angles of the robot
                    robot_gripper_state = robot_state[7:9]
                    panda_data = env.unwrapped.robot_env.data
                    eef_xpos = panda_data.body("panda0_link7").xpos
                    # alternatively, use 
                    # eef_xpos = (panda_data.body_xpos("panda0_leftfinger") + panda_data.body_xpos("panda0_rightfinger")) / 2
                    assert(eef_xpos.shape == (3,))
                    eef_R = panda_data.body("panda0_link7").xmat.reshape((3, 3))
                    Tep = SE3.Rt(eef_R, eef_xpos)
                    eef_rpy = Tep.rpy()
                    # action is 7dof end effector deltas (x, y, z, roll, pitch, yaw, gripper)
                    new_eef_xpos = eef_xpos + action[:3]
                    new_eef_rpy = eef_rpy + action[3:6]
                    new_Tep = SE3.RPY(new_eef_rpy)
                    new_Tep.t = new_eef_xpos
                    panda = rtb.models.Panda()
                    q = panda.ikine_LM(new_Tep)
                    q_vel = (q - robot_joint_state) / step_duration
                    action = q_vel
                                      


    
                    perform_franka_action(env, action)

                    # obs, _, _, _, _ = env.step(action)
                    t += 1

            except (KeyboardInterrupt, Exception) as e:
                if isinstance(e, KeyboardInterrupt):
                    print("\nCaught KeyboardInterrupt: Terminating episode early.")
                else:
                    print(f"\nCaught exception: {e}")
                break

        # Save a replay video of the episode
        save_rollout_video(replay_images, episode_idx)

        # [If saving rollout data] Save rollout data
        if cfg.save_data:
            save_rollout_data(replay_images, rollout_images, rollout_states, rollout_actions, idx=episode_idx)

        # Redo episode or continue
        if input("Enter 'r' if you want to redo the episode, or press Enter to continue: ") != "r":
            episode_idx += 1


if __name__ == "__main__":
    main()
