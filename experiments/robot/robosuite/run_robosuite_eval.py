import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Union, Any, Optional
import draccus
import wandb

sys.path.append(".")
from experiments.robot.robosuite.robosuite_utils import (
    get_robosuite_env,
    get_mimicgen_env,
    refresh_obs,
)
from experiments.robot.bridge.bridgev2_utils import (
    get_preprocessed_image,
    save_rollout_data,
    save_rollout_video,
)
from experiments.robot.openvla_utils import get_processor
from experiments.robot.robot_utils import (
    DATE_TIME,
    get_action,
    get_image_resize_size,
    get_model,
    invert_gripper_action,
    normalize_gripper_action,
)
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Assign the VLM to GPU 1


@dataclass
class GenerateConfig:

    # Model-specific parameters
    model_family: str = "openvla"
    pretrained_checkpoint: Union[str, Path] = "openvla/openvla-7b-finetuned-libero-spatial"
    lora_adapter: bool = False
    lora_exp_id: str = "openvla-7b+robosuite_dataset+b8+lr-0.0005+lora-r32+dropout-0.0+q-4bit--None--image_aug" # TODO: change this

    # Precision: default is bf16
    load_in_8bit: bool = False
    load_in_4bit: bool = True
    center_crop: bool = False

    # Environment-specific parameters

    # env_name: str = "PickPlace"
    # task_label: str = "Put the can into the box"
    # env_kwargs: Dict[str, Any] = field(default_factory=lambda: {"single_object_mode": 2, "object_type": "can"})
    # env_name: str = "Lift"
    # task_label: str = "Pick up the red cube"
    # env_kwargs: Dict[str, Any] = field(default_factory=dict)

    # Example of using a mimicgen environment
    env_name: str = "Mimicgen_Stack_D0"
    task_label: str = "Pick up the red block and place it on the green block."
    env_kwargs: Dict[str, Any] = field(default_factory=dict)

    camera_name: str = "agentview"
    camera_heights: int = 512
    camera_widths: int = 512

    max_episodes: int = 20
    max_steps: int = 200
    control_frequency: float = 5

    # Wandb parameters
    wandb_entity: str = "robot-vla"
    wandb_project: str = "openvla-evals"

    # Utils
    run_id_note: Optional[str] = None                # Extra note to add in run ID for logging
    local_log_dir: str = "./experiments/logs"        # Local directory for eval logs
    save_data: bool = False
    debug: bool = False # If debug, will need to manually rollout the videos, otherwise it will automatically save the results


@draccus.wrap()
def main(cfg: GenerateConfig) -> None:
    assert cfg.pretrained_checkpoint, "cfg.pretrained_checkpoint must be set."
    assert not cfg.center_crop, "`center_crop` should be disabled."

    # [OpenVLA] Set action un-normalization key
    cfg.unnorm_key = "" # doesn't do anything

    # Load model and processor
    model = get_model(cfg)
    processor = None
    if cfg.model_family == "openvla":
        processor = get_processor(cfg)

    # Initialize local logging
    run_id = f"EVAL-{cfg.env_name}-{cfg.model_family}-{DATE_TIME}"
    if cfg.lora_adapter:
        run_id += f"-lora"
    else:
        run_id += f"-baseline"
    if cfg.run_id_note is not None:
        run_id += f"--{cfg.run_id_note}"
    os.makedirs(cfg.local_log_dir, exist_ok=True)
    local_log_filepath = os.path.join(cfg.local_log_dir, run_id + ".txt")
    log_file = open(local_log_filepath, "w")
    print(f"Logging to local log file: {local_log_filepath}")

    # Check if is mimicgen environment
    if cfg.env_name.startswith("Mimicgen"):
        is_mimicgen = True
        # set the cfg.env_name to be the part after Mimicgen_
        cfg.env_name = cfg.env_name[len("mimicgen_"):]
    else:
        is_mimicgen = False

    # Initialize wandb
    if not cfg.debug:
        tags = ["robosuite", "eval"] if not is_mimicgen else ["mimicgen", "eval"]
        if cfg.lora_adapter:
            tags.append("lora")
        else:
            tags.append("baseline")
        wandb.init(
            project=cfg.wandb_project,
            entity=cfg.wandb_entity,
            config=cfg.__dict__,
            job_type="eval",
            tags=tags,
            name=run_id,
        )
    
    # Get environment
    env = get_robosuite_env(cfg) if not is_mimicgen else get_mimicgen_env(cfg)
    resize_size = get_image_resize_size(cfg)
    print(f"Environment: {cfg.env_name}, task: {cfg.task_label}")
    log_file.write(f"Environment: {cfg.env_name}, task: {cfg.task_label}\n")

    # Start evaluation
    episode_idx = 0
    success_count = 0
    while episode_idx < cfg.max_episodes:
        # Reset environment
        obs = env.reset()

        # Setup
        t = 0
        step_duration = 1.0 / cfg.control_frequency
        replay_images = []
        if cfg.save_data:
            rollout_images = []
            rollout_states = []
            rollout_actions = []
        is_success = False

        # Start episode
        if cfg.debug:
            input(f"Press Enter to start episode {episode_idx+1}...")
        print("Starting episode... Press Ctrl-C to terminate episode early!")
        log_file.write(f"Starting episode {episode_idx+1}...\n")
        last_tstamp = time.time()
        while t < cfg.max_steps:
            try:
                curr_tstamp = time.time()
                if curr_tstamp > last_tstamp + step_duration:
                    print(f"t: {t}")
                    print(f"Previous step elapsed time (sec): {curr_tstamp - last_tstamp:.2f}")
                    last_tstamp = time.time()

                    # Refresh the camera image and proprioceptive state
                    obs = refresh_obs(cfg, obs, env)

                    # Save full (not preprocessed) image for replay video
                    replay_images.append(obs["full_image"])

                    # Get preprocessed image
                    obs["full_image"] = get_preprocessed_image(obs, resize_size)

                    # Query model to get action
                    action = get_action(
                        cfg,
                        model,
                        obs,
                        cfg.task_label,
                        processor=processor,
                    )
                    action = normalize_gripper_action(action, binarize=True)
                    action = invert_gripper_action(action)

                    # [If saving rollout data] Save preprocessed image, robot state, and action
                    if cfg.save_data:
                        rollout_images.append(obs["full_image"])
                        rollout_states.append(obs["robot_state"])
                        rollout_actions.append(action)

                    # Execute action
                    print("action:", action)
                    log_file.write(f"action: {action}\n")
                    obs, _, done, info = env.step(action)
                    env.render()
                    t += 1
                    if env._check_success():
                        print(f"Task completed at t={t}!")
                        success_count += 1
                        is_success = True
                        break

            except (KeyboardInterrupt, Exception) as e:
                if isinstance(e, KeyboardInterrupt):
                    print("\nCaught KeyboardInterrupt: Terminating episode early.")
                else:
                    print(f"\nCaught exception: {e}")
                break

        # Save a replay video of the episode
        mp4_path = save_rollout_video(replay_images, episode_idx)
        if not cfg.debug:
            # log the video to wandb
            caption = f"Episode {episode_idx+1} (Success)" if is_success else f"Episode {episode_idx+1} (Failure)"
            wandb.log({"replay_video": wandb.Video(mp4_path, caption=caption)})            

        # [If saving rollout data] Save rollout data
        if cfg.save_data:
            save_rollout_data(replay_images, rollout_images, rollout_states, rollout_actions, idx=episode_idx)

        # Redo episode or continue
        if not cfg.debug or input("Enter 'r' if you want to redo the episode, or press Enter to continue: ") != "r":
            episode_idx += 1    

    # End evaluation
    success_rate = success_count / cfg.max_episodes
    print(f"Success rate: {success_rate}")
    log_file.write(f"Success rate: {success_rate}\n")
    log_file.close()
    if not cfg.debug:
        wandb.log({"success_rate": success_rate})
        wandb.finish()

if __name__ == "__main__":
    main()
