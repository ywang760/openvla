from draccus import ChoiceRegistry
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Union, Any, Optional
import wandb
from enum import Enum, unique

# Create a enum to determine robosuite/mimicgen
@unique
class RobosuiteEnvType(Enum):
    ROBOSUITE = "robosuite"
    MIMICGEN = "mimicgen"

@dataclass
class RobosuiteEnvConfig(ChoiceRegistry):
    env_name: str = ""
    env_type: RobosuiteEnvType = RobosuiteEnvType.ROBOSUITE
    task_label: str = ""
    env_kwargs: Dict[str, Any] = field(default_factory=dict) 
    camera_name: str = "agentview"
    camera_heights: int = 512
    camera_widths: int = 512

@RobosuiteEnvConfig.register_subclass("Lift")
@dataclass
class Lift(RobosuiteEnvConfig):
    env_name: str = "Lift"
    task_label: str = "Pick up the red cube"
    env_kwargs: Dict[str, Any] = field(default_factory=dict)

@RobosuiteEnvConfig.register_subclass("PickPlace")
class PickPlace(RobosuiteEnvConfig):
    env_name: str = "PickPlace"
    task_label: str = "Put the can into the box"
    env_kwargs: Dict[str, Any] = field(default_factory=lambda: {"single_object_mode": 2, "object_type": "can"})

@RobosuiteEnvConfig.register_subclass("Mimicgen_Stack_D0")
@dataclass
class Mimicgen_Stack_D0(RobosuiteEnvConfig):
    env_name: str = "Mimicgen_Stack_D0"
    env_type: RobosuiteEnvType = RobosuiteEnvType.MIMICGEN
    task_label: str = "Pick up the red block and place it on the green block."
    env_kwargs: Dict[str, Any] = field(default_factory=dict)

@dataclass
class RobosuiteEvalConfig():
    # Model-specific parameters
    model_family: str = "openvla"
    pretrained_checkpoint: Union[str, Path] = "/users/ywang760/scratch/openvla/runs/openvla-7b+robosuite_dataset+b8+lr-0.0005+lora-r32+dropout-0.0--clip+mae--image_aug"
    lora_adapter: bool = True # Currently is only used for tagging on wandb
    unnorm_key: str = "robosuite_dataset" # Edit based on the key in dataset_statistics.json

    env: RobosuiteEnvConfig = Lift()

    # Precision: default is bf16
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    center_crop: bool = False

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
    debug: bool = True # If debug, will need to manually rollout the videos, otherwise it will automatically save the results




