from dataclasses import dataclass
from enum import Enum, unique
from pathlib import Path
from typing import Optional, Union

from draccus import ChoiceRegistry

from prismatic.conf import VLAConfig


@dataclass
class Default(VLAConfig):
    vla_id: str = "default"
    base_vlm: Union[str, Path] = "siglip-224px+7b"
    freeze_vision_backbone: bool = False
    freeze_llm_backbone: bool = False
    unfreeze_last_llm_layer: bool = False

    # Data Mixture Parameters
    data_mix: str = "bridge"
    shuffle_buffer_size: int = 1024

    # Optimization Parameters
    epochs: int = 1000
    max_steps: Optional[int] = 20000

    expected_world_size: int = 2
    global_batch_size: int = 16
    per_device_batch_size: int = 4

    learning_rate: float = 2e-5
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    lr_scheduler_type: str = "constant"
    warmup_ratio: float = 0.0

    train_strategy: str = "fsdp-full-shard"

    enable_gradient_checkpointing: bool = True

    enable_mixed_precision_training: bool = True
    reduce_in_full_precision: bool = False


@dataclass
class Clip_Test(Default):
    # fmt: off
    vla_id: str = "clip-test"
    base_vlm: str = "prism-dinosiglip-224px+7b" # TODO: change this to a different vlm

    freeze_vision_backbone: bool = False
    freeze_llm_backbone: bool = True
    unfreeze_last_llm_layer: bool = True

    # Data Mixture Parameters
    data_mix: str = "robosuite"



@unique
class VLARegistryNew(Enum):
    # My configurations
    Default = Default
    Clip_Test = Clip_Test

    @property
    def vla_id(self) -> str:
        return self.value.vla_id


# Register VLAs in Choice Registry
for vla_variant in VLARegistryNew:
    VLAConfig.register_subclass(vla_variant.vla_id, vla_variant.value)
