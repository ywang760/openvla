import robosuite as suite
from robosuite.controllers import load_controller_config
import robosuite.macros as macros
from robosuite.utils.input_utils import *

macros.IMAGE_CONVENTION = "opencv"


def get_robosuite_env(cfg):
    controller_configs = load_controller_config(default_controller="OSC_POSE")
    env = suite.make(
        env_name=cfg.env_name,
        robots=["Panda"],
        gripper_types="default",
        has_renderer=cfg.debug, # Render only in debug mode, otherwise offscreen if launched from a ccv-script
        use_camera_obs=True,
        camera_names=cfg.camera_name,
        camera_heights=cfg.camera_heights,
        camera_widths=cfg.camera_widths,
        controller_configs=controller_configs,
        **cfg.env_kwargs
    )
    return env

def _check_mimicgen_env(env_name):
    # check if the environment is a valid mimicgen environment
    try:
        import robosuite_task_zoo
    except ImportError:
        pass

    # all base robosuite environments (and maybe robosuite task zoo)
    robosuite_envs = set(suite.ALL_ENVIRONMENTS)

    # all environments including mimicgen environments
    import mimicgen
    all_envs = set(suite.ALL_ENVIRONMENTS)

    # get only mimicgen envs
    only_mimicgen = sorted(all_envs - robosuite_envs)

    # keep only envs that correspond to the different reset distributions from the paper
    envs = [x for x in only_mimicgen if x[-1].isnumeric()]

    if env_name.startswith("Mimicgen_"):
        env_name = env_name[9:]

    if env_name not in envs:
        raise ValueError(f"{env_name} is not a valid environment.\n Valid environments are {envs}.")

    return env_name


def get_mimicgen_env(cfg):
    cfg.env_name = _check_mimicgen_env(cfg.env_name)
    controller_configs = load_controller_config(default_controller="OSC_POSE")

    env = suite.make(
        env_name=cfg.env_name,
        robots=["Panda"],
        gripper_types="default",
        has_renderer=cfg.debug, # Render only in debug mode, otherwise offscreen if launched from a ccv-script
        use_camera_obs=True,
        camera_names=cfg.camera_name,
        camera_heights=cfg.camera_heights,
        camera_widths=cfg.camera_widths,
        controller_configs=controller_configs,
        **cfg.env_kwargs
    )

    return env


def refresh_obs(cfg, obs, env):
    obs["full_image"] = obs[f"{cfg.camera_name}_image"]
    obs["robot_state"] = obs["robot0_proprio-state"]
    return obs
