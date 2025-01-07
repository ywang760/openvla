import robosuite as suite
import robosuite.macros as macros

macros.IMAGE_CONVENTION = "opencv"


def get_robosuite_env(cfg):
    controller_configs = suite.load_controller_config(default_controller="OSC_POSE")
    env = suite.make(
        env_name=cfg.env_name,
        robots=["Panda"],
        gripper_types="default",
        has_renderer=True,
        use_camera_obs=True,
        camera_names=cfg.camera_name,
        camera_heights=cfg.camera_heights,
        camera_widths=cfg.camera_widths,
        controller_configs=controller_configs, 
    )
    return env

def refresh_obs(cfg, obs, env):
    obs["full_image"] = obs[f"{cfg.camera_name}_image"]
    obs["robot_state"] = obs["robot0_proprio-state"]
    return obs
