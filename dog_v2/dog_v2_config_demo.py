from typing import Union

from params_proto import Meta
from go1_gym.envs.base.legged_robot_config import Cfg
from go1_gym.envs.go1.go1_config import config_go1


def config_dog_v2(Cnfg: Union[Cfg, Meta]):
    """
    Start from the public Go1 config, then overwrite only the robot-specific parts.
    This is the safest way to keep the Walk These Ways command layout / reward layout intact.
    """
    config_go1(Cnfg)

    # ---------- robot asset ----------
    _ = Cnfg.asset
    _.file = '{MINI_GYM_ROOT_DIR}/resources/robots/dog_v2/urdf/dog_v2.urdf'
    _.foot_name = 'Foot'  # substring match against LF_Foot_link, RF_Foot_link, ...
    _.penalize_contacts_on = ['HipF_link', 'Knee_link']
    _.terminate_after_contacts_on = []  # 暂时禁用，便于bring-up测试
    # fix_base_link 在各测试脚本中单独设置
    _.collapse_fixed_joints = False  # Keep foot links (they use fixed joints)

    # ---------- initial pose ----------
    _ = Cnfg.init_state
    _.pos = [0.0, 0.0, 0.32]

    # Default joint angles = 0
    _.default_joint_angles = {
        'LF_HipA_joint': 0.0,
        'LF_HipF_joint': 0.0,
        'LF_Knee_joint': 0.0,
        'LR_HipA_joint': 0.0,
        'LR_HipF_joint': 0.0,
        'LR_Knee_joint': 0.0,
        'RF_HipA_joint': 0.0,
        'RF_HipF_joint': 0.0,
        'RF_Knee_joint': 0.0,
        'RR_HipA_joint': 0.0,
        'RR_HipF_joint': 0.0,
        'RR_Knee_joint': 0.0,
    }

    # ---------- low-level control for bring-up ----------
    _ = Cnfg.control
    _.control_type = 'P'
    _.stiffness = {'joint': 40.0}
    _.damping = {'joint': 0.8}
    _.action_scale = 0.25
    _.hip_scale_reduction = 0.5
    _.decimation = 4

    # ---------- conservative terrain for debugging ----------
    _ = Cnfg.terrain
    _.mesh_type = 'plane'
    _.num_rows = 1
    _.num_cols = 1
    _.border_size = 0.0
    _.center_robots = True
    _.center_span = 1
    _.teleport_robots = False

    # ---------- single-env debug defaults ----------
    _ = Cnfg.env
    _.num_envs = 1
    _.num_recording_envs = 1
    _.num_privileged_obs = 0  # No privileged observations for bring-up testing

    # Disable all privileged observations for bring-up
    _.priv_observe_friction = False
    _.priv_observe_friction_indep = False
    _.priv_observe_ground_friction = False
    _.priv_observe_ground_friction_per_foot = False
    _.priv_observe_restitution = False
    _.priv_observe_base_mass = False
    _.priv_observe_com_displacement = False
    _.priv_observe_motor_strength = False
    _.priv_observe_motor_offset = False
    _.priv_observe_joint_friction = False
    _.priv_observe_Kp_factor = False
    _.priv_observe_Kd_factor = False
    _.priv_observe_contact_forces = False
    _.priv_observe_contact_states = False
    _.priv_observe_body_velocity = False
    _.priv_observe_foot_height = False
    _.priv_observe_body_height = False
    _.priv_observe_gravity = False
    _.priv_observe_terrain_type = False
    _.priv_observe_clock_inputs = False
    _.priv_observe_doubletime_clock_inputs = False
    _.priv_observe_halftime_clock_inputs = False
    _.priv_observe_desired_contact_states = False
    _.priv_observe_dummy_variable = False

    # ---------- turn off domain randomization for stand / sweep tests ----------
    _ = Cnfg.domain_rand
    _.push_robots = False
    _.randomize_friction = False
    _.randomize_friction_indep = False
    _.randomize_ground_friction = False
    _.randomize_restitution = False
    _.randomize_base_mass = False
    _.randomize_com_displacement = False
    _.randomize_motor_strength = False
    _.randomize_motor_offset = False
    _.randomize_Kp_factor = False
    _.randomize_Kd_factor = False
    _.randomize_lag_timesteps = False

    # Large resampling time keeps commands effectively constant during debug.
    _ = Cnfg.commands
    _.resampling_time = 1000.0
    _.command_curriculum = False  # Disable for bring-up testing (commands only 3-dim)

    # ---------- viewer camera for debugging ----------
    _ = Cnfg.viewer
    _.pos = [1.5, 1.5, 1.0]  # 相机位置 [m]
    _.lookat = [0.0, 0.0, 0.3]  # 看向机器人位置 [m]
