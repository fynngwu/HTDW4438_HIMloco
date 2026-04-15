from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

# Dog V2 物理参数：
# Base mass: 5.28 kg, 大腿长 ~0.216m, 小腿长 ~0.221m
# 关节: HipA(外展) + HipF(髋) + Knee(膝), 每腿 3 DOF
# 电机减速比30, 最高效率点速度95rpm, effort=33.5 Nm, velocity=3 rad/s

class DogV2Cfg(LeggedRobotCfg):
    # ==========================
    # 1. 环境与地形
    # ==========================
    class env(LeggedRobotCfg.env):
        num_envs = 4096

        num_one_step_observations = 45
        num_observations = num_one_step_observations * 6

        num_one_step_privileged_obs = 45 + 3 + 3 + 187
        num_privileged_obs = num_one_step_privileged_obs * 1

        episode_length_s = 20

    class terrain:
        mesh_type = 'plane'
        horizontal_scale = 0.1
        vertical_scale = 0.005
        border_size = 25
        curriculum = True
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.
        measure_heights = True
        measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]
        selected = False
        terrain_kwargs = None
        max_init_terrain_level = 5
        terrain_length = 8.
        terrain_width = 8.
        num_rows = 10
        num_cols = 20
        terrain_proportions = [0.1, 0.2, 0.3, 0.3, 0.1]
        slope_treshold = 0.75

    # ==========================
    # 2. 初始状态
    # ==========================
    class init_state(LeggedRobotCfg.init_state):
        # 默认全零角度即为弯腿站立姿态
        pos = [0.0, 0.0, 0.3]
        default_joint_angles = {
            'LF_HipA_joint': 0.0, 'LF_HipF_joint': 0.0, 'LF_Knee_joint': 0.0,
            'RF_HipA_joint': 0.0, 'RF_HipF_joint': 0.0, 'RF_Knee_joint': 0.0,
            'LR_HipA_joint': 0.0, 'LR_HipF_joint': 0.0, 'LR_Knee_joint': 0.0,
            'RR_HipA_joint': 0.0, 'RR_HipF_joint': 0.0, 'RR_Knee_joint': 0.0,
        }

    # ==========================
    # 3. 控制参数
    # ==========================
    class control(LeggedRobotCfg.control):
        control_type = 'P'
        stiffness = {'joint': 25.0}
        damping = {'joint': 0.5}
        action_scale = 0.25
        decimation = 2

    # ==========================
    # 4. 指令范围
    # ==========================
    class commands(LeggedRobotCfg.commands):
        curriculum = True
        max_curriculum = 1.0
        num_commands = 4
        resampling_time = 10.
        heading_command = True
        class ranges(LeggedRobotCfg.commands.ranges):
                lin_vel_x = [-4.0, 4.0]
                lin_vel_y = [-2.0, 2.0]
                ang_vel_yaw = [-1.56, 1.56]
                heading = [-3.14, 3.14]

    # ==========================
    # 5. 资产配置
    # ==========================
    class asset(LeggedRobotCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/dog_v2/urdf/dog_v2.urdf'
        name = "dog_v2"
        foot_name = "Foot"
        penalize_contacts_on = ["base", "HipA_link", "HipF_link", "Knee_link"]
        terminate_after_contacts_on = ["base"]
        self_collisions = 0
        self_collision_mode = "base_excluded"
        flip_visual_attachments = False
        collapse_fixed_joints = False

        density = 0.001
        angular_damping = 0.
        linear_damping = 0.
        max_angular_velocity = 9.5
        max_linear_velocity = 20.
        armature = 0.005

    # ==========================
    # 6. 域随机化
    # ==========================
    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.5, 1.25]

        randomize_payload_mass = False
        payload_mass_range = [-0.5, 1.0]

        push_robots = True
        push_interval_s = 15
        max_push_vel_xy = 0.5

        randomize_motor_strength = True
        motor_strength_range = [0.9, 1.1]

        randomize_kp = True
        kp_range = [0.8, 1.2]
        randomize_kd = True
        kd_range = [0.8, 1.2]

        disturbance = True
        disturbance_range = [-2.0, 2.0]
        disturbance_interval = 8

        delay = True

    # ==========================
    # 7. 奖励函数
    # ==========================
    class rewards(LeggedRobotCfg.rewards):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.27
        clearance_height_target = -0.135

        class scales(LeggedRobotCfg.rewards.scales):

            termination = -0.0
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.5
            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            orientation = -1.3
            dof_acc = -2.5e-7
            joint_power = -2e-5

            base_height = -1.0
            default_pos_linear = -0.02
            diagonal_sync = -0.1
            hip_mirror_symmetry = -0.1

            foot_clearance = -0.0
            action_rate = -0.01
            smoothness = -0.01
            feet_air_time = 0.05
            stumble = -0.0
            stand_still = -1.0
            torques = -0.0
            dof_vel = -0.0
            dof_pos_limits = -0.0
            dof_vel_limits = -0.0
            torque_limits = -0.0
            collision = -1.0

    # ==========================
    # 8. 归一化
    # ==========================
    class normalization(LeggedRobotCfg.normalization):
        class obs_scales(LeggedRobotCfg.normalization.obs_scales):
            lin_vel = 2.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            height_measurements = 5.0
        clip_observations = 100.
        clip_actions = 100.

    # ==========================
    # 9. 噪声
    # ==========================
    class noise(LeggedRobotCfg.noise):
        add_noise = True
        noise_level = 1.0
        class noise_scales(LeggedRobotCfg.noise.noise_scales):
            dof_pos = 0.01
            dof_vel = 0.1
            lin_vel = 0.1
            ang_vel = 0.2
            gravity = 0.05
            height_measurements = 0.1

    # ==========================
    # 10. 物理引擎
    # ==========================
    class sim(LeggedRobotCfg.sim):
        dt = 0.005
        substeps = 1
        gravity = [0., 0., -9.81]
        up_axis = 1
        class physx(LeggedRobotCfg.sim.physx):
            num_threads = 10
            solver_type = 1
            num_position_iterations = 4
            num_velocity_iterations = 1
            contact_offset = 0.01
            rest_offset = 0.0
            bounce_threshold_velocity = 0.5
            max_depenetration_velocity = 1.0
            default_buffer_size_multiplier = 5
            max_gpu_contact_pairs = 2**23

# ==========================
# 训练参数 (PPO Runner)
# ==========================
class DogV2CfgPPO(LeggedRobotCfgPPO):
    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01
        learning_rate = 1e-3

    class runner(LeggedRobotCfgPPO.runner):
        run_name = 'dog_v2_himloco_v1.0'
        experiment_name = 'flat_dog_v2'
        max_iterations = 300000
        save_interval = 200

        resume = False
        load_run = -1  # -1 = last run
        checkpoint = -1  # -1 = last saved model

        policy_class_name = 'HIMActorCritic'
        algorithm_class_name = 'HIMPPO'
        num_steps_per_env = 48
