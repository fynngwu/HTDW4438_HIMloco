"""
Go1 站立测试 - 使用Go1默认配置
用于对比Dog_v2的表现
"""
import time

import isaacgym
assert isaacgym

import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from go1_gym.envs import *
from go1_gym.envs.base.legged_robot_config import Cfg
from go1_gym.envs.go1.velocity_tracking import VelocityTrackingEasyEnv
from go1_gym.envs.go1.go1_config import config_go1


def make_env(headless=False):
    config_go1(Cfg)
    
    # 禁用终止条件，便于观察
    Cfg.asset.terminate_after_contacts_on = []
    
    # 禁用privileged observations
    Cfg.env.num_privileged_obs = 0
    Cfg.env.priv_observe_friction = False
    Cfg.env.priv_observe_friction_indep = False
    Cfg.env.priv_observe_ground_friction = False
    Cfg.env.priv_observe_ground_friction_per_foot = False
    Cfg.env.priv_observe_restitution = False
    Cfg.env.priv_observe_base_mass = False
    Cfg.env.priv_observe_com_displacement = False
    Cfg.env.priv_observe_motor_strength = False
    Cfg.env.priv_observe_motor_offset = False
    Cfg.env.priv_observe_joint_friction = False
    Cfg.env.priv_observe_Kp_factor = False
    Cfg.env.priv_observe_Kd_factor = False
    Cfg.env.priv_observe_contact_forces = False
    Cfg.env.priv_observe_contact_states = False
    Cfg.env.priv_observe_body_velocity = False
    Cfg.env.priv_observe_foot_height = False
    Cfg.env.priv_observe_body_height = False
    Cfg.env.priv_observe_gravity = False
    Cfg.env.priv_observe_terrain_type = False
    Cfg.env.priv_observe_clock_inputs = False
    Cfg.env.priv_observe_doubletime_clock_inputs = False
    Cfg.env.priv_observe_halftime_clock_inputs = False
    Cfg.env.priv_observe_desired_contact_states = False
    Cfg.env.priv_observe_dummy_variable = False
    
    # 禁用command curriculum
    Cfg.commands.command_curriculum = False
    
    # 禁用domain randomization
    Cfg.domain_rand.randomize_friction = False
    Cfg.domain_rand.randomize_restitution = False
    Cfg.domain_rand.randomize_base_mass = False
    Cfg.domain_rand.randomize_com_displacement = False
    Cfg.domain_rand.randomize_motor_strength = False
    Cfg.domain_rand.randomize_motor_offset = False
    Cfg.domain_rand.randomize_Kp_factor = False
    Cfg.domain_rand.randomize_Kd_factor = False
    Cfg.domain_rand.push_robots = False
    
    # 单环境测试
    Cfg.env.num_envs = 1
    
    # 平地
    Cfg.terrain.mesh_type = 'plane'
    Cfg.terrain.num_rows = 1
    Cfg.terrain.num_cols = 1
    Cfg.terrain.teleport_robots = False  # 禁用传送
    Cfg.terrain.curriculum = False
    
    # 相机位置
    Cfg.viewer.pos = [1.5, 1.5, 1.0]
    Cfg.viewer.lookat = [0.0, 0.0, 0.3]
    
    env = VelocityTrackingEasyEnv(sim_device='cuda:0', headless=headless, cfg=Cfg)
    return env


def main():
    env = make_env(headless=False)

    print('\n=== Go1 PD控制参数 ===')
    print(f'Kp (stiffness): {Cfg.control.stiffness}')
    print(f'Kd (damping): {Cfg.control.damping}')
    print(f'action_scale: {Cfg.control.action_scale}')
    print(f'hip_scale_reduction: {Cfg.control.hip_scale_reduction}')
    
    print('\n=== Go1 默认关节角度 ===')
    for name, angle in Cfg.init_state.default_joint_angles.items():
        print(f'  {name}: {angle} rad')

    print('\n[DOF order from simulator]')
    if hasattr(env, 'dof_names'):
        for i, name in enumerate(env.dof_names):
            print(f'{i:02d}: {name}')

    obs = env.reset()
    env.commands[:] = 0.0

    actions = torch.zeros((env.num_envs, Cfg.env.num_actions), device=env.device)

    num_steps = 3000
    print_every = 100

    print('\n[Standing test starts] zero action -> target pose = default_joint_angles')
    for step in range(num_steps):
        env.commands[:] = 0.0
        obs, rew, done, info = env.step(actions)

        if step % print_every == 0:
            base_h = env.root_states[0, 2].item()
            lin_vel = env.base_lin_vel[0].detach().cpu().numpy()
            joint_pos = env.dof_pos[0].detach().cpu().numpy()
            print(f'step={step:04d}  base_z={base_h:.3f}  base_lin_vel={lin_vel}')
            print('joint_pos =', [round(x, 3) for x in joint_pos.tolist()])

        if bool(done[0]):
            print(f'episode ended at step {step}, resetting...')
            obs = env.reset()
            env.commands[:] = 0.0

    print('\n[Standing test finished]')
    time.sleep(1.0)


if __name__ == '__main__':
    main()