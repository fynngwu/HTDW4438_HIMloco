import math
import time

import isaacgym
assert isaacgym

import torch
import sys
import os

# Add the directory to path for importing config
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from go1_gym.envs import *
from go1_gym.envs.base.legged_robot_config import Cfg
from go1_gym.envs.go1.velocity_tracking import VelocityTrackingEasyEnv

from dog_v2_config_demo import config_dog_v2


JOINT_NAMES = [
    'LF_HipA_joint', 'LF_HipF_joint', 'LF_Knee_joint',
    'LR_HipA_joint', 'LR_HipF_joint', 'LR_Knee_joint',
    'RF_HipA_joint', 'RF_HipF_joint', 'RF_Knee_joint',
    'RR_HipA_joint', 'RR_HipF_joint', 'RR_Knee_joint',
]

# In this repo, indices [0, 3, 6, 9] get extra hip_scale_reduction in torque computation.
HIPA_INDICES = {0, 3, 6, 9}


def make_env(headless=False):
    config_dog_v2(Cfg)
    Cfg.asset.fix_base_link = True  # 扫频测试：吊起来，便于观察关节
    env = VelocityTrackingEasyEnv(sim_device='cuda:0', headless=headless, cfg=Cfg)
    return env


def angle_delta_to_action(delta_rad: float, joint_index: int) -> float:
    denom = Cfg.control.action_scale
    if joint_index in HIPA_INDICES:
        denom *= Cfg.control.hip_scale_reduction
    return delta_rad / denom


def hold_pose(env, seconds=1.0):
    obs = env.reset()
    env.commands[:] = 0.0
    actions = torch.zeros((env.num_envs, Cfg.env.num_actions), device=env.device)
    steps = int(seconds / env.dt)
    for _ in range(steps):
        env.commands[:] = 0.0
        obs, rew, done, info = env.step(actions)
    return obs


def sweep_one_joint(env, joint_index, amplitude_rad=0.20, freq_hz=0.5, duration_s=6.0):
    obs = hold_pose(env, seconds=1.0)

    joint_name = JOINT_NAMES[joint_index]
    steps = int(duration_s / env.dt)
    actions = torch.zeros((env.num_envs, Cfg.env.num_actions), device=env.device)

    print(f'\n[Sweeping] joint_index={joint_index} joint_name={joint_name}')
    print(f'  amplitude={amplitude_rad:.3f} rad, frequency={freq_hz:.3f} Hz, duration={duration_s:.1f} s')
    print(f'  {"t(s)":>6} {"cmd_delta":>10} {"measured_q":>12} {"target_q":>10} {"torque(Nm)":>12}')

    for step in range(steps):
        t = step * env.dt
        delta = amplitude_rad * math.sin(2.0 * math.pi * freq_hz * t)
        actions.zero_()
        actions[:, joint_index] = angle_delta_to_action(delta, joint_index)

        env.commands[:] = 0.0
        obs, rew, done, info = env.step(actions)

        if step % max(1, steps // 10) == 0:
            q = env.dof_pos[0, joint_index].item()
            tau = env.torques[0, joint_index].item()
            target = env.joint_pos_target[0, joint_index].item()
            print(f'  {t:6.2f} {delta:+10.3f} {q:+12.3f} {target:+10.3f} {tau:+12.3f}')

        if bool(done[0]):
            print('  episode terminated, resetting and continuing this sweep...')
            obs = env.reset()
            env.commands[:] = 0.0


def main():
    env = make_env(headless=False)

    print('\n[DOF order from simulator]')
    if hasattr(env, 'dof_names'):
        for i, name in enumerate(env.dof_names):
            print(f'{i:02d}: {name}')

    # First, verify the robot can hold the nominal standing pose.
    hold_pose(env, seconds=2.0)

    # Sweep every joint one by one.
    for i in range(len(JOINT_NAMES)):
        sweep_one_joint(
            env,
            joint_index=i,
            amplitude_rad=0.15 if i in HIPA_INDICES else 0.25,
            freq_hz=0.5,
            duration_s=5.0,
        )
        time.sleep(0.5)

    print('\n[Joint sweep finished]')


if __name__ == '__main__':
    main()
