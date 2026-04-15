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


def make_env(headless=False):
    config_dog_v2(Cfg)
    Cfg.asset.fix_base_link = False  # 站立测试：放在地上
    env = VelocityTrackingEasyEnv(sim_device='cuda:0', headless=headless, cfg=Cfg)
    return env


def main():
    env = make_env(headless=False)

    print('\n[DOF order from simulator]')
    if hasattr(env, 'dof_names'):
        for i, name in enumerate(env.dof_names):
            print(f'{i:02d}: {name}')
    else:
        print('env.dof_names not found, please print the DOF order from the asset loader manually.')

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
