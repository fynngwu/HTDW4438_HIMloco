"""
IsaacGym play 数据记录脚本
记录观测、动作、力矩等数据用于与 MuJoCo 部署对比

用法 (需要 gym conda 环境):
  eval "$(conda shell.bash hook)" && conda activate gym
  python deploy/deploy_mujoco/record_isaacgym.py --task dog_v2 --headless
"""
import os, sys, json, argparse

# isaacgym MUST be imported before torch
import isaacgym
import torch
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Follow play.py import order to avoid circular import:
# 1) legged_gym (root __init__)  2) legged_gym.envs (registers tasks)  3) task_registry
from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry


def parse_args():
    # Pre-extract our custom args from sys.argv so isaacgym's get_args() doesn't choke
    steps = 200
    out = "deploy/deploy_mujoco/isaacgym_record.json"
    robot = 0
    cleaned_argv = []
    i = 0
    while i < len(sys.argv):
        if sys.argv[i] == "--steps":
            steps = int(sys.argv[i + 1])
            i += 2
        elif sys.argv[i] == "--out":
            out = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == "--robot":
            robot = int(sys.argv[i + 1])
            i += 2
        else:
            cleaned_argv.append(sys.argv[i])
            i += 1
    sys.argv = cleaned_argv
    args = get_args()
    args.steps = steps
    args.out = out
    args.robot = robot
    return args


def play_record(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 50)
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.disturbance = False
    env_cfg.domain_rand.randomize_payload_mass = False
    env_cfg.domain_rand.delay = False
    env_cfg.domain_rand.randomize_kp = False
    env_cfg.domain_rand.randomize_kd = False
    env_cfg.commands.heading_command = False

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    env.commands[:, 0] = 1.0
    env.commands[:, 1] = 0.0
    env.commands[:, 2] = 0.0

    obs = env.get_observations()

    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)

    r = args.robot
    records = []
    num_steps = args.steps

    print(f"Recording {num_steps} policy steps from robot {r}...")
    print(f"obs shape: {obs.shape}, num_one_step_obs: {env.num_one_step_obs}")

    for i in range(num_steps):
        # Record state BEFORE action
        pre_record = {
            "step": i,
            "base_pos": env.root_states[r, :3].cpu().numpy().tolist(),
            "base_quat": env.root_states[r, 3:7].cpu().numpy().tolist(),
            "base_lin_vel": env.root_states[r, 7:10].cpu().numpy().tolist(),
            "base_ang_vel": env.root_states[r, 10:13].cpu().numpy().tolist(),
            "dof_pos": env.dof_pos[r].cpu().numpy().tolist(),
            "dof_vel": env.dof_vel[r].cpu().numpy().tolist(),
            "obs_history": obs[r].cpu().numpy().tolist(),
            "actions_in_obs": env.actions[r].cpu().numpy().tolist(),
            "cmd": env.commands[r, :3].cpu().numpy().tolist(),
        }

        # Get action from policy
        actions = policy(obs.detach())

        # Record action
        pre_record["raw_action"] = actions[r].detach().cpu().numpy().tolist()
        pre_record["clipped_action"] = np.clip(actions[r].detach().cpu().numpy(), -100, 100).tolist()

        # Re-set commands before step (play.py does this every iteration
        # because env.step() -> post_physics_step() -> reset_idx() -> _resample_commands()
        env.commands[:, 0] = 1.0
        env.commands[:, 1] = 0.0
        env.commands[:, 2] = 0.0

        # Step environment
        obs, _, rews, dones, infos, _, _ = env.step(actions.detach())

        # Record post-step torques and targets
        post_record = {
            "torques": env.torques[r].cpu().numpy().tolist(),
            "joint_pos_target": env.joint_pos_target[r].cpu().numpy().tolist(),
            "dof_pos_after": env.dof_pos[r].cpu().numpy().tolist(),
            "dof_vel_after": env.dof_vel[r].cpu().numpy().tolist(),
        }

        record = {**pre_record, **post_record}
        records.append(record)

        if (i + 1) % 50 == 0:
            print(f"  Step {i+1}/{num_steps}, base_z={record['base_pos'][2]:.3f}")

    # Save
    out_path = os.path.join(PROJECT_ROOT, args.out)
    with open(out_path, "w") as f:
        json.dump(records, f, indent=2)
    print(f"\nRecorded {len(records)} steps to {out_path}")

    # Quick stats
    if records:
        actions = np.array([r['clipped_action'] for r in records])
        dof_pos = np.array([r['dof_pos'] for r in records])
        dof_vel = np.array([r['dof_vel'] for r in records])
        torques = np.array([r['torques'] for r in records])
        print(f"\nAction stats (clipped):")
        print(f"  mean: {actions.mean(axis=0)}")
        print(f"  std:  {actions.std(axis=0)}")
        print(f"  min:  {actions.min(axis=0)}")
        print(f"  max:  {actions.max(axis=0)}")
        print(f"\nDOF pos stats:")
        print(f"  mean: {dof_pos.mean(axis=0)}")
        print(f"  std:  {dof_pos.std(axis=0)}")
        print(f"  range: [{dof_pos.min(axis=0)}, {dof_pos.max(axis=0)}]")
        print(f"\nDOF vel stats:")
        print(f"  mean: {dof_vel.mean(axis=0)}")
        print(f"  std:  {dof_vel.std(axis=0)}")
        print(f"  range: [{dof_vel.min(axis=0)}, {dof_vel.max(axis=0)}]")
        print(f"\nTorque stats:")
        print(f"  mean: {torques.mean(axis=0)}")
        print(f"  std:  {torques.std(axis=0)}")
        print(f"  range: [{torques.min(axis=0)}, {torques.max(axis=0)}]")

        # Obs history check
        obs0 = np.array(records[0]['obs_history'])
        print(f"\nObs history at step 0 (270 dims):")
        print(f"  first 45 (newest): [{obs0[:45].min():.3f}, {obs0[:45].max():.3f}]")
        print(f"  last 45 (oldest):  [{obs0[-45:].min():.3f}, {obs0[-45:].max():.3f}]")
        print(f"  all zeros? {np.allclose(obs0[45:], 0)}")

        obs10 = np.array(records[10]['obs_history']) if len(records) > 10 else obs0
        print(f"\nObs history at step 10:")
        print(f"  first 45 (newest): [{obs10[:45].min():.3f}, {obs10[:45].max():.3f}]")
        print(f"  all zeros in old slots? {np.allclose(obs10[-45:], 0)}")


if __name__ == "__main__":
    args = parse_args()
    play_record(args)
