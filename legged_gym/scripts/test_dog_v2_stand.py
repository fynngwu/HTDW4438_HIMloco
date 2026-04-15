"""
dog_v2 站立验证脚本
用法: python legged_gym/scripts/test_dog_v2_stand.py [--headless] [--num_steps 500]
"""
import os
import sys
import argparse

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="dog_v2")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--num_steps", type=int, default=500)
    args_cli, _ = parser.parse_known_args()

    # 使用框架自带的 get_args 获取正确的 isaacgym 参数
    # 但用我们的 task/headless 覆盖
    sys.argv = [sys.argv[0], f"--task={args_cli.task}"]
    if args_cli.headless:
        sys.argv.append("--headless")
    args = get_args()

    # 获取配置并覆盖为调试用参数
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    env_cfg.env.num_envs = 1
    env_cfg.terrain.mesh_type = "plane"
    env_cfg.terrain.num_rows = 1
    env_cfg.terrain.num_cols = 1
    env_cfg.terrain.border_size = 0.0
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.randomize_payload_mass = False
    env_cfg.domain_rand.randomize_motor_strength = False
    env_cfg.domain_rand.randomize_kp = False
    env_cfg.domain_rand.randomize_kd = False
    env_cfg.domain_rand.disturbance = False
    env_cfg.domain_rand.delay = False
    env_cfg.commands.resampling_time = 1000.0
    env_cfg.commands.curriculum = False

    # 创建环境
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)

    # 设置相机正对机器人
    if not args_cli.headless:
        import numpy as np
        cam_pos = np.array([1.0, 1.0, 0.6], dtype=np.float64)
        cam_lookat = np.array([0.0, 0.0, 0.15], dtype=np.float64)
        env.set_camera(cam_pos, cam_lookat)

    print("=" * 60)
    print("dog_v2 站立验证")
    print("=" * 60)
    print(f"DOF names: {env.dof_names}")
    print(f"Feet indices: {env.feet_indices}")
    print(f"Num DOFs: {env.num_dof}")
    print(f"Default DOF pos: {env.default_dof_pos[0].cpu().numpy()}")
    print(f"Init base height: {env_cfg.init_state.pos[2]} m")
    print(f"Kp: {env_cfg.control.stiffness}, Kd: {env_cfg.control.damping}")
    print("=" * 60)

    # 用零动作步进（保持默认关节角度）
    obs = env.get_observations()
    num_steps = args_cli.num_steps

    print(f"\n步进 {num_steps} 步（零动作 = 默认站立姿态）...\n")

    for step in range(num_steps):
        actions = torch.zeros(env.num_envs, env.num_actions, device=env.device)
        obs, _, rews, dones, infos, _, _ = env.step(actions)

        # 每100步打印状态
        if step % 100 == 0 or step == num_steps - 1:
            base_pos = env.root_states[0, :3].cpu().numpy()
            base_quat = env.root_states[0, 3:7].cpu().numpy()
            base_lin_vel = env.root_states[0, 7:10].cpu().numpy()
            dof_pos = env.dof_pos[0].cpu().numpy()

            # 从四元数计算 roll 和 pitch
            w, x, y, z = base_quat
            import math
            roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x**2 + y**2))
            pitch = math.asin(max(-1, min(1, 2 * (w * y - z * x))))

            # 检查脚是否接触地面
            contact_z = env.contact_forces[0, env.feet_indices, 2].cpu().numpy()

            print(f"[Step {step:4d}] "
                  f"height={base_pos[2]:.4f}m  "
                  f"roll={roll:+.2f}°  pitch={pitch:+.2f}°  "
                  f"vel_z={base_lin_vel[2]:+.3f}m/s  "
                  f"contacts={[f'{c:.1f}' for c in contact_z]}")

            if step == num_steps - 1:
                print(f"\n关节位置 (deg):")
                for i, name in enumerate(env.dof_names):
                    angle_deg = dof_pos[i] * 180.0 / 3.14159
                    default_deg = env.default_dof_pos[0, i].item() * 180.0 / 3.14159
                    print(f"  {name:20s}: {angle_deg:+8.2f}° (default: {default_deg:+.2f}°)")

    # 最终评估
    base_height = env.root_states[0, 2].item()
    print("\n" + "=" * 60)
    print("评估结果:")
    print(f"  最终 base height: {base_height:.4f} m (目标: {env_cfg.init_state.pos[2]} m)")
    print(f"  高度偏差: {abs(base_height - env_cfg.init_state.pos[2])*100:.1f} cm")

    if abs(base_height - env_cfg.init_state.pos[2]) < 0.05:
        print("  [OK] 高度稳定，站立正常")
    elif base_height < env_cfg.init_state.pos[2] * 0.5:
        print("  [FAIL] 机器人可能已倒下")
    else:
        print("  [WARN] 高度有偏差，可能需要调整默认关节角度")

    contact_z = env.contact_forces[0, env.feet_indices, 2].cpu().numpy()
    feet_on_ground = sum(1 for c in contact_z if c > 1.0)
    print(f"  脚接触地面数: {feet_on_ground}/4 "
          f"(力: {[f'{c:.1f}N' for c in contact_z]})")
    print("=" * 60)

    # 保持viewer打开以便目视检查
    if not args_cli.headless:
        print("\nViewer 已打开，按 Ctrl+C 退出...")
        try:
            while True:
                env.gym.fetch_results(env.sim, True)
                env.gym.step_graphics(env.sim)
                env.gym.draw_viewer(env.viewer, env.sim, True)
        except KeyboardInterrupt:
            print("已退出。")


if __name__ == "__main__":
    main()
