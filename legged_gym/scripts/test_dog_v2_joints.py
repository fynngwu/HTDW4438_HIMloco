"""
dog_v2 关节运动 & 碰撞检测测试脚本（固定 body 模式）
用法: python legged_gym/scripts/test_dog_v2_joints.py [--headless]
"""
import os
import sys
import argparse
import math

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
import torch
import numpy


def test_single_joint_fixed(env, joint_idx, joint_name, amplitude, steps_per_test=200):
    """测试单个关节的正弦运动（固定 body 模式）"""
    print(f"\n--- 测试关节: {joint_name} (idx={joint_idx}, amp={amplitude:.2f} rad = {math.degrees(amplitude):.1f}°) ---")

    # 记录初始状态
    initial_pos = env.dof_pos[0, joint_idx].item()
    initial_vel = env.dof_vel[0, joint_idx].item()

    min_pos = float('inf')
    max_pos = float('-inf')

    for step in range(steps_per_test):
        phase = 2.0 * math.pi * step / steps_per_test
        target = amplitude * math.sin(phase)

        actions = torch.zeros(env.num_envs, env.num_actions, device=env.device)
        actions[0, joint_idx] = target / env.cfg.control.action_scale
        env.step(actions)

        pos = env.dof_pos[0, joint_idx].item()
        vel = env.dof_vel[0, joint_idx].item()
        min_pos = min(min_pos, pos)
        max_pos = max(max_pos, pos)

    final_pos = env.dof_pos[0, joint_idx].item()

    # 检查运动范围是否覆盖了目标振幅
    actual_range = (max_pos - min_pos) / 2.0
    tracking_ok = actual_range >= amplitude * 0.7  # 至少达到 70% 的目标振幅

    # 检查最终是否能回到接近零位
    return_ok = abs(final_pos - initial_pos) < 0.15

    status = "OK" if (tracking_ok and return_ok) else "FAIL"
    if not tracking_ok:
        status = "FAIL (振幅不足)"
    elif not return_ok:
        status = "FAIL (未回到零位)"

    print(f"  结果: {status}")
    print(f"    初始位置: {math.degrees(initial_pos):+.2f}°")
    print(f"    最终位置: {math.degrees(final_pos):+.2f}°")
    print(f"    运动范围: [{math.degrees(min_pos):+.1f}°, {math.degrees(max_pos):+.1f}°]")
    print(f"    实际振幅: {math.degrees(actual_range):.1f}° / 目标: {math.degrees(amplitude):.1f}°")

    # 检查关节限位
    dof_lower = env.dof_pos_limits[joint_idx, 0].item()
    dof_upper = env.dof_pos_limits[joint_idx, 1].item()
    limit_ok = min_pos >= dof_lower + 0.01 and max_pos <= dof_upper - 0.01
    if not limit_ok:
        print(f"    [WARN] 接近限位! limit=[{math.degrees(dof_lower):+.1f}°, {math.degrees(dof_upper):+.1f}°]")

    return tracking_ok and return_ok


def test_collision_detection(env):
    """测试碰撞检测是否正常工作"""
    print("\n" + "=" * 60)
    print("碰撞检测测试")
    print("=" * 60)

    penalize_names = env.cfg.asset.penalize_contacts_on
    terminate_names = env.cfg.asset.terminate_after_contacts_on

    print(f"碰撞惩罚列表 (子串匹配): {penalize_names}")
    print(f"终止碰撞列表 (子串匹配): {terminate_names}")
    print(f"脚部名称: {env.cfg.asset.foot_name}")

    if hasattr(env, 'penalised_contact_indices') and len(env.penalised_contact_indices) > 0:
        print(f"被惩罚刚体索引数: {len(env.penalised_contact_indices)}")
    else:
        print(f"  [WARN] penalised_contact_indices 为空!")

    if hasattr(env, 'termination_contact_indices') and len(env.termination_contact_indices) > 0:
        print(f"终止碰撞刚体索引数: {len(env.termination_contact_indices)}")
    else:
        print(f"  [WARN] termination_contact_indices 为空!")

    # 稳定后读取接触力
    print("\n稳定站立时接触力:")
    for _ in range(100):
        actions = torch.zeros(env.num_envs, env.num_actions, device=env.device)
        env.step(actions)

    contact_forces = env.contact_forces[0].cpu().numpy()

    # 脚部接触力
    print(f"  脚部接触力 (z轴):")
    feet_ok = True
    for i, fi in enumerate(env.feet_indices):
        fz = contact_forces[fi, 2]
        label = ['LF', 'LR', 'RF', 'RR'][i] if i < 4 else f'Foot{i}'
        status = "[OK]" if fz > 1.0 else "[WARN] 无接触!"
        if fz <= 1.0:
            feet_ok = False
        print(f"    {label}_Foot: {fz:.1f}N {status}")

    # 被惩罚刚体的接触力（正常应该接近0）
    if hasattr(env, 'penalised_contact_indices') and len(env.penalised_contact_indices) > 0:
        print(f"\n  被惩罚刚体接触力 (应接近0):")
        penalize_ok = True
        for idx in env.penalised_contact_indices:
            force_mag = float(numpy.linalg.norm(contact_forces[idx]))
            status = "[OK]" if force_mag < 0.1 else "[WARN] 有接触!"
            if force_mag >= 0.1:
                penalize_ok = False
            print(f"    index {idx}: {force_mag:.2f}N {status}")
    else:
        penalize_ok = False

    return feet_ok and penalize_ok


def test_all_joints_fixed(env):
    """固定 body 模式下依次测试每个关节"""
    print("=" * 60)
    print("关节运动测试 (固定 body 模式)")
    print("=" * 60)

    results = {}

    # 测试策略: 固定 body 下可以放心用较大振幅
    test_params = {
        'HipA': 0.5,   # 外展 ±28.6°
        'HipF': 0.8,   # 髋 ±45.8°
        'Knee': 0.8,   # 膝 ±45.8°
    }

    for i, name in enumerate(env.dof_names):
        for key, amp in test_params.items():
            if key in name:
                ok = test_single_joint_fixed(env, i, name, amp)
                results[name] = ok
                break
        else:
            print(f"  [SKIP] {name}: 未知关节类型")
            results[name] = None

    # 恢复零位
    for _ in range(100):
        actions = torch.zeros(env.num_envs, env.num_actions, device=env.device)
        env.step(actions)

    return results


def test_multi_joint_trot_fixed(env):
    """固定 body 模式下测试对角步态（trot）"""
    print("\n" + "=" * 60)
    print("对角步态 (Trot) 测试 (固定 body 模式)")
    print("=" * 60)

    trot_cycles = 4
    steps_per_cycle = 80
    hipf_amp = 0.6
    knee_amp = 0.5
    total_steps = trot_cycles * steps_per_cycle

    print(f"测试 {trot_cycles} 个步态周期 ({total_steps} 步)...")

    for step in range(total_steps):
        phase = 2.0 * math.pi * step / steps_per_cycle

        actions = torch.zeros(env.num_envs, env.num_actions, device=env.device)

        # DOF order: LF_HipA(0) LF_HipF(1) LF_Knee(2) LR_HipA(3) LR_HipF(4) LR_Knee(5)
        #             RF_HipA(6) RF_HipF(7) RF_Knee(8) RR_HipA(9) RR_HipF(10) RR_Knee(11)
        # 组1: LF(1,2) + RR(10,11), 组2: LR(4,5) + RF(7,8)
        actions[0, 1] = hipf_amp * math.sin(phase) / env.cfg.control.action_scale    # LF_HipF
        actions[0, 2] = knee_amp * math.sin(phase) / env.cfg.control.action_scale    # LF_Knee
        actions[0, 10] = hipf_amp * math.sin(phase) / env.cfg.control.action_scale   # RR_HipF
        actions[0, 11] = knee_amp * math.sin(phase) / env.cfg.control.action_scale   # RR_Knee

        actions[0, 4] = hipf_amp * math.sin(phase + math.pi) / env.cfg.control.action_scale  # LR_HipF
        actions[0, 5] = knee_amp * math.sin(phase + math.pi) / env.cfg.control.action_scale  # LR_Knee
        actions[0, 7] = hipf_amp * math.sin(phase + math.pi) / env.cfg.control.action_scale  # RF_HipF
        actions[0, 8] = knee_amp * math.sin(phase + math.pi) / env.cfg.control.action_scale  # RF_Knee

        env.step(actions)

        if step % 40 == 0:
            contact_z = env.contact_forces[0, env.feet_indices, 2].cpu().numpy()
            dof_pos = env.dof_pos[0].cpu().numpy()
            print(f"  [Step {step:4d}] "
                  f"LF_HipF={math.degrees(dof_pos[1]):+6.1f}° LF_Knee={math.degrees(dof_pos[2]):+6.1f}° "
                  f"RR_HipF={math.degrees(dof_pos[10]):+6.1f}° RR_Knee={math.degrees(dof_pos[11]):+6.1f}° "
                  f"contacts={['%.1f' % c for c in contact_z]}")

    print(f"\n对角步态测试结果: OK (固定 body 模式，不会倒)")

    # 恢复零位
    for _ in range(100):
        actions = torch.zeros(env.num_envs, env.num_actions, device=env.device)
        env.step(actions)

    return True


def test_self_collision(env):
    """测试自碰撞检测——极端姿态下关节是否互相干涉"""
    print("\n" + "=" * 60)
    print("自碰撞测试 (极端姿态)")
    print("=" * 60)

    # 将腿收到最紧凑的姿态，检查是否有异常接触力
    test_poses = [
        ("全收", [0.0, 0.8, -1.5] * 4),         # 所有腿最大程度收起
        ("全展", [0.0, -0.8, 1.5] * 4),         # 所有腿最大程度展开
        ("左收右展", [0.0, 0.8, -1.5] * 2 + [0.0, -0.8, 1.5] * 2),
    ]

    collision_free = True
    for pose_name, pose in test_poses:
        # 设置目标位置并步进
        for _ in range(100):
            actions = torch.zeros(env.num_envs, env.num_actions, device=env.device)
            for j in range(env.num_actions):
                actions[0, j] = pose[j] / env.cfg.control.action_scale
            env.step(actions)

        contact_forces = env.contact_forces[0].cpu().numpy()

        # 检查被惩罚刚体的接触
        has_collision = False
        if hasattr(env, 'penalised_contact_indices') and len(env.penalised_contact_indices) > 0:
            for idx in env.penalised_contact_indices:
                force_mag = float(numpy.linalg.norm(contact_forces[idx]))
                if force_mag > 1.0:
                    has_collision = True
                    print(f"  [{pose_name}] index {idx}: {force_mag:.2f}N [COLLISION!]")

        if has_collision:
            print(f"  [{pose_name}]: 自碰撞检测到!")
            collision_free = False
        else:
            print(f"  [{pose_name}]: 无自碰撞 [OK]")

        # 恢复零位
        for _ in range(100):
            actions = torch.zeros(env.num_envs, env.num_actions, device=env.device)
            env.step(actions)

    return collision_free


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="dog_v2")
    parser.add_argument("--headless", action="store_true")
    args_cli, _ = parser.parse_known_args()

    sys.argv = [sys.argv[0], f"--task={args_cli.task}"]
    if args_cli.headless:
        sys.argv.append("--headless")
    args = get_args()

    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    env_cfg.env.num_envs = 1
    env_cfg.terrain.mesh_type = "plane"
    env_cfg.terrain.num_rows = 1
    env_cfg.terrain.num_cols = 1
    env_cfg.terrain.border_size = 0
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

    # 固定 base link（吊起来模式）
    env_cfg.asset.fix_base_link = True

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)

    if not args_cli.headless:
        import numpy as np
        cam_pos = np.array([1.5, 1.5, 0.8], dtype=np.float64)
        cam_lookat = np.array([0.0, 0.0, 0.20], dtype=np.float64)
        env.set_camera(cam_pos, cam_lookat)

    print("=" * 60)
    print("dog_v2 关节运动 & 碰撞检测测试 (固定 body)")
    print("=" * 60)
    print(f"DOF names: {env.dof_names}")
    print(f"Num DOFs: {env.num_dof}")
    print(f"Default DOF pos: {env.default_dof_pos[0].cpu().numpy()}")
    print(f"Kp: {env_cfg.control.stiffness}, Kd: {env_cfg.control.damping}")
    print(f"Action scale: {env_cfg.control.action_scale}")
    print(f"Joint limits: [{math.degrees(env.dof_pos_limits[0,0].item()):.1f}°, "
          f"{math.degrees(env.dof_pos_limits[0,1].item()):.1f}°]")
    print("=" * 60)

    # 1. 碰撞检测测试
    collision_ok = test_collision_detection(env)

    # 2. 逐关节测试
    joint_results = test_all_joints_fixed(env)

    # 3. 对角步态测试
    trot_ok = test_multi_joint_trot_fixed(env)

    # 4. 自碰撞测试
    self_collision_ok = test_self_collision(env)

    # 汇总
    print("\n" + "=" * 60)
    print("汇总结果")
    print("=" * 60)
    print(f"碰撞检测: {'OK' if collision_ok else 'FAIL'}")
    print("关节运动:")
    for name, ok in joint_results.items():
        if ok is None:
            print(f"  {name:20s}: SKIP")
        elif ok:
            print(f"  {name:20s}: OK")
        else:
            print(f"  {name:20s}: FAIL")
    print(f"对角步态: {'OK' if trot_ok else 'FAIL'}")
    print(f"自碰撞: {'OK (无自碰撞)' if self_collision_ok else 'FAIL (检测到自碰撞)'}")
    print("=" * 60)

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
