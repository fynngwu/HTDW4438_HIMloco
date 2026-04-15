"""
MuJoCo sim2sim 数据记录脚本
记录观测、动作、力矩等数据用于与 IsaacGym play 对比

用法:
  uv run python deploy/deploy_mujoco/record_mujoco.py
  uv run python deploy/deploy_mujoco/record_mujoco.py --steps 500
"""
import time, os, argparse, json, numpy as np, mujoco, yaml
from collections import deque
from onnx_path_utils import resolve_onnx_path

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
YAML_PATH = os.path.join(SCRIPT_DIR, "configs", "dog_v2.yaml")

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--onnx", type=str, default=None)
    p.add_argument("--steps", type=int, default=500, help="Number of policy steps to record")
    p.add_argument("--out", type=str, default="deploy/deploy_mujoco/mujoco_record.json")
    return p.parse_args()

ARGS = parse_args()

with open(YAML_PATH, "r") as f:
    _cfg = yaml.load(f, Loader=yaml.FullLoader)
XML_PATH = os.path.abspath(_cfg["xml_path"].replace("{PROJECT_ROOT}", PROJECT_ROOT))
ONNX_PATH = resolve_onnx_path(
    project_root=PROJECT_ROOT, cli_onnx=ARGS.onnx,
    env_vars=["DOG_V2_ONNX_PATH"], onnx_glob="flat_dog_v2*.onnx", robot_name="dog_v2",
)

import onnxruntime as ort

def quat_rotate_inverse(q, v):
    q_w, q_vec = q[0], q[1:4]
    return v * (2.0 * q_w**2 - 1.0) - np.cross(q_vec, v) * q_w * 2.0 + q_vec * np.dot(q_vec, v) * 2.0

def run():
    with open(YAML_PATH, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    sim_dt = float(config['simulation_dt'])
    control_decimation = int(config['control_decimation'])
    num_obs = int(config['num_obs'])
    num_one_step_obs = int(config['num_one_step_obs'])
    init_base_height = float(config['init_base_height'])
    cmd_x_init = float(config.get('cmd_x_init', 1.0))

    kps = np.array(config['kps'], dtype=np.float32)
    kds = np.array(config['kds'], dtype=np.float32)
    default_dof_pos = np.array(config['default_angles'], dtype=np.float32)

    lin_vel_scale = config['lin_vel_scale']
    ang_vel_scale = config['ang_vel_scale']
    dof_pos_scale = config['dof_pos_scale']
    dof_vel_scale = config['dof_vel_scale']
    action_scale = config['action_scale']
    cmd_scale = np.array(config['cmd_scale'], dtype=np.float32)

    cmd = np.array([cmd_x_init, 0.0, 0.0], dtype=np.float32)

    print(f"Loading MuJoCo: {XML_PATH}")
    mj_model = mujoco.MjModel.from_xml_path(XML_PATH)
    mj_data = mujoco.MjData(mj_model)
    mj_model.opt.timestep = sim_dt

    print(f"Loading ONNX: {ONNX_PATH}")
    sess = ort.InferenceSession(ONNX_PATH)
    input_name = sess.get_inputs()[0].name
    input_dim = int(sess.get_inputs()[0].shape[-1])

    # Init state
    mj_data.qpos[7:] = default_dof_pos
    mj_data.qpos[2] = init_base_height
    mujoco.mj_forward(mj_model, mj_data)

    target_dof_pos = default_dof_pos.copy()
    action = np.zeros(12, dtype=np.float32)

    history_len = max(1, num_obs // num_one_step_obs)
    obs_history_buffer = deque(
        [np.zeros(num_one_step_obs, dtype=np.float32) for _ in range(history_len)],
        maxlen=history_len,
    )

    records = []

    step_counter = 0
    policy_step = 0
    pending_record = None

    print(f"Recording {ARGS.steps} policy steps...")

    while policy_step < ARGS.steps:
        # === Policy inference (at decimation boundary) ===
        if step_counter % control_decimation == 0:
            # 1. Build policy input from CURRENT history buffer
            # (starts as all zeros, matching IsaacGym obs_buf initialization)
            policy_input = np.concatenate(list(obs_history_buffer), axis=0).reshape(1, -1)

            # 2. Run inference
            outputs = sess.run(None, {input_name: policy_input})
            raw_action = outputs[0][0].astype(np.float32)

            clipped_action = np.clip(raw_action, -100.0, 100.0)
            action = clipped_action.copy()
            target_dof_pos = clipped_action * action_scale + default_dof_pos

            pending_record = {
                "step": policy_step,
                "sim_time": step_counter * sim_dt,
                "base_pos": mj_data.qpos[:3].tolist(),
                "base_quat": mj_data.qpos[3:7].tolist(),
                "dof_pos": mj_data.qpos[7:].tolist(),
                "dof_vel": mj_data.qvel[6:].tolist(),
                "policy_input": policy_input[0].tolist(),
                "raw_action": raw_action.tolist(),
                "clipped_action": clipped_action.tolist(),
                "target_dof_pos": target_dof_pos.tolist(),
                "cmd": cmd.tolist(),
            }
            policy_step += 1

        # === Physics step (runs every substep) ===
        tau = (target_dof_pos - mj_data.qpos[7:]) * kps - mj_data.qvel[6:] * kds
        tau_limit = np.abs(mj_model.actuator_ctrlrange[:, 1])
        tau = np.clip(tau, -tau_limit, tau_limit)
        mj_data.ctrl[:] = tau

        mujoco.mj_step(mj_model, mj_data)
        step_counter += 1

        # Check if robot fell
        if mj_data.qpos[2] < 0.1:
            print(f"Robot fell at policy step {policy_step}, sim_time={step_counter*sim_dt:.2f}s")
            break

        # === Post-physics observation (at decimation boundary) ===
        if pending_record is not None and step_counter % control_decimation == 0:
            qj = mj_data.qpos[7:].astype(np.float32)
            dqj = mj_data.qvel[6:].astype(np.float32)
            quat = mj_data.qpos[3:7].astype(np.float32)
            omega = mj_data.qvel[3:6].astype(np.float32)

            gravity_vec = np.array([0., 0., -1.], dtype=np.float32)
            proj_gravity = quat_rotate_inverse(quat, gravity_vec)

            # 3. Build obs from POST-physics state with the action just applied
            obs_raw = np.array([
                *cmd * cmd_scale,
                *omega * ang_vel_scale,
                *proj_gravity,
                *(qj - default_dof_pos) * dof_pos_scale,
                *dqj * dof_vel_scale,
                *action,  # the action that was just applied during physics
            ], dtype=np.float32)
            obs_history_buffer.appendleft(obs_raw.copy())

            pending_record["base_pos_after"] = mj_data.qpos[:3].tolist()
            pending_record["base_quat_after"] = mj_data.qpos[3:7].tolist()
            pending_record["dof_pos_after"] = mj_data.qpos[7:].tolist()
            pending_record["dof_vel_after"] = mj_data.qvel[6:].tolist()
            pending_record["obs_raw"] = obs_raw.tolist()
            records.append(pending_record)
            pending_record = None

    # Save
    out_path = os.path.join(PROJECT_ROOT, ARGS.out)
    with open(out_path, "w") as f:
        json.dump(records, f, indent=2)
    print(f"Recorded {len(records)} policy steps to {out_path}")
    print(f"Final sim_time: {records[-1]['sim_time']:.2f}s" if records else "No data recorded")

    # Quick stats
    if records:
        actions = np.array([r['clipped_action'] for r in records])
        dof_pos = np.array([r['dof_pos'] for r in records])
        dof_vel = np.array([r['dof_vel'] for r in records])
        print(f"\nAction stats:")
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

if __name__ == "__main__":
    run()
