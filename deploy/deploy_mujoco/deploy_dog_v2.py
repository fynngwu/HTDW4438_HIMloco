"""
dog_v2 ONNX policy MuJoCo sim2sim deployment
基于 deploy_4438_v2.py 适配, 所有参数与 IsaacGym dog_v2_config.py 对齐

用法:
  python deploy/deploy_mujoco/deploy_dog_v2.py
  python deploy/deploy_mujoco/deploy_dog_v2.py --onnx /path/to/model.onnx
"""
import time
import os
import argparse
import numpy as np
import mujoco
import mujoco.viewer
import onnxruntime as ort
import yaml
from collections import deque
from pynput import keyboard
from onnx_path_utils import resolve_onnx_path

# ================= 1. 路径配置 =================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
YAML_PATH = os.path.join(SCRIPT_DIR, "configs", "dog_v2.yaml")


def parse_args():
    parser = argparse.ArgumentParser(description="Deploy dog_v2 HIMLoco policy in MuJoCo.")
    parser.add_argument("--onnx", type=str, default=None, help="Path to ONNX policy file.")
    return parser.parse_args()


ARGS = parse_args()

# 先读取 YAML 获取 XML 路径
with open(YAML_PATH, "r") as f:
    _cfg = yaml.load(f, Loader=yaml.FullLoader)
XML_PATH = os.path.abspath(_cfg["xml_path"].replace("{PROJECT_ROOT}", PROJECT_ROOT))

ONNX_PATH = resolve_onnx_path(
    project_root=PROJECT_ROOT,
    cli_onnx=ARGS.onnx,
    env_vars=["DOG_V2_ONNX_PATH"],
    onnx_glob="flat_dog_v2*.onnx",
    robot_name="dog_v2",
)

print(f"YAML: {YAML_PATH}")
print(f"XML : {XML_PATH}")
print(f"ONNX: {ONNX_PATH}")

# ================= 2. 全局变量 =================
cmd = np.array([0.0, 0.0, 0.0], dtype=np.float32)
paused = False
default_dof_pos = None

# ================= 3. 辅助函数 =================

def quat_rotate_inverse(q, v):
    """将向量 v 从世界坐标系旋转到四元数 q 表示的机身坐标系
    q: [w, x, y, z] (MuJoCo 格式), v: [x, y, z]
    """
    q_w = q[0]
    q_vec = q[1:4]
    a = v * (2.0 * q_w ** 2 - 1.0)
    b = np.cross(q_vec, v) * q_w * 2.0
    c = q_vec * np.dot(q_vec, v) * 2.0
    return a - b + c


def pd_control(target_q, q, kp, target_dq, dq, kd):
    return (target_q - q) * kp + (target_dq - dq) * kd


def build_policy_input(history_buffer, input_dim, num_obs):
    """构建策略输入, 支持 270 维历史观测或 64 维单帧
    注意: 不修改 history_buffer, 调用方在推理后自行 appendleft
    """
    if input_dim == num_obs:
        return np.concatenate(list(history_buffer), axis=0).reshape(1, -1)
    if input_dim == 64:
        return np.zeros((1, 64), dtype=np.float32)
    if input_dim == 45:
        return history_buffer[0].reshape(1, -1)
    raise ValueError(f"Unsupported ONNX input dim: {input_dim}")


# ================= 4. 键盘控制 =================
def on_press(key):
    global cmd
    try:
        if key == keyboard.Key.up:
            cmd[0] = 0.5
        elif key == keyboard.Key.down:
            cmd[0] = -0.5
        elif key == keyboard.Key.left:
            cmd[2] = 1.0
        elif key == keyboard.Key.right:
            cmd[2] = -1.0
    except AttributeError:
        pass


def on_release(key):
    global cmd
    try:
        if key == keyboard.Key.up or key == keyboard.Key.down:
            cmd[0] = 0.0
        elif key == keyboard.Key.left or key == keyboard.Key.right:
            cmd[2] = 0.0
    except AttributeError:
        pass


def key_callback(keycode):
    global paused
    if chr(keycode) == ' ':
        paused = not paused
        print(f"Paused: {paused}")


# ================= 5. 主程序 =================
def run_simulation():
    global cmd, default_dof_pos

    # --- 加载 YAML 配置 ---
    with open(YAML_PATH, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    sim_dt = float(config.get('simulation_dt', 0.005))
    control_decimation = int(config.get('control_decimation', 2))
    num_obs = int(config.get('num_obs', 270))
    num_one_step_obs = int(config.get('num_one_step_obs', 45))
    init_base_height = float(config.get('init_base_height', 0.3))
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

    # 初始前进速度
    cmd[0] = cmd_x_init

    # --- 加载 MuJoCo & ONNX ---
    print("Loading MuJoCo model...")
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)
    model.opt.timestep = sim_dt

    # 检测 IMU 陀螺仪传感器
    use_gyro_sensor = True
    try:
        _ = data.sensor("angular-velocity").data
    except KeyError:
        use_gyro_sensor = False
        print("[WARN] No 'angular-velocity' sensor found, falling back to data.qvel[3:6].")

    print(f"Loading ONNX: {ONNX_PATH}")
    ort_session = ort.InferenceSession(ONNX_PATH)
    input_name = ort_session.get_inputs()[0].name
    input_shape = ort_session.get_inputs()[0].shape
    input_dim = int(input_shape[-1]) if isinstance(input_shape[-1], int) else num_obs
    print(f"ONNX input: {input_name}, shape: {input_shape}, dim: {input_dim}")

    # --- 初始化状态 ---
    data.qpos[7:] = default_dof_pos
    data.qpos[2] = init_base_height
    mujoco.mj_forward(model, data)

    target_dof_pos = default_dof_pos.copy()
    action = np.zeros(12, dtype=np.float32)

    # 观测历史缓冲 (6 帧, 与 IsaacGym history_length 一致)
    history_len = max(1, num_obs // num_one_step_obs)
    obs_dim = num_one_step_obs
    obs_history_buffer = deque(
        [np.zeros(obs_dim, dtype=np.float32) for _ in range(history_len)],
        maxlen=history_len,
    )

    # 打印关键参数确认
    print("=" * 50)
    print("dog_v2 sim2sim deployment")
    print(f"  sim_dt={sim_dt}, decimation={control_decimation}, policy_freq={1/(sim_dt*control_decimation):.0f}Hz")
    print(f"  Kp={kps[0]}, Kd={kds[0]}, action_scale={action_scale}")
    print(f"  obs={num_obs}, one_step={num_one_step_obs}, history={history_len}")
    print(f"  cmd_scale={cmd_scale}")
    print(f"  default_dof_pos={default_dof_pos}")
    print(f"  init_height={init_base_height}")
    print(f"  ctrlrange (from XML): {model.actuator_ctrlrange[:, 1]}")
    print("=" * 50)

    # 键盘监听
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()
    print("Arrow keys: move | Space: pause | Ctrl+C: quit")

    # --- 仿真循环 ---
    step_counter = 0
    need_obs_update = False
    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        while viewer.is_running():
            step_start = time.time()

            if not paused:
                # ================= 策略推理 (at decimation boundary) =================
                if step_counter % control_decimation == 0:
                    # Build policy input from CURRENT history buffer
                    # (starts as all zeros, matching IsaacGym obs_buf initialization)
                    policy_input = build_policy_input(
                        history_buffer=obs_history_buffer,
                        input_dim=input_dim,
                        num_obs=num_obs,
                    )

                    # ONNX 推理
                    outputs = ort_session.run(None, {input_name: policy_input})
                    raw_action = outputs[0][0].astype(np.float32)

                    # 动作后处理 (与 IsaacGym _compute_torques 一致)
                    # clip → scale → add default → PD control
                    raw_action = np.clip(raw_action, -100.0, 100.0)
                    action = raw_action.copy()
                    target_dof_pos = raw_action * action_scale + default_dof_pos

                    need_obs_update = True

                # ================= 物理步进 (runs every substep) =================
                tau = pd_control(target_dof_pos, data.qpos[7:], kps,
                                 np.zeros_like(kds), data.qvel[6:], kds)
                # 裁剪到执行器 ctrlrange (从 XML 读取)
                tau_limit = np.abs(model.actuator_ctrlrange[:, 1])
                tau = np.clip(tau, -tau_limit, tau_limit)
                data.ctrl[:] = tau

                mujoco.mj_step(model, data)
                step_counter += 1

                # ================= 观测更新 (after physics, at decimation boundary) =================
                # matching IsaacGym: compute_observations runs in post_physics_step
                if need_obs_update and step_counter % control_decimation == 0:
                    qj = data.qpos[7:].astype(np.float32)
                    dqj = data.qvel[6:].astype(np.float32)
                    quat = data.qpos[3:7].astype(np.float32)

                    if use_gyro_sensor:
                        omega = data.sensor("angular-velocity").data.astype(np.float32)
                    else:
                        omega = data.qvel[3:6].astype(np.float32)

                    gravity_vec = np.array([0., 0., -1.], dtype=np.float32)
                    proj_gravity = quat_rotate_inverse(quat, gravity_vec)

                    # Build obs from POST-physics state with the action just applied
                    obs_list = []
                    obs_list.extend(cmd * cmd_scale)                             # 0-2:  cmd
                    obs_list.extend(omega * ang_vel_scale)                        # 3-5:  ang_vel
                    obs_list.extend(proj_gravity)                                 # 6-8:  gravity (no scale)
                    obs_list.extend((qj - default_dof_pos) * dof_pos_scale)       # 9-20: dof_pos offset
                    obs_list.extend(dqj * dof_vel_scale)                          # 21-32: dof_vel
                    obs_list.extend(action)                                       # 33-44: last action

                    obs_raw = np.array(obs_list, dtype=np.float32)
                    obs_history_buffer.appendleft(obs_raw.copy())
                    need_obs_update = False

            viewer.sync()

            # 实时步进 (不等待, 让 MuJoCo viewer 控制帧率)
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)


if __name__ == "__main__":
    run_simulation()
