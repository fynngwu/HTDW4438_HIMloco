# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import os
import sys
import copy
from datetime import datetime

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from legged_gym import LEGGED_GYM_ROOT_DIR

import isaacgym
from isaacgym import gymapi
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger

import numpy as np
import torch
import torch.nn.functional as F


class PolicyExporterHIMOnnx(torch.nn.Module):
    def __init__(self, actor_critic):
        super().__init__()
        self.actor = copy.deepcopy(actor_critic.actor)
        self.estimator = copy.deepcopy(actor_critic.estimator.encoder)

    def forward(self, obs_history):
        parts = self.estimator(obs_history)[:, 0:19]
        vel, z = parts[..., :3], parts[..., 3:]
        z = F.normalize(z, dim=-1, p=2.0)
        return self.actor(torch.cat((obs_history[:, 0:45], vel, z), dim=1))


def local_export_policy_as_onnx(actor_critic, path, obs_example):
    if hasattr(actor_critic, "estimator"):
        model = PolicyExporterHIMOnnx(actor_critic)
    elif hasattr(actor_critic, "actor"):
        model = actor_critic.actor
    else:
        model = actor_critic
    model.eval()

    os.makedirs(os.path.dirname(path), exist_ok=True)

    if hasattr(actor_critic, "estimator"):
        input_dim = obs_example.shape[-1]
    elif hasattr(model, "__getitem__") and hasattr(model[0], "in_features"):
        input_dim = model[0].in_features
    else:
        input_dim = obs_example.shape[-1]
    dummy_input = torch.zeros(
        1,
        input_dim,
        device=obs_example.device,
        dtype=obs_example.dtype,
    )

    print(f"Exporting policy as ONNX to: {path}")
    torch.onnx.export(
        model,
        dummy_input,
        path,
        opset_version=11,
        verbose=False,
        input_names=["obs"],
        output_names=["actions"],
        dynamic_axes={
            "obs": {0: "batch"},
            "actions": {0: "batch"},
        },
    )
    print("ONNX export done.")


def _safe_filename_part(value):
    value = str(value).strip()
    for ch in ("/", "\\", " ", "\t", "\n", ":", ";"):
        value = value.replace(ch, "_")
    return value


def build_onnx_filename(project_name, checkpoint):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ckpt_title = f"model_{checkpoint}" if checkpoint is not None else "model_auto"
    return f"{_safe_filename_part(project_name)}_{timestamp}_{_safe_filename_part(ckpt_title)}.onnx"


def _subscribe_play_keyboard(env):
    if env.viewer is None:
        return
    keymap = {
        gymapi.KEY_UP: "forward",
        gymapi.KEY_W: "forward",
        gymapi.KEY_DOWN: "backward",
        gymapi.KEY_S: "backward",
        gymapi.KEY_Q: "left",
        gymapi.KEY_E: "right",
        gymapi.KEY_LEFT: "yaw_left",
        gymapi.KEY_A: "yaw_left",
        gymapi.KEY_RIGHT: "yaw_right",
        gymapi.KEY_D: "yaw_right",
        gymapi.KEY_SPACE: "zero_cmd",
    }
    for key, action in keymap.items():
        env.gym.subscribe_viewer_keyboard_event(env.viewer, key, action)


def _update_play_command_from_keyboard(env, cmd, pressed, lin_x, lin_y, yaw):
    if env.viewer is None:
        return
    for evt in env.gym.query_viewer_action_events(env.viewer):
        if evt.action == "QUIT" and evt.value > 0:
            sys.exit()
        if evt.action == "toggle_viewer_sync" and evt.value > 0:
            env.enable_viewer_sync = not env.enable_viewer_sync
            continue
        if evt.action == "zero_cmd" and evt.value > 0:
            for key in pressed:
                pressed[key] = False
            cmd[:] = 0.0
            continue
        if evt.action in pressed:
            pressed[evt.action] = evt.value > 0

    cmd[0] = lin_x if pressed["forward"] else (-lin_x if pressed["backward"] else 0.0)
    cmd[1] = lin_y if pressed["left"] else (-lin_y if pressed["right"] else 0.0)
    cmd[2] = yaw if pressed["yaw_left"] else (-yaw if pressed["yaw_right"] else 0.0)


def play(args, x_vel=0.3, y_vel=0.0, yaw_vel=0.0):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = 1
    env_cfg.terrain.num_rows = 10
    env_cfg.terrain.num_cols = 8
    env_cfg.terrain.curriculum = True
    env_cfg.terrain.max_init_terrain_level = 9
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    args = get_args()
    env_cfg.domain_rand.disturbance = False
    env_cfg.domain_rand.randomize_payload_mass = False
    env_cfg.commands.heading_command = False
    # env_cfg.terrain.mesh_type = 'plane'
    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    camera_offset = np.array(env_cfg.viewer.pos, dtype=np.float64) - np.array(env_cfg.viewer.lookat, dtype=np.float64)
    robot_position = env.root_states[0, :3].detach().cpu().numpy().astype(np.float64)
    env.set_camera(robot_position + camera_offset, robot_position)
    cmd = np.array([x_vel, y_vel, yaw_vel], dtype=np.float32)
    env.commands[:, 0] = cmd[0]
    env.commands[:, 1] = cmd[1]
    env.commands[:, 2] = cmd[2]

    pressed = {
        "forward": False,
        "backward": False,
        "left": False,
        "right": False,
        "yaw_left": False,
        "yaw_right": False,
    }
    _subscribe_play_keyboard(env)
    lin_x_cmd = 0.8
    lin_y_cmd = 0.4
    yaw_cmd = 1.0
    print("Keyboard control: Up/W forward | Down/S backward | Q/E strafe | Left/A yaw+ | Right/D yaw- | Space zero")

    obs = env.get_observations()
    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)


    # export policy as a jit module (used to run it from C++)
    if EXPORT_POLICY:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
        export_policy_as_jit(ppo_runner.alg.actor_critic, path)
        print('Exported policy as jit script to: ', path)

    if EXPORT_ONNX:
        target_dir = os.path.join(LEGGED_GYM_ROOT_DIR, "onnx")
        project_name = getattr(train_cfg.runner, "experiment_name", None) or args.task
        onnx_filename = build_onnx_filename(project_name, args.checkpoint)
        onnx_full_path = os.path.join(target_dir, onnx_filename)
        local_export_policy_as_onnx(ppo_runner.alg.actor_critic, onnx_full_path, obs)

    logger = Logger(env.dt)
    robot_index = 0 # which robot is used for logging
    joint_index = 1 # which joint is used for logging
    stop_state_log = 100 # number of steps before plotting states
    stop_rew_log = env.max_episode_length + 1 # number of steps before print average episode rewards
    camera_position = np.array(env_cfg.viewer.pos, dtype=np.float64)
    camera_vel = np.array([1., 1., 0.])
    camera_direction = np.array(env_cfg.viewer.lookat) - np.array(env_cfg.viewer.pos)
    img_idx = 0

    for i in range(10*int(env.max_episode_length)):
        _update_play_command_from_keyboard(env, cmd, pressed, lin_x_cmd, lin_y_cmd, yaw_cmd)
        actions = policy(obs.detach())
        env.commands[:, 0] = cmd[0]
        env.commands[:, 1] = cmd[1]
        env.commands[:, 2] = cmd[2]
        obs, _, rews, dones, infos, _, _ = env.step(actions.detach())

        if RECORD_FRAMES:
            if i % 2:
                filename = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'frames', f"{img_idx}.png")
                env.gym.write_viewer_image_to_file(env.viewer, filename)
                img_idx += 1
        if MOVE_CAMERA:
            camera_position += camera_vel * env.dt
            env.set_camera(camera_position, camera_position + camera_direction)

        if i < stop_state_log:
            logger.log_states(
                {
                    'dof_pos_target': actions[robot_index, joint_index].item() * env.cfg.control.action_scale + env.default_dof_pos[robot_index, joint_index].item(),
                    'dof_pos': env.dof_pos[robot_index, joint_index].item(),
                    'dof_vel': env.dof_vel[robot_index, joint_index].item(),
                    'dof_torque': env.torques[robot_index, joint_index].item(),
                    'command_x': env.commands[robot_index, 0].item(),
                    'command_y': env.commands[robot_index, 1].item(),
                    'command_yaw': env.commands[robot_index, 2].item(),
                    'base_vel_x': env.base_lin_vel[robot_index, 0].item(),
                    'base_vel_y': env.base_lin_vel[robot_index, 1].item(),
                    'base_vel_z': env.base_lin_vel[robot_index, 2].item(),
                    'base_vel_yaw': env.base_ang_vel[robot_index, 2].item(),
                    'contact_forces_z': env.contact_forces[robot_index, env.feet_indices, 2].cpu().numpy()
                }
            )
        elif i==stop_state_log:
            logger.plot_states()
        if  0 < i < stop_rew_log:
            if infos["episode"]:
                num_episodes = torch.sum(env.reset_buf).item()
                if num_episodes>0:
                    logger.log_rewards(infos["episode"], num_episodes)
        elif i==stop_rew_log:
            logger.print_rewards()


if __name__ == '__main__':
    EXPORT_POLICY = True
    EXPORT_ONNX = True
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    args = get_args()
    play(args, x_vel=1.0, y_vel=0.0, yaw_vel=0.0)
