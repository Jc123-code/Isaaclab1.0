# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
import isaaclab.utils.math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def switch_joint_reached_position(
    env: ManagerBasedRLEnv,
    target_position_deg: float = -14.11,
    tolerance_deg: float = 8.0,
    velocity_threshold_deg_s: float | None = 1.0,
    hold_time_s: float = 0.2,
    joint_name: str = "RevoluteJoint_1",
    debug_print: bool = False,
    debug_env_id: int = 0,
    debug_every_n_steps: int = 10,
) -> torch.Tensor:
    """Succeed when a container joint reaches a target position and stops moving."""
    container: Articulation = env.scene["container"]

    if not hasattr(env, "_container_joint_idx_map"):
        env._container_joint_idx_map = {}
    if joint_name not in env._container_joint_idx_map:
        env._container_joint_idx_map[joint_name] = (
            container.joint_names.index(joint_name) if joint_name in container.joint_names else 0
        )
    joint_idx = env._container_joint_idx_map[joint_name]

    joint_pos = container.data.joint_pos[:, joint_idx]
    joint_vel = container.data.joint_vel[:, joint_idx]
    target = math.radians(target_position_deg)
    tolerance = math.radians(tolerance_deg)
    delta = math_utils.wrap_to_pi(joint_pos - target)

    position_reached = torch.abs(delta) <= tolerance
    if velocity_threshold_deg_s is None:
        stopped = torch.ones_like(position_reached, dtype=torch.bool)
    else:
        stopped = torch.abs(joint_vel) <= math.radians(velocity_threshold_deg_s)

    reset_mask = env.episode_length_buf <= 1
    if not hasattr(env, "_container_target_reached_state") or env._container_target_reached_state.shape != position_reached.shape:
        env._container_target_reached_state = torch.zeros_like(position_reached, dtype=torch.bool)
    if not hasattr(env, "_container_target_hold_time"):
        env._container_target_hold_time = torch.zeros(env.scene.num_envs, device=env.device)
    if torch.any(reset_mask):
        env._container_target_reached_state[reset_mask] = False
        env._container_target_hold_time[reset_mask] = 0.0
    env._container_target_reached_state = env._container_target_reached_state | position_reached
    reached = env._container_target_reached_state & stopped
    env._container_target_hold_time = torch.where(
        reached, env._container_target_hold_time + env.step_dt, torch.zeros_like(env._container_target_hold_time)
    )
    success = env._container_target_hold_time >= hold_time_s

    if debug_print and 0 <= debug_env_id < joint_pos.shape[0]:
        if not hasattr(env, "_target_success_debug_counter") or env._target_success_debug_counter.shape[0] != env.scene.num_envs:
            env._target_success_debug_counter = torch.zeros(env.scene.num_envs, device=env.device, dtype=torch.long)
        env._target_success_debug_counter += 1
        should_print = reset_mask[debug_env_id] or (
            env._target_success_debug_counter[debug_env_id] % debug_every_n_steps == 0
        )
        if should_print:
            print(
                "[target-success-debug]",
                f"env={debug_env_id}",
                f"joint={joint_name}",
                f"joint_deg={math.degrees(joint_pos[debug_env_id].item()):.2f}",
                f"joint_vel_deg_s={math.degrees(joint_vel[debug_env_id].item()):.2f}",
                f"target_deg={target_position_deg:.2f}",
                f"delta_deg={math.degrees(delta[debug_env_id].item()):.2f}",
                f"tolerance_deg={tolerance_deg:.2f}",
                f"velocity_threshold_deg_s={velocity_threshold_deg_s}",
                f"position_reached={bool(position_reached[debug_env_id].item())}",
                f"target_reached_latched={bool(env._container_target_reached_state[debug_env_id].item())}",
                f"stopped={bool(stopped[debug_env_id].item())}",
                f"hold_time={env._container_target_hold_time[debug_env_id].item():.3f}",
                f"success={bool(success[debug_env_id].item())}",
                f"episode_step={int(env.episode_length_buf[debug_env_id].item())}",
            )

    return success
