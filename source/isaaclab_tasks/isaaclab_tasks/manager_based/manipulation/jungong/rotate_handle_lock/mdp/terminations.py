# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to activate certain terminations for the lift task.

The functions can be passed to the :class:`isaaclab.managers.TerminationTermCfg` object to enable
the termination introduced by the function.
"""

from __future__ import annotations

import torch
import math
from typing import TYPE_CHECKING

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer, FrameTransformerData
import isaaclab.utils.math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def switch_joint_rotated_success(
    env: ManagerBasedRLEnv,
    angle_threshold_deg: float = 90.0,
    hold_time_s: float = 0.5,
    joint_name: str | None = None,
    direction: str = "abs",
) -> torch.Tensor:
    """Terminate when the container articulation joint rotates beyond a threshold and is held."""
    container: Articulation = env.scene["container"]
    # resolve joint index once
    if not hasattr(env, "_container_joint_idx"):
        if joint_name and joint_name in container.joint_names:
            env._container_joint_idx = container.joint_names.index(joint_name)
        else:
            # fallback to first joint
            env._container_joint_idx = 0
    joint_idx = env._container_joint_idx

    joint_pos = container.data.joint_pos[:, joint_idx]
    # baseline at (early) episode start to avoid reset timing offsets
    if not hasattr(env, "_container_joint_baseline") or env._container_joint_baseline.shape != joint_pos.shape:
        env._container_joint_baseline = joint_pos.clone()
    reset_mask = env.episode_length_buf <= 1
    if torch.any(reset_mask):
        env._container_joint_baseline[reset_mask] = joint_pos[reset_mask]
    delta = joint_pos - env._container_joint_baseline

    threshold = math.radians(angle_threshold_deg)
    if direction == "cw":
        rotated = delta <= -threshold
    elif direction == "ccw":
        rotated = delta >= threshold
    else:
        rotated = torch.abs(delta) >= threshold

    if not hasattr(env, "_container_hold_time"):
        env._container_hold_time = torch.zeros(env.scene.num_envs, device=env.device)
    if torch.any(reset_mask):
        env._container_hold_time[reset_mask] = 0.0
    env._container_hold_time = torch.where(
        rotated, env._container_hold_time + env.step_dt, torch.zeros_like(env._container_hold_time)
    )
    success = env._container_hold_time >= hold_time_s
    return success
