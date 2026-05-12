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
) -> torch.Tensor:
    """Terminate when the switch articulation joint rotates beyond a threshold and is held."""
    switch: Articulation = env.scene["switch"]
    # resolve joint index once
    if not hasattr(env, "_switch_joint_idx"):
        if joint_name and joint_name in switch.joint_names:
            env._switch_joint_idx = switch.joint_names.index(joint_name)
        else:
            # fallback to first joint
            env._switch_joint_idx = 0
    joint_idx = env._switch_joint_idx

    joint_pos = switch.data.joint_pos[:, joint_idx]
    # baseline at episode start
    if not hasattr(env, "_switch_joint_baseline") or env._switch_joint_baseline.shape != joint_pos.shape:
        env._switch_joint_baseline = joint_pos.clone()
    reset_mask = env.episode_length_buf == 0
    if torch.any(reset_mask):
        env._switch_joint_baseline[reset_mask] = joint_pos[reset_mask]
    delta = torch.abs(joint_pos - env._switch_joint_baseline)

    rotated = delta >= math.radians(angle_threshold_deg)

    if not hasattr(env, "_switch_hold_time"):
        env._switch_hold_time = torch.zeros(env.scene.num_envs, device=env.device)
    env._switch_hold_time = torch.where(
        rotated, env._switch_hold_time + env.step_dt, torch.zeros_like(env._switch_hold_time)
    )
    success = env._switch_hold_time >= hold_time_s
    return success
