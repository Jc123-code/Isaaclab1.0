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

def gun_handle_pulled_success(
    env: ManagerBasedRLEnv,
    press_joint_name: str = "PrismaticJoint",
    pull_threshold: float = -0.3,
    hold_time_s: float = 0.5,
) -> torch.Tensor:
    """Terminate when gun_handle is pulled along the prismatic joint by a threshold and held.

    Notes:
    - Uses per-episode baseline, so success is based on relative displacement.
    - With axis set to local Z, pulling backward is typically negative displacement.
    """
    gun: Articulation = env.scene["switch"]

    if not hasattr(env, "_gun_pull_joint_idx"):
        if press_joint_name in gun.joint_names:
            env._gun_pull_joint_idx = gun.joint_names.index(press_joint_name)
        elif len(gun.joint_names) == 1:
            env._gun_pull_joint_idx = 0
        else:
            raise ValueError(f"Cannot find joint '{press_joint_name}' in gun joints: {gun.joint_names}")
    joint_idx = env._gun_pull_joint_idx

    joint_pos = gun.data.joint_pos[:, joint_idx]
    if not hasattr(env, "_gun_pull_baseline") or env._gun_pull_baseline.shape != joint_pos.shape:
        env._gun_pull_baseline = joint_pos.clone()
    reset_mask = env.episode_length_buf == 0
    if torch.any(reset_mask):
        env._gun_pull_baseline[reset_mask] = joint_pos[reset_mask]

    delta = joint_pos - env._gun_pull_baseline
    pulled = delta <= pull_threshold

    # Debug print every step: report bolt joint displacement of env_0.
    jp0 = float(joint_pos[0].item())
    b0 = float(env._gun_pull_baseline[0].item())
    d0 = float(delta[0].item())
    print(
        f"[gun_bolt_joint] pos={jp0:.4f}, baseline={b0:.4f}, "
        f"delta={d0:.4f}, pulled={bool(pulled[0].item())}"
    )

    if hold_time_s <= 0.0:
        return pulled

    if not hasattr(env, "_gun_pull_hold_time"):
        env._gun_pull_hold_time = torch.zeros(env.scene.num_envs, device=env.device)
    env._gun_pull_hold_time = torch.where(
        pulled, env._gun_pull_hold_time + env.step_dt, torch.zeros_like(env._gun_pull_hold_time)
    )
    success = env._gun_pull_hold_time >= hold_time_s
    return success
