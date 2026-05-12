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
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import RigidObject
from isaaclab.sensors import FrameTransformerData

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def clean_mortarbarrel_success(
    env: ManagerBasedRLEnv,
    default_mortar_pose: tuple[float, ...] = (0.97829, -0.3658, 1.02122, 0.69901, 0.7005, 0.10065, 0.10276),
    success_center_w: tuple[float, float, float] = (0.65053, -0.46199, 1.24521),
    sphere_radius: float = 0.05,
) -> torch.Tensor:
    drawer_tf_data: FrameTransformerData = env.scene["table_frame"].data
    bursh_pos_w = drawer_tf_data.target_pos_w[..., 0, :]
    mortar: RigidObject = env.scene["mortar"]
    mortar_root_state_w = mortar.data.root_state_w[: bursh_pos_w.shape[0]]

    default_mortar_pose_tensor = torch.tensor(default_mortar_pose[:7], device=bursh_pos_w.device, dtype=bursh_pos_w.dtype)
    default_mortar_pos = default_mortar_pose_tensor[:3]
    default_mortar_quat = default_mortar_pose_tensor[3:7].unsqueeze(0)
    success_center_w_tensor = torch.tensor(success_center_w, device=bursh_pos_w.device, dtype=bursh_pos_w.dtype)
    success_center_local = math_utils.quat_rotate_inverse(
        default_mortar_quat, (success_center_w_tensor - default_mortar_pos).unsqueeze(0)
    ).squeeze(0)

    center = mortar_root_state_w[:, :3] + math_utils.quat_rotate(
        mortar_root_state_w[:, 3:7],
        success_center_local.unsqueeze(0).expand(bursh_pos_w.shape[0], -1),
    )
    dist_to_center = torch.linalg.vector_norm(bursh_pos_w - center, dim=-1)
    inside_sphere = dist_to_center <= sphere_radius

    state_name = "_clean_mortarbarrel_bursh_entered_sphere"
    entered_sphere = getattr(env, state_name, None)
    if entered_sphere is None or entered_sphere.shape != inside_sphere.shape:
        entered_sphere = torch.zeros_like(inside_sphere, dtype=torch.bool)

    if hasattr(env, "episode_length_buf"):
        prev_len_name = "_clean_mortarbarrel_prev_episode_length_buf"
        prev_len = getattr(env, prev_len_name, None)
        cur_len = env.episode_length_buf

        if prev_len is None or prev_len.shape != cur_len.shape:
            prev_len = cur_len.clone()
            entered_sphere[:] = False
        else:
            # Reset boundary for each sub-env: episode length wraps from a larger value to a smaller one.
            reset_ids = cur_len <= prev_len
            entered_sphere[reset_ids] = False

        setattr(env, prev_len_name, cur_len.clone())

    entered_sphere |= inside_sphere
    setattr(env, state_name, entered_sphere)

    success = entered_sphere & (~inside_sphere)
    return success
