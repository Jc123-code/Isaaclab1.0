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

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer, FrameTransformerData

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def clean_mortarbarrel_success(
    env: ManagerBasedRLEnv,
    sphere_center: tuple[float, float, float] = (0.62295, -0.50855, 1.3843),
    sphere_radius: float = 0.05,
) -> torch.Tensor:
    drawer_tf_data: FrameTransformerData = env.scene["table_frame"].data
    bursh_pos_w = drawer_tf_data.target_pos_w[..., 0, :]

    center = torch.tensor(sphere_center, device=bursh_pos_w.device, dtype=bursh_pos_w.dtype)
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
