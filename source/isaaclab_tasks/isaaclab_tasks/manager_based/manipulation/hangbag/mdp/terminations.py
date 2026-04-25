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

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def get_bag_success(
    env: ManagerBasedRLEnv,
    target_pos: tuple[float, float, float] = (0.64845, 0.52333, 1.1895),
    success_radius: float = 0.06,
    wait_time: float = 0.5,
):
    """Return termination 0.5s after the bag first enters the target sphere.

    Notes:
        - The target center is interpreted in each sub-environment's local frame.
        - Entering the sphere is treated as success immediately and latched for the episode.
        - The environment resets only after waiting ``wait_time`` seconds after that success.
    """

    bag_pos_local = env.scene["bag"].data.root_pos_w[:, :3] - env.scene.env_origins[:, :3]
    target = torch.tensor(target_pos, device=bag_pos_local.device, dtype=bag_pos_local.dtype).unsqueeze(0)
    dist_to_target = torch.linalg.norm(bag_pos_local - target, dim=-1)
    inside_target = dist_to_target <= success_radius

    latched_name = "_bag_success_latched"
    timer_name = "_bag_success_timer"
    prev_len_name = "_bag_prev_episode_length_buf"

    success_latched = getattr(env, latched_name, None)
    success_timer = getattr(env, timer_name, None)

    if success_latched is None or success_latched.shape != inside_target.shape:
        success_latched = torch.zeros_like(inside_target, dtype=torch.bool)
    if success_timer is None or success_timer.shape != inside_target.shape:
        success_timer = torch.zeros_like(inside_target, dtype=torch.float32)

    if hasattr(env, "episode_length_buf"):
        cur_len = env.episode_length_buf
        prev_len = getattr(env, prev_len_name, None)

        if prev_len is None or prev_len.shape != cur_len.shape:
            success_latched.zero_()
            success_timer.zero_()
        else:
            reset_ids = cur_len <= prev_len
            success_latched[reset_ids] = False
            success_timer[reset_ids] = 0.0

        setattr(env, prev_len_name, cur_len.clone())

    newly_successful = inside_target & (~success_latched)
    success_latched |= inside_target

    dt = 1.0 / 60.0
    if hasattr(env, "step_dt"):
        dt = env.step_dt
    elif hasattr(env, "physics_dt"):
        dt = env.physics_dt

    success_timer[success_latched] += dt
    success_timer[newly_successful] = 0.0

    setattr(env, latched_name, success_latched)
    setattr(env, timer_name, success_timer)

    return success_latched & (success_timer >= wait_time)
