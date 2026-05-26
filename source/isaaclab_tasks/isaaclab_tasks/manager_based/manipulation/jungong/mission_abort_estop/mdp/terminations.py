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

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer, FrameTransformerData

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv



def plate_success(env: ManagerBasedRLEnv, dist_threshold: float = 0.05):
    drawer_tf_data: FrameTransformerData = env.scene["table_frame"].data
    
    # 计算各轴距离
    dist_x = torch.norm(drawer_tf_data.target_pos_w[..., 0, [0]] - drawer_tf_data.target_pos_w[..., 1, [0]], dim=-1)
    dist_y = torch.norm(drawer_tf_data.target_pos_w[..., 0, [1]] - drawer_tf_data.target_pos_w[..., 1, [1]], dim=-1)
    dist_z = torch.norm(drawer_tf_data.target_pos_w[..., 0, [2]] - drawer_tf_data.target_pos_w[..., 1, [2]], dim=-1)
    
    # 三个条件
    x_condition = dist_x < (0.05)
    y_condition = dist_y < (dist_threshold )
    z_condition = dist_z <= 0.05
    # print(f"{dist_z}")
    # 组合条件
    success = x_condition & y_condition & z_condition
    
    print(f"X Distance: {dist_x}, Y Distance: {dist_y}, Z Distance: {dist_z}, Success: {success}")
    
    return success


def button_press_hold_success(
    env: ManagerBasedRLEnv,
    press_joint_name: str = "PrismaticJoint",
    press_threshold: float = -0.01,
    hold_time_s: float = 1.0,
    default_color: tuple[float, float, float] = (0.5, 0.0, 1.0),
    active_color: tuple[float, float, float] = (0.0, 1.0, 0.0),
) -> torch.Tensor:
    """Terminate when the button is pressed beyond a threshold and held for a duration.

    Also changes the ReferencePoint color while the button is pressed.
    """
    # -- resolve button articulation and joint index (cache per env instance)
    button: Articulation = env.scene["button"]
    if not hasattr(env, "_button_press_joint_idx"):
        if press_joint_name in button.joint_names:
            env._button_press_joint_idx = button.joint_names.index(press_joint_name)
        elif len(button.joint_names) == 1:
            # fallback: single joint articulation
            env._button_press_joint_idx = 0
        else:
            raise ValueError(f"Cannot find joint '{press_joint_name}' in button joints: {button.joint_names}")
    joint_idx = env._button_press_joint_idx

    # -- compute pressed condition (relative to per-episode baseline)
    joint_pos = button.data.joint_pos[:, joint_idx]
    if not hasattr(env, "_button_press_baseline") or env._button_press_baseline.shape != joint_pos.shape:
        env._button_press_baseline = joint_pos.clone()
    # reset baseline at the start of each episode
    reset_mask = env.episode_length_buf == 0
    if torch.any(reset_mask):
        env._button_press_baseline[reset_mask] = joint_pos[reset_mask]
    baseline = env._button_press_baseline
    pressed = (joint_pos - baseline) <= press_threshold
    print(
        f"[button] joint_pos={joint_pos.detach().cpu().numpy()} "
        f"baseline={baseline.detach().cpu().numpy()} "
        f"delta={(joint_pos - baseline).detach().cpu().numpy()} "
        f"pressed={pressed.detach().cpu().numpy()}"
    )

    # -- setup hold timer buffer
    if not hasattr(env, "_button_press_hold_time"):
        env._button_press_hold_time = torch.zeros(env.scene.num_envs, device=env.device)

    # -- update timer
    env._button_press_hold_time = torch.where(
        pressed, env._button_press_hold_time + env.step_dt, torch.zeros_like(env._button_press_hold_time)
    )

    # -- change ReferencePoint color based on pressed state
    _set_referencepoint_color(env, pressed, default_color, active_color)

    # -- success when held long enough
    success = env._button_press_hold_time >= hold_time_s
    return success


def _set_referencepoint_color(
    env: ManagerBasedRLEnv,
    pressed_mask: torch.Tensor,
    default_color: tuple[float, float, float],
    active_color: tuple[float, float, float],
) -> None:
    """Bind visual material to ReferencePoint per-env based on pressed state."""
    ref_obj: RigidObject = env.scene["ReferencePoint"]
    ref_prim_expr = ref_obj.cfg.prim_path
    # Build prim path per env
    # Cache material creation so we don't recreate every step
    if not hasattr(env, "_refpoint_materials_created"):
        env._refpoint_materials_created = set()

    for i, env_prim in enumerate(env.scene.env_prim_paths):
        ref_prim = ref_prim_expr.replace(env.scene.env_regex_ns, env_prim)
        # Prepare material paths
        default_mtl_path = f"{ref_prim}/Looks/DefaultMat"
        active_mtl_path = f"{ref_prim}/Looks/ActiveMat"

        # Create materials if missing (once per env)
        if ref_prim not in env._refpoint_materials_created:
            default_cfg = sim_utils.PreviewSurfaceCfg(diffuse_color=default_color)
            active_cfg = sim_utils.PreviewSurfaceCfg(diffuse_color=active_color)
            default_cfg.func(default_mtl_path, default_cfg)
            active_cfg.func(active_mtl_path, active_cfg)
            env._refpoint_materials_created.add(ref_prim)

        # Bind based on pressed state
        if bool(pressed_mask[i].item()):
            sim_utils.bind_visual_material(ref_prim, active_mtl_path)
        else:
            sim_utils.bind_visual_material(ref_prim, default_mtl_path)
