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
import omni.usd
from pxr import Gf, UsdGeom, UsdShade
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer, FrameTransformerData

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv



def button_press_hold_success(
    env: ManagerBasedRLEnv,
    press_threshold: float,
    press_joint_name: str = "PrismaticJoint",
    hold_time_s: float = 2.0,
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
    for i, env_prim in enumerate(env.scene.env_prim_paths):
        ref_prim = ref_prim_expr.replace(env.scene.env_regex_ns, env_prim)
        # Set displayColor only on state change to avoid any perceived gradual updates
        if not hasattr(env, "_refpoint_color_state"):
            env._refpoint_color_state = {}
        prev_state = env._refpoint_color_state.get(ref_prim, None)
        cur_state = bool(pressed_mask[i].item())
        if prev_state is None or prev_state != cur_state:
            _bind_referencepoint_material(ref_prim, default_color, active_color, cur_state)
            env._refpoint_color_state[ref_prim] = cur_state


def _bind_referencepoint_material(
    ref_prim_path: str,
    default_color: tuple[float, float, float],
    active_color: tuple[float, float, float],
    use_active: bool,
) -> None:
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(ref_prim_path)
    if not prim.IsValid():
        return
    # Ensure materials exist (create once)
    default_mat_path = f"{ref_prim_path}/Looks/DefaultMat"
    active_mat_path = f"{ref_prim_path}/Looks/ActiveMat"
    default_mat = _ensure_preview_surface(default_mat_path, default_color)
    active_mat = _ensure_preview_surface(active_mat_path, active_color)
    target_mat = active_mat if use_active else default_mat

    # Bind to all Gprims under this prim (instant swap)
    gprims = []
    def _collect_gprims(p):
        if p.IsA(UsdGeom.Gprim):
            gprims.append(UsdGeom.Gprim(p))
        for child in p.GetChildren():
            _collect_gprims(child)

    _collect_gprims(prim)
    if not gprims:
        return
    for gprim in gprims:
        UsdShade.MaterialBindingAPI(gprim).Bind(target_mat)


def _ensure_preview_surface(prim_path: str, color: tuple[float, float, float]) -> UsdShade.Material:
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        cfg = sim_utils.PreviewSurfaceCfg(diffuse_color=color)
        sim_utils.spawn_preview_surface(prim_path, cfg)
        prim = stage.GetPrimAtPath(prim_path)
    return UsdShade.Material(prim)
