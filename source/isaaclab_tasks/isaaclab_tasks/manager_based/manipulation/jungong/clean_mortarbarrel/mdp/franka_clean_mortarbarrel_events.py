# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import math
import random
import warnings
import torch
from typing import TYPE_CHECKING, Sequence, Union
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, AssetBase
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, PhysxSchema
import omni.usd


def _as_seq(x):
    return (x,) if not isinstance(x, (list, tuple)) else x


def _extract_pose_from_matrix(transform: Gf.Matrix4d) -> tuple[Gf.Vec3f, Gf.Quatf]:
    """Convert a USD transform matrix into translation/quaternion attributes for joint frames."""
    decomposed = Gf.Transform(transform)
    translation = decomposed.GetTranslation()
    quatd = decomposed.GetRotation().GetQuat()
    return (
        Gf.Vec3f(float(translation[0]), float(translation[1]), float(translation[2])),
        Gf.Quatf(
            float(quatd.GetReal()),
            float(quatd.GetImaginary()[0]),
            float(quatd.GetImaginary()[1]),
            float(quatd.GetImaginary()[2]),
        ),
    )


def create_fixed_joint_between_assets(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    parent_asset_cfg: SceneEntityCfg,
    child_asset_cfg: SceneEntityCfg,
    joint_name: str = "table_fixed_joint",
):
    """Create a fixed joint between two rigid bodies for every environment instance.

    The child body keeps its current pose relative to the parent body when the joint is created.
    """

    stage = omni.usd.get_context().get_stage()
    parent_asset = env.scene[parent_asset_cfg.name]
    child_asset = env.scene[child_asset_cfg.name]

    if env_ids is None:
        env_id_list = list(range(env.num_envs))
    elif isinstance(env_ids, torch.Tensor):
        env_id_list = env_ids.detach().cpu().tolist()
    else:
        env_id_list = list(env_ids)

    parent_paths = list(parent_asset.root_physx_view.prim_paths[: parent_asset.num_instances])
    child_paths = list(child_asset.root_physx_view.prim_paths[: child_asset.num_instances])
    xform_cache = UsdGeom.XformCache()

    for env_id in env_id_list:
        parent_path = parent_paths[env_id]
        child_path = child_paths[env_id]
        fixed_joint_path = f"{child_path}/{joint_name}"

        if stage.GetPrimAtPath(fixed_joint_path).IsValid():
            continue

        parent_prim = stage.GetPrimAtPath(parent_path)
        child_prim = stage.GetPrimAtPath(child_path)
        if not parent_prim.IsValid() or not child_prim.IsValid():
            warnings.warn(
                f"[create_fixed_joint_between_assets] Invalid prims: parent={parent_path}, child={child_path}",
                stacklevel=2,
            )
            continue

        parent_world = xform_cache.GetLocalToWorldTransform(parent_prim)
        child_world = xform_cache.GetLocalToWorldTransform(child_prim)

        # Choose the child's current body frame as the joint frame so the existing relative pose is preserved.
        parent_to_joint = child_world * parent_world.GetInverse()
        child_to_joint = Gf.Matrix4d(1.0)
        local_pos0, local_rot0 = _extract_pose_from_matrix(parent_to_joint)
        local_pos1, local_rot1 = _extract_pose_from_matrix(child_to_joint)

        joint = UsdPhysics.FixedJoint.Define(stage, Sdf.Path(fixed_joint_path))
        joint.CreateBody0Rel().SetTargets([Sdf.Path(parent_path)])
        joint.CreateBody1Rel().SetTargets([Sdf.Path(child_path)])
        joint.CreateLocalPos0Attr().Set(local_pos0)
        joint.CreateLocalRot0Attr().Set(local_rot0)
        joint.CreateLocalPos1Attr().Set(local_pos1)
        joint.CreateLocalRot1Attr().Set(local_rot1)
        joint.CreateJointEnabledAttr().Set(True)



def set_default_joint_pose(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    default_pose: Union[torch.Tensor, Sequence[torch.Tensor]] | None,
    asset_cfg: Union[SceneEntityCfg, Sequence[SceneEntityCfg]] = SceneEntityCfg("robot"),
):
    """
    """
    cfgs = _as_seq(asset_cfg)

    if default_pose is not None:
        poses = _as_seq(default_pose)
        for pose, cfg  in zip(poses, cfgs):
            asset: Articulation = env.scene[cfg.name]
            pose = torch.as_tensor(pose, device=env.device).view(1, -1)  # (1, DoF)
            dof = asset.data.default_joint_pos.shape[-1]
            if pose.shape[-1] != dof:
                if pose.shape[-1] > dof:
                    pose = pose[..., :dof]
                else:
                    pose_padded = asset.data.default_joint_pos.clone()  # (1, dof)
                    pose_padded[..., : pose.shape[-1]] = pose
                    pose = pose_padded
                warnings.warn(
                    f"[set_default_joint_pose] default_pose dim mismatch for '{cfg.name}': "
                    f"got {pose.shape[-1]} vs dof {dof}. Adapted to match DOF.",
                    stacklevel=2,
                )
            assert pose.shape[-1] == dof
            asset.data.default_joint_pos = (pose.repeat(env.num_envs, 1))
    else:
        for cfg  in cfgs:
            asset: Articulation = env.scene[cfg.name]
            pose = asset.data.default_joint_pos  # (1, DoF)
            dof = asset.data.default_joint_pos.shape[-1]
            assert pose.shape[-1] == dof 
            asset.data.default_joint_pos = (pose.repeat(env.num_envs, 1))

def reset_to_default_joint_pose(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    default_pose: Union[torch.Tensor, Sequence[torch.Tensor]] | None,
    asset_cfg: Union[SceneEntityCfg, Sequence[SceneEntityCfg]] = SceneEntityCfg("robot"),
):
    """
    """
    cfgs = _as_seq(asset_cfg)

    if default_pose is not None:
        poses = _as_seq(default_pose)
        for pose, cfg  in zip(poses, cfgs):
            asset: Articulation = env.scene[cfg.name]
            pose = torch.as_tensor(pose, device=env.device).view(1, -1)  # (1, DoF)
            dof = asset.data.default_joint_pos.shape[-1]
            if pose.shape[-1] != dof:
                if pose.shape[-1] > dof:
                    pose = pose[..., :dof]
                else:
                    pose_padded = asset.data.default_joint_pos.clone()  # (1, dof)
                    pose_padded[..., : pose.shape[-1]] = pose
                    pose = pose_padded
                warnings.warn(
                    f"[reset_to_default_joint_pose] default_pose dim mismatch for '{cfg.name}': "
                    f"adapted to dof {dof}.",
                    stacklevel=2,
                )
            assert pose.shape[-1] == dof
            asset.write_joint_position_to_sim(pose.repeat(env.num_envs, 1))
    else:
        for cfg  in cfgs:
            asset: Articulation = env.scene[cfg.name]
            pose = asset.data.default_joint_pos  # (1, DoF)
            dof = asset.data.default_joint_pos.shape[-1]
            assert pose.shape[-1] == dof 
            asset.write_joint_position_to_sim(pose.repeat(env.num_envs, 1))

def randomize_joint_by_gaussian_offset(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    mean: float,
    std: float,
    asset_cfg: Union[SceneEntityCfg, Sequence[SceneEntityCfg]] = SceneEntityCfg("robot"),
    keep_last_two_as_gripper: bool = True,
):
    """对一个或多个机器人的关节状态在 default 附近加高斯噪声并写回仿真。
    - 对每个 asset：
        * q := default_joint_pos[env_ids] + N(mean, std)
        * clamp 到 soft_joint_pos_limits
        * 可选：最后两个关节（通常是 gripper）不加噪声
        * 速度用 default_joint_vel[env_ids]
        * 同步 set_position/velocity_target + write_joint_state_to_sim
    """
    cfgs = _as_seq(asset_cfg)

    for cfg in cfgs:
        asset: Articulation = env.scene[cfg.name]

        # 取默认值
        q0 = asset.data.default_joint_pos[env_ids].clone()          # (E, DoF)
        qd0 = asset.data.default_joint_vel[env_ids].clone()         # (E, DoF)

        # 加噪
        noise = math_utils.sample_gaussian(mean, std, q0.shape, q0.device)
        q = q0 + noise

        # 限幅
        limits = asset.data.soft_joint_pos_limits[env_ids]          # (E, DoF, 2)
        q = q.clamp_(limits[..., 0], limits[..., 1])

        # 夹爪关节保持默认（假设为最后两个 DoF；若不是请改成显式索引）
        if keep_last_two_as_gripper and q.shape[-1] >= 9:
            q[:, -2:] = q0[:, -2:]

        # 写入仿真 & 目标
        asset.set_joint_position_target(q, env_ids=env_ids)
        asset.set_joint_velocity_target(qd0, env_ids=env_ids)
        asset.write_joint_state_to_sim(q, qd0, env_ids=env_ids)



def reset_pose_to_default(env, env_ids, default_pose, asset_cfg):
    """
    Reset one or multiple book objects to their default poses and zero velocities.
    
    Args:
        env: 当前仿真环境对象
        env_ids: 要重置的环境 ID 列表 (tensor or list)
        default_pose: tensor 或 list[ tensor ]
            每个书本的目标姿态。可为单个 (13,) tensor 或每个 asset 对应一个 (13,)。
        asset_cfg: list[SceneEntityCfg]
            对应的书本资产配置，如 [book_red, book_blue]
    """

    # 如果 default_pose 是单个 Tensor，则统一封装成列表
    if isinstance(default_pose, torch.Tensor):
        default_pose = [default_pose]
    elif isinstance(default_pose, list):
        if not all(isinstance(p, torch.Tensor) for p in default_pose):
            raise TypeError("Each default_pose element must be a torch.Tensor.")
    else:
        raise TypeError("default_pose must be a Tensor or list of Tensors.")

    # 遍历每个书本资产及其对应的默认姿态
    for i, cfg in enumerate(asset_cfg):
        asset = env.scene[cfg.name]
        pose = default_pose[min(i, len(default_pose) - 1)]  # 若 default_pose 数量不足，则重复最后一个

        # 确保 pose 为二维
        if pose.ndim == 1:
            pose = pose.unsqueeze(0)

        # 若仅包含 7 维 (pos+quat)，则补上速度 6 维
        if pose.shape[-1] == 7:
            zeros = torch.zeros((pose.shape[0], 6), device=pose.device)
            pose = torch.cat([pose, zeros], dim=-1)

        num_envs = len(env_ids)
        # 扩展到每个 env
        if pose.shape[0] == 1:
            root_state = pose.repeat(num_envs, 1)
        elif pose.shape[0] == num_envs:
            root_state = pose
        else:
            raise ValueError(
                f"default_pose[{i}] shape {pose.shape} incompatible with env_ids length {num_envs}"
            )

        # 写入仿真并更新缓存
        asset.write_root_state_to_sim(root_state, env_ids=env_ids)
        asset._data.root_state_w[env_ids] = root_state.clone()

        print(f"[Reset] object '{cfg.name}' reset for env_ids={env_ids.tolist()} to {pose[0, :7].cpu().numpy()}.")


def reset_pose_randomize_in_xy_ellipse(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    default_pose: Union[torch.Tensor, Sequence[torch.Tensor]],
    asset_cfg: Union[SceneEntityCfg, Sequence[SceneEntityCfg]],
    x_radius: float = 0.04,
    y_radius: float = 0.04,
    roll_range: tuple[float, float] = (0.0, 0.0),
    pitch_range: tuple[float, float] = (0.0, 0.0),
    yaw_range: tuple[float, float] = (0.0, 0.0),
    parent_asset_cfg: SceneEntityCfg | None = None,
    joint_name: str | None = None,
):
    """Reset object pose with slight XY and orientation randomization.

    If the object is welded to another asset through a fixed joint, pass the parent asset config and
    joint name so the joint frame is updated to the new randomized pose as well.
    """

    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    elif not isinstance(env_ids, torch.Tensor):
        env_ids = torch.tensor(env_ids, device=env.device, dtype=torch.long)
    else:
        env_ids = env_ids.to(device=env.device, dtype=torch.long)

    poses = _as_seq(default_pose)
    cfgs = _as_seq(asset_cfg)
    num_envs = env_ids.numel()

    for i, cfg in enumerate(cfgs):
        asset = env.scene[cfg.name]
        pose = torch.as_tensor(poses[min(i, len(poses) - 1)], device=env.device)

        if pose.ndim == 1:
            pose = pose.unsqueeze(0)

        if pose.shape[-1] == 7:
            pose = torch.cat([pose, torch.zeros((pose.shape[0], 6), device=env.device)], dim=-1)
        elif pose.shape[-1] != 13:
            raise ValueError(f"default_pose must have 7 or 13 values, but got shape {pose.shape}.")

        if pose.shape[0] == 1:
            root_state = pose.repeat(num_envs, 1)
        elif pose.shape[0] == num_envs:
            root_state = pose.clone()
        else:
            raise ValueError(f"default_pose shape {pose.shape} incompatible with env_ids length {num_envs}.")

        theta = 2.0 * math.pi * torch.rand(num_envs, device=env.device)
        radius_scale = torch.sqrt(torch.rand(num_envs, device=env.device))
        root_state[:, 0] += x_radius * radius_scale * torch.cos(theta)
        root_state[:, 1] += y_radius * radius_scale * torch.sin(theta)

        roll = torch.empty(num_envs, device=env.device).uniform_(roll_range[0], roll_range[1])
        pitch = torch.empty(num_envs, device=env.device).uniform_(pitch_range[0], pitch_range[1])
        yaw = torch.empty(num_envs, device=env.device).uniform_(yaw_range[0], yaw_range[1])
        delta_quat = math_utils.quat_from_euler_xyz(roll, pitch, yaw)
        root_state[:, 3:7] = math_utils.quat_mul(delta_quat, root_state[:, 3:7])
        root_state[:, 7:13] = 0.0

        asset.write_root_state_to_sim(root_state, env_ids=env_ids)
        asset._data.root_state_w[env_ids] = root_state.clone()

        if parent_asset_cfg is not None and joint_name is not None:
            _update_fixed_joint_pose_for_reset(
                env=env,
                env_ids=env_ids,
                parent_asset_cfg=parent_asset_cfg,
                child_asset_cfg=cfg,
                joint_name=joint_name,
                child_root_state=root_state,
            )


def _update_fixed_joint_pose_for_reset(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    parent_asset_cfg: SceneEntityCfg,
    child_asset_cfg: SceneEntityCfg,
    joint_name: str,
    child_root_state: torch.Tensor,
):
    """Update a fixed joint's local frame so it matches the child's newly reset pose."""

    stage = omni.usd.get_context().get_stage()
    parent_asset = env.scene[parent_asset_cfg.name]
    child_asset = env.scene[child_asset_cfg.name]

    child_paths = list(child_asset.root_physx_view.prim_paths[: child_asset.num_instances])
    parent_root_state = parent_asset.data.root_state_w[env_ids]
    rel_pos = math_utils.quat_rotate_inverse(
        parent_root_state[:, 3:7], child_root_state[:, :3] - parent_root_state[:, :3]
    )
    rel_quat = math_utils.quat_mul(math_utils.quat_conjugate(parent_root_state[:, 3:7]), child_root_state[:, 3:7])

    for local_idx, env_id in enumerate(env_ids.detach().cpu().tolist()):
        fixed_joint_path = f"{child_paths[env_id]}/{joint_name}"
        joint_prim = stage.GetPrimAtPath(fixed_joint_path)
        if not joint_prim.IsValid():
            warnings.warn(
                f"[reset_pose_randomize_in_xy_ellipse] Fixed joint not found at: {fixed_joint_path}",
                stacklevel=2,
            )
            continue

        joint = UsdPhysics.FixedJoint(joint_prim)
        rel_pos_i = rel_pos[local_idx]
        rel_quat_i = rel_quat[local_idx]
        joint.CreateLocalPos0Attr().Set(Gf.Vec3f(float(rel_pos_i[0]), float(rel_pos_i[1]), float(rel_pos_i[2])))
        joint.CreateLocalRot0Attr().Set(
            Gf.Quatf(float(rel_quat_i[0]), float(rel_quat_i[1]), float(rel_quat_i[2]), float(rel_quat_i[3]))
        )
        joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
        joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))


def randomize_scene_lighting_domelight(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    intensity_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("light"),
):
    asset: AssetBase = env.scene[asset_cfg.name]
    light_prim = asset.prims[0]

    # Sample new light intensity
    new_intensity = random.uniform(intensity_range[0], intensity_range[1])

    # Set light intensity to light prim
    intensity_attr = light_prim.GetAttribute("inputs:intensity")
    intensity_attr.Set(new_intensity)


def sample_object_poses(
    num_objects: int,
    min_separation: float = 0.0,
    pose_range: dict[str, tuple[float, float]] = {},
    max_sample_tries: int = 5000,
):
    range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    pose_list = []

    for i in range(num_objects):
        for j in range(max_sample_tries):
            sample = [random.uniform(range[0], range[1]) for range in range_list]

            # Accept pose if it is the first one, or if reached max num tries
            if len(pose_list) == 0 or j == max_sample_tries - 1:
                pose_list.append(sample)
                break

            # Check if pose of object is sufficiently far away from all other objects
            separation_check = [math.dist(sample[:3], pose[:3]) > min_separation for pose in pose_list]
            if False not in separation_check:
                pose_list.append(sample)
                break

    return pose_list


def randomize_object_pose(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfgs: list[SceneEntityCfg],
    min_separation: float = 0.0,
    pose_range: dict[str, tuple[float, float]] = {},
    max_sample_tries: int = 5000,
):
    if env_ids is None:
        return

    # Randomize poses in each environment independently
    for cur_env in env_ids.tolist():
        pose_list = sample_object_poses(
            num_objects=len(asset_cfgs),
            min_separation=min_separation,
            pose_range=pose_range,
            max_sample_tries=max_sample_tries,
        )

        # Randomize pose for each object
        for i in range(len(asset_cfgs)):
            asset_cfg = asset_cfgs[i]
            asset = env.scene[asset_cfg.name]

            # Write pose to simulation
            pose_tensor = torch.tensor([pose_list[i]], device=env.device)
            positions = pose_tensor[:, 0:3] + env.scene.env_origins[cur_env, 0:3]
            orientations = math_utils.quat_from_euler_xyz(pose_tensor[:, 3], pose_tensor[:, 4], pose_tensor[:, 5])
            asset.write_root_pose_to_sim(
                torch.cat([positions, orientations], dim=-1), env_ids=torch.tensor([cur_env], device=env.device)
            )
            asset.write_root_velocity_to_sim(
                torch.zeros(1, 6, device=env.device), env_ids=torch.tensor([cur_env], device=env.device)
            )
