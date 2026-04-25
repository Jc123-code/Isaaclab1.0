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

from pxr import Usd, UsdGeom, UsdPhysics, PhysxSchema
import omni.usd


def _as_seq(x):
    return (x,) if not isinstance(x, (list, tuple)) else x


def set_default_joint_pose(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    default_pose: Union[torch.Tensor, Sequence[torch.Tensor]] | None,
    asset_cfg: Union[SceneEntityCfg, Sequence[SceneEntityCfg]] = SceneEntityCfg("robot"),
):
    """设置机器人的默认关节位置"""
    cfgs = _as_seq(asset_cfg)

    if default_pose is not None:
        poses = _as_seq(default_pose)
        for pose, cfg in zip(poses, cfgs):
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
        for cfg in cfgs:
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
    stabilization_steps: int = 10,  # 添加稳定步数参数
):
    """重置关节型机器人到默认位置，并清零速度。
    
    Args:
        env: 仿真环境对象
        env_ids: 要重置的环境ID列表
        default_pose: 默认关节位置，如果为None则使用asset自带的default_joint_pos
        asset_cfg: 资产配置，可以是单个或多个
        stabilization_steps: 稳定步数，执行多少个仿真步以让机械臂稳定
    """
    cfgs = _as_seq(asset_cfg)

    if default_pose is not None:
        poses = _as_seq(default_pose)
        for pose, cfg in zip(poses, cfgs):
            asset: Articulation = env.scene[cfg.name]
            pose_tensor = torch.as_tensor(pose, device=env.device).view(1, -1)
            dof = asset.data.default_joint_pos.shape[-1]
            if pose_tensor.shape[-1] != dof:
                if pose_tensor.shape[-1] > dof:
                    pose_tensor = pose_tensor[..., :dof]
                else:
                    pose_padded = asset.data.default_joint_pos.clone()  # (1, dof)
                    pose_padded[..., : pose_tensor.shape[-1]] = pose_tensor
                    pose_tensor = pose_padded
                warnings.warn(
                    f"[reset_to_default_joint_pose] default_pose dim mismatch for '{cfg.name}': "
                    f"adapted to dof {dof}.",
                    stacklevel=2,
                )
            assert pose_tensor.shape[-1] == dof
            
            num_reset = len(env_ids)
            joint_pos = pose_tensor.expand(num_reset, -1)
            joint_vel = torch.zeros(num_reset, dof, device=env.device)
            
            # 1. 直接写入状态（绕过控制器）
            asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
            
            # 2. 设置目标位置和速度
            asset.set_joint_position_target(joint_pos, env_ids=env_ids)
            asset.set_joint_velocity_target(joint_vel, env_ids=env_ids)
            
            # 3. **关键步骤**：执行几步仿真让系统稳定
            if stabilization_steps > 0:
                for _ in range(stabilization_steps):
                    # 保持相同的目标
                    asset.set_joint_position_target(joint_pos, env_ids=env_ids)
                    asset.set_joint_velocity_target(joint_vel, env_ids=env_ids)
                    # 推进仿真一步
                    env.sim.step(render=False)
                    # 更新数据缓冲
                    asset.update(dt=env.physics_dt if hasattr(env, 'physics_dt') else 1.0/60.0)
    else:
        for cfg in cfgs:
            asset: Articulation = env.scene[cfg.name]
            
            joint_pos = asset.data.default_joint_pos[env_ids]
            joint_vel = torch.zeros_like(joint_pos)
            
            # 同样的步骤
            asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
            asset.set_joint_position_target(joint_pos, env_ids=env_ids)
            asset.set_joint_velocity_target(joint_vel, env_ids=env_ids)
            
            if stabilization_steps > 0:
                for _ in range(stabilization_steps):
                    asset.set_joint_position_target(joint_pos, env_ids=env_ids)
                    asset.set_joint_velocity_target(joint_vel, env_ids=env_ids)
                    env.sim.step(render=False)
                    asset.update(dt=env.physics_dt if hasattr(env, 'physics_dt') else 1.0/60.0)


def reset_pose_to_default(env, env_ids, default_pose, asset_cfg):
    """
    Reset one or multiple rigid body objects to their default poses and zero velocities.
    
    Args:
        env: 当前仿真环境对象
        env_ids: 要重置的环境 ID 列表 (tensor or list)
        default_pose: tensor 或 list[tensor]
            每个物体的目标姿态。可为单个 (13,) tensor 或每个 asset 对应一个 (13,)。
        asset_cfg: list[SceneEntityCfg]
            对应的资产配置，如 [broom, dustpan]
    """
    # 如果 default_pose 是单个 Tensor，则统一封装成列表
    if isinstance(default_pose, torch.Tensor):
        default_pose = [default_pose]
    elif isinstance(default_pose, list):
        if not all(isinstance(p, torch.Tensor) for p in default_pose):
            raise TypeError("Each default_pose element must be a torch.Tensor.")
    else:
        raise TypeError("default_pose must be a Tensor or list of Tensors.")

    # 遍历每个资产及其对应的默认姿态
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


def reset_test_episode_with_delay(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    robot_default_pose: Union[torch.Tensor, Sequence[torch.Tensor]] | None,
    robot_asset_cfg: Union[SceneEntityCfg, Sequence[SceneEntityCfg]] = SceneEntityCfg("robot"),
    object_default_poses: list[torch.Tensor] = None,
    object_asset_cfgs: list[SceneEntityCfg] = None,
    delay_seconds: float = 0.5,
):
    """
    测试轮次结束后重置所有物体（机械臂、扫把、簸箕）到初始位置并等待指定时间。
    
    Args:
        env: 仿真环境对象
        env_ids: 要重置的环境ID列表
        robot_default_pose: 机械臂的默认关节位置
        robot_asset_cfg: 机械臂资产配置
        object_default_poses: 刚体对象（扫把、簸箕等）的默认姿态列表
        object_asset_cfgs: 刚体对象资产配置列表
        delay_seconds: 等待时间（秒）
    """
    
    # 1. 重置机械臂到默认关节位置并清零速度
    print(f"[Test Reset] Resetting robot arm for env_ids={env_ids.tolist()}")
    reset_to_default_joint_pose(
        env=env,
        env_ids=env_ids,
        default_pose=robot_default_pose,
        asset_cfg=robot_asset_cfg
    )
    
    # 2. 重置扫把和簸箕到默认姿态并清零速度
    if object_default_poses is not None and object_asset_cfgs is not None:
        print(f"[Test Reset] Resetting objects (broom, dustpan, etc.) for env_ids={env_ids.tolist()}")
        reset_pose_to_default(
            env=env,
            env_ids=env_ids,
            default_pose=object_default_poses,
            asset_cfg=object_asset_cfgs
        )
    
    # 3. 执行仿真步骤以等待指定时间
    # 计算需要执行的步数（假设物理步频率存储在 env.physics_dt 中）
    if hasattr(env, 'physics_dt'):
        num_steps = int(delay_seconds / env.physics_dt)
    elif hasattr(env, 'step_dt'):
        num_steps = int(delay_seconds / env.step_dt)
    else:
        # 默认假设 60Hz 物理频率
        num_steps = int(delay_seconds * 60)
    
    print(f"[Test Reset] Waiting {delay_seconds}s ({num_steps} steps) before next test...")
    
    # 执行空动作以推进仿真
    for _ in range(num_steps):
        # 保持当前位置（零速度命令）
        cfgs = _as_seq(robot_asset_cfg)
        for cfg in cfgs:
            asset: Articulation = env.scene[cfg.name]
            # 保持当前关节位置目标
            current_pos = asset.data.joint_pos[env_ids]
            zero_vel = torch.zeros_like(current_pos)
            asset.set_joint_position_target(current_pos, env_ids=env_ids)
            asset.set_joint_velocity_target(zero_vel, env_ids=env_ids)
        
        # 推进一步仿真
        env.sim.step(render=False)
    
    print(f"[Test Reset] Reset complete. Ready for next test episode.")


def reset_test_episode_simple(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    robot_asset_cfg: Union[SceneEntityCfg, Sequence[SceneEntityCfg]] = SceneEntityCfg("robot"),
    object_asset_cfgs: list[SceneEntityCfg] = None,
    delay_seconds: float = 0.5,
):
    """
    简化版本：使用各资产自带的默认位置进行重置。
    
    Args:
        env: 仿真环境对象
        env_ids: 要重置的环境ID列表
        robot_asset_cfg: 机械臂资产配置
        object_asset_cfgs: 刚体对象资产配置列表（扫把、簸箕等）
        delay_seconds: 等待时间（秒）
    """
    
    # 1. 重置机械臂（使用其自带的 default_joint_pos）
    print(f"[Test Reset] Resetting robot arm to default pose for env_ids={env_ids.tolist()}")
    reset_to_default_joint_pose(
        env=env,
        env_ids=env_ids,
        default_pose=None,  # 使用资产自带的默认位置
        asset_cfg=robot_asset_cfg
    )
    
    # 2. 重置刚体对象（扫把、簸箕等）
    if object_asset_cfgs is not None:
        print(f"[Test Reset] Resetting {len(object_asset_cfgs)} objects for env_ids={env_ids.tolist()}")
        for cfg in object_asset_cfgs:
            asset: AssetBase = env.scene[cfg.name]
            
            # 获取默认根状态（位置+姿态+速度）
            if hasattr(asset.data, 'default_root_state'):
                default_state = asset.data.default_root_state[env_ids].clone()
            else:
                # 如果没有默认状态，创建零速度状态
                current_pose = asset.data.root_state_w[env_ids, :7].clone()
                zero_vel = torch.zeros((len(env_ids), 6), device=env.device)
                default_state = torch.cat([current_pose, zero_vel], dim=-1)
            
            # 写入仿真
            asset.write_root_state_to_sim(default_state, env_ids=env_ids)
            print(f"  - Reset '{cfg.name}' to default pose")
    
    # 3. 等待指定时间
    num_steps = int(delay_seconds / getattr(env, 'physics_dt', 1.0/60.0))
    print(f"[Test Reset] Waiting {delay_seconds}s ({num_steps} steps)...")
    
    for _ in range(num_steps):
        env.sim.step(render=False)
    
    print(f"[Test Reset] Reset complete. Ready for next test.\n")


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
