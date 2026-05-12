# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import isaaclab.sim as sim_utils
import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg, AssetBaseCfg
from isaaclab.assets.articulation import ArticulationCfg

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import CameraCfg, FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg

from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab_tasks.manager_based.manipulation.pour_ball_cup import mdp
from isaaclab_tasks.manager_based.manipulation.pour_ball_cup.mdp import franka_pourballcup_events
from isaaclab_tasks.manager_based.manipulation.pour_ball_cup.pourballcup_env_cfg import PourBallCupEnvCfg


def _make_reference_point_pose(dx: float, dy: float, dz: float = 0.0) -> torch.Tensor:
    return torch.tensor(
        [
            0.47 + dx,
            -0.5 + dy,
            1.05 + dz,
            0.70711,
            0.70711,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        device="cuda",
    )


##
# Pre-defined configs
##
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG  # isort: skip


@configclass
class EventCfg:
    """Configuration for events."""
    # 机械臂初始位置
    init_franka_arm_pose = EventTerm(
        func=franka_pourballcup_events.set_default_joint_pose,
        mode="startup",
        params={
            "default_pose": [[0.0, -0.247, 0.35, -2.0, 1.5, 1.8, 0.4, 0.04, 0.04],
                            [0.0, -0.247, -0.35, -2.0, -1.13, 1.8, 0.4, 0.04, 0.04]],
            "asset_cfg": [SceneEntityCfg("robot_left"),SceneEntityCfg("robot_right")],
        },
    )

    init_table_joint_pose = EventTerm(
        func=franka_pourballcup_events.set_default_joint_pose,
        mode="startup",
        params={
            "default_pose": [[0.0 ,0.0 ,0.0 ,0.0]],
            "asset_cfg": [SceneEntityCfg("table")],
        },
    )



    reset_container_pose = EventTerm(
        func=franka_pourballcup_events.reset_pose_to_default,
        mode="reset",
        params={
            "default_pose": torch.tensor([
                0.5, 0.0, 1.05, 0, 0, 0.70711, 0.70711,   # container 位置 + 姿态
                0, 0, 0, 0, 0, 0
            ], device="cuda"),
            "asset_cfg": [SceneEntityCfg("container")],
        },
    )

    reset_table_joint_pose = EventTerm(
        func=franka_pourballcup_events.reset_to_default_joint_pose,
        mode="reset",
        params={
            "default_pose": None,
            "asset_cfg": [SceneEntityCfg("table")],
        },
    )

    randomize_franka_joint_state = EventTerm(
        func=franka_pourballcup_events.randomize_joint_by_gaussian_offset,
        mode="reset",
        params={
            "mean": 0.0,
            "std": 0.02,
            "asset_cfg": [SceneEntityCfg("robot_left"),SceneEntityCfg("robot_right")],
        },
    )

    reset_cup_pose = EventTerm(
        func=franka_pourballcup_events.reset_pose_to_default,
        mode="reset",
        params={
            "default_pose": torch.tensor([
                0.47, -0.5, 1.05, 0, 0, 0.70711, 0.70711,   # cup 位置 + 姿态
                0, 0, 0, 0, 0, 0              # 静止
            ], device="cuda"),
            "asset_cfg": [SceneEntityCfg("cup")],
        },
    )  

    reset_ReferencePoint_pose = EventTerm(
        func=franka_pourballcup_events.reset_pose_to_default,
        mode="reset",
        params={
            "default_pose": [
                _make_reference_point_pose(-0.015, -0.015, 0.1),
                _make_reference_point_pose(-0.005, -0.015, 0.1),
                _make_reference_point_pose(0.005, -0.015, 0.1),
                _make_reference_point_pose(0.015, -0.015, 0.1),
                _make_reference_point_pose(-0.015, -0.005, 0.1),
                _make_reference_point_pose(-0.005, -0.005, 0.1),
                _make_reference_point_pose(0.005, -0.005, 0.1),
                _make_reference_point_pose(0.015, -0.005, 0.1),
                _make_reference_point_pose(-0.010, 0.005, 0.1),
                _make_reference_point_pose(0.010, 0.005, 0.1),
            ],
            "asset_cfg": [
                SceneEntityCfg("ReferencePoint"),
                SceneEntityCfg("ReferencePoint_1"),
                SceneEntityCfg("ReferencePoint_2"),
                SceneEntityCfg("ReferencePoint_3"),
                SceneEntityCfg("ReferencePoint_4"),
                SceneEntityCfg("ReferencePoint_5"),
                SceneEntityCfg("ReferencePoint_6"),
                SceneEntityCfg("ReferencePoint_7"),
                SceneEntityCfg("ReferencePoint_8"),
                SceneEntityCfg("ReferencePoint_9"),
            ],
        },
    )   

@configclass
class FrankaPourBallCupEnvCfg(PourBallCupEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set events
        self.events = EventCfg()

        # Set Franka as robot
        self.scene.robot_left = FRANKA_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/panda_left",
                                init_state=ArticulationCfg.InitialStateCfg(
                                    pos=[0, -0.05, 1.6],
                                    rot=[0.707107, 0.707107, 0.0, 0.0],
                                    joint_pos={
                                        "panda_joint1": 0.0,
                                        "panda_joint2": -0.247,
                                        "panda_joint3": 0.35,
                                        "panda_joint4": -2.0,
                                        "panda_joint5": 1.5,
                                        "panda_joint6": 1.8,
                                        "panda_joint7": 0.4,
                                        "panda_finger_joint1": 0.04,
                                        "panda_finger_joint2": 0.04,
                                    },
                                ))#wxyz    

        self.scene.robot_left.spawn.semantic_tags = [("class", "robot"),("instance", "robot_left")]

        self.scene.robot_right = FRANKA_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/panda_right",
                                init_state=ArticulationCfg.InitialStateCfg(
                                    pos=[0, 0.05, 1.6],
                                    rot=[0.707107, -0.707107,0,0],
                                    joint_pos={
                                        "panda_joint1": 0.0,
                                        "panda_joint2": -0.247,
                                        "panda_joint3": -0.35,
                                        "panda_joint4": -2.0,
                                        "panda_joint5": -1.13,
                                        "panda_joint6": 1.8,
                                        "panda_joint7": 0.4,
                                        "panda_finger_joint1": 0.04,
                                        "panda_finger_joint2": 0.04,
                                    },
                                ))#wxyz    

        self.scene.robot_right.spawn.semantic_tags = [("class", "robot"),("instance", "robot_right")]

        # Increase gripper strength (panda_hand actuator)
        for robot_cfg in (self.scene.robot_left, self.scene.robot_right):
            hand_actuator = robot_cfg.actuators.get("panda_hand")
            if hand_actuator is not None:
                hand_actuator.effort_limit_sim = 1600.0 #物理引擎允许的最大夹爪关节力（或力矩）。越大，夹爪越“有劲”。
                hand_actuator.stiffness = 4000.0 #位置控制的刚度（P 增益）。越大，夹爪越“硬”，越用力逼近目标位置。
                hand_actuator.damping = 200.0 #速度阻尼（D 增益）。越大，越不容易抖动，但响应会慢一些。

        # Add semantics to table
        self.scene.table.spawn.semantic_tags = [("class", "table")]

        # Add semantics to ground
        self.scene.plane.semantic_tags = [("class", "ground")]

        # Set actions for the specific robot type (franka)
        self.actions.arm_action_left = mdp.JointPositionActionCfg(
            asset_name="robot_left", joint_names=["panda_joint.*"], scale=0.5, use_default_offset=True
        )

        self.actions.arm_action_right = mdp.JointPositionActionCfg(
            asset_name="robot_right", joint_names=["panda_joint.*"], scale=0.5, use_default_offset=True
        )


        self.actions.gripper_action_left = mdp.BinaryJointPositionActionCfg(
            asset_name="robot_left",
            joint_names=["panda_finger.*"],
            open_command_expr={"panda_finger_.*": 0.04},
            close_command_expr={"panda_finger_.*": 0.0},
        )
        self.actions.gripper_action_right = mdp.BinaryJointPositionActionCfg(
            asset_name="robot_right",
            joint_names=["panda_finger.*"],
            open_command_expr={"panda_finger_.*": 0.04},
            close_command_expr={"panda_finger_.*": 0.0},
        )

        self.scene.wrist_cam_left = CameraCfg(
            prim_path="{ENV_REGEX_NS}/panda_left/panda_hand/wrist_cam",
            update_period=0.0333,
            height=480,
            width=640,
            data_types=["rgb", "distance_to_image_plane"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
            ),
            offset=CameraCfg.OffsetCfg(pos=(0.025, 0.0, 0.0), rot=(0.707, 0.0, 0.0, 0.707), convention="ros"),
        )
 
        self.scene.wrist_cam_right = CameraCfg(
            prim_path="{ENV_REGEX_NS}/panda_right/panda_hand/wrist_cam",
            update_period=0.0333,
            height=480,
            width=640,
            data_types=["rgb", "distance_to_image_plane"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
            ),
            offset=CameraCfg.OffsetCfg(pos=(0.025, 0.0, 0.0), rot=(0.707, 0.0, 0.0, 0.707), convention="ros"),
        )


        # Set table view camera
        self.scene.table_cam = CameraCfg(
            prim_path="{ENV_REGEX_NS}/table_cam",
            update_period=0.0333,
            height=480,
            width=640,
            data_types=["rgb", "distance_to_image_plane"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
            ),
            offset=CameraCfg.OffsetCfg(pos=(0.25318, -0.03488, 3.659), rot=(0.02656,-0.70663,0.70695,-0.01408), convention="ros"),
        )

        # Listens to the required transforms
        marker_cfg_left = FRAME_MARKER_CFG.copy()
        marker_cfg_left.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg_left.prim_path = "/Visuals/FrameTransformerLeft"

        marker_cfg_right = FRAME_MARKER_CFG.copy()
        marker_cfg_right.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg_right.prim_path = "/Visuals/FrameTransformerRight"


        self.scene.ee_frame_left = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/panda_left/panda_link0",
            debug_vis=False,
            visualizer_cfg=marker_cfg_left,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/panda_left/panda_hand",
                    name="end_effector",
                    offset=OffsetCfg(
                        pos=(0.0, 0.0, 0.027),
                    ),
                ),
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/panda_left/panda_rightfinger",
                    name="tool_rightfinger",
                    offset=OffsetCfg(
                        pos=(0.0, 0.0, 0.046),
                    ),
                ),
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/panda_left/panda_leftfinger",
                    name="tool_leftfinger",
                    offset=OffsetCfg(
                        pos=(0.0, 0.0, 0.046),
                    ),
                ),
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/panda_left/panda_hand",
                    name="end_link",
                    offset=OffsetCfg(
                        pos=(0.0, 0.0, 0.0),
                    ),
                ),

            ],
        )

        self.scene.ee_frame_right = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/panda_right/panda_link0",
            debug_vis=False,
            visualizer_cfg=marker_cfg_right,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/panda_right/panda_hand",
                    name="end_effector",
                    offset=OffsetCfg(
                        pos=(0.0, 0.0, 0.1034),
                    ),
                ),
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/panda_right/panda_rightfinger",
                    name="tool_rightfinger",
                    offset=OffsetCfg(
                        pos=(0.0, 0.0, 0.046),
                    ),
                ),
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/panda_right/panda_leftfinger",
                    name="tool_leftfinger",
                    offset=OffsetCfg(
                        pos=(0.0, 0.0, 0.046),
                    ),
                ),
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/panda_right/panda_hand",
                    name="end_link",
                    offset=OffsetCfg(
                        pos=(0.0, 0.0, 0.0),
                    ),
                ),

            ],
        )
