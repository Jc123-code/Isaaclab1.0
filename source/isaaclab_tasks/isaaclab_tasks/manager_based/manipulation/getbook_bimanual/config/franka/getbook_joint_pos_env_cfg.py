# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import torch
from random import randint, uniform


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

from isaaclab_tasks.manager_based.manipulation.getbook_bimanual import mdp
from isaaclab_tasks.manager_based.manipulation.getbook_bimanual.mdp import franka_getbook_events
from isaaclab_tasks.manager_based.manipulation.getbook_bimanual.getbook_env_cfg import GetBookEnvCfg


##
# Pre-defined configs
##
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
# from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG
from tacex_assets.robots.franka import FRANKA_PANDA_ARM_GSMINI_GRIPPER_HIGH_PD_RIGID_CFG  # isort: skip
from tacex_assets.sensors.gelsight_mini.gsmini_cfg import GelSightMiniCfg  # isort: skip
from tacex.simulation_approaches.fots import FOTSMarkerSimulatorCfg  # isort: skip



@configclass
class EventCfg:
    """Configuration for events."""
    # 机械臂初始位置
    init_franka_arm_pose = EventTerm(
        func=franka_getbook_events.set_default_joint_pose,
        mode="startup",
        params={
            "default_pose": [[0.0, -0.247, 0.35, -2.0, 1.5, 1.8, 0.4, 0.04, 0.04],
                            [0.0, -0.247, -0.35, -2.0, -1.13, 1.8, 0.4, 0.04, 0.04]],
            "asset_cfg": [SceneEntityCfg("robot_left"),SceneEntityCfg("robot_right")],
        },
    )

    init_table_joint_pose = EventTerm(
        func=franka_getbook_events.set_default_joint_pose,
        mode="startup",
        params={
            "default_pose": [[0.0 ,0.0 ,0.0 ,0.0,0.0]],
            "asset_cfg": [SceneEntityCfg("table")],
        },
    )

#重置逻辑

    reset_table_joint_pose = EventTerm(
        func=franka_getbook_events.reset_to_default_joint_pose,
        mode="reset",
        params={
            "default_pose": None,
            "asset_cfg": [SceneEntityCfg("table")],
        },
    )

    randomize_franka_joint_state = EventTerm(
        func=franka_getbook_events.randomize_joint_by_gaussian_offset,
        mode="reset",
        params={
            "mean": 0.0,
            "std": 0.02,
            "asset_cfg": [SceneEntityCfg("robot_left"),SceneEntityCfg("robot_right")],
        },
    )

#目标物体：
    # reset_red_book_pose = EventTerm(
    #     func=franka_getbook_events.reset_book_pose_to_default,
    #     mode="reset",
    #     params={
    #         "default_pose": torch.tensor([
    #             # 0.8, 0.2, 1.5, 0.70711, 0.70711, 0, 0,  # 位置 + 四元数,0.3, 0.0, 0.75, 0, 0, 0, 1；-0.5, -0.5, 0.5, 0.5
    #             0.8, uniform(0.2-0.1, 0.2+0.1), 1.5, 0.70711, 0.70711, 0, 0,
    #             # 0.8, y_red_rand.item(), 1.5, 0.70711, 0.70711, 0, 0,
    #             0, 0, 0, 0, 0, 0              # 线速度 + 角速度
    #         ], device="cuda"),
    #         "asset_cfg": [SceneEntityCfg("book_red")],
    #     },
    # )


    reset_red_book_pose = EventTerm(
        func=franka_getbook_events.reset_book_pose_to_default,
        mode="reset",
        params={
            "pose_range": {
                "x": (0.8, 0.8),           # x 固定
                "y": (0.2, 0.2),           # y 固定
                "z": (1.5, 1.5),           # z 固定
            },
            "fixed_quat": torch.tensor([0.70711, 0.70711, 0.0, 0.0]),  # 固定的四元数
            "asset_cfg": [SceneEntityCfg("book_red")],
        },
    )


    # reset_blue_book_pose = EventTerm(
    #     func=franka_getbook_events.reset_book_pose_to_default,
    #     mode="reset",
    #     params={
    #         "default_pose": torch.tensor([
    #             # 0.8, -0.2, 1.4, 0.70711, 0.70711, 0, 0,  # 位置 + 四元数,0.3, 0.0, 0.75, 0, 0, 0, 1
    #             0.8, uniform(-0.2-0.1, -0.2+0.1), 1.4, 0.70711, 0.70711, 0, 0,
    #             # 0.8, -0.2, 1.4, 0.70711, 0.70711, 0, 0,
    #             0, 0, 0, 0, 0, 0              # 线速度 + 角速度
    #         ], device="cuda"),
    #         "asset_cfg": [SceneEntityCfg("book_blue")],
    #     },
    # )

    reset_blue_book_pose = EventTerm(
        func=franka_getbook_events.reset_book_pose_to_default,
        mode="reset",
        params={
            "pose_range": {
                "x": (0.8, 0.8),           # x 固定
                "y": (-0.25, -0.25),         # y 固定
                "z": (1.5, 1.5),           # z 固定
            },
            "fixed_quat": torch.tensor([0.70711, 0.70711, 0.0, 0.0]),  # 固定的四元数
            "asset_cfg": [SceneEntityCfg("book_blue")],
        },
    )

    
@configclass
class FrankaGetBookEnvCfg(GetBookEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set events
        self.events = EventCfg()

        # Set Franka as robot
        self.scene.robot_right = FRANKA_PANDA_ARM_GSMINI_GRIPPER_HIGH_PD_RIGID_CFG.replace(
            prim_path="{ENV_REGEX_NS}/panda_right",
            init_state=ArticulationCfg.InitialStateCfg(
                pos=[0, -0.05, 1.6],
                rot=[0.707107, 0.707107, 0.0, 0.0],
                joint_pos=FRANKA_PANDA_ARM_GSMINI_GRIPPER_HIGH_PD_RIGID_CFG.init_state.joint_pos,
            ),
        )  # wxyz
        self.scene.robot_left = FRANKA_PANDA_ARM_GSMINI_GRIPPER_HIGH_PD_RIGID_CFG.replace(
            prim_path="{ENV_REGEX_NS}/panda_left",
            init_state=ArticulationCfg.InitialStateCfg(
                pos=[0, 0.05, 1.6],
                rot=[0.707107, -0.707107, 0.0, 0.0],
                joint_pos=FRANKA_PANDA_ARM_GSMINI_GRIPPER_HIGH_PD_RIGID_CFG.init_state.joint_pos,
            ),
        )  # wxyz

        # Refresh tactile sensors after the IK robot configs are swapped in.
        self._configure_gsmini_sensors() 
 

        self.scene.robot_right.spawn.semantic_tags = [("class", "robot"),("instance", "robot_right")]

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
    def _marker_cfg(self, gelpad_prim: str) -> FOTSMarkerSimulatorCfg:
        return FOTSMarkerSimulatorCfg(
            lamb=[0.00125, 0.00021, 0.00038],
            pyramid_kernel_size=[51, 21, 11, 5],
            kernel_size=5,
            marker_params=FOTSMarkerSimulatorCfg.MarkerParams(
                num_markers_col=9,
                num_markers_row=11,
                num_markers=99,
                x0=15,
                y0=26,
                dx=26,
                dy=29,
            ),
            tactile_img_res=(240, 180),  # rgb图分辨率
            device="cuda",
            frame_transformer_cfg=FrameTransformerCfg(
                prim_path=gelpad_prim,
                target_frames=[
                    FrameTransformerCfg.FrameCfg(
                        prim_path="/World/envs/env_.*/Table/book_red",
                        name="book_red",
                    )
                ],
                debug_vis=False,
            ),
        )  # fots maker图

    def _configure_gsmini_sensors(self):
        gsmini_template = GelSightMiniCfg(
            prim_path="{ENV_REGEX_NS}/panda_left/gelsight_mini_case_left",
            sensor_camera_cfg=GelSightMiniCfg.SensorCameraCfg(
                prim_path_appendix="/Camera",
                update_period=0,
                resolution=(240, 180),  # 深度图
                data_types=["depth"],
                clipping_range=(0.024, 0.034),
            ),
            device="cuda",
            debug_vis=True,
            # debug_vis=False, #弹出fots_marker
            marker_motion_sim_cfg=self._marker_cfg("/World/envs/env_.*/panda_left/gelpad_left"),
            data_types=["tactile_rgb", "marker_motion"],
        )
        gsmini_template.optical_sim_cfg = gsmini_template.optical_sim_cfg.replace(
            with_shadow=False,
            device="cuda",
            tactile_img_res=(240, 180),  # fotsmarker图
        )

        # One GelSight per hand (left finger only) to avoid duplicate windows.
        self.scene.gsmini_left_left = gsmini_template
        self.scene.gsmini_right_left = gsmini_template.replace(
            prim_path="{ENV_REGEX_NS}/panda_right/gelsight_mini_case_left",
            marker_motion_sim_cfg=self._marker_cfg("/World/envs/env_.*/panda_right/gelpad_left"),
        )  # 触觉图像
