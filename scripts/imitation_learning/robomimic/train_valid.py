# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
The main entry point for training ACT policies from pre-collected data.

This script loads dataset(s), creates an ACT model, and trains it.
It supports training on IsaacLab environments with the ACT algorithm from robomimic.

Args:
    algo: Name of the algorithm to run (should be 'act').
    task: Name of the environment.
    name: If provided, override the experiment name defined in the config.
    dataset: If provided, override the dataset path defined in the config.
    log_dir: Directory to save logs.
    normalize_training_actions: Whether to normalize actions in the training data.
    epochs: Number of training epochs.
"""

# ===== CRITICAL: Add ACT to Python path BEFORE any imports =====
import sys
sys.path.insert(0, '/home/abc/IsaacLab/act')
sys.path.insert(0, '/home/abc/IsaacLab/act/detr')

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher

# launch omniverse app
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

"""Rest everything follows."""

# Standard library imports
import argparse
import json
import os
import time
import traceback
from collections import OrderedDict

# Third-party imports
import gymnasium as gym
import h5py
import numpy as np
import psutil
import shutil
import torch
from torch.utils.data import DataLoader

# Robomimic imports
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.train_utils as TrainUtils
from robomimic.algo import algo_factory
from robomimic.config import Config, config_factory
from robomimic.utils.log_utils import DataLogger, PrintLogger

# Isaac Lab imports (needed so that environment is registered)
import isaaclab_tasks  # noqa: F401
import isaaclab_tasks.manager_based.manipulation.opendrawer_bimanual  # noqa: F401
import isaaclab_tasks.manager_based.manipulation.openleftdrawer  # noqa: F401


def normalize_hdf5_actions(config: Config, log_dir: str) -> str:
    """Normalizes actions in hdf5 dataset to [-1, 1] range.

    Args:
        config: The configuration object containing dataset path.
        log_dir: Directory to save normalization parameters.

    Returns:
        Path to the normalized dataset.
    """
    # Get the dataset path - handle both string and list formats
    if isinstance(config.train.data, list):
        dataset_path = config.train.data[0]["path"]
    else:
        dataset_path = config.train.data
    
    base, ext = os.path.splitext(dataset_path)
    normalized_path = base + "_normalized" + ext

    # Copy the original dataset
    print(f"Creating normalized dataset at {normalized_path}")
    shutil.copyfile(dataset_path, normalized_path)

    # Open the new dataset and normalize the actions
    with h5py.File(normalized_path, "r+") as f:
        dataset_paths = [f"/data/demo_{str(i)}/actions" for i in range(len(f["data"].keys()))]

        # Compute the min and max of the dataset
        dataset = np.array(f[dataset_paths[0]]).flatten()
        for i, path in enumerate(dataset_paths):
            if i != 0:
                data = np.array(f[path]).flatten()
                dataset = np.append(dataset, data)

        max_val = np.max(dataset)
        min_val = np.min(dataset)

        # Normalize the actions
        for i, path in enumerate(dataset_paths):
            data = np.array(f[path])
            normalized_data = 2 * ((data - min_val) / (max_val - min_val)) - 1  # Scale to [-1, 1] range
            del f[path]
            f[path] = normalized_data

        # Save the min and max values to log directory
        with open(os.path.join(log_dir, "normalization_params.txt"), "w") as param_file:
            param_file.write(f"min: {min_val}\n")
            param_file.write(f"max: {max_val}\n")

    return normalized_path


def get_env_metadata_from_dataset(dataset_path, set_env_specific_obs_processors=True):
    """
    Retrieves env metadata from dataset.

    Args:
        dataset_path (str): path to dataset

        set_env_specific_obs_processors (bool): environment might have custom rules for how to process
            observations - if this flag is true, make sure ObsUtils will use these custom settings.

    Returns:
        env_meta (dict): environment metadata. Contains 3 keys:
            :`'env_name'`: name of environment
            :`'type'`: type of environment, should be a value in EB.EnvType
            :`'env_kwargs'`: dictionary of keyword arguments to pass to environment constructor
    """
    dataset_path = os.path.expanduser(dataset_path)
    f = h5py.File(dataset_path, "r")
    env_meta = json.loads(f["data"].attrs["env_args"])
    if "env_kwargs" in env_meta.keys():
        if "env_lang" in env_meta["env_kwargs"]:
            del env_meta["env_kwargs"]["env_lang"]
    else:
        env_meta["env_kwargs"] = {}

    f.close()
    if set_env_specific_obs_processors:
        # handle env-specific custom observation processing logic
        EnvUtils.set_env_specific_obs_processing(env_meta=env_meta)
    return env_meta


def train(config: Config, device: str, log_dir: str, ckpt_dir: str, video_dir: str):
    """Train an ACT model using the algorithm specified in config.

    Args:
        config: Configuration object.
        device: PyTorch device to use for training.
        log_dir: Directory to save logs.
        ckpt_dir: Directory to save checkpoints.
        video_dir: Directory to save videos.
    """
    # first set seeds
    np.random.seed(config.train.seed)
    torch.manual_seed(config.train.seed)

    print("\n============= New ACT Training Run with Config =============")
    print(config)
    print("")

    print(f">>> Saving logs into directory: {log_dir}")
    print(f">>> Saving checkpoints into directory: {ckpt_dir}")
    print(f">>> Saving videos into directory: {video_dir}")

    if config.experiment.logging.terminal_output_to_txt:
        # log stdout and stderr to a text file
        logger = PrintLogger(os.path.join(log_dir, "log.txt"))
        sys.stdout = logger
        sys.stderr = logger

    # read config to set up metadata for observation modalities (e.g. detecting rgb observations)
    ObsUtils.initialize_obs_utils_with_config(config)

    # make sure the dataset exists
    if isinstance(config.train.data, list):
        dataset_path = os.path.expanduser(config.train.data[0]["path"])
    else:
        dataset_path = os.path.expanduser(config.train.data)
    
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset at provided path {dataset_path} not found!")

    if isinstance(config.train.data, str):
        with config.values_unlocked():
            config.train.data = [{"path": config.train.data}]

    # load basic metadata from training file
    print("\n============= Loaded Environment Metadata =============")
    env_meta = get_env_metadata_from_dataset(dataset_path=dataset_path)

    shape_meta = FileUtils.get_shape_metadata_from_dataset(
        dataset_config=config.train.data[0],
        action_keys=config.train.action_keys,
        all_obs_keys=config.all_obs_keys,
        verbose=True
    )
    
    if config.experiment.env is not None:
        env_meta["env_name"] = config.experiment.env
        print("=" * 30 + "\n" + "Replacing Env to {}\n".format(env_meta["env_name"]) + "=" * 30)

    # create environment
    envs = OrderedDict()
    if config.experiment.rollout.enabled:
        # create environments for validation runs
        env_names = [env_meta["env_name"]]

        if config.experiment.additional_envs is not None:
            for name in config.experiment.additional_envs:
                env_names.append(name)

        for env_name in env_names:
            env = EnvUtils.create_env_from_metadata(
                env_meta=env_meta,
                env_name=env_name,
                render=False,
                render_offscreen=config.experiment.render_video,
                use_image_obs=shape_meta["use_images"],
            )
            envs[env.name] = env
            print(envs[env.name])

    print("")

    # load training data BEFORE using trainset
    print("\n============= Loading Training Data =============")
    trainset, validset = TrainUtils.load_data_for_training(config, obs_keys=shape_meta["all_obs_keys"])
    train_sampler = trainset.get_dataset_sampler()
    print("\n============= Training Dataset =============")
    print(trainset)
    print("")

    # setup for a new training run
    data_logger = DataLogger(log_dir, config=config, log_tb=config.experiment.logging.log_tb)
    
    # number of learning steps per epoch (defaults to a full dataset pass)
    train_num_steps = config.experiment.epoch_every_n_steps
    valid_num_steps = config.experiment.validation_epoch_every_n_steps

    # add info to optim_params
    with config.values_unlocked():
        if "optim_params" in config.algo:
            # add info to optim_params of each net
            for k in config.algo.optim_params:
                config.algo.optim_params[k]["num_train_batches"] = len(trainset) if train_num_steps is None else train_num_steps
                config.algo.optim_params[k]["num_epochs"] = config.train.num_epochs
        
        # handling for "hbc" and "iris" algorithms
        if config.algo_name == "hbc":
            for sub_algo in ["planner", "actor"]:
                # add info to optim_params of each net
                for k in config.algo[sub_algo].optim_params:
                    config.algo[sub_algo].optim_params[k]["num_train_batches"] = len(trainset) if train_num_steps is None else train_num_steps
                    config.algo[sub_algo].optim_params[k]["num_epochs"] = config.train.num_epochs
        if config.algo_name == "iris":
            for sub_algo in ["planner", "value"]:
                # add info to optim_params of each net
                for k in config.algo["value_planner"][sub_algo].optim_params:
                    config.algo["value_planner"][sub_algo].optim_params[k]["num_train_batches"] = len(trainset) if train_num_steps is None else train_num_steps
                    config.algo["value_planner"][sub_algo].optim_params[k]["num_epochs"] = config.train.num_epochs

    # create ACT model
    print("\n============= Creating ACT Model =============")
    model = algo_factory(
        algo_name=config.algo_name,
        config=config,
        obs_key_shapes=shape_meta["all_shapes"],
        ac_dim=shape_meta["ac_dim"],
        device=device,
    )

    # save the config as a json file
    with open(os.path.join(log_dir, "..", "config.json"), "w") as outfile:
        json.dump(config, outfile, indent=4)

    print("\n============= ACT Model Summary =============")
    print(model)  # print model summary
    print("")

    # maybe retrieve statistics for normalizing observations
    obs_normalization_stats = None
    if config.train.hdf5_normalize_obs:
        obs_normalization_stats = trainset.get_obs_normalization_stats()

    # initialize data loaders
    train_loader = DataLoader(
        dataset=trainset,
        sampler=train_sampler,
        batch_size=config.train.batch_size,
        shuffle=(train_sampler is None),
        num_workers=config.train.num_data_workers,
        drop_last=True,
    )

    if config.experiment.validate:
        # cap num workers for validation dataset at 1
        num_workers = min(config.train.num_data_workers, 1)
        valid_sampler = validset.get_dataset_sampler()
        valid_loader = DataLoader(
            dataset=validset,
            sampler=valid_sampler,
            batch_size=config.train.batch_size,
            shuffle=(valid_sampler is None),
            num_workers=num_workers,
            drop_last=True,
        )
    else:
        valid_loader = None

    # main training loop
    best_valid_loss = None
    best_success_rate = None
    last_ckpt_time = time.time()

    for epoch in range(1, config.train.num_epochs + 1):  # epoch numbers start at 1
        step_log = TrainUtils.run_epoch(model=model, data_loader=train_loader, epoch=epoch, num_steps=train_num_steps)
        model.on_epoch_end(epoch)

        # setup checkpoint path
        epoch_ckpt_name = f"model_epoch_{epoch}"

        # check for recurring checkpoint saving conditions
        should_save_ckpt = False
        if config.experiment.save.enabled:
            time_check = (config.experiment.save.every_n_seconds is not None) and (
                time.time() - last_ckpt_time > config.experiment.save.every_n_seconds
            )
            epoch_check = (
                (config.experiment.save.every_n_epochs is not None)
                and (epoch > 0)
                and (epoch % config.experiment.save.every_n_epochs == 0)
            )
            epoch_list_check = epoch in config.experiment.save.epochs
            last_epoch_check = epoch == config.train.num_epochs
            should_save_ckpt = time_check or epoch_check or epoch_list_check or last_epoch_check
        
        ckpt_reason = None
        if should_save_ckpt:
            last_ckpt_time = time.time()
            ckpt_reason = "time"

        print(f"Train Epoch {epoch}")
        print(json.dumps(step_log, sort_keys=True, indent=4))
        for k, v in step_log.items():
            if k.startswith("Time_"):
                data_logger.record(f"Timing_Stats/Train_{k[5:]}", v, epoch)
            else:
                data_logger.record(f"Train/{k}", v, epoch)

        # Evaluate the model on validation set
        if config.experiment.validate:
            with torch.no_grad():
                step_log = TrainUtils.run_epoch(
                    model=model, data_loader=valid_loader, epoch=epoch, validate=True, num_steps=valid_num_steps
                )
            for k, v in step_log.items():
                if k.startswith("Time_"):
                    data_logger.record(f"Timing_Stats/Valid_{k[5:]}", v, epoch)
                else:
                    data_logger.record(f"Valid/{k}", v, epoch)

            print(f"Validation Epoch {epoch}")
            print(json.dumps(step_log, sort_keys=True, indent=4))

            # save checkpoint if achieve new best validation loss
            valid_check = "Loss" in step_log
            if valid_check and (best_valid_loss is None or (step_log["Loss"] <= best_valid_loss)):
                best_valid_loss = step_log["Loss"]
                if config.experiment.save.enabled and config.experiment.save.on_best_validation:
                    epoch_ckpt_name += f"_best_validation_{best_valid_loss}"
                    should_save_ckpt = True
                    ckpt_reason = "valid" if ckpt_reason is None else ckpt_reason

        # Rollout evaluation
        if config.experiment.rollout.enabled and (epoch % config.experiment.rollout.rate == 0):
            print(f"\n============= Rollout Evaluation at Epoch {epoch} =============")
            rollout_info = TrainUtils.rollout_with_stats(
                policy=model,
                envs=envs,
                horizon=config.experiment.rollout.horizon,
                use_goals=config.use_goals,
                num_episodes=config.experiment.rollout.n,
                render=False,
                video_dir=video_dir if config.experiment.render_video else None,
                epoch=epoch,
                video_skip=config.experiment.video_skip,
                terminate_on_success=config.experiment.rollout.terminate_on_success,
            )

            # log rollout results
            for env_name in rollout_info:
                rollout_logs = rollout_info[env_name]
                for k, v in rollout_logs.items():
                    if k.startswith("Time_"):
                        data_logger.record(f"Timing_Stats/Rollout_{env_name}_{k[5:]}", v, epoch)
                    else:
                        data_logger.record(f"Rollout/{env_name}/{k}", v, epoch)

                print(f"\nRollout Results for {env_name}:")
                print(json.dumps(rollout_logs, sort_keys=True, indent=4))

                # check for best success rate
                if "Success_Rate" in rollout_logs:
                    success_rate = rollout_logs["Success_Rate"]
                    if best_success_rate is None or success_rate > best_success_rate:
                        best_success_rate = success_rate
                        if config.experiment.save.enabled and config.experiment.save.on_best_rollout_success_rate:
                            epoch_ckpt_name += f"_best_success_rate_{best_success_rate:.3f}"
                            should_save_ckpt = True
                            ckpt_reason = "success" if ckpt_reason is None else ckpt_reason

        # Save model checkpoints based on conditions (success rate, validation loss, etc)
        if should_save_ckpt:
            print(f"\n>>> Saving checkpoint (reason: {ckpt_reason})")
            TrainUtils.save_model(
                model=model,
                config=config,
                env_meta=env_meta,
                shape_meta=shape_meta,
                ckpt_path=os.path.join(ckpt_dir, epoch_ckpt_name + ".pth"),
                obs_normalization_stats=obs_normalization_stats,
            )

        # Finally, log memory usage in MB
        process = psutil.Process(os.getpid())
        mem_usage = int(process.memory_info().rss / 1000000)
        data_logger.record("System/RAM Usage (MB)", mem_usage, epoch)
        print(f"\nEpoch {epoch} Memory Usage: {mem_usage} MB\n")

    # terminate logging
    data_logger.close()


def main(args: argparse.Namespace):
    """Train an ACT model on a task using robomimic.

    Args:
        args: Command line arguments.
    """
    # load config
    if args.task is not None:
        cfg_entry_point_key = f"robomimic_{args.algo}_cfg_entry_point"
        task_name = args.task.split(":")[-1]

        print(f"Loading ACT configuration for task: {task_name}")
        print(f"Looking for config entry point: {cfg_entry_point_key}")
        print(" ")
        
        cfg_entry_point_file = gym.spec(task_name).kwargs.pop(cfg_entry_point_key)
        
        if cfg_entry_point_file is None:
            raise ValueError(
                f"Could not find ACT configuration for the environment: '{task_name}'."
                f" Please check that the gym registry has the entry point: '{cfg_entry_point_key}'."
            )

        with open(cfg_entry_point_file) as f:
            ext_cfg = json.load(f)
            config = config_factory(ext_cfg["algo_name"])
        
        with config.values_unlocked():
            config.update(ext_cfg)
            config.experiment.rollout.enabled = False
            print("\n>>> Rollout evaluation disabled (training only mode)")
    else:
        raise ValueError("Please provide a task name through CLI arguments.")

    if args.dataset is not None:
        config.train.data = args.dataset

    if args.name is not None:
        config.experiment.name = args.name

    if args.epochs is not None:
        config.train.num_epochs = args.epochs

    # ========== 启用验证集配置 ==========
    with config.values_unlocked():
        # 1. 启用验证
        config.experiment.validate = True
        config.experiment.validation_epoch_every_n_steps = 100
        
        # 2. 确保data是列表格式
        if isinstance(config.train.data, str):
            data_path = config.train.data
            config.train.data = [{"path": data_path}]
        
        #3. 获取数据集路径
        dataset_path = config.train.data[0]["path"] if isinstance(config.train.data, list) else config.train.data
        
    # 4. 检查数据集是否已经有train/valid分割
        import h5py
        with h5py.File(dataset_path, "r") as f:
            demo_keys = list(f["data"].keys())
            has_train_split = False
            if len(demo_keys) > 0:
                first_demo = f[f"data/{demo_keys[0]}"]
                if "train" in first_demo.attrs:
                    has_train_split = True

     # 5. 如果没有分割，自动分割数据集
        if not has_train_split:
            print(f"\n{'='*60}")
            print(">>> Dataset does not have train/valid split")
            print(">>> Automatically splitting dataset...")
            print(f"{'='*60}\n")

            # 使用robomimic的split功能
        from robomimic.scripts.split_train_val import split_train_val_from_hdf5
        split_train_val_from_hdf5(
            hdf5_path=dataset_path,
            val_ratio=0.1,
            filter_key=None
        )
        print(">>> Dataset split completed!\n")

        # 6. 设置filter keys
        config.train.hdf5_filter_key = "train"
        config.train.hdf5_validation_filter_key = "valid"
        # 7. 打印配置信息
        print(f"\n{'='*60}")
        print(f">>> Validation ENABLED")
        print(f">>> Filter key: train")
        print(f">>> Validation filter key: valid")
        print(f">>> Validation every {config.experiment.validation_epoch_every_n_steps} steps")
        print(f"{'='*60}\n")
            

    # change location of experiment directory
    config.train.output_dir = os.path.abspath(os.path.join("./logs", args.log_dir, args.task))

    log_dir, ckpt_dir, video_dir, time_dir = TrainUtils.get_exp_dir(config)

    # ===== 动作归一化：在这里调用 =====
    if args.normalize_training_actions:
        print("\n============= Normalizing Training Actions =============")
        normalized_path = normalize_hdf5_actions(config, log_dir)
        
        # Update the config with normalized dataset path
        with config.values_unlocked():
            if isinstance(config.train.data, list):
                config.train.data[0]["path"] = normalized_path
            else:
                config.train.data = normalized_path
        
        print(f"Using normalized dataset: {normalized_path}")
        print("Normalization parameters saved to: {}/normalization_params.txt".format(log_dir))
        print("=" * 60 + "\n")

    # get torch device
    device = TorchUtils.get_torch_device(try_to_use_cuda=config.train.cuda)

    config.lock()

    # catch error during training and print it
    res_str = "finished run successfully!"
    try:
        train(config, device, log_dir, ckpt_dir, video_dir)
    except Exception as e:
        res_str = f"run failed with error:\n{e}\n\n{traceback.format_exc()}"
    print(res_str)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Experiment Name (for tensorboard, saving models, etc.)
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="(optional) if provided, override the experiment name defined in the config",
    )

    # Dataset path, to override the one in the config
    parser.add_argument(
        "--dataset",
        type=str,
        default="/home/abc/hdf5/open_left_drawer.hdf5/open_left_drawer.hdf5",
        help="(optional) if provided, override the dataset path defined in the config",
    )

    parser.add_argument(
        "--task", 
        type=str, 
        default="Isaac-Open-Left-Drawer-Franka-Bimanual-IK-Abs-v0", 
        help="Name of the task."
    )
    
    parser.add_argument(
        "--algo", 
        type=str, 
        default="act", 
        help="Name of the algorithm (use 'act' for Action Chunking Transformer)."
    )
    
    parser.add_argument(
        "--log_dir", 
        type=str, 
        default="robomimic_act", 
        help="Path to log directory"
    )
    
    parser.add_argument(
        "--normalize_training_actions", 
        action="store_true", 
        default=False, 
        help="Normalize actions"
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help=(
            "Optional: Number of training epochs. If specified, overrides the number of epochs from the JSON training"
            " config."
        ),
    )

    args = parser.parse_args()

    # run training
    main(args)
    # close sim app
    simulation_app.close()
