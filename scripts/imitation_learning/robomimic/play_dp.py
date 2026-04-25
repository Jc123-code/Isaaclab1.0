# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play and evaluate a trained policy from robomimic."""

"""Launch Isaac Sim Simulator first."""

import argparse
from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Evaluate robomimic policy for Isaac Lab environment.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--task", type=str, default="Isaac-Get-BlueBook-Franka-Bimanual-IK-Abs-v0", help="Name of the task.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to trained robomimic checkpoint (.pth).")
parser.add_argument("--horizon", type=int, default=800, help="Step horizon of each rollout.")
parser.add_argument("--num_rollouts", type=int, default=1, help="Number of rollouts.")
parser.add_argument("--seed", type=int, default=101, help="Random seed.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)

# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch
from collections import deque

import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils

from isaaclab_tasks.utils import parse_env_cfg


def _extract_policy_obs(obs_policy: dict, policy) -> dict:
    """Keep only observation keys expected by the loaded robomimic policy."""
    required_obs_keys = list(policy.policy.global_config.all_obs_keys)
    missing_keys = [key for key in required_obs_keys if key not in obs_policy]
    if missing_keys:
        raise KeyError(
            "Environment observation is missing keys required by checkpoint: "
            + ", ".join(missing_keys)
        )
    return {key: obs_policy[key] for key in required_obs_keys}


def _build_temporal_obs(obs_history: deque, obs_horizon: int) -> dict:
    """Build [B, T, ...] observations from recent per-step [B, ...] observations."""
    history = list(obs_history)
    if len(history) == 0:
        raise RuntimeError("Observation history is empty.")
    if len(history) < obs_horizon:
        pad_count = obs_horizon - len(history)
        history = [history[0]] * pad_count + history
    else:
        history = history[-obs_horizon:]

    stacked = {}
    for key in history[0]:
        stacked[key] = torch.stack([step_obs[key] for step_obs in history], dim=1)
    return stacked


def rollout(policy, env, horizon, device):
    policy.start_episode()
    obs_dict, _ = env.reset()
    traj = dict(actions=[], obs=[], next_obs=[])
    obs_horizon = int(policy.policy.algo_config.horizon.observation_horizon)
    obs_history = deque(maxlen=obs_horizon)
    
    for i in range(horizon):
        # Prepare observations
        obs_step = _extract_policy_obs(obs_dict["policy"], policy)
        obs_history.append(obs_step)
        obs = _build_temporal_obs(obs_history, obs_horizon)
        traj["obs"].append(obs_step)
        
        # Compute actions (Isaac Lab already provides batched observations with batch size 1)
        actions_np = policy(obs, batched_ob=True)
        actions = torch.from_numpy(actions_np).to(device=device, dtype=torch.float32)
        if actions.ndim == 1:
            actions = actions.unsqueeze(0)
        
        # Apply actions
        obs_dict, _, terminated, truncated, _ = env.step(actions)
        obs = _extract_policy_obs(obs_dict["policy"], policy)
        
        # Record trajectory
        traj["actions"].append(actions.tolist())
        traj["next_obs"].append(obs)
        
        terminated_flag = bool(terminated.item()) if torch.is_tensor(terminated) else bool(terminated)
        truncated_flag = bool(truncated.item()) if torch.is_tensor(truncated) else bool(truncated)

        if terminated_flag:
            return True, traj
        elif truncated_flag:
            return False, traj
    
    return False, traj


def main():
    """Run a trained policy from robomimic with Isaac Lab environment."""
    if args_cli.checkpoint is None:
        raise ValueError("Please provide --checkpoint for evaluation.")

    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, 
        device=args_cli.device, 
        num_envs=1, 
        use_fabric=not args_cli.disable_fabric
    )
    
    # Set observations to dictionary mode for Robomimic
    env_cfg.observations.policy.concatenate_terms = False
    
    # Set termination conditions
    env_cfg.terminations.time_out = None
    
    # Disable recorder
    env_cfg.recorders = None
    
    # Create environment
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    
    # Set seed
    torch.manual_seed(args_cli.seed)
    env.seed(args_cli.seed)
    
    # Acquire device
    device = TorchUtils.get_torch_device(try_to_use_cuda=True)
    
    # Load policy (supports diffusion_policy checkpoints saved by train dp.py)
    print(f"\n[INFO] Loading checkpoint: {args_cli.checkpoint}")
    policy, _ = FileUtils.policy_from_checkpoint(ckpt_path=args_cli.checkpoint, device=device, verbose=True)
    print("[INFO] Policy loaded successfully!\n")
    
    # Run policy
    results = []
    for trial in range(args_cli.num_rollouts):
        print(f"[INFO] Starting trial {trial + 1}/{args_cli.num_rollouts}")
        terminated, traj = rollout(policy, env, args_cli.horizon, device)
        results.append(terminated)
        status = "SUCCESS" if terminated else "FAILED"
        print(f"[INFO] Trial {trial + 1}: {status}\n")
    
    # Print summary
    print("=" * 60)
    print(f"Successful trials: {results.count(True)} / {len(results)}")
    print(f"Success rate: {results.count(True) / len(results) * 100:.1f}%")
    print(f"Trial Results: {results}")
    print("=" * 60)
    
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
