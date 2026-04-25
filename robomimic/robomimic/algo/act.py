"""
Implementation of Action Chunking with Transformers (ACT).
"""
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

import robomimic.utils.tensor_utils as TensorUtils

from robomimic.algo import register_algo_factory_func, PolicyAlgo
from robomimic.algo.bc import BC_VAE


@register_algo_factory_func("act")
def algo_config_to_class(algo_config):
    """
    Maps algo config to the BC algo class to instantiate, along with additional algo kwargs.
    Args:
        algo_config (Config instance): algo config
    Returns:
        algo_class: subclass of Algo
        algo_kwargs (dict): dictionary of additional kwargs to pass to algorithm
    """
    act_enabled = algo_config.ACT.enabled
    assert act_enabled
    algo_class, algo_kwargs = ACT, {}

    return algo_class, algo_kwargs


class ACT(BC_VAE):
    """
    BC training with a VAE policy.
    """
    def _create_networks(self):
        """
        Creates networks and places them into @self.nets.
        """
        import sys
        from detr.main import build_ACT_model_and_optimizer

        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        self.nets = nn.ModuleDict()
        self.chunk_size = self.global_config["train"]["seq_length"]
        self.camera_keys = self.obs_config['modalities']['obs']['rgb'].copy()
        self.proprio_keys = self.obs_config['modalities']['obs']['low_dim'].copy()
        self.obs_keys = self.proprio_keys + self.camera_keys

        self.proprio_dim = 0
        for k in self.proprio_keys:
            self.proprio_dim += self.obs_key_shapes[k][0]

        # 动作维度自动检测
        print(f"\n[ACT DEBUG] ========== Action Dimension Detection ==========")
        print(f"[ACT DEBUG] self.ac_dim: {self.ac_dim}")
        print(f"[ACT DEBUG] Available attributes: {[attr for attr in dir(self) if 'action' in attr.lower()]}")
        
        # 尝试多种方式获取正确的动作维度
        actual_action_dim = self.ac_dim  # 默认值
        
        if hasattr(self, 'action_shapes') and 'actions' in self.action_shapes:
            actual_action_dim = self.action_shapes['actions'][0]
            print(f"[ACT DEBUG] ✓ Using action_dim from self.action_shapes: {actual_action_dim}")
        elif 'actions' in self.obs_key_shapes:
            actual_action_dim = self.obs_key_shapes['actions'][0]
            print(f"[ACT DEBUG] ✓ Using action_dim from self.obs_key_shapes: {actual_action_dim}")
        else:
            print(f"[ACT DEBUG] ⚠ Using default action_dim from self.ac_dim: {actual_action_dim}")
        
        print(f"[ACT DEBUG] Final action dimension: {actual_action_dim}")
        print(f"[ACT DEBUG] ================================================\n")

        # 构建配置字典
        policy_config = {
            'num_queries': self.chunk_size,
            'hidden_dim': self.algo_config['ACT']['hidden_dim'],
            'dim_feedforward': self.algo_config['ACT']['dim_feedforward'],
            'backbone': self.algo_config['ACT']['backbone'],
            'enc_layers': self.algo_config['ACT']['enc_layers'],
            'dec_layers': self.algo_config['ACT']['dec_layers'],
            'nheads': self.algo_config['ACT']['nheads'],
            'latent_dim': self.algo_config['ACT']['latent_dim'],
            'a_dim': actual_action_dim,  # 使用检测到的正确维度
            'state_dim': self.proprio_dim,
            'camera_names': self.camera_keys
        }
        self.kl_weight = self.algo_config['ACT']['kl_weight']
        
        # 保存原始 sys.argv
        original_argv = sys.argv.copy()
        
        # 构造伪命令行参数以满足 argparse 要求
        sys.argv = [
            'train.py',
            '--ckpt_dir', '/tmp/act_ckpt',
            '--policy_class', 'ACT',
            '--task_name', 'robomimic_task',
            '--seed', str(self.global_config.train.seed),
            '--num_epochs', str(self.global_config.train.num_epochs),
            '--kl_weight', str(self.kl_weight),
            '--chunk_size', str(self.chunk_size),
            '--hidden_dim', str(policy_config['hidden_dim']),
            '--dim_feedforward', str(policy_config['dim_feedforward']),
            '--backbone', policy_config['backbone'],
            '--enc_layers', str(policy_config['enc_layers']),
            '--dec_layers', str(policy_config['dec_layers']),
            '--nheads', str(policy_config['nheads']),
            '--num_queries', str(policy_config['num_queries']),
            '--camera_names', ','.join(policy_config['camera_names'])
        ]
        
        try:
            # 调用 build_ACT_model_and_optimizer
            model, optimizer = build_ACT_model_and_optimizer(policy_config)
        finally:
            # 恢复原始 sys.argv
            sys.argv = original_argv
        
        self.nets["policy"] = model
        self.nets = self.nets.float().to(self.device)

        self.temporal_agg = False
        self.query_frequency = self.chunk_size

        self._step_counter = 0
        self.a_hat_store = None

    def process_batch_for_training(self, batch):
        """
        Processes input batch from a data loader to filter out
        relevant information and prepare the batch for training.
        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader
        Returns:
            input_batch (dict): processed and filtered batch that
                will be used for training
        """
        input_batch = dict()
        input_batch["obs"] = {k: batch["obs"][k][:, 0, :] for k in batch["obs"] if k != 'pad_mask'}
        input_batch["obs"]['pad_mask'] = batch["obs"]['pad_mask']
        input_batch["goal_obs"] = batch.get("goal_obs", None)
        input_batch["actions"] = batch["actions"][:, :, :]
        return TensorUtils.to_float(TensorUtils.to_device(input_batch, self.device))

    def train_on_batch(self, batch, epoch, validate=False):
        """
        Update from superclass to set categorical temperature, for categorcal VAEs.
        """
        return super(BC_VAE, self).train_on_batch(batch, epoch, validate=validate)

    def _forward_training(self, batch):
        """
        Internal helper function for BC algo class. Compute forward pass
        and return network outputs in @predictions dict.
        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training
        Returns:
            predictions (dict): dictionary containing network outputs
        """
        proprio = [batch["obs"][k] for k in self.proprio_keys]
        proprio = torch.cat(proprio, axis=1)
        qpos = proprio

        images = []
        for cam_name in self.camera_keys:
            image = batch['obs'][cam_name]
            image = self.normalize(image)
            image = image.unsqueeze(axis=1)
            images.append(image)
        images = torch.cat(images, axis=1)

        env_state = torch.zeros([qpos.shape[0], 10]).cuda()

        actions = batch['actions']
        is_pad = batch['obs']['pad_mask'] == 0
        is_pad = is_pad.squeeze(dim=-1)

        a_hat, is_pad_hat, (mu, logvar) = self.nets["policy"](qpos, images, env_state, actions, is_pad)
        total_kld, dim_wise_kld, mean_kld = self.kl_divergence(mu, logvar)
        loss_dict = dict()
        all_l1 = F.l1_loss(actions, a_hat, reduction='none')
        l1 = (all_l1 * ~is_pad.unsqueeze(-1)).mean()
        loss_dict['l1'] = l1
        loss_dict['kl'] = total_kld[0]

        predictions = OrderedDict(
            actions=actions,
            kl_loss=loss_dict['kl'],
            reconstruction_loss=loss_dict['l1'],
        )

        return predictions

    def get_action(self, obs_dict, goal_dict=None):
        """
        Get policy action outputs.
        Args:
            obs_dict (dict): current observation
            goal_dict (dict): (optional) goal
        Returns:
            action (torch.Tensor): action tensor
        """
        assert not self.nets.training

        proprio = [obs_dict[k] for k in self.proprio_keys]
        proprio = torch.cat(proprio, axis=1)
        qpos = proprio

        images = []
        for cam_name in self.camera_keys:
            image = obs_dict[cam_name]
            image = self.normalize(image)
            image = image.unsqueeze(axis=1)
            images.append(image)
        images = torch.cat(images, axis=1)

        env_state = torch.zeros([qpos.shape[0], 10]).cuda()

        if self._step_counter % self.query_frequency == 0:
            a_hat, is_pad_hat, (mu, logvar) = self.nets["policy"](qpos, images, env_state)
            self.a_hat_store = a_hat

        action = self.a_hat_store[:, self._step_counter % self.query_frequency, :]
        self._step_counter += 1
        return action

    def reset(self):
        """
        Reset algo state to prepare for environment rollouts.
        """
        self._step_counter = 0

    def _compute_losses(self, predictions, batch):
        """
        Internal helper function for BC algo class. Compute losses based on
        network outputs in @predictions dict, using reference labels in @batch.
        Args:
            predictions (dict): dictionary containing network outputs, from @_forward_training
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training
        Returns:
            losses (dict): dictionary of losses computed over the batch
        """
        kl_loss = predictions["kl_loss"]
        recons_loss = predictions["reconstruction_loss"]
        action_loss = recons_loss + self.kl_weight * kl_loss
        return OrderedDict(
            recons_loss=recons_loss,
            kl_loss=kl_loss,
            action_loss=action_loss,
        )

    def log_info(self, info):
        """
        Process info dictionary from @train_on_batch to summarize
        information to pass to tensorboard for logging.
        Args:
            info (dict): dictionary of info
        Returns:
            loss_log (dict): name -> summary statistic
        """
        log = PolicyAlgo.log_info(self, info)
        log["Loss"] = info["losses"]["action_loss"].item()
        log["KL_Loss"] = info["losses"]["kl_loss"].item()
        log["Reconstruction_Loss"] = info["losses"]["recons_loss"].item()
        if "policy_grad_norms" in info:
            log["Policy_Grad_Norms"] = info["policy_grad_norms"]
        return log

    def kl_divergence(self, mu, logvar):
        batch_size = mu.size(0)
        assert batch_size != 0
        if mu.data.ndimension() == 4:
            mu = mu.view(mu.size(0), mu.size(1))
        if logvar.data.ndimension() == 4:
            logvar = logvar.view(logvar.size(0), logvar.size(1))

        klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        total_kld = klds.sum(1).mean(0, True)
        dimension_wise_kld = klds.mean(0)
        mean_kld = klds.mean(1).mean(0, True)

        return total_kld, dimension_wise_kld, mean_kld
