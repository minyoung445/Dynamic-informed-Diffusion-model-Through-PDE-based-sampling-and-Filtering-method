from typing import Any, List

import numpy as np
import torch
import torchmetrics
from einops import rearrange
from torch import Tensor
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigh
import math
import pprint
import torch.nn.functional as F

from src.experiment_types._base_experiment import BaseExperiment

lambda_reg = 0.2

class InterpolationExperiment(BaseExperiment):
    r"""Base class for all interpolation experiments."""

    

    def __init__(self, stack_window_to_channel_dim: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.save_hyperparameters(ignore=["model"])
        assert self.horizon >= 2, "horizon must be >=2 for interpolation experiments"

       
        self.register_buffer("_A_frac", None)

    @property
    def horizon_range(self) -> List[int]:
        # h = horizon
        # We use timesteps  w-l+1, ..., w-1, w+h to predict timesteps w, ..., w+h-1
        # interpolate between step t=0 and t=horizon
        return list(np.arange(1, self.horizon))

    @property
    def true_horizon(self) -> int:
        return self.horizon

    @property
    def horizon_name(self) -> str:
        s = f"{self.true_horizon}h"
        return s

    @property
    def short_description(self) -> str:
        name = super().short_description
        name += f" (h={self.horizon_name})"
        return name

    @property
    def WANDB_LAST_SEP(self) -> str:
        return "/ipol/"

    def actual_num_input_channels(self, num_input_channels: int) -> int:
        if self.hparams.stack_window_to_channel_dim:
            return num_input_channels * self.window + num_input_channels
        return 2 * num_input_channels  # inputs and targets are concatenated
    
   

    def _expected_cond_channels(self) -> int:
       
        for key in ("conditional_channels", "num_conditional_channels"):
            if hasattr(self.model, key):
                val = getattr(self.model, key)
                if isinstance(val, int):
                    return val
        
        return 0

    
    def _build_condition_from_batch(self, batch: Any, H: int, W: int, device: torch.device) -> torch.Tensor:
        exp_c = self._expected_cond_channels()
        if exp_c == 0:
            return None 

        B = batch["dynamics"].shape[0]

        ready = batch.get("static", None) or batch.get("condition", None)
        if ready is not None:
            cond = ready
            if cond.dim() == 3:  # (B,H,W) -> (B,1,H,W)
                cond = cond.unsqueeze(1)
            if cond.dim() != 4:
                raise RuntimeError(f"Condition must be 4D, got shape={tuple(cond.shape)}")
          
            if cond.shape[-2:] != (H, W):
                cond = F.interpolate(cond.float(), size=(H, W), mode="nearest")
            if cond.shape[0] != B:
                raise RuntimeError(f"Condition batch size mismatch: {cond.shape[0]} vs {B}")

           
            c = cond.shape[1]
            if c > exp_c:
                cond = cond[:, :exp_c]  
            elif c < exp_c:
                pad = torch.zeros(B, exp_c - c, H, W, device=cond.device, dtype=cond.dtype)
                cond = torch.cat([cond, pad], dim=1)
            return cond.to(device)

       
        pieces = []

       
        mask = batch.get("obstacle_mask", None) or batch.get("mask", None)
        if mask is not None:
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
            elif mask.dim() != 4:
                raise RuntimeError(f"Unsupported mask shape: {tuple(mask.shape)}")
            if mask.shape[1] != 1:
                raise RuntimeError(f"mask must be 1 channel, got {mask.shape}")
            if mask.shape[-2:] != (H, W):
                mask = F.interpolate(mask.float(), size=(H, W), mode="nearest")
            pieces.append(mask.to(device).float())

        nu = batch.get("viscosity", None) or batch.get("nu", None)
        if nu is not None:
            if   nu.dim() == 1: nu = nu.view(-1,1,1,1).expand(-1,1,H,W)
            elif nu.dim() == 2: nu = nu.view(-1,1,1,1).expand(-1,1,H,W)
            elif nu.dim() == 4:
                if nu.shape[1] != 1: raise RuntimeError(f"nu must be 1 channel, got {nu.shape}")
                if nu.shape[-2:] != (H, W):
                    nu = F.interpolate(nu.float(), size=(H, W), mode="nearest")
            else:
                raise RuntimeError(f"Unsupported nu shape: {tuple(nu.shape)}")
            pieces.append(nu.to(device).float())

       
        if not pieces:
            raise RuntimeError(
                f"No condition found to build. Expected {exp_c} channel(s). "
                f"Provide 'static'/'condition' or fields like 'mask'/'obstacle_mask', 'viscosity'/'nu'."
            )

        cond = torch.cat(pieces, dim=1)  # (B, sumC, H, W)

        c = cond.shape[1]
        if c > exp_c:
            cond = cond[:, :exp_c]       
        elif c < exp_c:
            pad = torch.zeros(B, exp_c - c, H, W, device=cond.device, dtype=cond.dtype)
            cond = torch.cat([cond, pad], dim=1)

        return cond


    def _build_fractional_operator(self, H: int, W: int) -> torch.Tensor:
        
        d    = 2
        v    = 0.5
        rho  = 12.0
        ell  = np.sqrt(2 * v / rho)      # ℓ = sqrt(2v/ρ)
        alpha = (d / 2) + v

        M   = H * W
        dx2 = 1.0

        def idx(i, j):
            return i * W + j

        rows, cols, vals = [], [], []
        for i in range(H):
            for j in range(W):
                k = idx(i, j)
               
                rows.append(k); cols.append(k); vals.append(4.0 / dx2)

             
                for di, dj in [(1,0), (-1,0), (0,1), (0,-1)]:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < H and 0 <= nj < W:
                        rows.append(k)
                        cols.append(idx(ni, nj))
                        vals.append(-1.0 / dx2)
                    

        
        L = coo_matrix((vals, (rows, cols)), shape=(M, M), dtype=np.float64)

        
        I = coo_matrix((np.ones(M), (np.arange(M), np.arange(M))), shape=(M, M))
        A = I - (ell ** -2) * L

        
        A_mat = A.toarray()
        eig_vals, eig_vecs = eigh(A_mat)                 # λ_i, Q
        eig_vals = torch.from_numpy(eig_vals).float()    # (M,)
        eig_vecs = torch.from_numpy(eig_vecs).float()    # (M,M)

        # (I - ℓ²Δ)^(α/2) = Q diag(λ_i^(α/2)) Q^T
        diag_pow = eig_vals.clamp(min=0.0) ** (alpha / 2)
        A_frac = (eig_vecs * diag_pow.unsqueeze(0)) @ eig_vecs.t()  # (M,M)
        return A_frac

    # --------------------------------- Metrics
    def get_metrics(self, split: str, split_name: str, **kwargs) -> torch.nn.ModuleDict:
        metrics = {
            f"{split_name}/{self.horizon_name}_avg{self.WANDB_LAST_SEP}mse": torchmetrics.MeanSquaredError(
                squared=True
            )
        }
        for h in self.horizon_range:
            metrics[f"{split_name}/t{h}{self.WANDB_LAST_SEP}mse"] = torchmetrics.MeanSquaredError(squared=True)
        return torch.nn.ModuleDict(metrics)

    @property
    def default_monitor_metric(self) -> str:
        return f"val/{self.horizon_name}_avg{self.WANDB_LAST_SEP}mse"

    @torch.no_grad()
    def _evaluation_step(
        self,
        batch: Any,
        batch_idx: int,
        split: str,
        dataloader_idx: int = None,
        return_only_preds_and_targets: bool = False,
    ):
        log_dict = dict()
        compute_metrics = split != "predict"
        split_metrics = getattr(self, f"{split}_metrics") if compute_metrics else None
        dynamics = batch["dynamics"]  # dynamics is a (b, t, c, h, w) tensor

        return_dict = dict()
        avg_mse_key = f"{split}/{self.horizon_name}_avg{self.WANDB_LAST_SEP}mse"
        avg_mse_tracker = split_metrics[avg_mse_key] if compute_metrics else None

        inputs = self.get_evaluation_inputs(dynamics, split=split)
        extra_kwargs = {}
        for k, v in batch.items():
            if k != "dynamics":
                extra_kwargs[k] = self.get_ensemble_inputs(v, split=split, add_noise=False)

        for t_step in self.horizon_range:
            # dynamics[, self.window] is already the first target frame (t_step=1)
            targets = dynamics[:, self.window + t_step - 1, ...]  # (b, c, h, w)
            time = torch.full((inputs.shape[0],), t_step, device=self.device, dtype=torch.long)

            results = self.predict(inputs, time=time, **extra_kwargs)
            results["targets"] = targets
            preds = results["preds"]
            results = {f"t{t_step}_{k}": v for k, v in results.items()}

            if return_only_preds_and_targets:
                return_dict[f"t{t_step}_preds"] = preds
                return_dict[f"t{t_step}_targets"] = targets
            else:
                return_dict = {**return_dict, **results}
            if not compute_metrics:
                continue
            if self.use_ensemble_predictions(split):
                preds = preds.mean(dim=0)  # average over ensemble
           
            # Compute mse (force contiguous to satisfy torchmetrics’ .view(-1))
            metric_name = f"{split}/t{t_step}{self.WANDB_LAST_SEP}mse"
            metric = split_metrics[metric_name]
            preds   = preds.reshape(-1)
            targets = targets.reshape(-1)
            metric(preds, targets)
            # metric_name = f"{split}/t{t_step}{self.WANDB_LAST_SEP}mse"
            # metric = split_metrics[metric_name]
            # metric(preds, targets)  # compute metrics (need to be in separate line to the following line!)
            log_dict[metric_name] = metric

            # Add contribution to the average mse from this time step's MSE
            avg_mse_tracker(preds, targets)

        if compute_metrics:
            log_kwargs = dict()
            log_kwargs["sync_dist"] = True  # for DDP training
            # Log the average MSE
            log_dict[avg_mse_key] = avg_mse_tracker
            self.log_dict(log_dict, on_step=False, on_epoch=True, **log_kwargs)  # log metric objects

        return return_dict

    def get_inputs_from_dynamics(self, dynamics: Tensor, split: str, **kwargs) -> Tensor:
        """Get the inputs from the dynamics tensor.
        Since we are doing interpolation, this consists of the first window frames plus the last frame.
        """
        assert dynamics.shape[1] == self.window + self.horizon, "dynamics must have shape (b, t, c, h, w)"
        past_steps = dynamics[:, : self.window, ...]  # (b, window, c, lat, lon) at time 0
        last_step = dynamics[:, -1, ...]  # (b, c, lat, lon) at time t=window+horizon
        if self.hparams.stack_window_to_channel_dim:
            past_steps = rearrange(past_steps, "b window c lat lon -> b (window c) lat lon")
        else:
            last_step = last_step.unsqueeze(1)  # (b, 1, c, lat, lon)
        inputs = torch.cat([past_steps, last_step], dim=1)  # (b, window*c + c, lat, lon)
        return inputs

    def get_evaluation_inputs(self, dynamics: Tensor, split: str, **kwargs) -> Tensor:
        inputs = self.get_inputs_from_dynamics(dynamics, split)
        inputs = self.get_ensemble_inputs(inputs, split)
        return inputs

    

    
    def get_loss(self, batch: Any) -> Tensor:
        dynamics = batch["dynamics"]
        split = "train" if self.training else "val"
        inputs = self.get_inputs_from_dynamics(dynamics, split=split)
        b, _, c, H, W = dynamics.shape

        possible_times = torch.tensor(self.horizon_range, device=self.device, dtype=torch.long)
        t = possible_times[torch.randint(len(possible_times), (b,), device=self.device)]
        targets = dynamics[torch.arange(b), self.window + t - 1, ...]

        exp_c = self._expected_cond_channels()
        cond = self._build_condition_from_batch(batch, H, W, device=inputs.device) if exp_c > 0 else None

        extra = {k: v for k, v in batch.items()
                if k not in ("dynamics", "static", "condition", "obstacle_mask", "mask", "viscosity", "nu")}

        if exp_c > 0:
            base_mse = self.model.get_loss(inputs=inputs, targets=targets, time=t, condition=cond, **extra)
        else:
            base_mse = self.model.get_loss(inputs=inputs, targets=targets, time=t, **extra)

       
        if (self._A_frac is None) or (self._A_frac.shape[0] != H * W):
            A_frac = self._build_fractional_operator(H, W).to(self.device)
            object.__setattr__(self, "_A_frac", A_frac)
        else:
            A_frac = self._A_frac

      
        if exp_c > 0:
            phi = self.model(inputs, time=t, condition=cond)
        else:
            phi = self.model(inputs, time=t)

        phi_flat = phi.view(b * c, H * W)
        reg_flat = phi_flat @ A_frac.t()
        reg_term = reg_flat.pow(2).mean()

        loss = base_mse + lambda_reg * reg_term
        return loss