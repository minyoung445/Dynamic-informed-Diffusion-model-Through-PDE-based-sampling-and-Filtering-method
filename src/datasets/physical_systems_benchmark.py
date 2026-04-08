import json
import pathlib
from collections import namedtuple

from typing import Any, Dict, List, Union

import numpy as np
from einops import rearrange
from torch.utils import data


Trajectory = namedtuple(
    "Trajectory",
    [
        "name",
        "features",
        "dp_dt",
        "dq_dt",
        "t",
        "trajectory_meta",
        "p_noiseless",
        "q_noiseless",
        "masses",
        "edge_index",
        "vertices",
        "fixed_mask",
        "condition",
        "static_nodes",
    ],
)


class TrajectoryDataset(data.Dataset):
    """Returns batches of full trajectories.
    dataset[idx] -> a set of snapshots for a full trajectory"""

    def __init__(self, data_dir, subsample: int = 1, max_samples: int = None):
        super().__init__()
        data_dir = pathlib.Path(data_dir)
        self.subsample = subsample
        self.max_samples = max_samples

        with open(data_dir / "system_meta.json", "r", encoding="utf8") as meta_file:
            metadata = json.load(meta_file)
        self.system = str(metadata["system"]).lower()
        self.system_metadata = metadata["metadata"]
        self._trajectory_meta: List[Dict[str, Any]] = metadata["trajectories"]
        self._npz_file = np.load(data_dir / "trajectories.npz")

        if self.system == "navier-stokes":
            self.h, self.w = 221, 42
            self._ndims_p = 2
            self._ndims_q = 1
        elif self.system == "spring-mesh":
            self.h, self.w = 10, 10
            self._ndims_p = 2
            self._ndims_q = 2
        elif self.system == "wave":
           
            self.h, self.w = 125, 1
            self._ndims_p = 1
            self._ndims_q = 1
        else:
            raise ValueError(f"Unknown system: {self.system}")

 
    def _pos_axis(self, ndim: int, axis: int) -> int:
        return axis if axis >= 0 else ndim + axis

    def _ensure_last_channel(self, arr: np.ndarray) -> np.ndarray:
        
        if arr.ndim == 2:
            return arr[..., None]
        return arr

    def concatenate_features(self, p: np.ndarray, q: np.ndarray, channel_dim: int = -1) -> np.ndarray:
        
        
        if self.system == "wave":
            p = self._ensure_last_channel(p)
            q = self._ensure_last_channel(q)
            cax_p = self._pos_axis(p.ndim, channel_dim)
            cax_q = self._pos_axis(q.ndim, channel_dim)
            assert p.shape[cax_p] == 1, f"[wave] Expected p 1ch, got {p.shape}"
            assert q.shape[cax_q] == 1, f"[wave] Expected q 1ch, got {q.shape}"
            return np.concatenate([p, q], axis=channel_dim)  # (..., 2, ...)

        # ns / spring-mesh 등: 기존 규칙
        p = self._ensure_last_channel(p) if p.ndim <= 2 else p
        q = self._ensure_last_channel(q) if q.ndim <= 2 else q

        cax_p = self._pos_axis(p.ndim, channel_dim)
        cax_q = self._pos_axis(q.ndim, channel_dim)
        assert p.shape[cax_p] == self._ndims_p, f"Expected p to have {self._ndims_p} channels but got {p.shape}"
        assert q.shape[cax_q] == self._ndims_q, f"Expected q to have {self._ndims_q} channel(s), but got {q.shape}"
        return np.concatenate([p, q], axis=channel_dim)

    # ---------- Dataset API ----------
    def __len__(self):
        return len(self._trajectory_meta) if self.max_samples is None else self.max_samples

    def __getitem__(self, idx):
        meta = self._trajectory_meta[idx]
        name = meta["name"]

    
        p = self._npz_file[meta["field_keys"]["p"]]
        q = self._npz_file[meta["field_keys"]["q"]]
        dp_dt = self._npz_file[meta["field_keys"]["dpdt"]]
        dq_dt = self._npz_file[meta["field_keys"]["dqdt"]]
        t = self._npz_file[meta["field_keys"]["t"]]

        # noiseless
        if "p_noiseless" in meta["field_keys"] and "q_noiseless" in meta["field_keys"]:
            p_noiseless = self._npz_file[meta["field_keys"]["p_noiseless"]]
            q_noiseless = self._npz_file[meta["field_keys"]["q_noiseless"]]
        else:
            p_noiseless, q_noiseless = p, q

        # masses
        if "masses" in meta["field_keys"]:
            masses = self._npz_file[meta["field_keys"]["masses"]]
        else:
            masses = np.ones(self.h * self.w, dtype=np.float32)

        # edge_index
        if "edge_indices" in meta["field_keys"]:
            edge_index = self._npz_file[meta["field_keys"]["edge_indices"]]
            if edge_index.ndim == 2 and edge_index.shape[0] != 2:
                edge_index = edge_index.T
        else:
            edge_index = []

        # vertices
        if "vertices" in meta["field_keys"]:
            vertices = self._npz_file[meta["field_keys"]["vertices"]]
        else:
            vertices = []

        # masks / conditions
        if "fixed_mask_p" in meta["field_keys"]:
            fixed_mask_p = np.expand_dims(self._npz_file[meta["field_keys"]["fixed_mask_p"]], 0)
        else:
            fixed_mask_p = [[]]
        if "fixed_mask_q" in meta["field_keys"]:
            fixed_mask_q = np.expand_dims(self._npz_file[meta["field_keys"]["fixed_mask_q"]], 0)
        else:
            fixed_mask_q = [[]]

        
        if "extra_fixed_mask" in meta["field_keys"]:
            extra_fixed_mask = np.expand_dims(self._npz_file[meta["field_keys"]["extra_fixed_mask"]], 0)
        
            if isinstance(extra_fixed_mask, np.ndarray) and extra_fixed_mask.ndim == 2:
                extra_fixed_mask = extra_fixed_mask[..., None]  # (1, H*W, 1)
        else:
            extra_fixed_mask = np.zeros((1, self.h * self.w, 1), dtype=np.float32)  # (1, H*W, 1)

        if "enumerated_fixed_mask" in meta["field_keys"]:
            static_nodes = np.expand_dims(self._npz_file[meta["field_keys"]["enumerated_fixed_mask"]], 0)
        else:
            static_nodes = [[]]

      
        features = self.concatenate_features(p, q, channel_dim=-1)

        # (time, (h w), C) -> (time, C, h, w)
        features = rearrange(features, "time (h w) c -> time c h w", h=self.h, w=self.w).astype(np.float32)

        
        dp_dt = self._ensure_last_channel(dp_dt)
        dq_dt = self._ensure_last_channel(dq_dt)
        p_noiseless = self._ensure_last_channel(p_noiseless)
        q_noiseless = self._ensure_last_channel(q_noiseless)

        dp_dt = rearrange(dp_dt, "time (h w) c -> time c h w", h=self.h, w=self.w).astype(np.float32)
        dq_dt = rearrange(dq_dt, "time (h w) c -> time c h w", h=self.h, w=self.w).astype(np.float32)
        p_noiseless = rearrange(p_noiseless, "time (h w) c -> time c h w", h=self.h, w=self.w).astype(np.float32)
        q_noiseless = rearrange(q_noiseless, "time (h w) c -> time c h w", h=self.h, w=self.w).astype(np.float32)

        # masses, vertices, static_nodes
        masses = rearrange(masses, "(h w) -> h w", h=self.h, w=self.w)
        vertices = (
            rearrange(vertices, "(h w) c -> c h w", h=self.h, w=self.w).astype(np.float32) if len(vertices) > 0 else []
        )
        static_nodes = (
            rearrange(static_nodes.squeeze(), "(h w) -> h w", h=self.h, w=self.w) if len(static_nodes[0]) > 0 else []
        )

        
        if len(fixed_mask_p[0]) > 0:
            if fixed_mask_p.ndim == 2:  # (1, H*W)
                fixed_mask_p = fixed_mask_p[..., None]  # (1, H*W, 1)
            fixed_mask_p = rearrange(fixed_mask_p.squeeze(), "(h w) c -> c h w", h=self.h, w=self.w)
        else:
            fixed_mask_p = np.zeros((self._ndims_p, self.h, self.w), dtype=bool)

        if len(fixed_mask_q[0]) > 0:
            if fixed_mask_q.ndim == 2:  # (1, H*W)
                fixed_mask_q = fixed_mask_q[..., None]  # (1, H*W, 1)
            fixed_mask_q = rearrange(fixed_mask_q.squeeze(), "(h w) c -> c h w", h=self.h, w=self.w)
        else:
            fixed_mask_q = np.zeros((self._ndims_q, self.h, self.w), dtype=bool)

        fixed_mask = self.concatenate_features(fixed_mask_p, q=fixed_mask_q, channel_dim=0)

       
        extra_fixed_mask = rearrange(extra_fixed_mask, "1 (h w) c -> c h w", h=self.h, w=self.w).astype(np.float32)

        # subsample
        if self.subsample > 1:
            step = int(self.subsample)
            meta = dict(meta)
            meta["time_step_size"] = meta.get("time_step_size", 1.0) * step

            features = features[::step]
            dp_dt = dp_dt[::step]
            dq_dt = dq_dt[::step]
            p_noiseless = p_noiseless[::step]
            q_noiseless = q_noiseless[::step]
            t = t[::step]
            meta["num_time_steps"] = len(t)

        return Trajectory(
            name=name,
            trajectory_meta=meta,
            features=features,
            dp_dt=dp_dt,
            dq_dt=dq_dt,
            t=t,
            p_noiseless=p_noiseless,
            q_noiseless=q_noiseless,
            masses=masses,
            edge_index=edge_index,
            vertices=vertices,
            fixed_mask=fixed_mask.astype(bool) if fixed_mask.dtype != bool else fixed_mask,
            condition=extra_fixed_mask,   
            static_nodes=static_nodes,
        )
    def __len__(self):
        return len(self._trajectory_meta) if self.max_samples is None else self.max_samples
