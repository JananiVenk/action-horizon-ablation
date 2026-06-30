import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
import os

class LiftDataset(Dataset):
    def __init__(self, data_dir, action_horizon=1):
        self.data_dir = data_dir
        self.files = sorted([f for f in os.listdir(data_dir) if f.endswith(".hdf5")])
        self.action_horizon = action_horizon
        
        # Build index: (file_idx, timestep)
        self.index = []
        for fi, fname in enumerate(self.files):
            with h5py.File(os.path.join(data_dir, fname), "r") as f:
                T = f["actions"].shape[0]
                for t in range(T - action_horizon):
                    self.index.append((fi, t))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        fi, t = self.index[idx]
        fname = self.files[fi]
        with h5py.File(os.path.join(self.data_dir, fname), "r") as f:
            eef_pos = f["observations/robot0_eef_pos"][t]
            eef_quat = f["observations/robot0_eef_quat"][t]
            joint_pos = f["observations/robot0_joint_pos"][t]
            
            obs = np.concatenate([eef_pos, eef_quat, joint_pos])
            action = f["actions"][t : t + self.action_horizon]
        
        return {
            "obs": torch.tensor(obs, dtype=torch.float32),
            "action": torch.tensor(action, dtype=torch.float32),
        }