import torch.nn as nn
import torch
from torch.utils.data import DataLoader, random_split
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset import LiftDataset
import numpy as np
import os
import matplotlib.pyplot as plt

class MLPPolicyChunked(nn.Module):
    def __init__(self, obs_dim=14, action_dim=7, horizon=32, hidden_dim=256):
        super().__init__()
        self.horizon = horizon
        self.action_dim = action_dim
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim * horizon),
            nn.Tanh(),
        )

    def forward(self, obs):
        out = self.net(obs)
        return out.view(-1, self.horizon, self.action_dim)  # [B, horizon, 7]
    
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

HORIZON = 32

full_dataset = LiftDataset(r"C:\\Users\\vjana\\Desktop\\Projects\\Action Representation Ablation Study for Robot Manipulation Policies\\demos_with_obs", action_horizon=HORIZON)
train_size = int(0.9 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)

model = MLPPolicyChunked(horizon=HORIZON).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
loss_fn = torch.nn.MSELoss()

num_epochs = 100
best_val_loss = float("inf")
os.makedirs("checkpoints", exist_ok=True)
os.makedirs("plots", exist_ok=True)

train_loss_history, val_loss_history = [], []

for epoch in range(num_epochs):
    model.train()
    train_losses = []
    for batch in train_loader:
        obs = batch["obs"].to(device)
        action = batch["action"].to(device)  # [B, horizon, 7]

        pred = model(obs)
        loss = loss_fn(pred, action)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

    model.eval()
    val_losses = []
    with torch.no_grad():
        for batch in val_loader:
            obs = batch["obs"].to(device)
            action = batch["action"].to(device)
            pred = model(obs)
            loss = loss_fn(pred, action)
            val_losses.append(loss.item())

    train_loss = np.mean(train_losses)
    val_loss = np.mean(val_losses)
    train_loss_history.append(train_loss)
    val_loss_history.append(val_loss)

    print(f"Epoch {epoch+1}/{num_epochs} | train_loss: {train_loss:.5f} | val_loss: {val_loss:.5f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "checkpoints/chunked_horizon32.pt")

print("Training done! Best model saved to checkpoints/chunked_horizon32.pt")
