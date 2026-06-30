import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import robosuite as suite
import numpy as np
import imageio

device = "cuda" if torch.cuda.is_available() else "cpu"
HORIZON = 32

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
        return out.view(-1, self.horizon, self.action_dim)

model = MLPPolicyChunked(horizon=HORIZON).to(device)
model.load_state_dict(torch.load("checkpoints/chunked_horizon32.pt"))
model.eval()

env = suite.make(
    "Lift",
    robots="Panda",
    has_renderer=False,
    has_offscreen_renderer=True,
    use_camera_obs=True,
    camera_names=["agentview"],
    camera_heights=480,
    camera_widths=480,
    control_freq=20,
    reward_shaping=True,
)

best_frames = None
best_reward = -1
num_tries = 10

for ep in range(num_tries):
    obs = env.reset()
    frames = []
    max_reward = 0
    steps_taken = 0

    while steps_taken < 300:
        frame = obs["agentview_image"]
        frames.append(frame)

        obs_vec = np.concatenate([
            obs["robot0_eef_pos"],
            obs["robot0_eef_quat"],
            obs["robot0_joint_pos"]
        ])
        obs_tensor = torch.tensor(obs_vec, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            action_chunk = model(obs_tensor).cpu().numpy().squeeze(0)

        for action in action_chunk:
            frame = obs["agentview_image"]
            frames.append(frame)

            obs, reward, done, info = env.step(action)
            max_reward = max(max_reward, reward)
            steps_taken += 1
            if done or steps_taken >= 300:
                break

        if done:
            break

    print(f"Episode {ep+1}: max_reward={max_reward:.3f}")

    if max_reward > best_reward:
        best_reward = max_reward
        best_frames = frames

env.close()

imageio.mimsave("chunked32_demo.mp4", best_frames, fps=20)
print(f"Saved video with best reward {best_reward:.3f} to chunked_demo.mp4")