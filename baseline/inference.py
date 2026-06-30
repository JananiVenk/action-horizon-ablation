import torch
import torch.nn as nn
import robosuite as suite
import numpy as np
import imageio

device = "cuda" if torch.cuda.is_available() else "cpu"

class MLPPolicy(nn.Module):
    def __init__(self, obs_dim=14, action_dim=7, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, obs):
        return self.net(obs)

model = MLPPolicy().to(device)
model.load_state_dict(torch.load("checkpoints/baseline_joint_single.pt"))
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

# Try multiple episodes, save the best one
best_frames = None
best_reward = -1

num_tries = 10
for ep in range(num_tries):
    obs = env.reset()
    frames = []
    max_reward = 0

    for step in range(300):
        frame = obs["agentview_image"]
        frames.append(frame)

        obs_vec = np.concatenate([
            obs["robot0_eef_pos"],
            obs["robot0_eef_quat"],
            obs["robot0_joint_pos"]
        ])
        obs_tensor = torch.tensor(obs_vec, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            action = model(obs_tensor).cpu().numpy().squeeze()

        obs, reward, done, info = env.step(action)
        max_reward = max(max_reward, reward)

        if done:
            break

    print(f"Episode {ep+1}: max_reward={max_reward:.3f}")

    if max_reward >=best_reward:
        best_reward = max_reward
        best_frames = frames

env.close()

# Save video
imageio.mimsave("baseline_demo.mp4", best_frames, fps=20)
print(f"Saved video with best reward {best_reward:.3f} to baseline_demo.mp4")