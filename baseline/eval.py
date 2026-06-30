import torch
import torch.nn as nn
import robosuite as suite
import numpy as np

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
            nn.Tanh()
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
    has_offscreen_renderer=False,
    use_camera_obs=False,
    control_freq=20,
    reward_shaping=True,
)

num_eval_episodes = 50
successes = 0

for ep in range(num_eval_episodes):
    obs = env.reset()
    max_reward = 0
    
    for step in range(300):
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
    
    if max_reward > 0.9:
        successes += 1
        print(f"Episode {ep+1}: ✓ SUCCESS (max_reward={max_reward:.3f})")
    else:
        print(f"Episode {ep+1}: ✗ fail (max_reward={max_reward:.3f})")

success_rate = successes / num_eval_episodes * 100
print(f"\nBaseline Success Rate: {success_rate:.1f}% ({successes}/{num_eval_episodes})")

env.close()