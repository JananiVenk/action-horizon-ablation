import robosuite as suite
import numpy as np
import h5py
import os

# Load official demos
src = h5py.File("official_demos/lift_demos.hdf5", "r")
save_dir = "demos_with_obs"
os.makedirs(save_dir, exist_ok=True)

env = suite.make(
    "Lift",
    robots="Panda",
    has_renderer=False,
    has_offscreen_renderer=True,
    use_camera_obs=True,
    camera_names=["agentview", "robot0_eye_in_hand"],
    camera_heights=84,
    camera_widths=84,
    control_freq=20,
    reward_shaping=True,
)

num_demos = 150
print(f"Replaying {num_demos} demos...")

for i in range(num_demos):
    demo = src[f"data/demo_{i}"]
    states = demo["states"][:]
    actions = demo["actions"][:]

    observations, rewards, dones = [], [], []

    # Reset env and set initial state
    env.reset()
    env.sim.set_state_from_flattened(states[0])
    env.sim.forward()
    obs = env._get_observations()

    for j, action in enumerate(actions):
        observations.append(obs)
        obs, reward, done, info = env.step(action)
        rewards.append(reward)
        dones.append(done)

    # Save
    fname = f"{save_dir}/demo_{i+1}.hdf5"
    with h5py.File(fname, "w") as f:
        f.create_dataset("actions", data=actions)
        f.create_dataset("rewards", data=np.array(rewards))
        f.create_dataset("dones", data=np.array(dones))
        og = f.create_group("observations")
        og.create_dataset("agentview_image", data=np.array([o["agentview_image"] for o in observations]))
        og.create_dataset("robot0_eye_in_hand_image", data=np.array([o["robot0_eye_in_hand_image"] for o in observations]))
        og.create_dataset("robot0_joint_pos", data=np.array([o["robot0_joint_pos"] for o in observations]))
        og.create_dataset("robot0_eef_pos", data=np.array([o["robot0_eef_pos"] for o in observations]))
        og.create_dataset("robot0_eef_quat", data=np.array([o["robot0_eef_quat"] for o in observations]))

    print(f"✓ Demo {i+1}/{num_demos} | max reward: {max(rewards):.3f}")

src.close()
env.close()
print("Done!")