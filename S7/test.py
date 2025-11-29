import json
import os
import numpy as np
import matplotlib.pyplot as plt

with open('test_trajectories.json', 'r') as f:
    trajectories = json.load(f)

out_dir = 'figures'
os.makedirs(out_dir, exist_ok=True)

def validate_episode(traj, ep_idx):
    states = traj.get('states', [])
    actions = traj.get('actions', [])
    rewards = traj.get('rewards', [])
    n_states = len(states)
    n_actions = len(actions)
    n_rewards = len(rewards)
    ok = True
    if n_actions != n_states - 1:
        print(f"[WARN] Episode {ep_idx}: actions ({n_actions}) != states-1 ({n_states-1})")
        ok = False
    if n_rewards != n_actions:
        print(f"[WARN] Episode {ep_idx}: rewards ({n_rewards}) != actions ({n_actions})")
        ok = False
    return ok

for ep_idx, traj in enumerate(trajectories):
    if not validate_episode(traj, ep_idx):
        print(f"Episode {ep_idx} validation failed — check shapes.")
    states = np.array(traj['states'], dtype=float)  # shape (T+1, 5)
    actions = np.array(traj['actions'], dtype=int)
    rewards = np.array(traj['rewards'], dtype=float)

    # Basic summary
    print(f"Episode {ep_idx}: timesteps={states.shape[0]-1}, unique_actions={np.unique(actions)}, total_reward={rewards.sum():.3f}")
    print(f"  initial state: {states[0]}")
    print(f"  final state:   {states[-1]}")

    # Plot all states
    labels = ['heart_rate','systolic_bp','diastolic_bp','temperature','lactate']
    fig, axs = plt.subplots(2, 3, figsize=(14, 8))
    axs = axs.flatten()
    for i in range(states.shape[1]):
        axs[i].plot(states[:, i], marker='o')
        axs[i].set_title(labels[i])
        axs[i].set_xlabel('time step')
    axs[4].set_ylabel(labels[4])
    # Actions plot
    axs[5].plot(np.arange(len(actions)), actions, marker='x', color='orange')
    axs[5].set_title('Actions (index)')
    axs[5].set_xlabel('time step')
    plt.tight_layout()
    fig_path = os.path.join(out_dir, f'episode_{ep_idx}_states_actions.png')
    plt.savefig(fig_path)
    plt.close(fig)

    # Rewards and cumulative reward
    plt.figure(figsize=(8,3))
    plt.subplot(1,2,1)
    plt.plot(rewards, marker='o')
    plt.title(f'Episode {ep_idx} - Rewards')
    plt.xlabel('time step')
    plt.subplot(1,2,2)
    plt.plot(np.cumsum(rewards), marker='o', color='green')
    plt.title(f'Episode {ep_idx} - Cumulative Reward')
    plt.xlabel('time step')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'episode_{ep_idx}_rewards.png'))
    plt.close()

    # Action histogram
    plt.figure(figsize=(5,3))
    plt.hist(actions, bins=range(min(actions)-1, max(actions)+2), color='C1')
    plt.title(f'Episode {ep_idx} - Action distribution')
    plt.xlabel('action index')
    plt.ylabel('count')
    plt.savefig(os.path.join(out_dir, f'episode_{ep_idx}_actions_hist.png'))
    plt.close()

print(f"Figures saved to {out_dir}/")