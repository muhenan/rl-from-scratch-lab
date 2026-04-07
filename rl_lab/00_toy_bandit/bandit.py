import random
import matplotlib.pyplot as plt

# ─── Problem Setup ────────────────────────────────────────────────────────────
# 3 arms, each with a fixed (unknown to agent) reward probability
arm_probs = [0.2, 0.5, 0.8]   # true reward probabilities

# ─── Policy ───────────────────────────────────────────────────────────────────
# policy[i] = probability of choosing arm i
# starts uniform: agent has no preference
policy = [1/3, 1/3, 1/3]

learning_rate = 0.1
num_steps = 100

# ─── History (for plotting) ───────────────────────────────────────────────────
history = [[], [], []]   # history[i] = policy[i] at each step

# ─── Run Loop ─────────────────────────────────────────────────────────────────
for step in range(1, num_steps + 1):

    print(f"\nStep {step}")
    print(f"Policy: [{policy[0]:.4f}, {policy[1]:.4f}, {policy[2]:.4f}]")

    # ── Action Sampling ──────────────────────────────────────────────────────
    # sample action according to current policy distribution
    r = random.random()
    if r < policy[0]:
        action = 0
    elif r < policy[0] + policy[1]:
        action = 1
    else:
        action = 2

    print(f"Action: {action}")

    # ── Reward Simulation ────────────────────────────────────────────────────
    # reward = 1 with probability arm_probs[action], else 0
    reward = 1 if random.random() < arm_probs[action] else 0

    print(f"Reward: {reward}")

    # ── REINFORCE-style Policy Update ────────────────────────────────────────
    # increase probability of the chosen action if it got rewarded
    # formula: policy[action] += lr * reward
    policy[action] += learning_rate * reward

    # normalize so policy sums to 1
    total = policy[0] + policy[1] + policy[2]
    policy[0] /= total
    policy[1] /= total
    policy[2] /= total

    print(f"Updated Policy: [{policy[0]:.4f}, {policy[1]:.4f}, {policy[2]:.4f}]")

    history[0].append(policy[0])
    history[1].append(policy[1])
    history[2].append(policy[2])

# ─── Final Result ─────────────────────────────────────────────────────────────
print("\n" + "="*50)
print("Final Policy:")
print(f"  arm 0 (prob=0.2): {policy[0]:.4f}")
print(f"  arm 1 (prob=0.5): {policy[1]:.4f}")
print(f"  arm 2 (prob=0.8): {policy[2]:.4f}")
print(f"\nBest arm chosen by policy: arm {policy.index(max(policy))}")

# ─── Plot ─────────────────────────────────────────────────────────────────────
steps = list(range(1, num_steps + 1))
plt.plot(steps, history[0], label="arm 0 (true p=0.2)")
plt.plot(steps, history[1], label="arm 1 (true p=0.5)")
plt.plot(steps, history[2], label="arm 2 (true p=0.8)")
plt.xlabel("Step")
plt.ylabel("Policy Probability")
plt.title("Multi-Armed Bandit: Policy over Time")
plt.legend()
plt.tight_layout()
plt.savefig("rl_lab/00_toy_bandit/policy_plot.png", dpi=150)
print("\nPlot saved to rl_lab/00_toy_bandit/policy_plot.png")
plt.show()
