import random

# ─── Problem Setup ────────────────────────────────────────────────────────────
# 3 arms, each with a fixed (unknown to agent) reward probability
arm_probs = [0.2, 0.5, 0.8]   # true reward probabilities

# ─── Policy ───────────────────────────────────────────────────────────────────
# policy[i] = probability of choosing arm i
# starts uniform: agent has no preference
policy = [1/3, 1/3, 1/3]

learning_rate = 0.1
num_steps = 100

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

# ─── Final Result ─────────────────────────────────────────────────────────────
print("\n" + "="*50)
print("Final Policy:")
print(f"  arm 0 (prob=0.2): {policy[0]:.4f}")
print(f"  arm 1 (prob=0.5): {policy[1]:.4f}")
print(f"  arm 2 (prob=0.8): {policy[2]:.4f}")
print(f"\nBest arm chosen by policy: arm {policy.index(max(policy))}")
