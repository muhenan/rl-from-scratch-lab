import random
import math
import matplotlib.pyplot as plt

# ─── Problem Setup ────────────────────────────────────────────────────────────
arm_probs = [0.2, 0.5, 0.8]   # true reward probabilities (unknown to agent)

# ─── Policy ───────────────────────────────────────────────────────────────────
logits = [0.0, 0.0, 0.0]

learning_rate = 0.3
epsilon = 0.2   # PPO clipping range: ratio stays within [1-ε, 1+ε]
num_steps = 100
K = 5           # gradient update epochs per rollout — THIS is what makes ratio != 1
                # after epoch 1, logits change → new_logprob drifts from old_logprob
                # clip kicks in when drift exceeds epsilon

# ─── History (for plotting) ───────────────────────────────────────────────────
history = [[], [], []]

# ─── Softmax ──────────────────────────────────────────────────────────────────
def softmax(logits):
    max_l = max(logits)
    exps = [math.exp(l - max_l) for l in logits]
    total = sum(exps)
    return [e / total for e in exps]

# ─── Run Loop ─────────────────────────────────────────────────────────────────
for step in range(1, num_steps + 1):

    print(f"\n{'='*60}")
    print(f"Step {step}")

    # ── Phase 1: Collect ONE Rollout (old policy snapshot) ───────────────────
    # Sample action and reward using CURRENT policy.
    # Freeze old_logprob — this is the anchor for all K update epochs below.
    #
    # Key idea: old policy is fixed here and never changes within this step.
    # new policy will drift away from it across K epochs.

    old_probs = softmax(logits)
    print(f"  Old Probs:   [{old_probs[0]:.4f}, {old_probs[1]:.4f}, {old_probs[2]:.4f}]")

    r = random.random()
    if r < old_probs[0]:
        action = 0
    elif r < old_probs[0] + old_probs[1]:
        action = 1
    else:
        action = 2

    reward = 1 if random.random() < arm_probs[action] else 0
    advantage = reward   # no critic/baseline in this toy example

    old_logprob = math.log(old_probs[action])

    print(f"  Action:      {action}   Reward: {reward}   Advantage: {advantage}")
    print(f"  Old Logprob: {old_logprob:.4f}")

    # ── Phase 2: K Epochs of Gradient Updates on the Same Rollout ────────────
    # We reuse (action, reward, old_logprob) for K steps.
    # Each epoch updates logits → new_logprob drifts from old_logprob → ratio != 1
    # When ratio drifts beyond [1-ε, 1+ε], clip kicks in and limits the update.

    for epoch in range(1, K + 1):

        # recompute new policy from CURRENT (already partially updated) logits
        new_probs = softmax(logits)
        new_logprob = math.log(new_probs[action])

        # ratio = pi_new(action) / pi_old(action)
        # = 1.0  → policy hasn't moved from old
        # > 1.0  → new policy gives MORE prob to this action than old did
        # < 1.0  → new policy gives LESS prob to this action than old did
        ratio = math.exp(new_logprob - old_logprob)

        # clipped objective
        # surr1: raw policy gradient term
        # surr2: same but ratio is clamped — prevents too-large updates
        # objective = min(surr1, surr2) → always takes the conservative estimate
        clipped_ratio = max(1 - epsilon, min(1 + epsilon, ratio))
        surr1 = ratio * advantage
        surr2 = clipped_ratio * advantage
        objective = min(surr1, surr2)

        # gradient of log pi(action) w.r.t. logits ("理想 - 现实"):
        #   grad[action] = 1 - new_probs[action]
        #   grad[other]  = -new_probs[other]
        grad_logprob = [-new_probs[i] for i in range(3)]
        grad_logprob[action] += 1.0

        grad_logits = [advantage * clipped_ratio * grad_logprob[i] for i in range(3)]

        logits[0] += learning_rate * grad_logits[0]
        logits[1] += learning_rate * grad_logits[1]
        logits[2] += learning_rate * grad_logits[2]

        print(f"\n  -- Epoch {epoch} --")
        print(f"     New Logprob:   {new_logprob:.4f}")
        print(f"     Ratio:         {ratio:.4f}   (= exp(new_logprob - old_logprob))")
        print(f"     Clipped Ratio: {clipped_ratio:.4f}")
        print(f"     Surr1:         {surr1:.4f}   Surr2: {surr2:.4f}   Objective: {objective:.4f}")
        print(f"     Grad Logits:   [{grad_logits[0]:.4f}, {grad_logits[1]:.4f}, {grad_logits[2]:.4f}]")
        print(f"     Logits:        [{logits[0]:.4f}, {logits[1]:.4f}, {logits[2]:.4f}]")

    updated_probs = softmax(logits)
    print(f"\n  Updated Probs: [{updated_probs[0]:.4f}, {updated_probs[1]:.4f}, {updated_probs[2]:.4f}]")

    history[0].append(updated_probs[0])
    history[1].append(updated_probs[1])
    history[2].append(updated_probs[2])

# ─── Final Result ─────────────────────────────────────────────────────────────
final_probs = softmax(logits)
print("\n" + "="*60)
print("Final Logits:", [f"{l:.4f}" for l in logits])
print("Final Probs:")
print(f"  arm 0 (true p=0.2): {final_probs[0]:.4f}")
print(f"  arm 1 (true p=0.5): {final_probs[1]:.4f}")
print(f"  arm 2 (true p=0.8): {final_probs[2]:.4f}")
print(f"\nBest arm chosen by policy: arm {final_probs.index(max(final_probs))}")

# ─── Plot ─────────────────────────────────────────────────────────────────────
steps = list(range(1, num_steps + 1))
plt.plot(steps, history[0], label="arm 0 (true p=0.2)")
plt.plot(steps, history[1], label="arm 1 (true p=0.5)")
plt.plot(steps, history[2], label="arm 2 (true p=0.8)")
plt.xlabel("Step")
plt.ylabel("Policy Probability")
plt.title("PPO Bandit: Policy over Time (K=5 epochs per rollout)")
plt.legend()
plt.tight_layout()
plt.savefig("rl_lab/02_ppo/policy_plot.png", dpi=150)
print("\nPlot saved to rl_lab/02_ppo/policy_plot.png")
plt.show()
