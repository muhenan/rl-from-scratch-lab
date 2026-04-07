import random
import math
import matplotlib.pyplot as plt

# ─── Problem Setup ────────────────────────────────────────────────────────────
# Same bandit environment as 00_toy_bandit
arm_probs = [0.2, 0.5, 0.8]   # true reward probabilities (unknown to agent)

# ─── Policy: Parameterized by Logits ─────────────────────────────────────────
# Key upgrade from bandit.py:
#   bandit.py    → directly stored probs, updated probs manually
#   reinforce.py → store logits, derive probs via softmax, update via gradient
#
# Why logits instead of probs?
#   - probs must sum to 1 and stay in [0,1] — hard to update freely
#   - logits are unconstrained real numbers — gradient can move them anywhere
#   - softmax always converts logits → valid probs automatically
logits = [0.0, 0.0, 0.0]

learning_rate = 0.1
num_steps = 100

# ─── History (for plotting) ───────────────────────────────────────────────────
history = [[], [], []]   # history[i] = probs[i] at each step

# ─── Softmax ──────────────────────────────────────────────────────────────────
# probs[i] = exp(logits[i]) / S,  where S = sum of all exp(logits)
# subtract max for numerical stability (doesn't change the result)
def softmax(logits):
    max_l = max(logits)
    exps = [math.exp(l - max_l) for l in logits]
    total = sum(exps)
    return [e / total for e in exps]

# ─── Run Loop ─────────────────────────────────────────────────────────────────
for step in range(1, num_steps + 1):

    print(f"\nStep {step}")
    print(f"Logits:  [{logits[0]:.4f}, {logits[1]:.4f}, {logits[2]:.4f}]")

    # ── Compute Probabilities ────────────────────────────────────────────────
    probs = softmax(logits)
    print(f"Probs:   [{probs[0]:.4f}, {probs[1]:.4f}, {probs[2]:.4f}]")

    # ── Action Sampling ──────────────────────────────────────────────────────
    r = random.random()
    if r < probs[0]:
        action = 0
    elif r < probs[0] + probs[1]:
        action = 1
    else:
        action = 2
    print(f"Action:  {action}")

    # ── Log Probability ──────────────────────────────────────────────────────
    # logprob = log(probs[action])
    #
    # Why does logprob appear in RL at all?
    #
    # Our goal is to maximize expected reward:
    #   J = E[reward] = sum_a probs[a] * reward(a)
    #
    # We want dJ/d(logits). But we can only SAMPLE one action — we don't
    # know reward(a) for all a at once.
    #
    # The log-derivative trick solves this:
    #   d(probs[a]) / d(logits[i])
    #   = probs[a] * d(log probs[a]) / d(logits[i])
    #
    # So: dJ/d(logits[i]) = E[ reward * d(log probs[action]) / d(logits[i]) ]
    #
    # This is now an expectation → estimate it with one sample:
    #   ≈ reward * d(log probs[action]) / d(logits[i])
    #
    # logprob is the bridge: in PyTorch you'd write
    #   loss = -logprob * reward
    #   loss.backward()
    # and autograd computes d(logprob)/d(logits) automatically.
    # Here we compute that gradient by hand below.
    logprob = math.log(probs[action])
    print(f"Logprob: {logprob:.4f}")

    # ── Reward ───────────────────────────────────────────────────────────────
    reward = 1 if random.random() < arm_probs[action] else 0
    print(f"Reward:  {reward}")

    # ── Gradient: d(log probs[action]) / d(logits[i]) ───────────────────────
    #
    # Step 1 — expand log probs[action] using softmax definition:
    #   probs[a] = exp(logits[a]) / S,   S = sum_k exp(logits[k])
    #   log probs[action] = logits[action] - log(S)
    #
    # Step 2 — differentiate w.r.t. logits[i]:
    #   d(log probs[action]) / d(logits[i])
    #   = d(logits[action]) / d(logits[i])  -  d(log S) / d(logits[i])
    #
    # First term:
    #   i == action  →  1
    #   i != action  →  0
    #
    # Second term — chain rule on log(S):
    #   d(log S) / d(logits[i])
    #   = (1/S) * d(S) / d(logits[i])     ← d(log S)/dS = 1/S
    #   = (1/S) * exp(logits[i])           ← only the i-th term in S depends on logits[i]
    #   = probs[i]                         ← this is exactly the softmax definition
    #
    # Combine:
    #   i == action:  1 - probs[i]         (positive → push this logit up)
    #   i != action:  0 - probs[i]         (negative → push other logits down)
    grad_logits = [-probs[i] for i in range(3)]   # start with -probs[i] for all i
    grad_logits[action] += 1.0                     # add 1 to chosen action
    #
    # 记不住公式？用"理想 - 现实"来记：
    #   reward=1 时，你在说"这个 action 应该得到全部概率"
    #   理想分布 target = [0, 0, 1]（action=2 的情况）
    #   grad = target - probs
    #        = [0-probs[0], 0-probs[1], 1-probs[2]]
    #        = [-probs[0],  -probs[1],  1-probs[2]]   ← 和公式完全一致

    print(f"Grad:    [{grad_logits[0]:.4f}, {grad_logits[1]:.4f}, {grad_logits[2]:.4f}]")

    # ── REINFORCE Update ─────────────────────────────────────────────────────
    # Gradient ascent on expected reward:
    #   logits[i] += lr * reward * grad_logits[i]
    #
    # reward=1 → nudge logits so chosen action becomes more likely
    # reward=0 → no update (reward * grad = 0, nothing to learn from failure)
    logits[0] += learning_rate * reward * grad_logits[0]
    logits[1] += learning_rate * reward * grad_logits[1]
    logits[2] += learning_rate * reward * grad_logits[2]

    print(f"Updated Logits: [{logits[0]:.4f}, {logits[1]:.4f}, {logits[2]:.4f}]")

    # record probs AFTER update
    updated_probs = softmax(logits)
    history[0].append(updated_probs[0])
    history[1].append(updated_probs[1])
    history[2].append(updated_probs[2])

# ─── Final Result ─────────────────────────────────────────────────────────────
final_probs = softmax(logits)
print("\n" + "="*50)
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
plt.title("REINFORCE: Policy over Time (via logits + softmax)")
plt.legend()
plt.tight_layout()
plt.savefig("rl_lab/01_policy_gradient/policy_plot.png", dpi=150)
print("\nPlot saved to rl_lab/01_policy_gradient/policy_plot.png")
plt.show()
