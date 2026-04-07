# rl-from-scratch-lab

Understand Reinforcement Learning by Implementing Every Step.

A minimal, learning-first Reinforcement Learning (RL) lab.

This project is NOT a framework.
This project is NOT for production.
This project is NOT optimized.

This project exists for ONE purpose:

> Understand RL deeply by implementing every algorithm step-by-step from scratch.

---

## 🎯 Project Goal

This repository is designed to help a single developer (or coding agent) understand RL algorithms by:

1. Implementing them from scratch (no abstraction, no magic)
2. Printing all intermediate variables
3. Keeping code small and readable (200~300 lines per file)
4. Mapping formulas directly to code

---

## 🧠 What This Project Is

* A sandbox for RL algorithms
* A place to inspect rollout, reward, logprob, KL, advantage
* A collection of minimal implementations

---

## ❌ What This Project Is NOT

* Not a reusable RL library
* Not a scalable system
* Not using distributed training (no Ray / no FSDP)
* Not focused on performance
* Not focused on UI

---

## 📁 Project Structure

rl_lab/
│
├── 00_toy_bandit/
├── 01_policy_gradient/
├── 02_ppo/
├── 03_dpo/
├── 04_grpo/
├── 05_gspo/
│
├── common/
│   ├── model.py
│   ├── utils.py
│   └── logger.py
│
└── notes/

---

## 🧪 Implementation Principles

Every algorithm MUST follow:

### 1. Minimal Code

* Each file < 300 lines
* No over-engineering

### 2. Full Transparency

Print all key variables:

* logprob (old / new)
* reward
* advantage
* ratio
* KL divergence
* loss

### 3. Formula → Code Mapping

Each key formula must appear explicitly in code.

Example (PPO):

ratio = exp(new_logprob - old_logprob)

### 4. No Hidden Magic

* No trainer frameworks
* No black-box APIs (except optional small models later)

---

## 🧭 Learning Roadmap

### Phase 1 — Toy RL (No LLM)

#### 00_toy_bandit

* Multi-armed bandit
* Learn exploration vs exploitation

#### 01_policy_gradient

* REINFORCE
* Understand:

  * logprob
  * reward signal
  * gradient direction

---

### Phase 2 — Core RL Algorithms

#### 02_ppo

Implement PPO from scratch:

Key components:

* old_logprob
* new_logprob
* ratio
* clipped objective
* advantage

Must print:

* ratio
* clipped ratio
* advantage
* final loss

---

### Phase 3 — Preference Learning

#### 03_dpo

Implement Direct Preference Optimization:

Input:

* (prompt, chosen, rejected)

Core:
loss = -log(sigmoid(beta * (logp_chosen - logp_rejected)))

Understand:

* Why no rollout
* Why no critic
* Why KL is implicit

---

### Phase 4 — Group-based RL

#### 04_grpo

Implement Group Relative Policy Optimization:

Steps:

1. Sample multiple responses per prompt
2. Compute reward for each
3. Compute group mean
4. Advantage = reward - mean

---

#### 05_gspo

Implement GSPO:

Core idea:

* Convert rewards into probabilities using softmax
* Each sample contributes to learning

---

## 🤖 Coding Agent Instructions

If you are a coding agent working on this repo:

### Your priorities:

1. Correctness over performance
2. Clarity over abstraction
3. Explicit over implicit

---

### When implementing a new algorithm:

You MUST:

* Keep code under 300 lines
* Print all intermediate variables
* Add comments explaining:

  * what each variable represents
  * how it relates to the formula

---

### DO NOT:

* Introduce complex class hierarchies
* Add config systems
* Add CLI frameworks
* Add unnecessary dependencies

---

### Example Workflow

For each algorithm:

1. Implement minimal version
2. Run with a tiny dataset
3. Print all intermediate values
4. Verify math manually

---

## 📌 Model Usage (Later Phase Only)

Only after understanding toy RL:

Use small models like:

* distilgpt2

DO NOT:

* use large models
* use GPU clusters
* optimize throughput

---

## 🧠 Notes System

Each algorithm MUST have a corresponding note file:

notes/ppo.md
notes/dpo.md
notes/grpo.md
notes/gspo.md

Each note must answer:

1. What problem does this algorithm solve?
2. What is the core formula?
3. What are the most confusing variables?

---

## 🚀 Final Goal

By completing this repo, you should be able to answer:

* What is rollout really doing?
* Why PPO needs clipping?
* Why DPO doesn't need a critic?
* Why GRPO removes value model?
* What GSPO improves?

---

## 🧩 Suggested First Task

Start with:

00_toy_bandit/bandit.py

Then move to:

01_policy_gradient/reinforce.py

---

## 🔥 Key Philosophy

This project is not about writing code.

It is about making RL impossible to misunderstand.

---
