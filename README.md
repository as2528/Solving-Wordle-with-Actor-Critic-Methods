# Wordle RL — PPO + GAE with Masked Softmax

Reinforcement Learning agent that solves **Wordle** using **Actor–Critic (PPO)** with **Generalized Advantage Estimation (GAE)** and a **hard action mask** (posterior-consistent words only). Includes entropy floor, KL target/penalty, trust-region brake, reward clipping, and richer logging.

* **Environment**: Wordle with 6-turn horizon
* **Policy**: MLP over a fixed-length state encoder (letters + feedback)
* **Masking**: Invalid guesses (contradicting known feedback) are zeroed out via masked softmax
* **Algorithm**: PPO (clipped) + GAE, optional KL penalty + adaptive LR controller

> This repo mirrors a Kaggle implementation, but is structured for GitHub.

---

## Features

* PPO + GAE with:

  * Clipped surrogate objective
  * Entropy bonus (annealed to a floor)
  * KL target + adaptive LR + optional KL penalty
  * TRPO-style “brake” when minibatch KL spikes
* **Hard action mask** for Wordle consistency
* **Information gain** via posterior-entropy shaping
* Train/val/test **split by target words** (not by action vocabulary)
* **Greedy evaluation** with binomial **95% CI**
* **Visualize episodes** with emoji feedback

---

**CUDA**: If you have an Ampere+ GPU, PyTorch may use TF32 (via `torch.set_float32_matmul_precision('high')`) for faster matmuls. No action needed.

---

What it does:

* Downloads/caches **allowed** and **solutions** word lists into `.wordlists/`
* Splits solutions into **train/val/test** (80/10/10)
* Trains PPO with safer defaults (see `PPOConfig`)
* Saves the **best checkpoint** to `ppo_wordle_best.pt`
* Runs a **final test** and prints SR ± CI
* Optionally **visualizes** a handful of held-out targets


---

## Key Components

### Word lists

* **Allowed** (actions): [tabatkins/wordle-list](https://github.com/tabatkins/wordle-list)
* **Solutions** (targets): cfreshman’s Wordle answers gist

Loader behavior:

1. Try cache (`.wordlists/allowed.txt`, `.wordlists/solutions.txt`)
2. If missing, **download** them and cache
3. If download fails, **fallback** to `wordfreq` or `nltk` (quality differs)

### State encoder

Encodes up to `history_len` steps (letters one-hot: `5×26`, feedback one-hot: `5×3`) → `145 × history_len` vector.

### Masked softmax

Invalid actions (posterior-inconsistent words) get `-1e9` logits; prob mass is renormalized over valid actions. We also log **pre-mask invalid mass** for diagnostics.

### Reward shaping (high level)

* Per-step cost
* Diversity bonus (unique letters)
* Repeat penalty
* Preserve-green **quadratic** bonus
* Reuse-previous-yellow bonus
* Entropy shaping: `λ * (H_prev − H_now)` from `entropy_start_turn`
* Early-win time-shaped bonus
* Timeout penalty

---

## Configuration

All knobs live in `PPOConfig`:

* **Learning rates**: `actor_lr`, `critic_lr`, `weight_decay`
* **Exploration**: `entropy_coef_start/end`, `temperature`
* **Batching**: `episodes_per_batch`, `ppo_epochs`, `minibatch_size`
* **Credit**: `gamma`, `gae_lambda`
* **Trust region**: `clip_eps`, `max_grad_norm`
* **KL control**: `kl_target`, `kl_adapt`, `kl_coef_*`, `trpo_brake`
* **Reward clipping**: `reward_clip_low/high`
* **Eval/Early stop**: `eval_every`, `early_stop_patience`, etc.

Reward shaping constants are in `RewardCoeffs`.

---

## Logging

* Console logs:

  * KL vs target + LR adjustments
  * Entropy, policy/value loss
  * Pre-mask invalid mass (diagnostic)
  * Eval SR ± CI and avg turns

---

## Citations / Credits

* Word lists:

  * Allowed: **tabatkins** — [https://github.com/tabatkins/wordle-list](https://github.com/tabatkins/wordle-list)
  * Solutions: **cfreshman** — (Wordle answers gist)
* PPO/GAE: Schulman et al., 2016–2017

---

## Notes

* **Repro**: Set `DETERMINISTIC=1` to reduce kernel nondeterminism. RL is still noisy; SR will vary slightly.
* **Hardware**: Works on CPU; GPU recommended for faster training.
* **History length**: If you switch encoder to 6, update model input dims accordingly.

---

## License

MIT (see `LICENSE`).
