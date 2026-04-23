# RLVR Math Reasoning — Project Spec & Implementation Plan

**Project title:** Hint-Free Difficulty-Adaptive Curriculum with Calibrated Abstention for Mathematical Reasoning via RLVR  
**Novel contribution:** Option A + B — adaptive curriculum through pure problem selection (no hints) + calibrated uncertainty across MATH difficulty tiers  
**Hardware:** NVIDIA RTX A4000 (16GB VRAM)  
**Base model:** `Qwen/Qwen2.5-Math-7B-Instruct`  
**Target benchmark:** MATH-500 Pass@1 + ECE (Expected Calibration Error) per difficulty level

---

## Research Claim (the one sentence you defend)

> A hint-free, rollout-accuracy-driven curriculum selector matches or exceeds hinted scaffolding (SEELE) on MATH-500 Pass@1, while a ternary reward structure produces significantly better-calibrated uncertainty estimates across difficulty levels 1–5, without any human trace collection.

---

## Repository Structure

```
rlvr-math/
├── AGENTS.md                  ← this file (living spec for the project)
├── configs/
│   └── grpo_a4000.yaml        ← all hyperparameters in one place
├── data/
│   ├── prepare_dataset.py     ← merge MATH + GSM8K, filter, stratify
│   └── difficulty_sampler.py  ← adaptive curriculum logic (core novelty B)
├── verifier/
│   ├── math_verifier.py       ← sympy-based verifier with timeout (core of everything)
│   └── test_verifier.py       ← verifier unit tests — run before any training
├── rewards/
│   ├── binary_reward.py       ← phase 2 reward
│   └── ternary_reward.py      ← phase 3 reward with warmup (core novelty A)
├── training/
│   ├── phase2_grpo.py         ← GRPO elicitation run
│   └── phase3_calibration.py  ← ternary reward calibration run
├── eval/
│   ├── eval_fast.py           ← Pass@1 greedy only — runs every 200 steps (~7 min)
│   └── eval_calibration.py    ← K=8 ECE eval — runs every 1000 steps (~53 min)
├── monitoring/
│   └── wandb_callbacks.py     ← entropy, rollout_acc, ECE logged every 50 steps
└── notebooks/
    └── results_analysis.ipynb ← plots for the paper
```

---

## VRAM Budget (RTX A4000 — 16GB)

Unsloth handles the reference model via adapter toggling on the same model object — no second full model copy in VRAM. With `gradient_checkpointing=True` (mandatory), the budget breaks down as:

| Component | Memory |
|---|---|
| 7B weights NF4 4-bit | ~3.5 GB |
| LoRA adapters (r=64, 7 modules) | ~0.3 GB |
| KV cache for G=4 × 512 token completions (Qwen2.5 GQA) | ~0.4 GB |
| Gradient + optimizer state (LoRA only) | ~1.2 GB |
| Activation memory with gradient checkpointing | ~3.5–5.0 GB |
| **Estimated total** | **~9–10.5 GB** |

The config values of G=4 and max_completion_length=512 are load-bearing — do not increase either without re-estimating VRAM. If OOM occurs, reduce in this order: (1) max_completion_length 512→256, (2) num_generations 4→2, (3) lora_r 64→32.

---

## AGENTS.md — Behavioral Contract

This section defines what each module must do, its interface, and its acceptance criteria. Treat each section as a mini-spec before writing any code.

---

### Agent 0 — Verifier (`verifier/math_verifier.py`)

**Purpose:** Given a model completion and a ground-truth answer string, return a float reward in {0.0, 1.0}.

**Critical requirement:** Must handle all answer equivalences:
- Fraction forms: `\frac{1}{2}` == `0.5` == `1/2`
- Radicals: `\sqrt{4}` == `2`
- Negative signs: `{-3}` == `-3`
- Implicit multiplication: `2x` == `2*x` in sympy
- Percentage: `50\%` == `0.5` (normalize both sides before comparison)

**Public interface — always call these from training and eval code:**
```python
def extract_boxed(text: str) -> str | None:
    """Extract content from \boxed{...}. Returns None if not found."""

def verify_with_timeout(completion: str, ground_truth: str,
                        timeout: float = 2.0) -> float:
    """
    Public API for all reward functions and eval code.
    Returns 1.0 if correct, 0.0 otherwise.
    Guaranteed to return within timeout seconds — never hangs.
    """

def is_abstention(completion: str) -> bool:
    """Returns True if the completion expresses explicit uncertainty and has no boxed answer."""
```

**Internal only — never call from training:**
```python
def _verify_internal(completion: str, ground_truth: str) -> float:
    """Raw sympy verification — may hang on pathological inputs."""
```

**Why timeout is mandatory:** Sympy's `simplify()` is computationally unbounded. A model generating `\boxed{e^{e^{e^{x}}}}` or a deeply nested malformed expression causes `simplify()` to hang indefinitely, freezing the entire training loop with no error or stacktrace. `ProcessPoolExecutor` is used (not `ThreadPoolExecutor` or `signal.alarm`) because only process isolation guarantees the sympy thread is actually killed on timeout.

**Acceptance criteria — all must pass before any GPU work:**
```python
assert verify_with_timeout(r"The answer is \boxed{\frac{1}{2}}", "0.5") == 1.0
assert verify_with_timeout(r"\boxed{-3}", "{-3}") == 1.0
assert verify_with_timeout(r"\boxed{\sqrt{4}}", "2") == 1.0
assert verify_with_timeout(r"\boxed{2x + 1}", "2x+1") == 1.0
assert verify_with_timeout(r"\boxed{0.333}", "1/3") == 1.0       # numeric fallback
assert verify_with_timeout(r"\boxed{50\%}", "0.5") == 1.0        # percentage normalization
assert verify_with_timeout(r"\boxed{25\%}", "1/4") == 1.0        # percentage → fraction
assert verify_with_timeout(r"I think it might be 5", "5") == 0.0 # no boxed = 0
assert verify_with_timeout(r"\boxed{6}", "5") == 0.0
assert verify_with_timeout(r"\boxed{}", "5") == 0.0              # empty boxed = 0
# Timeout test — must complete in under 3 seconds total
import time
t = time.time()
verify_with_timeout(r"\boxed{e^{e^{e^{e^{x}}}}}", "1")
assert time.time() - t < 3.0, "Timeout not working — check ProcessPoolExecutor"
```

**Full implementation:**
```python
import re
import concurrent.futures
import numpy as np
from sympy import simplify, N
from latex2sympy2 import latex2sympy

# --- Percentage normalization ---

def _normalize_percentage(s: str) -> str:
    """Convert '50\\%' or '50%' to '0.5'. No-op if not a percentage string."""
    match = re.match(r'^\\?(\d+(?:\.\d+)?)\\?%$', s.strip())
    if match:
        return str(float(match.group(1)) / 100)
    return s

# --- Boxed extractor ---

def extract_boxed(text: str) -> str | None:
    """Extract content from \\boxed{...}, handling nested braces."""
    match = re.search(r'\\boxed\{', text)
    if not match:
        return None
    start = match.end()
    depth = 1
    for i, ch in enumerate(text[start:], start):
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                content = text[start:i].strip()
                return content if content else None
    return None

# --- Abstention detection ---

IDK_PATTERNS = [
    r"i('m| am) not sure",
    r"i don'?t know",
    r"cannot (determine|solve|compute)",
    r"insufficient information",
    r"\\boxed\{\\text\{(IDK|unknown|unclear)\}\}",
    r"no (unique |definitive )?solution",
]

def is_abstention(completion: str) -> bool:
    """True if model expressed uncertainty AND gave no boxed answer."""
    if extract_boxed(completion) is not None:
        return False
    lower = completion.lower()
    return any(re.search(p, lower) for p in IDK_PATTERNS)

# --- Internal verifier (never call directly from training) ---

def _verify_internal(completion: str, ground_truth: str) -> float:
    pred = extract_boxed(completion)
    if pred is None:
        return 0.0

    pred = _normalize_percentage(pred)
    gt   = _normalize_percentage(ground_truth.strip())

    if pred == gt:
        return 1.0

    try:
        pred_expr  = latex2sympy(pred)
        truth_expr = latex2sympy(gt)
        if simplify(pred_expr - truth_expr) == 0:
            return 1.0
        pred_num  = complex(N(pred_expr))
        truth_num = complex(N(truth_expr))
        if abs(pred_num - truth_num) < 1e-5:
            return 1.0
    except Exception:
        pass

    return 0.0

# --- Public API: always use this in training and eval ---

def verify_with_timeout(completion: str, ground_truth: str,
                        timeout: float = 2.0) -> float:
    """
    Calls _verify_internal in a subprocess with a hard timeout.
    Returns 0.0 on timeout or any error. Never raises. Never hangs.
    ProcessPoolExecutor (not ThreadPoolExecutor) — only process isolation
    guarantees the sympy computation is actually killed on timeout.
    """
    with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_verify_internal, completion, ground_truth)
        try:
            return future.result(timeout=timeout)
        except (concurrent.futures.TimeoutError, Exception):
            return 0.0
```

**Performance note:** `ProcessPoolExecutor` has ~20–50ms overhead per call when sympy is actually invoked. The fast-path string match (`pred == gt`) returns in microseconds and never hits the subprocess. Batch your reward calls — reward functions already receive lists of completions, so the overhead is amortized over batch_size completions per subprocess spawn.

---

### Agent 1 — Dataset Preparation (`data/prepare_dataset.py`)

**Purpose:** Build the training corpus. MATH-500 is the eval set — never include it in training.

**Sources:**
- `lighteval/MATH` train split (12,499 problems, levels 1–5, 7 subjects)
- `gsm8k` main train split (7,473 problems, all mapped to level 1)

**Output format per sample:**
```python
{
    "problem": str,     # raw problem text
    "answer": str,      # ground truth answer string
    "level": int,       # 1–5 (GSM8K mapped to level 1)
    "subject": str,     # algebra, geometry, etc.
    "source": str,      # "math" or "gsm8k"
}
```

**Pre-filtering — scope and compute cost:**

Only pre-filter MATH Levels 4 and 5 (~4,000 problems). Levels 1–3 and all GSM8K have near-zero chance of zero-pass problems on Qwen2.5-Math-7B-Instruct. Filtering only L4/L5 with n=4 samples takes approximately 2 hours — run overnight before training starts. Cache the result to disk immediately.

```python
def prefilter_hard_problems(dataset_levels_4_5, model, tokenizer,
                             n_samples: int = 4) -> list:
    """
    Remove problems where the base model gets 0/4 correct.
    Only called on MATH levels 4 and 5.
    Run ONCE and cache result with dataset.save_to_disk().
    Runtime: ~2 hours on A4000. Expected retention: ~80% L4, ~60% L5.
    """
    keep = []
    for item in tqdm(dataset_levels_4_5, desc="Pre-filtering L4/L5"):
        completions = generate_n(model, tokenizer, item["problem"],
                                 n=n_samples, temperature=0.7, max_new_tokens=512)
        if any(verify_with_timeout(c, item["answer"]) == 1.0 for c in completions):
            keep.append(item)
    return keep

def prepare_dataset(cache_path: str = "./data/train_filtered.hf") -> dict:
    """Returns {level: [problems]}. Loads from cache if it exists."""
    import os
    if os.path.exists(cache_path):
        from datasets import load_from_disk
        return load_from_disk(cache_path)
    # ... build, prefilter, save — see prepare_dataset.py for full implementation
```

**One-time cost.** Run `prepare_dataset.py` once on Day 1. Never run again.

---

### Agent 2 — Difficulty Sampler (`data/difficulty_sampler.py`) ← NOVELTY B

**Purpose:** Dynamically select training problems based on the model's current rollout accuracy per difficulty tier. No hints. No IRT model. Pure selection.

**Core invariant:** At any training step, rollout accuracy per level should be between 0.30 and 0.70.

**Algorithm:**

```
State: rolling_accuracy = {level: deque(maxlen=100) for level in 1..5}
       Initialized to [0.5] * 20 for all levels (neutral start)

Every training step:
1. Compute p_correct[level] = mean of rolling_accuracy[level]
2. For each level, compute inclusion_weight:
   weight = max(0.05, 1.0 - |p_correct - 0.50| * 2.0)
   → peaks at 1.0 when p_correct = 0.50
   → falls to ~0.1 at p_correct = 0.05 or 0.95
   → floor of 0.05 ensures NO level is ever permanently excluded
3. Sample problems proportionally to weight × level_dataset_size
4. After each rollout, update rolling_accuracy[level]
```

**The 0.05 floor is non-negotiable.** Without it, a level that drops below 10% accuracy reaches weight=0.0, is never sampled again, and becomes permanently locked out. The floor keeps every level in rotation.

**Implementation:**
```python
from collections import deque
import numpy as np

class AdaptiveDifficultySampler:
    def __init__(self, dataset_by_level: dict, window: int = 100):
        self.data   = dataset_by_level
        self.levels = sorted(dataset_by_level.keys())
        self.acc    = {l: deque([0.5] * 20, maxlen=window) for l in self.levels}

    def _weight(self, level: int) -> float:
        p = float(np.mean(self.acc[level]))
        return max(0.05, 1.0 - abs(p - 0.5) * 2.0)

    def sample_batch(self, batch_size: int) -> list:
        raw_weights = np.array([
            self._weight(l) * len(self.data[l]) for l in self.levels
        ], dtype=float)
        probs = raw_weights / raw_weights.sum()

        batch = []
        for _ in range(batch_size):
            level = np.random.choice(self.levels, p=probs)
            item  = np.random.choice(self.data[level])
            batch.append({**item, "_sampled_level": level})
        return batch

    def update(self, level: int, correct: bool):
        self.acc[level].append(float(correct))

    def get_stats(self) -> dict:
        stats = {}
        for l in self.levels:
            p = float(np.mean(self.acc[l]))
            stats[f"curriculum/acc_level_{l}"]    = p
            stats[f"curriculum/weight_level_{l}"] = self._weight(l)
        return stats
```

**Paper figure:** `curriculum/weight_level_1..5` over training steps is Figure 2 — it visually demonstrates the curriculum responding to the model's capability growth.

---

### Agent 3 — Reward Functions

#### Phase 2: Binary Reward (`rewards/binary_reward.py`)

**Length penalty is delayed to step 300.** Before step 300, the model is still learning to use chain-of-thought. Penalizing length too early destroys CoT before it forms. The free threshold is 200 tokens — standard math CoT fits in 100–200 tokens; anything beyond is overthinking.

```python
from verifier.math_verifier import verify_with_timeout, extract_boxed

def binary_reward(completions: list[str], answers: list[str],
                  tokenizer, current_step: int = 0, **kwargs) -> list[float]:
    rewards = []
    for completion, answer in zip(completions, answers):
        score = verify_with_timeout(completion, answer)

        # Format bonus: incentivize \boxed{} usage
        if extract_boxed(completion) is not None:
            score += 0.1

        # Length penalty: delayed until step 300, free threshold 200 tokens
        if current_step >= 300:
            n_tokens = len(tokenizer.encode(completion))
            score -= 0.0008 * max(0, n_tokens - 200)

        rewards.append(score)
    return rewards
```

#### Phase 3: Ternary Reward with Warmup (`rewards/ternary_reward.py`) ← NOVELTY A

**Why warmup is required:** GRPO computes advantages as `(R - mean(group)) / std(group)`. An abrupt switch from binary to ternary reward shifts the reward distribution violently — a batch that previously had rewards `[1.0, 0.0, 0.0, 1.0]` (mean=0.5, std=0.5) might suddenly have `[-1.5, -1.5, 0.15, 1.0]` (mean=-0.71, std=1.07). The advantage spike causes a massive KL divergence jump that can collapse Phase 2's learned policy in a single update.

The fix: linearly warm up both penalty and abstention reward over 50 steps of Phase 3. At step 0 of Phase 3, the reward is effectively binary. By step 50, it's the full ternary system.

```python
import re
import functools
from verifier.math_verifier import verify_with_timeout, extract_boxed, is_abstention

def _ternary_reward_core(completions: list[str], answers: list[str],
                          tokenizer, current_step: int,
                          phase3_start_step: int,
                          warmup_steps: int = 50, **kwargs) -> list[float]:

    steps_into_phase3 = current_step - phase3_start_step
    alpha = min(1.0, steps_into_phase3 / warmup_steps)  # 0.0 → 1.0

    hallucination_penalty = -1.5 * alpha   # 0.0 → -1.5
    abstention_reward     =  0.15 * alpha  # 0.0 →  0.15

    rewards = []
    for completion, answer in zip(completions, answers):

        if is_abstention(completion):
            rewards.append(abstention_reward)
            continue

        score = verify_with_timeout(completion, answer)

        if score == 1.0:
            rewards.append(1.0)
        elif extract_boxed(completion) is not None:
            # Confident wrong answer — heavy asymmetric penalty
            rewards.append(hallucination_penalty)
        else:
            # No boxed, not an abstention — mild penalty, also warmed up
            rewards.append(-0.3 * alpha)

        # Length penalty always active in phase 3, only on non-correct answers
        if rewards[-1] != 1.0:
            n_tokens = len(tokenizer.encode(completion))
            rewards[-1] -= 0.0008 * max(0, n_tokens - 200)

    return rewards

def make_ternary_reward_fn(phase3_start_step: int, warmup_steps: int = 50):
    """
    Factory function. Bakes phase3_start_step into the reward function.
    Usage: reward_fn = make_ternary_reward_fn(phase3_start_step=1200)
    """
    return functools.partial(
        _ternary_reward_core,
        phase3_start_step=phase3_start_step,
        warmup_steps=warmup_steps,
    )
```

**Abstention prompt injection (Phase 3 only):** 30% of Phase 3 training batches use the uncertainty-permissive prompt. This trains the model to generalize abstention to both prompted and unprompted contexts.

```python
import random

def build_prompt(problem: str, allow_abstention: bool = False) -> str:
    if allow_abstention:
        instruction = (
            "Solve the following math problem step by step. "
            "If you are genuinely uncertain, you may respond with "
            "'I don't know' instead of guessing. "
            "Put your final answer in \\boxed{}.\n\n"
        )
    else:
        instruction = (
            "Solve the following math problem step by step. "
            "Put your final answer in \\boxed{}.\n\n"
        )
    return instruction + f"Problem: {problem}\n\nSolution:"

def sample_prompt(problem: str, phase: int) -> str:
    if phase == 3 and random.random() < 0.30:
        return build_prompt(problem, allow_abstention=True)
    return build_prompt(problem, allow_abstention=False)
```

---

### Agent 4 — Training Configuration (`configs/grpo_a4000.yaml`)

```yaml
model:
  name: "Qwen/Qwen2.5-Math-7B-Instruct"
  load_in_4bit: true
  max_seq_length: 1024
  gradient_checkpointing: true    # MANDATORY — do not disable

lora:
  r: 64
  lora_alpha: 64
  target_modules:
    - q_proj
    - k_proj
    - v_proj
    - o_proj
    - gate_proj
    - up_proj
    - down_proj
  lora_dropout: 0.0
  bias: "none"

grpo:
  num_generations: 4              # group size G — keep at 4 for 16GB
  max_completion_length: 512      # do not increase without re-checking VRAM
  learning_rate: 5.0e-6
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 4  # effective batch = 8
  kl_coeff: 0.01                  # increase to 0.05 if entropy collapse
  temperature: 0.9
  warmup_steps: 50
  max_grad_norm: 0.1
  save_steps: 250
  logging_steps: 10

phase3:
  learning_rate: 2.0e-6           # lower LR for fine calibration
  max_completion_length: 384      # force conciseness in phase 3
  reward_warmup_steps: 50         # ternary reward warmup duration
  additional_steps: 400           # steps beyond phase 2 best checkpoint

monitoring:
  report_to: "wandb"
  project: "rlvr-math-adaptive"
  fast_eval_every_steps: 200      # Pass@1 greedy only (~7 min)
  calibration_eval_every_steps: 1000  # K=8 ECE eval (~53 min)

output_dir: "./checkpoints"
```

---

### Agent 5 — Two-Tier Evaluation

**Why two tiers:** K=8 calibration eval on 500 problems = 4,000 generations ≈ 53 minutes. Running every 200 steps would consume ~37% of total compute on evaluation. The two-tier approach keeps eval overhead under 8% while capturing all data needed for the paper.

#### Fast eval (`eval/eval_fast.py`) — every 200 training steps

```python
from verifier.math_verifier import verify_with_timeout, is_abstention
from datasets import load_dataset
import numpy as np

def eval_pass1(model, tokenizer) -> dict:
    """
    Greedy decode on all 500 MATH-500 problems. ~7 minutes on A4000.
    Call every 200 training steps.
    """
    dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")
    results = {
        "overall": [],
        "by_level": {1:[], 2:[], 3:[], 4:[], 5:[]},
        "abstention_by_level": {1:[], 2:[], 3:[], 4:[], 5:[]},
    }

    model.eval()
    for item in dataset:
        level     = int(item["level"].split()[-1])
        prompt    = build_prompt(item["problem"], allow_abstention=False)
        completion = generate_one(model, tokenizer, prompt, temperature=0.0)
        correct   = verify_with_timeout(completion, item["answer"]) == 1.0
        abstained = is_abstention(completion)

        results["overall"].append(correct)
        results["by_level"][level].append(correct)
        results["abstention_by_level"][level].append(abstained)

    model.train()
    return {
        "eval/pass@1_overall": np.mean(results["overall"]),
        **{f"eval/pass@1_level_{l}": np.mean(v)
           for l, v in results["by_level"].items()},
        **{f"eval/abstention_rate_level_{l}": np.mean(v)
           for l, v in results["abstention_by_level"].items()},
    }
```

#### Calibration eval (`eval/eval_calibration.py`) — every 1000 training steps

```python
from collections import defaultdict
import numpy as np

def eval_calibration(model, tokenizer, K: int = 8, n_bins: int = 10) -> dict:
    """
    K=8 sampling on all 500 MATH-500 problems. ~53 minutes on A4000.
    Call only at major checkpoints (every 1000 steps, and on final models).
    Produces ECE per difficulty level — this is the core Novelty A metric.
    """
    dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")
    bins_by_level = defaultdict(lambda: [[] for _ in range(n_bins)])

    model.eval()
    for item in dataset:
        level      = int(item["level"].split()[-1])
        completions = generate_n(model, tokenizer, item["problem"],
                                  n=K, temperature=0.7)
        n_correct  = sum(verify_with_timeout(c, item["answer"]) == 1.0
                         for c in completions)
        confidence = n_correct / K       # proxy: fraction correct out of K
        accuracy   = float(n_correct > 0)  # Pass@K binary label

        bin_idx = min(int(confidence * n_bins), n_bins - 1)
        bins_by_level[level][bin_idx].append((confidence, accuracy))

    ece_by_level = {}
    for level, bins in bins_by_level.items():
        total = sum(len(b) for b in bins)
        ece = sum(
            (len(b) / total) * abs(
                np.mean([x[1] for x in b]) - np.mean([x[0] for x in b])
            )
            for b in bins if b
        )
        ece_by_level[f"eval/ece_level_{level}"] = ece

    model.train()
    return ece_by_level
```

---

## Experiment Tracking — WandB Dashboard

| Metric | Frequency | Source | Purpose |
|---|---|---|---|
| `train/reward_mean` | Every step | GRPO trainer | Overall training signal |
| `train/reward_std` | Every step | GRPO trainer | Must stay > 0 |
| `train/entropy` | Every step | GRPO trainer | Detect entropy collapse |
| `train/kl_divergence` | Every step | GRPO trainer | Detect policy drift |
| `curriculum/acc_level_1..5` | Every step | Difficulty sampler | Verify curriculum adapts |
| `curriculum/weight_level_1..5` | Every step | Difficulty sampler | Paper Figure 2 |
| `eval/pass@1_overall` | Every 200 steps | Fast eval | Primary training metric |
| `eval/pass@1_level_1..5` | Every 200 steps | Fast eval | Tier-stratified performance |
| `eval/abstention_rate_level_1..5` | Every 200 steps | Fast eval | Ternary reward validation |
| `eval/ece_level_1..5` | Every 1000 steps | Calibration eval | Core novelty A metric |

**WandB alert conditions:**

| Alert | Threshold | Action |
|---|---|---|
| Entropy collapse | `train/entropy < 0.5` | Stop run, increase `kl_coeff` 0.01→0.05 |
| Sampler deadlock | Any `curriculum/weight_level_X < 0.04` for 200 steps | Debug — floor not triggering |
| KL blow-up | `train/kl_divergence > 50` | Reduce LR; check phase 3 warmup alpha |
| **Abstention overuse** | `eval/abstention_rate_level_2 > 0.30` | Stop Phase 3 — abstention collapse, not calibration |
| NaN in loss | Any NaN | Add `eps=1e-8` in GRPO advantage denominator |

**The abstention overuse alert is the most important Phase 3 guard.** If the model learns to abstain on Level 1–2 problems (where it should be 70–90% accurate), the ternary reward has caused learned cowardice rather than calibration. The expected behavior is abstention rate monotonically increasing with difficulty level. Any violation of this ordering is a failure signal.

---

## Phase Timeline

### Week 1 — Foundation (Days 1–4)

- [ ] Set up repo structure exactly as specified
- [ ] `pip install unsloth trl datasets sympy latex2sympy2 wandb torch`
- [ ] Implement `verifier/math_verifier.py` — full timeout logic with `ProcessPoolExecutor`
- [ ] Run `test_verifier.py` — **every test including the timeout test must pass before touching the GPU**
- [ ] Run `prepare_dataset.py` overnight — prefilter L4/L5 only with n=4, cache to disk
- [ ] While prefilter runs: implement and unit-test `AdaptiveDifficultySampler` on CPU only
- [ ] Smoke test: load model, generate 10 completions, verify manually, confirm no OOM at config values
- [ ] Record base model baseline: run `eval_pass1()` on unmodified Qwen2.5-Math-7B-Instruct

### Week 2 — Phase 2 Training (Days 5–9)

- [ ] Run Phase 2 GRPO for 500 steps. Check WandB: `reward_mean` trending up, `entropy` stable
- [ ] **Run ablation in parallel (or sequentially):** identical config but uniform static sampling — save as `phase2_static/`. This is required for the Novelty B comparison.
- [ ] Continue adaptive run to 1200 steps, `eval_pass1()` every 200 steps automatically
- [ ] Save best Pass@1 checkpoint as `phase2_best/`; note the step number

### Week 3 — Phase 3 + Calibration (Days 10–14)

- [ ] Load `phase2_best/`, set `phase3_start_step` to its exact step number
- [ ] Verify warmup is working: first-batch rewards in Phase 3 should be close to binary reward values (alpha ≈ 0)
- [ ] Monitor `eval/abstention_rate_level_2` every eval — must stay below 0.30
- [ ] Run `eval_calibration()` on: base model, `phase2_best`, `phase3_final`
- [ ] Run adaptive vs static ablation comparison for Plot 1

### Week 4 — Analysis + Writing (Days 15–20)

Produce three plots in `results_analysis.ipynb`:

- **Plot 1:** Pass@1 over training steps — adaptive vs static curriculum. Two lines, X=steps, Y=MATH-500 Pass@1. Novelty B result.
- **Plot 2:** ECE per difficulty level — base model vs binary RLVR vs ternary RLVR. Grouped bars, X=Level 1–5, Y=ECE. Novelty A result.
- **Plot 3:** Abstention rate per difficulty level for ternary model. Should be monotonically increasing — this validates the calibration story.

Write in order: Method → Experiments → Results → Related Work → Introduction (last).

---

## Failure Modes and Fixes

| Symptom | Likely cause | Fix |
|---|---|---|
| CUDA OOM | Activation memory | Reduce `max_completion_length` 512→256 first; then `num_generations` 4→2 |
| `train/entropy < 0.5` | Entropy collapse | Increase `kl_coeff` 0.01→0.05 |
| `reward/std == 0` in batches | All correct or all wrong in group | Sampler not maintaining 30–70% target — debug `_weight()` |
| `nan` in advantages | `std(group_rewards) == 0` | Add `eps=1e-8` in GRPO advantage denominator |
| Pass@1 stagnant after 800 steps | LR wrong | Try 2e-6 (lower) or 1e-5 (higher) |
| Training loop hangs silently | Sympy timeout not working | Confirm `ProcessPoolExecutor` not `ThreadPoolExecutor` or `signal` |
| `abstention_rate_level_2 > 0.30` | Abstention collapse | Reduce `abstention_reward` to 0.05 or increase warmup_steps to 100 |
| Phase 3 KL spike at step 0 | Warmup not applying | Check `phase3_start_step` is set to correct checkpoint step number |
| CoT disappears, bare `\boxed{}` only | Length penalty too early/aggressive | Confirm `current_step >= 300` gate is active; coefficient at 0.0008 |

---

## Baselines Required for the Paper

Four numbers. All four are mandatory:

1. **Base model zero-shot:** Qwen2.5-Math-7B-Instruct, greedy decode, no training. `eval_pass1()` + `eval_calibration()`. Pass@1 per level + ECE per level.
2. **Binary RLVR + static curriculum:** Phase 2 with uniform sampling. Same steps, same config, no adaptive sampler. Isolates Novelty B.
3. **Binary RLVR + adaptive curriculum:** Phase 2 best checkpoint with adaptive sampler. Isolates curriculum contribution.
4. **Ternary RLVR + adaptive curriculum:** Phase 3 final. Full system.

**Reading the results:**
- (3) Pass@1 > (2) Pass@1 → adaptive curriculum works (Novelty B holds)
- (4) ECE < (3) ECE on levels 4–5 → ternary reward calibrates (Novelty A holds)
- (4) abstention rate monotonically increasing with level → calibration is genuine, not collapse

---

## Paper Abstract Draft

> We present an approach to training mathematical reasoning in LLMs via RLVR that eliminates two common dependencies: (1) hint scaffolding for curriculum design, and (2) binary reward functions that systematically incentivize overconfident hallucinations. Our difficulty-adaptive curriculum dynamically resamples training problems based on rolling rollout accuracy per difficulty tier, maintaining a 30–70% success rate without any annotated solution hints. Our ternary reward structure explicitly rewards correct answers, neutrally handles abstentions, and asymmetrically penalizes confident incorrect answers. On MATH-500, our full system achieves [X]% Pass@1 (+[Y]pp over the base model), while reducing Expected Calibration Error on Level 4–5 problems from [Z] to [W]. Ablations confirm that adaptive curriculum selection contributes [A]pp of the accuracy gain independently of reward shaping.

Fill brackets after running experiments.

---

## Key Papers to Cite

- DeepSeek-R1 (GRPO baseline and "aha moment" motivation)
- DAPO (asymmetric clipping — compare if compute allows)
- TruthRL (ternary reward motivation)
- SEELE (adaptive scaffolding — the related work your hint-free curriculum replaces)
- Tsinghua RLVR "search compression" (motivation for measuring calibration beyond accuracy)
- MATH dataset (Hendrycks et al. 2021)
- Unsloth / TRL (implementation)

---

## Change Log

| Version | Change |
|---|---|
| v1.0 | Initial spec |
| v1.1 | Added VRAM budget table with correct Unsloth adapter-toggle accounting |
| v1.1 | `gradient_checkpointing: true` added to config (was missing) |
| v1.1 | Pre-filter scope reduced to L4/L5 only with n=4 samples; ~2hr not ~13hr |
| v1.1 | Two-tier eval: `eval_fast.py` every 200 steps (~7 min), `eval_calibration.py` every 1000 steps (~53 min) |
| v1.1 | Sampler pseudocode corrected to match code — `max(0.05, ...)` floor documented explicitly |
| v1.1 | Verifier: `verify_with_timeout()` added as the only public API; `_verify_internal()` marked internal |
| v1.1 | Verifier: `_normalize_percentage()` added; percentage test cases added to acceptance criteria |
| v1.1 | Phase 3 reward: `alpha` warmup over 50 steps added to prevent KL spike on transition |
| v1.1 | Binary reward: length penalty delayed to step 300, coefficient 0.001→0.0008, free threshold 300→200 tokens |
| v1.1 | WandB: `abstention_rate_level_2 > 0.30` alert added for abstention collapse detection |

*Last updated: v1.1 — post-review fixes applied.*
