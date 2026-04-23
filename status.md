# RLVR Math Project — Status & Implementation Log

## Phase 0: Environment Setup ✓

**Goal**: Configure the project environment and verify dependencies

The repo is now set up and operational:
- Network-restricted environment required manual permission for installs
- Created working `.venv` with helper scripts
- Bootstrap: `source .venv/bin/activate` and `.venv/bin/pip` both work

**Installed & Importable**:
- datasets, wandb, latex2sympy2, trl, transformers, bitsandbytes, xformers
- unsloth (with compatibility work in progress)

---

## Phase 1: Core Implementation & Testing ✓

**Goal**: Implement core RLVR components and verify with unit tests

### Modules Implemented

Verifier with boxed extraction, percentage normalization, abstention detection, symbolic/numeric checking, and hard timeout isolation in verifier/math_verifier.py (line 1)
Adaptive curriculum sampler in data/difficulty_sampler.py (line 1)
Dataset normalization/cache pipeline for lighteval/MATH and gsm8k in data/prepare_dataset.py (line 1)
Reward/config scaffolding in rewards/ternary_reward.py (line 1), rewards/binary_reward.py (line 1), and configs/grpo_a4000.yaml (line 1)
I also added runnable tests, and they pass inside the venv:

verifier/test_verifier.py (line 1)
data/test_difficulty_sampler.py (line 1)
data/test_prepare_dataset.py (line 1)
Verification:

.venv/bin/python -m unittest verifier.test_verifier data.test_difficulty_sampler data.test_prepare_dataset -> 13 tests passed
Installed/importable in the venv: datasets, wandb, latex2sympy2, trl, transformers, bitsandbytes, xformers
unsloth is installed, but importing it in this sandbox reports no visible supported GPU; torch.cuda.is_available() is False and nvidia-smi can’t see the driver here, so I did not start a GPU training smoke test yet
The next best step is wiring the actual phase2_grpo.py training entry point around Unsloth/TRL and doing a model-load smoke test once we have GPU visibility.
That helps a lot: the GPU and driver are healthy on the host, so the earlier unsloth failure was a sandbox visibility issue rather than a broken machine. I’m going to re-run the GPU checks with escalated access and then push the training-side implementation forward.


Ran 3 commands
The host GPU path is opening up now. I’m waiting on the Python-side checks so I can confirm torch and unsloth see the A4000 before I wire the next training step.

torch now sees the A4000 correctly, so the remaining issue is purely a package mismatch: the installed unsloth expects a newer torch._inductor API than torch 2.4.1 provides. I’m checking the installed metadata and then I’ll either pin a compatible unsloth pair or keep moving with a clean fallback training path.


Ran 3 commands
I confirmed the mismatch: unsloth 2025.11.1 is installed on top of torch 2.4.1, and the crash comes from unsloth_zoo assuming a newer torch._inductor.config API. I’m checking the installed code path so I can choose the safest fix instead of guessing at versions.


Explored 2 files, 1 search
I found the exact crash site in unsloth_zoo: it assumes torch._inductor.config is always present at import time. I’m checking whether a small compatibility patch will unblock import cleanly on torch 2.4.1, which is safer than churning the whole stack again.


Explored 1 file, ran 1 command
I found a cleaner path than downgrading half the stack: we can shim the missing torch._inductor.config API before importing unsloth. I’m adding that compatibility layer into the repo and then I’ll test a real unsloth import against the GPU.


Created 1 file
The shim is in place. I’m testing it directly with the host GPU visible, and if it holds I’ll thread it into the training entry points so imports are stable from the start.

Ran python3 -m unittest verifier.test_verifier data.test_difficulty_sampler data.test_prepare_dataset
1 file changed
Undo

Review
training/runtime_compat.py
continue
1:04 PM

I’m picking up from the last checkpoint by fixing the verifier file first, since that indentation error is blocking the basic test loop. After that I’ll re-run the local tests and continue the Unsloth compatibility path.


Explored 2 files, ran 1 command
I found the blocker: verifier/math_verifier.py has a stray leading space before the opening docstring, so Python treats line 1 as unexpectedly indented. I’m fixing that and cleaning the tiny unused helper in the runtime shim at the same time.


Edited 2 files
The syntax blocker is fixed. I’m rerunning the core test suite now, then I’ll retry the Unsloth import path through the compatibility shim.


Ran 2 commands
The local codebase is healthy again. I’m retrying the host-GPU Unsloth import with the shim in place; if that lands, I’ll wire it into the training entry points immediately.

Ran .venv/bin/python - <<'PY' from training.runtime_compat import ensure_torch_inductor_config_compat ensure_torch_inductor_config_compat() import torch print('cuda_available', torch.cuda.is_available()) print('device_name', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'n/a') import unsloth print('unsloth import ok') PY
The shim is close, but unsloth_zoo is using inspect.getsource() on the config module, so it wants a real source-backed module rather than an in-memory one. I’m swapping the shim to a file-backed module, which should satisfy that import path cleanly.

I’ll stay anchored to RLVR_MATH_PROJECT.md. The only non-spec addition I’m carrying is a small local runtime compatibility shim for the installed torch/unsloth mismatch; it does not change the RLVR method, rewards, sampler, verifier behavior, or training design.


Ran 3 commands
The current spec-critical CPU pieces still pass: 13 tests green and the edited files compile. Next I’m going to test the Unsloth import shim without changing files; then I’ll tell you before touching the training/eval entry points.

Ran .venv/bin/python - <<'PY' from training.runtime_compat import ensure_torch_inductor_config_compat ensure_torch_inductor_config_compat() import torch print('cuda_available', torch.cuda.is_available()) print('device_name', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'n/a') import unsloth print('unsloth import ok') PY
The file-backed shim advanced the import, and the next missing compatibility symbol is is_fbcode from torch._inductor.config. This is still environment glue rather than an RLVR logic change; I’m going to add that missing symbol to the shim and retry.


Explored 1 file, 1 search
Better find: this Torch install actually has a real torch._inductor.config.py; it just is not attached to torch._inductor until imported. I’m updating the compatibility helper to prefer the real Torch config module and only use our shim as a fallback, which is closer to the actual installed stack and less invasive.


Edited 1 file
The compatibility helper now uses the real Torch config module when available. I’m running the tests again and retrying Unsloth import with GPU access.


Ran 2 commands
The RLVR spec itself is still intact; the blocker is the local training stack. The current unsloth import is now failing because torchao 0.17.0 expects newer Torch dtypes like torch.int1, which torch 2.4.1 does not have.


Explored 2 files, ran 1 command
Before I change anything: I’m going to make one environment-only adjustment, not an RLVR algorithm change. I want to remove torchao from the venv and retry unsloth, because the currently installed torchao 0.17.0 is explicitly incompatible with torch 2.4.1 and is blocking imports before we even reach the training code.

Ran python3 -m py_compile training/runtime_compat.py training/_torch_inductor_config_shim.py
Ran .venv/bin/python -m pip uninstall -y torchao
1 file changed
Undo

Review
training/runtime_compat.py
pick up whre you left off
1:18 PM


Worked for 11m 19s
I stayed close to RLVR_MATH_PROJECT.md and only made one non-spec change: a local Unsloth/Torch compatibility helper so the installed stack works on your A4000. After that, I kept the implementation aligned with the spec.

What’s now in place:

Verifier, sampler, dataset normalization, and rewards are working and tested.
Unsloth now imports successfully on the host GPU.
Phase 2 and Phase 3 training entry points are implemented in training/phase2_grpo.py (line 1) and training/phase3_calibration.py (line 1).
Shared training wiring for config loading, Unsloth model setup, adaptive iterable dataset, and reward wrappers is in training/common.py (line 1).
Fast eval and calibration eval are implemented in eval/eval_fast.py (line 1) and eval/eval_calibration.py (line 1).
Verification:

13 unit tests pass.
New training/eval modules compile.
Imports work inside .venv.
unsloth imports correctly with host GPU access.
The next spec-aligned step is to run the first real smoke path:

prepare_dataset.py against HF datasets.
A model load + one-generation smoke test.
Then the first short Phase 2 GRPO run.
Before I make that next change or run, I’ll keep telling you exactly what I’m about to modify.

Three things worth fixing before you run smoke tests
1. training/common.py — the sampler update logic is wrong
Look at make_phase2_reward:
pythonfor level, reward in zip(levels, rewards):
    sampler.update(int(level), reward >= 1.0)
The condition reward >= 1.0 is checking the reward value, not whether the answer was correct. In Phase 2, a correct answer with the format bonus scores 1.1, which passes. But a correct verbose answer might score 0.9 after the length penalty, which would tell the sampler "wrong" even though the model got it right. The sampler needs to know correctness, not reward magnitude.
Fix:
pythonfrom verifier.math_verifier import verify_with_timeout

for completion, answer, level in zip(completions, answers, levels):
    correct = verify_with_timeout(completion, answer) == 1.0
    sampler.update(int(level), correct)
You need to pass completions into the reward function closure — it already has access to them since they're the first argument.
2. training/common.py — levels extraction will silently return an empty list
pythonlevels = kwargs.get("_sampled_level") or kwargs.get("level") or []
TRL's GRPOTrainer passes batch fields as keyword arguments to the reward function. The key for sampled level will be _sampled_level if you set remove_unused_columns=False, but the actual key name depends on how TRL unpacks the dataset. If neither key matches, levels is [], the zip produces nothing, and the sampler never updates — silently. This would make the adaptive curriculum completely inoperative with no error.
Add a WandB log warning when levels is empty, or better, assert it during the smoke test:
pythonlevels = kwargs.get("_sampled_level") or kwargs.get("level") or []
if not levels:
    import warnings
    warnings.warn("No level data found in reward kwargs — sampler not updating. Check TRL batch key names.")
Then in your smoke test, verify that curriculum/acc_level_X metrics are actually changing in WandB after 10 steps.
3. eval/eval_calibration.py imports build_prompt from rewards/ternary_reward.py
pythonfrom rewards.ternary_reward import build_prompt
eval_fast.py does the same. This creates a dependency from eval on the ternary reward module — eval runs during Phase 2 training where ternary reward isn't loaded yet. It works because build_prompt is a plain function with no state, but it's architecturally messy. When you run Phase 2 fast eval, you're technically importing Phase 3 code.
Minor, but move build_prompt and sample_prompt to a shared utils/prompts.py module and import from there in both the reward and eval files.

The three smoke tests to run in order
Smoke test 1 — dataset only (no GPU):
bashpython -c "
from data.prepare_dataset import load_and_normalize_datasets, group_by_level
items = load_and_normalize_datasets()
grouped = group_by_level(items)
for level, problems in sorted(grouped.items()):
    print(f'Level {level}: {len(problems)} problems')
"
Expected: levels 1–5 with roughly 7500+, 2000, 2000, 2000, 2000 problems.
Smoke test 2 — model load + verifier (no training):
bashpython -c "
from training.common import load_config, load_model_and_tokenizer
from eval.eval_fast import generate_one
from verifier.math_verifier import verify_with_timeout
from rewards.ternary_reward import build_prompt

cfg = load_config('configs/grpo_a4000.yaml')
model, tokenizer = load_model_and_tokenizer(cfg)
prompt = build_prompt('What is 2 + 2?')
completion = generate_one(model, tokenizer, prompt, temperature=0.0)
print('Completion:', completion[:200])
print('Reward:', verify_with_timeout(completion, '4'))
"
Expected: model loads under 11GB VRAM, completion contains \boxed{4}, reward is 1.0.
Smoke test 3 — 20 GRPO steps:
bashpython training/phase2_grpo.py --max-steps 20 --config configs/grpo_a4000.yaml
Expected: no OOM, WandB shows curriculum/acc_level_X metrics updating, reward_mean is non-zero. If curriculum/acc_level_X is all zeros or missing, the levels extraction bug (item 2 above) is active.

I’m going to make exactly the three fixes you called out before I run the smoke tests: correct sampler updates in training/common.py, make missing level data explicit instead of silent, and move prompt helpers into a shared module so eval no longer imports Phase 3 code.


Explored 4 files
I’m making the three requested changes now: moving prompt helpers into utils/prompts.py, fixing sampler updates to use verifier correctness instead of reward magnitude, and warning loudly when level metadata is missing so the curriculum can’t silently stall.


Created 3 files, edited 4 files
I caught a test issue in the new training/test_common.py: the assertion shape was wrong for the “correct but penalized” case. I’m tightening that test so it proves the sampler uses verifier correctness even when the reward value drops below 1.0.


Edited 1 file
I’ve finished the three fixes and added a focused unit test for the sampler update path. I’m running the smoke tests in the order you asked, starting with dataset-only, then model load + verifier, and only then the 20-step GRPO run.


Ran 2 commands
Reconnecting... 1/5
Ran HF_HOME=/tmp/hf_home .venv/bin/python - <<'PY' from data.prepare_dataset import load_and_normalize_datasets, group_by_level items = load_and_normalize_datasets() grouped = group_by_level(items) for level, problems in sorted(grouped.items()): print(f'Level {level}: {len(problems)} problems') PY
The unit tests are green again. Smoke test 1 is downloading and normalizing the datasets now, so I’m waiting on the real level counts before moving to the model load.

Smoke test 1 exposed a spec drift rather than a code bug: lighteval/MATH is not accessible from the Hub as written. I’m not changing the code yet; I’m first checking the currently available dataset IDs so I can tell you exactly what would need to be updated if we want the smoke test to run end-to-end.


Ran 2 commands
I found the current Hub situation. HuggingFaceH4/MATH-500 is fine, and gsm8k works with the "main" config, but the training-source dataset ID in the spec is stale. I’m checking which accessible MATH dataset matches the expected schema best before I change the loader.

Ran HF_HOME=/tmp/hf_home .venv/bin/python - <<'PY' from datasets import load_dataset candidates = [ ('DigitalLearningGmbH/MATH-lighteval', 'train'), ('xDAN2099/lighteval-MATH', 'train'), ('verl-team/lighteval-MATH-preprocessed', 'train'), ('hendrycks/competition_math', 'train'), ] for path, split in candidates: try: ds = load_dataset(path, split=f'{split}[:2]') print('\n===', path, '===') print(ds.column_names) print(ds[0]) except Exception as e: print('\n===', path, 'FAIL ===') print(repr(e)) PY
$ HF_HOME=/tmp/hf_home .venv/bin/python - <<'PY'
from datasets import load_dataset
candidates = [
    ('DigitalLearningGmbH/MATH-lighteval', 'train'),
    ('xDAN2099/lighteval-MATH', 'train'),
    ('verl-team/lighteval-MATH-preprocessed', 'train'),
    ('hendrycks/competition_math', 'train'),
]
for path, split in candidates:
    try:
        ds = load_dataset(path, split=f'{split}[:2]')
        print('\n===', path, '===')
        print(ds.column_names)
        print(ds[0])
    except Exception as e:
        print('\n===', path, 'FAIL ===')
        print(repr(e))
PY

Generating train split: 100%|██████████| 7500/7500 [00:00<00:00, 101764.63 examples/s]
Generating test split: 100%|██████████| 5000/5000 [00:00<00:00, 124295.56 examples/s]

=== DigitalLearningGmbH/MATH-lighteval ===
['problem', 'level', 'solution', 'type']
{'problem': 'Let \\[f(x) = \\left\\{\n\\begin{array}{cl} ax+3, &\\text{ if }x>2, \\\\\nx-5 &\\text{ if } -2 \\le x \\le 2, \\\\\n2x-b &\\text{ if } x <-2.\n\\end{array}\n\\right.\\]Find $a+b$ if the piecewise function is continuous (which means that its graph can be drawn without lifting your pencil from the paper).', 'level': 'Level 5', 'solution': 'For the piecewise function to be continuous, the cases must "meet" at $2$ and $-2$. For example, $ax+3$ and $x-5$ must be equal when $x=2$. This implies $a(2)+3=2-5$, which we solve to get $2a=-6 \\Rightarrow a=-3$. Similarly, $x-5$ and $2x-b$ must be equal when $x=-2$. Substituting, we get $-2-5=2(-2)-b$, which implies $b=3$. So $a+b=-3+3=\\boxed{0}$.', 'type': 'Algebra'}
Generating train split: 100%|██████████| 7500/7500 [00:00<00:00, 115244.17 examples/s]
Generating test split: 100%|██████████| 5000/5000 [00:00<00:00, 131651.27 examples/s]

=== xDAN2099/lighteval-MATH ===
['problem', 'level', 'type', 'solution']
{'problem': 'Let \\[f(x) = \\left\\{\n\\begin{array}{cl} ax+3, &\\text{ if }x>2, \\\\\nx-5 &\\text{ if } -2 \\le x \\le 2, \\\\\n2x-b &\\text{ if } x <-2.\n\\end{array}\n\\right.\\]Find $a+b$ if the piecewise function is continuous (which means that its graph can be drawn without lifting your pencil from the paper).', 'level': 'Level 5', 'type': 'Algebra', 'solution': 'For the piecewise function to be continuous, the cases must "meet" at $2$ and $-2$. For example, $ax+3$ and $x-5$ must be equal when $x=2$. This implies $a(2)+3=2-5$, which we solve to get $2a=-6 \\Rightarrow a=-3$. Similarly, $x-5$ and $2x-b$ must be equal when $x=-2$. Substituting, we get $-2-5=2(-2)-b$, which implies $b=3$. So $a+b=-3+3=\\boxed{0}$.'}
Generating train split: 100%|██████████| 7500/7500 [00:00<00:00, 210981.09 examples/s]
Generating test split: 100%|██████████| 5000/5000 [00:00<00:00, 205407.80 examples/s]

=== verl-team/lighteval-MATH-preprocessed ===
['level', 'type', 'data_source', 'prompt', 'ability', 'reward_model', 'extra_info']
{'level': 'Level 5', 'type': 'Algebra', 'data_source': 'DigitalLearningGmbH/MATH-lighteval', 'prompt': [{'content': "Let \\[f(x) = \\left\\{\n\\begin{array}{cl} ax+3, &\\text{ if }x>2, \\\\\nx-5 &\\text{ if } -2 \\le x \\le 2, \\\\\n2x-b &\\text{ if } x <-2.\n\\end{array}\n\\right.\\]Find $a+b$ if the piecewise function is continuous (which means that its graph can be drawn without lifting your pencil from the paper). Let's think step by step and output the final answer within \\boxed{}.", 'role': 'user'}], 'ability': 'math', 'reward_model': {'ground_truth': '0', 'style': 'rule'}, 'extra_info': {'index': 0, 'split': 'train'}}

=== hendrycks/competition_math FAIL ===
DatasetNotFoundError("Dataset 'hendrycks/competition_math' doesn't exist on the Hub or cannot be accessed.")

Success
I found the least-invasive way to keep us aligned with the spec: I’m going to keep lighteval/MATH as the first attempted source in prepare_dataset.py, but add a fallback to DigitalLearningGmbH/MATH-lighteval, which is currently accessible and has the same problem/level/solution/type schema.


Edited 1 file
The dataset loader now preserves the spec path first and only falls back when the Hub no longer serves that ID. I’m rerunning smoke test 1 with that change and then I’ll move directly to the model-load smoke test.


Ran 2 commands
Smoke test 1 got past dataset IDs and found a real normalization edge case: some MATH solutions in the accessible dataset don’t contain a boxed answer, so _extract_math_answer is too strict for the actual source. I’m inspecting the first failing examples before I change extraction logic.

Ran HF_HOME=/tmp/hf_home .venv/bin/python - <<'PY' from datasets import load_dataset from data.prepare_dataset import _extract_math_answer for dataset_name in ['DigitalLearningGmbH/MATH-lighteval']: ds = load_dataset(dataset_name, split='train') for idx, item in enumerate(ds): try: _extract_math_answer(item) except Exception: print('dataset', dataset_name, 'idx', idx) print(item) break PY
I’m keeping the verifier spec unchanged here. The dataset issue is specifically in answer extraction, so I’m only relaxing _extract_math_answer to handle MATH solutions that use the common shorthand \boxed 2 instead of \boxed{2}.


Edited 1 file
The dataset extractor now handles the shorthand boxed form without changing verifier behavior. I’m rerunning smoke test 1 and, if that clears, I’ll go straight into the model-load smoke test.


Ran 2 commands
Smoke test 1 found one more real-world dataset quirk: some rows have level == "Level ?" in the accessible MATH dataset. I’m inspecting how common that is before deciding whether to drop or remap those rows, since that would be a small data-policy change relative to the spec.

Ran HF_HOME=/tmp/hf_home .venv/bin/python - <<'PY' from collections import Counter from datasets import load_dataset ds = load_dataset('DigitalLearningGmbH/MATH-lighteval', split='train') counts = Counter(item['level'] for item in ds) print(counts) examples = [item for item in ds if item['level'] == 'Level ?'] print('unknown_count', len(examples)) for item in examples[:3]: print(item) PY
The Level ? rows are only 2 out of 7500 MATH training examples, so the cleanest spec-aligned move is to skip them with a warning rather than invent a difficulty mapping. I’m making that small data hygiene change now.


Edited 1 file
The loader now skips only the malformed MATH rows and warns about it. I’m rerunning smoke test 1 again; if it clears, I’ll immediately run the model-load smoke test.


Ran 2 commands
Smoke test 1 now passes. The counts are a bit different from the rough spec estimate because the accessible MATH mirror has 564 Level 1 problems and a couple malformed rows we skip, but the 1–5 grouping is there and usable. I’m moving to smoke test 2 now: model load, one generation, and verifier check.

Ran HF_HOME=/tmp/hf_home .venv/bin/python - <<'PY' from training.common import load_config, load_model_and_tokenizer from eval.eval_fast import generate_one from verifier.math_verifier import verify_with_timeout from utils.prompts import build_prompt cfg = load_config('configs/grpo_a4000.yaml') model, tokenizer = load_model_and_tokenizer(cfg) prompt = build_prompt('What is 2 + 2?') completion = generate_one(model, tokenizer, prompt, temperature=0.0) print('Completion:', completion[:200]) print('Reward:', verify_with_timeout(completion, '4')) PY
The model smoke test is in the heavy part now, which is likely the initial model/tokenizer download and load. I’m letting it settle and will report the exact completion and verifier result once it finishes.

The load is taking longer than a normal warm start, so this looks like a first-time model fetch rather than a Python-side hang. I’m giving it a bit more time before I decide whether to probe the cache or split the smoke test.

The model-load smoke test passed: the model loaded on the A4000, generated a correct answer path, and the verifier returned 1.0. I’m running the 20-step Phase 2 GRPO smoke test now without changing code first, so we can see the next real runtime failure if there is one.

Ran HF_HOME=/tmp/hf_home .venv/bin/python training/phase2_grpo.py --max-steps 20 --config configs/grpo_a4000.yaml
The 20-step smoke test exposed a script-entry issue, not a GRPO issue: running python training/phase2_grpo.py from the repo root doesn’t put the repo root on sys.path, so from training.common import ... fails. I’m going to add a tiny path bootstrap to the training entry scripts so the exact smoke test command in the spec works as written.


Edited 2 files
The entry-point bootstrap is in. I’m rerunning the exact 20-step command now so we can get to the actual trainer/runtime behavior.


Ran 2 commands
The run is past dataset prep and into model initialization. One useful thing surfaced already: transformers is getting imported before unsloth in the training entry path, so I’m watching for whether that stays just a warning or becomes a performance/compatibility issue in the trainer boot.

The 20-step run reached the real trainer and exposed two environment/runtime mismatches with this installed Unsloth version:

It rejects fp16=True because the loaded model is running in bfloat16 on the A4000, so the training args need to use bf16=True and fp16=False.
This Unsloth build expects per_device_train_batch_size to be a multiple of num_generations, so the spec’s 2 with num_generations=4 gets auto-upgraded to 4. To stay as close as possible to the spec’s effective batch size of 8, I’m going to make that explicit in code by using per_device_train_batch_size=4 and reducing gradient_accumulation_steps from 4 to 2 at runtime for this environment.
Those are runtime compatibility adjustments, not RLVR-method changes.

I’m also going to move the TrainerCallback import out of module top-level in the training scripts so unsloth gets imported first and we stop tripping the patch-order warning during startup.


Edited 3 files
Worked for 18m 18s
The runtime compatibility adjustments are in: training precision now follows actual GPU support, batch sizing is normalized for this Unsloth build, and the transformers callback import no longer happens before unsloth. I’m doing a quick compile/test pass and then rerunning the 20-step GRPO smoke test.

Ran python3 -m py_compile training/common.py training/phase2_grpo.py training/phase3_calibration.py
Ran python3 -m unittest verifier.test_verifier data.test_difficulty_sampler data.test_prepare_dataset training.test_common
Ran HF_HOME=/tmp/hf_home .venv/bin/python training/phase2_grpo.py --max-steps 20 --config configs/grpo_a4000.yaml
