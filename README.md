# Aligning Language Models with SFT vs DPO (UltraFeedback) — Base vs SFT vs DPO on Mistral-7B-Instruct

This repo demonstrates how **Supervised Fine-Tuning (SFT)** and **Direct Preference Optimization (DPO)** change a base instruction model’s behavior using:

- **Base model**: `mistralai/Mistral-7B-Instruct-v0.2`
- **SFT adapter**: trained to improve instruction-following + structured outputs
- **DPO adapter**: trained on pairwise preferences (“chosen vs rejected”)

The repo includes:

- Data preparation scripts for SFT and DPO
- Training scripts for SFT and DPO adapters
- Two evaluation tracks:
  1. **Format / schema compliance** (fast, deterministic validators)
  2. **Preference alignment accuracy** (DPO-relevant, automatic)

---

## Evaluation 1: Format / schema compliance (automatic validators)

### What was scored

Each prompt includes explicit `FORMAT RULES`, and the evaluation applies lightweight validators:

- **exact_bullets**: “Output EXACTLY N bullets” (optionally enforcing a prefix like `- `)
- **first_n_lines_questions**: first N lines must be questions
- **json_schema**: output must parse as JSON and match a schema

### Results (deterministic decoding)

From `artifacts/example_generations_eval_summary.json` / `artifacts/example_generations.json`:

**Pass-all (all checks satisfied per prompt):**

- **Base:** 16 / 46 = **34.8%**
- **SFT:** 25 / 46 = **54.3%** ✅ best overall compliance
- **DPO:** 18 / 46 = **39.1%**

**Per-check pass rates:**

- `exact_bullets`: Base **38.6%** · SFT **52.3%** · DPO **43.2%**
- `first_n_lines_questions`: Base **80%** · SFT **100%** · DPO **60%**
- `json_schema`: Base **0%** · SFT **100%** · DPO **100%**

**Key takeaway:** SFT is strongest on strict “do exactly what I asked” formatting.  
Both SFT and DPO dramatically outperform Base on structured JSON tasks.

---

## Evaluation 2: Preference alignment accuracy (DPO-relevant metric)

This evaluation measures whether each model assigns higher log-likelihood to the **chosen** completion than the **rejected** completion (pairwise preferences).

**Note on evaluation split (important):**  
For this portfolio iteration, SFT and DPO were trained using the full UltraFeedback dataset.  
The preference evaluation therefore uses a **post-hoc, deterministic evaluation subset** (seeded split + fixed sampling) to produce **reproducible diagnostic numbers** across Base/SFT/DPO (not a strict generalization benchmark).

**Production-grade variant (future work):** create train/valid/test splits **before training**, save indices/IDs, and keep the test set fully untouched.

### Metric

For each pair:

- compute log-likelihood of `chosen` completion given the prompt
- compute log-likelihood of `rejected` completion given the prompt
- prediction is correct if `logp(chosen) > logp(rejected)`
- report **preference accuracy**

### Results (4-bit inference, 2000 pairs)

**Run A: `max_completion_tokens = 256`**
| Model | Preference accuracy | Avg logp margin (chosen - rejected) | Ties / 2000 |
| ----- | ------------------- | ----------------------------------- | ----------- |
| Base | 0.2705 | -51.47 | 723 (36.1%) |
| SFT | 0.2580 | -45.87 | 723 (36.1%) |
| DPO | **0.2715** | **-43.75** | 723 (36.1%) |

Tie-aware metrics (ties are common in this setting):

- `accuracy_excl_ties = correct / (n - ties)`
- `accuracy_w_ties_0.5 = (correct + 0.5 * ties) / n`

| Model | Accuracy excl. ties | Accuracy w/ ties=0.5 |
| ----- | ------------------- | -------------------- |
| Base  | 0.4236              | 0.4512               |
| SFT   | 0.4041              | 0.4387               |
| DPO   | **0.4252**          | **0.4522**           |

**Run B: `max_completion_tokens = 512` (sensitivity check)**
| Model | Preference accuracy | Avg logp margin (chosen - rejected) | Ties / 2000 |
| ----- | ------------------- | ----------------------------------- | ----------- |
| Base | 0.2625 | -50.30 | 756 (37.8%) |
| SFT | 0.2545 | -45.14 | 756 (37.8%) |
| DPO | **0.2670** | **-43.08** | 756 (37.8%) |

Tie-aware metrics:
| Model | Accuracy excl. ties | Accuracy w/ ties=0.5 |
| ----- | ------------------- | -------------------- |
| Base | 0.4220 | 0.4515 |
| SFT | 0.4092 | 0.4435 |
| DPO | **0.4293** | **0.4560** |

**Interpretation (what you can say in interviews):**

- **SFT** wins on **format compliance** (Evaluation 1).
- **DPO** is directionally best on the **preference-alignment diagnostic**: best (least-negative) logp margin and best tie-aware accuracy in both token settings.

---

## Artifacts (results you can inspect)

- `artifacts/example_generations.json`  
  Full generations for Base/SFT/DPO under deterministic + sampled decoding, plus per-item validator results.
- `artifacts/example_generations_eval_summary.json`  
  Summary pass rates and per-check rates.
- `artifacts/preference_accuracy_4bit_2000_tok256.json`  
  Preference accuracy @ 2000 pairs, `max_completion_tokens=256`.
- `artifacts/preference_accuracy_4bit_2000_tok512.json`  
  Preference accuracy @ 2000 pairs, `max_completion_tokens=512`.

---

## How to reproduce

### 1) Generate samples + automatic format scoring

```bash
python src/evaluation/generate_samples.py \
  --base_model mistralai/Mistral-7B-Instruct-v0.2 \
  --sft_dir models/sft \
  --dpo_dir models/dpo \
  --prompts_file data/eval/prompts_with_checks.json \
  --output_file artifacts/example_generations.json \
  --batch_size 8 \
  --max_new_tokens 256 \
  --max_prompt_tokens 1024
```

### 2) Preference alignment diagnostic (log-likelihood accuracy)

**A) 256 completion tokens**

```bash
python src/evaluation/eval_preference_accuracy.py \
  --base_model mistralai/Mistral-7B-Instruct-v0.2 \
  --sft_dir models/sft \
  --dpo_dir models/dpo \
  --n_samples 2000 \
  --max_prompt_tokens 1024 \
  --max_completion_tokens 256 \
  --load_in_4bit \
  --output_file artifacts/preference_accuracy_4bit_2000_tok256.json
```

**B) 512 completion tokens**

```bash
python src/evaluation/eval_preference_accuracy.py \
  --base_model mistralai/Mistral-7B-Instruct-v0.2 \
  --sft_dir models/sft \
  --dpo_dir models/dpo \
  --n_samples 2000 \
  --max_prompt_tokens 1024 \
  --max_completion_tokens 512 \
  --load_in_4bit \
  --output_file artifacts/preference_accuracy_4bit_2000_tok512.json
```

---

## Future improvements (nice-to-have)

- Use a true held-out split (save dataset indices/IDs before training)
- Reduce tie rate (often improved by template consistency + truncation choices)
- Clean training/template artifacts that cause “dirty prefixes” in generations
- Add a small replay UI (Gradio) that loads `artifacts/example_generations.json` (no GPU needed)

---

## What this project demonstrates (interview-ready)

- End-to-end pipeline for **SFT vs DPO** adapters
- Practical evaluation:
  - **format compliance** (fast, deterministic, portfolio-friendly)
  - **preference alignment** (DPO-relevant, automatic)
- Honest tradeoffs:
  - SFT excels at strict formatting
  - DPO targets preference alignment (and can improve it)

## Demo (Gradio replay — no GPU required)

This demo replays saved generations from `artifacts/example_generations.json`.

### Run locally

```bash
pip install gradio
python demo/app.py

```
