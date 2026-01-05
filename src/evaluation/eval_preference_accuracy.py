

import argparse
import json
import random
import re
from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from transformers import BitsAndBytesConfig


# ---------- Cleaning (same idea as eval cleaner) ----------
_CLEAN_PREFIX_PATTERNS = [
    r"^\ufeff",
    r"^[�\s]+",
    r"^\s*(?:🌐\s*)?assistant\s*:\s*",
    r"^\s*im_assistant\s*:\s*",
    r"^\s*limelight\s*:\s*",
    r"^\s*EM\d+\.\s*",
]

def clean_target(text: str) -> str:
    if text is None:
        return ""
    t = text.strip()
    changed = True
    while changed:
        changed = False
        for pat in _CLEAN_PREFIX_PATTERNS:
            new_t = re.sub(pat, "", t, count=1, flags=re.IGNORECASE)
            if new_t != t:
                t = new_t.lstrip()
                changed = True
    t = t.lstrip("`\"' \n\t")
    t = re.sub(r"\r\n?", "\n", t).strip()
    return t

#-----------BitsAndBytesConfig --------------------------
def _make_bnb_config(load_in_4bit: bool):
     if not load_in_4bit:
         return None
     if BitsAndBytesConfig is None:
         raise ImportError(
             "BitsAndBytesConfig is not available. Install bitsandbytes to use --load_in_4bit."
         )
     return BitsAndBytesConfig(
         load_in_4bit=True,
         bnb_4bit_use_double_quant=True,
         bnb_4bit_quant_type="nf4",
         bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16,
     )
# ---------- Robust field extraction ----------
def extract_prompt_chosen_rejected(ex: Dict[str, Any]) -> Tuple[str, str, str]:
    """
    Tries common schemas for preference datasets.
    Expected end result: prompt (user text), chosen (assistant completion), rejected (assistant completion).
    """
    # Common case
    if "prompt" in ex and "chosen" in ex and "rejected" in ex:
        prompt = ex["prompt"]
        chosen = ex["chosen"]
        rejected = ex["rejected"]
        # Sometimes prompt is a list of messages; handle lightly
        if isinstance(prompt, list) and prompt and isinstance(prompt[0], dict) and "content" in prompt[0]:
            # take user content if present
            user_msgs = [m for m in prompt if m.get("role") == "user"]
            prompt = user_msgs[-1]["content"] if user_msgs else prompt[-1]["content"]
        return str(prompt), str(chosen), str(rejected)

    # Some datasets store conversation in "messages" and chosen/rejected separately
    if "messages" in ex and "chosen" in ex and "rejected" in ex:
        msgs = ex["messages"]
        if isinstance(msgs, list):
            user_msgs = [m for m in msgs if isinstance(m, dict) and m.get("role") == "user"]
            prompt = user_msgs[-1]["content"] if user_msgs else (msgs[-1].get("content", "") if msgs else "")
        else:
            prompt = str(msgs)
        return str(prompt), str(ex["chosen"]), str(ex["rejected"])

    raise KeyError(f"Could not extract fields from example keys: {list(ex.keys())}")


# ---------- Prompt formatting ----------
def make_prompt_text(tokenizer, user_prompt: str) -> str:
    messages = [{"role": "user", "content": user_prompt}]
    # add_generation_prompt makes the string end right where assistant should begin
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


# ---------- Log-likelihood of completion given prompt ----------
@torch.no_grad()
def batch_loglik_of_completions(
    model,
    tokenizer,
    prompt_texts: List[str],
    completions: List[str],
    max_prompt_tokens: int,
    max_completion_tokens: int,
) -> torch.Tensor:
    """
    Returns log-likelihood (sum of log probs) of completion tokens only, per example in batch.
    """
    assert len(prompt_texts) == len(completions)

    # Tokenize prompt only (so we can mask)
    prompt_enc = tokenizer(
        prompt_texts,
        truncation=True,
        max_length=max_prompt_tokens,
        add_special_tokens=False,
    )

    # Tokenize full prompt+completion
    full_texts = [p + c for p, c in zip(prompt_texts, completions)]
    full_enc = tokenizer(
        full_texts,
        truncation=True,
        max_length=max_prompt_tokens + max_completion_tokens,
        padding=True,
        add_special_tokens=False,
        return_tensors="pt",
    )

    input_ids = full_enc["input_ids"].to(model.device)
    attention_mask = full_enc["attention_mask"].to(model.device)

    # Build labels: mask out prompt tokens, keep completion tokens
    labels = input_ids.clone()
    labels[:] = -100

    for i in range(len(prompt_texts)):
        p_len = len(prompt_enc["input_ids"][i])
        # Completion region = [p_len, seq_len)
        seq_len = int(attention_mask[i].sum().item())
        # guard: if prompt got truncated to full length, completion may be empty
        if p_len < seq_len:
            labels[i, p_len:seq_len] = input_ids[i, p_len:seq_len]

    # Forward pass
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
        out = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = out.logits  # [B, T, V]

    # Shift for causal LM
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    shift_mask = (shift_labels != -100) & (attention_mask[:, 1:] == 1)

    # log softmax then gather labels
    log_probs = F.log_softmax(shift_logits, dim=-1)  # [B, T-1, V]
    # replace -100 with 0 for gather safety
    safe_labels = shift_labels.clone()
    safe_labels[safe_labels == -100] = 0
    token_logp = log_probs.gather(dim=-1, index=safe_labels.unsqueeze(-1)).squeeze(-1)  # [B, T-1]

    # sum only completion tokens
    token_logp = token_logp * shift_mask.float()
    seq_logp = token_logp.sum(dim=-1)  # [B]
    return seq_logp


def load_model_variant(base_model: str, adapter_dir: str | None, load_in_4bit: bool):
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_cfg = _make_bnb_config(load_in_4bit)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=quant_cfg,
        torch_dtype=(torch.bfloat16 if torch.cuda.is_available() and not load_in_4bit else None),
        device_map="auto",
    )
    if adapter_dir:
        model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()
    return model, tokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", required=True)
    ap.add_argument("--sft_dir", default=None)
    ap.add_argument("--dpo_dir", default=None)

    ap.add_argument("--dataset", default="argilla/ultrafeedback-binarized-preferences-cleaned")
    ap.add_argument("--split", default="train")  # dataset split to pull from
    ap.add_argument("--n_samples", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--max_prompt_tokens", type=int, default=1024)
    ap.add_argument("--max_completion_tokens", type=int, default=256)
    ap.add_argument("--load_in_4bit", action="store_true",help="Load base model in 4-bit NF4 (requires bitsandbytes).")

    ap.add_argument("--output_file", default="artifacts/preference_accuracy.json")
    args = ap.parse_args()

    random.seed(args.seed)

    # Load dataset and create held-out split
    ds = load_dataset(args.dataset, split=args.split)

    # Ensure the test split is large enough to supply n_samples
    n_total = len(ds)
    target_test = int(args.n_samples * 1.2)  # buffer so sampling is clean
    test_size = max(0.02, min(0.2, target_test / max(1, n_total)))

    ds = ds.train_test_split(test_size=test_size, seed=args.seed)
    eval_ds = ds["test"]
    #sample deterministically
    eval_ds = eval_ds.shuffle(seed=args.seed).select(range(min(args.n_samples, len(eval_ds))))

    # Sample examples
    idxs = list(range(len(eval_ds)))
    random.shuffle(idxs)
    idxs = idxs[: min(args.n_samples, len(idxs))]
    examples = [eval_ds[i] for i in idxs]

    # Extract and format
    triplets = []
    for ex in examples:
        prompt, chosen, rejected = extract_prompt_chosen_rejected(ex)
        prompt = prompt.strip()
        chosen = clean_target(chosen)
        rejected = clean_target(rejected)
        triplets.append((prompt, chosen, rejected))

    variants = [
        ("base", None),
        ("sft", args.sft_dir),
        ("dpo", args.dpo_dir),
    ]

    results = {}
    for name, adapter in variants:
        if adapter is None and name != "base":
            continue  # skip if not provided
        
        model, tokenizer = load_model_variant(args.base_model, adapter, args.load_in_4bit)
        correct = 0
        margins = []
        ties = 0

        # Batch loop
        for start in range(0, len(triplets), args.batch_size):
            batch = triplets[start:start + args.batch_size]

            prompts = [make_prompt_text(tokenizer, p) for (p, _, _) in batch]

            chosen_texts = [c for (_, c, _) in batch]
            rejected_texts = [r for (_, _, r) in batch]

            # compute logp for chosen and rejected
            logp_chosen = batch_loglik_of_completions(
                model, tokenizer, prompts, chosen_texts,
                args.max_prompt_tokens, args.max_completion_tokens
            )
            logp_rejected = batch_loglik_of_completions(
                model, tokenizer, prompts, rejected_texts,
                args.max_prompt_tokens, args.max_completion_tokens
            )

            m = (logp_chosen - logp_rejected).detach().cpu().tolist()
            margins.extend(m)

            for mm in m:
                if abs(mm) < 1e-6:
                    ties += 1
                if mm > 0:
                    correct += 1

        acc = correct / len(triplets) if triplets else 0.0
        avg_margin = sum(margins) / len(margins) if margins else 0.0

        results[name] = {
            "n": len(triplets),
            "preference_accuracy": acc,
            "avg_logp_margin": avg_margin,
            "ties": ties,
            "adapter": adapter,
        }

        # free GPU memory between variants
        del model
        torch.cuda.empty_cache()

    out = {
        "dataset": args.dataset,
        "split_used": args.split,
        "n_samples": len(triplets),
        "seed": args.seed,
        "max_prompt_tokens": args.max_prompt_tokens,
        "max_completion_tokens": args.max_completion_tokens,
        "results": results,
    }

    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
