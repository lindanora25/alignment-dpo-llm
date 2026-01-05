

# src/evaluate/generate_samples.py

import os
import json
import argparse
from typing import List, Dict, Any, Optional
from json import JSONDecodeError
import re
from collections import defaultdict


import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    BitsAndBytesConfig,
)
from peft import PeftConfig, PeftModel

def build_quant_cfg(load_in_4bit: bool = True) -> Optional[BitsAndBytesConfig]:
    if not load_in_4bit:
        return None
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

def enable_a100_speedups():
    # Fast matmul on Ampere+ (A100)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

def load_base_model(model_name: str, quant_cfg):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        quantization_config=quant_cfg,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    model.eval()
    # For generation, cache helps a lot
    model.config.use_cache = True
    return model

def load_peft_model(adapter_dir: str, quant_cfg):
    # Adapter-only dir: read which base model it came from
    peft_cfg = PeftConfig.from_pretrained(adapter_dir)
    base_name = peft_cfg.base_model_name_or_path

    base = load_base_model(base_name, quant_cfg)
    model = PeftModel.from_pretrained(base, adapter_dir)
    model.eval()
    model.config.use_cache = True
    return model

def format_chat_prompt(tokenizer, user_prompt: str) -> str:
    # Works for Mistral Instruct tokenizers
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": user_prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    # fallback
    return f"User: {user_prompt}\nAssistant:"

def pretokenize_and_sort(tokenizer, prompts, max_length=None):
    enc = tokenizer(
        prompts,
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
        return_attention_mask=True,
    )
    items = [{"input_ids": ids, "attention_mask": mask}
             for ids, mask in zip(enc["input_ids"], enc["attention_mask"])]

    order = sorted(range(len(items)), key=lambda i: len(items[i]["input_ids"]), reverse=True)
    items_sorted = [items[i] for i in order]
    return items_sorted, order

@torch.inference_mode()
def batch_generate_pretok(model, tokenizer, pretok_items_sorted, order, gen_cfg, batch_size=4):
    device = next(model.parameters()).device
    outs_sorted = []

    for i in range(0, len(pretok_items_sorted), batch_size):
        batch_items = pretok_items_sorted[i:i+batch_size]
        batch = tokenizer.pad(
            batch_items,
            padding=True,
            pad_to_multiple_of=8,
            return_tensors="pt",
        ).to(device)

        out = model.generate(**batch, generation_config=gen_cfg)

        prompt_lens = batch["attention_mask"].sum(dim=1).tolist()
        for j, plen in enumerate(prompt_lens):
            outs_sorted.append(tokenizer.decode(out[j, plen:], skip_special_tokens=True))

    # unsort
    outs = [None] * len(outs_sorted)
    for sorted_i, original_i in enumerate(order):
        outs[original_i] = outs_sorted[sorted_i]
    return outs
#Helper functions
# ----------------------------
# Output cleaner (display/eval)
# ----------------------------

_CLEAN_PREFIX_PATTERNS = [
    r"^\ufeff",                # BOM
    r"^[�\s]+",                 # Unicode replacement chars
    r"^\s*(?:🌐\s*)?Assistant\s*:\s*",   # "🌐 Assistant:"
    r"^\s*assistant\s*:\s*",
    r"^\s*im_assistant\s*:\s*",
    r"^\s*limelight\s*:\s*",
    r"^\s*EM\d+\.\s*",          # "EM2." / "EM3."
]

def clean_output(text: str) -> str:
    """Lightweight normalization for nicer eval + HF demo display (does NOT change meaning)."""
    if text is None:
        return ""
    t = text.strip()

    # Normalize common prefix garbage/preambles
    changed = True
    while changed:
        changed = False
        for pat in _CLEAN_PREFIX_PATTERNS:
            new_t = re.sub(pat, "", t, count=1, flags=re.IGNORECASE)
            if new_t != t:
                t = new_t.lstrip()
                changed = True

    # Remove leading stray quotes/backticks that sometimes appear
    t = t.lstrip("`\"' \n\t")

    # Normalize newlines
    t = re.sub(r"\r\n?", "\n", t).strip()
    return t


# ----------------------------
# Validators / scoring
# ----------------------------

_BULLET_LINE_RE = re.compile(r"^\s*(?:[-*•]\s+|\d+[\.\)]\s+|[A-Za-z][\.\)]\s+)", re.UNICODE)

def _infer_exact_bullets(prompt: str) -> int | None:
    # "exactly 5 bullets" / "exactly five bullets" (digits only for now)
    m = re.search(r"\bexactly\s+(\d+)\s+bullets?\b", prompt, flags=re.IGNORECASE)
    if m:
        return int(m.group(1))
    # "in 5 bullets"
    m = re.search(r"\bin\s+(\d+)\s+bullets?\b", prompt, flags=re.IGNORECASE)
    if m:
        return int(m.group(1))
    return None

def _count_bullets(text: str, prefix: str | None = None) -> int:
    lines = [ln.rstrip() for ln in text.split("\n") if ln.strip()]

    if prefix:
        # Count only lines that start with the exact prefix (after leading whitespace)
        return sum(1 for ln in lines if ln.lstrip().startswith(prefix))

    return sum(1 for ln in lines if _BULLET_LINE_RE.match(ln))

def validate_exact_bullets(text: str, n: int, prefix: str | None = None) -> dict:
    cnt = _count_bullets(text, prefix=prefix)
    return {"pass": cnt == n, "found": cnt, "expected": n, "prefix": prefix}


def _infer_first_n_questions(prompt: str) -> int | None:
    # "Ask 3 questions" / "Ask me 3 questions"
    m = re.search(r"\bask(?:\s+me)?\s+(\d+)\s+questions?\b", prompt, flags=re.IGNORECASE)
    if m:
        return int(m.group(1))
    # "first 3 questions"
    m = re.search(r"\bfirst\s+(\d+)\s+questions?\b", prompt, flags=re.IGNORECASE)
    if m:
        return int(m.group(1))
    return None

def validate_first_n_lines_are_questions(text: str, n: int) -> dict:
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    first = lines[:n]
    if len(first) < n:
        return {"pass": False, "reason": "too_few_lines", "found": len(first), "expected": n}
    # Accept numbered questions like "1. ...?"
    def is_question(ln: str) -> bool:
        return "?" in ln

    ok = all(is_question(ln) for ln in first)
    return {"pass": ok, "checked_lines": first}


def _extract_json_candidate(text: str) -> str | None:
    """
    Try to extract a JSON object/array from:
    - a ```json ... ``` block (preferred)
    - otherwise, first '{' ... last '}' (object)
    - otherwise, first '[' ... last ']' (array)
    """
    # fenced code block
    m = re.search(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).strip()

    # object
    i = text.find("{")
    j = text.rfind("}")
    if i != -1 and j != -1 and j > i:
        return text[i:j+1].strip()

    # array
    i = text.find("[")
    j = text.rfind("]")
    if i != -1 and j != -1 and j > i:
        return text[i:j+1].strip()

    return None

def _infer_required_json_keys(prompt: str) -> list[str]:
    """
    Heuristic: if prompt contains a schema/example with keys in quotes, treat them as required candidates.
    Example: {"intent": "...", "urgency": "..."} -> intent, urgency
    """
    keys = re.findall(r"\"([A-Za-z_][A-Za-z0-9_]*)\"\s*:", prompt)
    # De-dup preserving order
    seen = set()
    out = []
    for k in keys:
        if k not in seen:
            out.append(k)
            seen.add(k)
    return out

def _infer_enum_constraints(prompt: str) -> dict[str, list[str]]:
    """
    Heuristic for enums like:
      urgency: "low"|"medium"|"high"
    """
    enums = {}
    for m in re.finditer(r"\b([A-Za-z_][A-Za-z0-9_]*)\b\s*:\s*(\"[^\"]+\"(?:\s*\|\s*\"[^\"]+\"){1,})", prompt):
        key = m.group(1)
        raw = m.group(2)
        vals = re.findall(r"\"([^\"]+)\"", raw)
        if vals:
            enums[key] = vals
    return enums

def validate_json_parses_and_matches(prompt: str, text: str, schema: dict | None = None) -> dict:
    """
    schema (optional) supports:
      {
        "type": "object",
        "required": [...],
        "enums": {"urgency": ["low","medium","high"]}
      }
    If schema is None, infer required keys + enums heuristically from the prompt.
    """
    candidate = _extract_json_candidate(text)
    if not candidate:
        return {"pass": False, "reason": "no_json_found"}

    try:
        obj = json.loads(candidate)
    except Exception as e:
        return {"pass": False, "reason": "json_parse_error", "error": str(e)}

    sch = schema or {}
    required = sch.get("required") or _infer_required_json_keys(prompt)
    enums = sch.get("enums") or _infer_enum_constraints(prompt)

    # type check (light)
    if sch.get("type") == "object" and not isinstance(obj, dict):
        return {"pass": False, "reason": "wrong_type", "expected": "object", "found": type(obj).__name__}

    missing = [k for k in required if isinstance(obj, dict) and k not in obj]
    if missing:
        return {"pass": False, "reason": "missing_keys", "missing": missing}

    bad_enums = {}
    if isinstance(obj, dict):
        for k, allowed in enums.items():
            if k in obj and isinstance(obj[k], str) and obj[k] not in allowed:
                bad_enums[k] = {"found": obj[k], "allowed": allowed}
    if bad_enums:
        return {"pass": False, "reason": "enum_violation", "details": bad_enums}

    return {"pass": True, "required_checked": required, "enums_checked": enums}

def build_checks(prompt_item: dict) -> list[dict]:
    # Only use explicit checks; no inference.
    checks = prompt_item.get("checks")
    return checks if isinstance(checks, list) else []


def run_checks(prompt_item: dict, cleaned_text: str) -> dict:
    prompt = prompt_item["prompt"]
    checks = build_checks(prompt_item)

    results = {"checks": [], "pass_all": True}
    for chk in checks:
        ctype = chk["type"]
        if ctype == "exact_bullets":
            r = validate_exact_bullets(cleaned_text, chk["n"], prefix=chk.get("prefix"))
        elif ctype == "first_n_lines_questions":
            r = validate_first_n_lines_are_questions(cleaned_text, chk["n"])
        elif ctype == "json_schema":
            r = validate_json_parses_and_matches(prompt, cleaned_text, schema=chk.get("schema"))
        else:
            r = {"pass": False, "reason": f"unknown_check:{ctype}"}

        results["checks"].append({"type": ctype, **r})
        if not r.get("pass", False):
            results["pass_all"] = False

    # If no checks inferred, treat as "not_applicable" (don’t penalize)
    if not checks:
        results["pass_all"] = None

    return results


def compute_eval_summary(items: list[dict], prefer_variant: str = "det") -> dict:
    """
    Summarize pass rates + a simple 'format compliance win-rate' story.
    """
    models = ["base", "sft", "dpo"]
    pass_all_counts = {m: 0 for m in models}
    applicable_counts = {m: 0 for m in models}
    per_check = {m: defaultdict(lambda: {"pass": 0, "total": 0}) for m in models}

    # winner counts (most passed checks; tie allowed)
    wins = defaultdict(int)
    ties = 0

    for item in items:
        # Score each model on the chosen variant if available
        scored = {}
        for m in models:
            ev = item.get("auto_eval", {}).get(m, {}).get(prefer_variant)
            if not ev:
                continue
            scored[m] = ev

            # pass_all aggregation
            if ev["pass_all"] is not None:
                applicable_counts[m] += 1
                if ev["pass_all"]:
                    pass_all_counts[m] += 1

            # per-check aggregation
            for chk in ev["checks"]:
                t = chk["type"]
                per_check[m][t]["total"] += 1
                if chk.get("pass"):
                    per_check[m][t]["pass"] += 1

        # win-rate (only for items where at least one model had checks)
        # Winner = model with max number of passed checks for that item.
        # If all have pass_all None -> skip
        if scored:
            # Count pass per model
            def count_pass(ev):
                return sum(1 for c in ev["checks"] if c.get("pass"))
            scores = {m: count_pass(ev) for m, ev in scored.items()}

            max_score = max(scores.values()) if scores else None
            best = [m for m, s in scores.items() if s == max_score]

            if max_score is None:
                pass
            elif len(best) == 1:
                wins[best[0]] += 1
            else:
                ties += 1

    # Convert to rates
    summary = {
        "prefer_variant": prefer_variant,
        "pass_all_rate": {
            m: (pass_all_counts[m] / applicable_counts[m] if applicable_counts[m] else None)
            for m in models
        },
        "per_check_rate": {
            m: {
                t: (d["pass"] / d["total"] if d["total"] else None)
                for t, d in per_check[m].items()
            }
            for m in models
        },
        "wins": dict(wins),
        "ties": ties,
        "counts": {"pass_all": pass_all_counts, "applicable": applicable_counts},
    }
    return summary
###########

def free_model(model):
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", default="mistralai/Mistral-7B-Instruct-v0.2")
    parser.add_argument("--sft_dir", default="models/sft")
    parser.add_argument("--dpo_dir", default="models/dpo")
    parser.add_argument("--prompts_file", default="data/eval/prompts.json")
    parser.add_argument("--output_file", default="artifacts/example_generations.json")

    parser.add_argument("--load_in_4bit", action="store_true")
    parser.set_defaults(load_in_4bit=True)

    parser.add_argument("--batch_size", type=int, default=8) #4-8-16
    parser.add_argument("--max_new_tokens", type=int, default=256) #256
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--max_prompt_tokens", type=int, default=1024)
    parser.add_argument("--do_sample", action="store_true")
    parser.set_defaults(do_sample=False)

    parser.add_argument("--two_pass", action="store_true")
    parser.set_defaults(two_pass=False)
    
    parser.add_argument("--sample_temperature", type=float, default=0.9)
    parser.add_argument("--sample_top_p", type=float, default=0.9)


    # Optional: don’t generate unsafe categories for public artifacts
    parser.add_argument("--skip_categories", nargs="*", default=["safety_harm"])

    parser.add_argument("--clean_outputs", action="store_true")
    parser.set_defaults(clean_outputs=True)

    parser.add_argument("--auto_eval", action="store_true")
    parser.set_defaults(auto_eval=True)

    parser.add_argument("--eval_variant", choices=["det", "samp"], default="det")

    args = parser.parse_args()

    if torch.cuda.is_available():
        enable_a100_speedups()

    quant_cfg = build_quant_cfg(args.load_in_4bit)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    with open(args.prompts_file, "r", encoding="utf-8") as f:
        try:
            prompt_items = json.load(f)
        except JSONDecodeError as e:
            raise SystemExit(
                f"Invalid JSON in {args.prompts_file}: {e.msg} at line {e.lineno}, col {e.colno}"
            )

    # Filter categories (optional)
    if args.skip_categories:
        prompt_items = [p for p in prompt_items if p.get("category") not in set(args.skip_categories)]

    chat_prompts = [format_chat_prompt(tokenizer, p["prompt"]) for p in prompt_items]

    assert all(isinstance(x, str) for x in chat_prompts), f"Bad prompt types: {set(type(x) for x in chat_prompts)}"


    # keep both configs
    # gen_cfg_det stays deterministic
    # gen_cfg_samp stays sampled
    gen_cfg_det = GenerationConfig(
      max_new_tokens=args.max_new_tokens,
      do_sample=False,
      temperature=None,
      top_p=None,
      pad_token_id=tokenizer.pad_token_id,
      eos_token_id=tokenizer.eos_token_id,
      repetition_penalty=1.1,
    )

    gen_cfg_samp = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        do_sample=True,
        temperature=args.sample_temperature,
        top_p=args.sample_top_p,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        repetition_penalty=1.1,
    )


    results = [{
      "id": p["id"],
      "category": p.get("category"),
      "prompt": p["prompt"],
      "rubric": {
        "instruction_following": None,
        "helpfulness": None,
        "truthfulness": None,
        "tone": None,
        "safety": None,
        "overall": None,
        "winner": None,   # "base" | "sft" | "dpo" | "tie"
        "notes": ""
      },
      "outputs": {
        "base": {"det": None, "samp": None},
        "sft":  {"det": None, "samp": None},
        "dpo":  {"det": None, "samp": None},
      }
    } for p in prompt_items]

    # Pretokenize once
    pretok_items_sorted, order = pretokenize_and_sort(tokenizer, chat_prompts, max_length=args.max_prompt_tokens) 
    # ---- BASE ----
    base_model = load_base_model(args.base_model, quant_cfg)
    
    print("Generating BASE outputs...")
    base_det = batch_generate_pretok(base_model, tokenizer, pretok_items_sorted, order, gen_cfg_det, batch_size=args.batch_size)
    for r, o in zip(results, base_det):
         r["outputs"]["base"]["det"] = o

    if args.two_pass:
        base_samp = batch_generate_pretok(base_model, tokenizer, pretok_items_sorted, order, gen_cfg_samp, batch_size=args.batch_size)
        for r, o in zip(results, base_samp):
            r["outputs"]["base"]["samp"] = o

    free_model(base_model)

  
    print("Generating SFT outputs...")
    sft_model = load_peft_model(args.sft_dir, quant_cfg)
    sft_det = batch_generate_pretok(sft_model, tokenizer, pretok_items_sorted, order, gen_cfg_det, batch_size=args.batch_size)
    for r, o in zip(results, sft_det):
        r["outputs"]["sft"]["det"] = o

    if args.two_pass:
        sft_samp = batch_generate_pretok(sft_model, tokenizer, pretok_items_sorted, order, gen_cfg_samp, batch_size=args.batch_size)
        for r, o in zip(results, sft_samp):
            r["outputs"]["sft"]["samp"] = o 
    free_model(sft_model)

    print("Generating DPO outputs...")
    dpo_model = load_peft_model(args.dpo_dir, quant_cfg)
    dpo_det = batch_generate_pretok(dpo_model, tokenizer, pretok_items_sorted, order, gen_cfg_det, batch_size=args.batch_size)
    for r, o in zip(results, dpo_det):
        r["outputs"]["dpo"]["det"] = o

    if args.two_pass:
        dpo_samp = batch_generate_pretok(dpo_model, tokenizer, pretok_items_sorted, order, gen_cfg_samp, batch_size=args.batch_size)
        for r, o in zip(results, dpo_samp):
            r["outputs"]["dpo"]["samp"] = o

    free_model(dpo_model)
    ########
    # ----------------------------
    # Clean + auto-eval
    # ----------------------------
    models = ["base", "sft", "dpo"]
    variants = ["det"] + (["samp"] if args.two_pass else [])

    for i, p in enumerate(prompt_items):
        # store inferred checks for transparency
        results[i]["checks_inferred"] = build_checks(p)

        results[i]["outputs_clean"] = {m: {} for m in models}
        results[i]["auto_eval"] = {m: {} for m in models}

        for m in models:
            for v in variants:
                raw = results[i]["outputs"][m][v]
                if raw is None:
                    continue

                cleaned = clean_output(raw) if args.clean_outputs else raw
                results[i]["outputs_clean"][m][v] = cleaned

                if args.auto_eval:
                    results[i]["auto_eval"][m][v] = run_checks(p, cleaned)

    eval_summary = compute_eval_summary(results, prefer_variant=args.eval_variant)

    #########
    payload = {
    "run_meta": {
      "base_model": args.base_model,
      "sft_dir": args.sft_dir,
      "dpo_dir": args.dpo_dir,
      "batch_size": args.batch_size,
      "max_new_tokens": args.max_new_tokens,
      "max_prompt_tokens": args.max_prompt_tokens,
      "deterministic": {"do_sample": False},
      "sampled": {"do_sample": True, "temperature": args.sample_temperature, "top_p": args.sample_top_p} if args.two_pass else None,
    },
    "auto_eval_summary": eval_summary,
    "items": results
  }

    out_dir = os.path.dirname(args.output_file) or "."
    os.makedirs(out_dir, exist_ok=True)
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"Saved generations to: {args.output_file}")

    ###
    eval_out = args.output_file.replace(".json", "_eval_summary.json")
    with open(eval_out, "w", encoding="utf-8") as f:
        json.dump(eval_summary, f, indent=2, ensure_ascii=False)
    print(f"Saved eval summary to: {eval_out}")


    ###

if __name__ == "__main__":
    main()
