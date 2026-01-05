
# src/training/train_dpo.py
import os
import argparse
from dataclasses import dataclass
from typing import Optional, Dict, Any

import torch
from datasets import load_dataset
from transformers import set_seed


from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig
from trl import DPOConfig, DPOTrainer

# key imports
from peft import PeftConfig, PeftModel

def load_policy_and_ref_from_sft(sft_dir: str, quant_cfg):
    # Read adapter metadata to know the base model name
    peft_cfg = PeftConfig.from_pretrained(sft_dir)
    base_name = peft_cfg.base_model_name_or_path

    # Load base model twice (policy + ref)
    policy_base = AutoModelForCausalLM.from_pretrained(
        base_name,
        quantization_config=quant_cfg,
        device_map="auto",
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else None,
        trust_remote_code=True,
        # attn_implementation="flash_attention_2",
    )
    ref_base = AutoModelForCausalLM.from_pretrained(
        base_name,
        quantization_config=quant_cfg,
        device_map="auto",
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else None,
        trust_remote_code=True,
        # attn_implementation="flash_attention_2",
    )

    # Attach the SAME SFT adapters
    policy = PeftModel.from_pretrained(policy_base, sft_dir, is_trainable=True)
    ref = PeftModel.from_pretrained(ref_base, sft_dir, is_trainable=False)

    ref.eval()
    #freezing the reference model
    for p in ref.parameters():
        p.requires_grad = False

    policy.config.use_cache = False
    ref.config.use_cache = False
    return base_name, policy, ref


@dataclass
class DPOTrainCfg:
    # Start DPO from the SFT checkpoint
    sft_model_dir: str = "models/sft"
    ref_model_name_or_path: Optional[str] = None # if None, uses base model of sft dir

    train_file: str = "data/processed/dpo_dataset.json"
    eval_file:Optional[str] = None
    
    seed : int = 42

    output_dir: str = "models/dpo"
    # output_dir: str = "/workspace/models/dpo" for pod
    max_length: int = 384#512
    max_prompt_length: int = 192#256

    beta: float = 0.05#0.1 
    num_train_epochs: int=1
    per_device_train_batch_size: int = 4#2
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 2 #4
    learning_rate: float = 1e-5#1e-4
    warmup_steps: int = 300
    lr_scheduler_type: str="cosine"
    max_grad_norm: float =1.0
    weight_decay: float=0.0
    disable_dropout: bool= True #removing drop out noise

    logging_steps: int = 100
    eval_strategy: str = "no"
  
    save_steps: int = 1000 #4000#1000 #200
    save_total_limit: int = 2

    precompute_ref_log_probs: bool = True
    precompute_ref_batch_size: int = 16


    dataset_num_proc = 4
    fp16: bool = False
    bf16: bool = True
    # DPO continues training the SFT LoRA adapters directly
    # No new LoRA config is created here
    load_in_4bit: bool = True


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sft_model_dir", type=str, default="models/sft")
    p.add_argument("--train_file", type=str, default="data/processed/dpo_dataset.json")
    p.add_argument("--eval_file", type=str, default=None)
    p.add_argument("--output_dir", type=str, default="models/dpo")

    p.add_argument("--beta", type=float, default=0.05)
    p.add_argument("--num_train_epochs", type=int, default=1)
    p.add_argument("--learning_rate", type=float, default=1e-5)

    p.add_argument("--per_device_train_batch_size", type=int, default=4)
    p.add_argument("--gradient_accumulation_steps", type=int, default=2)

    p.add_argument("--warmup_steps", type=int, default=300)
    p.add_argument("--lr_scheduler_type", type=str, default="cosine")

    p.add_argument("--max_grad_norm", type=float, default=1.0)

    p.add_argument("--save_steps", type=int, default=1000)
    p.add_argument("--save_total_limit", type=int, default=2)
    p.add_argument("--logging_steps", type=int, default=100)

    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--max_length", type=int, default=384)
    p.add_argument("--max_prompt_length", type=int, default=192)

    p.add_argument("--use_lora", action="store_true")
    p.add_argument("--no_lora", dest="use_lora", action="store_false")
    p.set_defaults(use_lora=False)

    p.add_argument("--load_in_4bit", action="store_true")
    p.add_argument("--no_4bit", dest="load_in_4bit", action="store_false")
    p.set_defaults(load_in_4bit=True)

    return p.parse_args()


def build_tokenizer(model_path: str):
    tok = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    return tok

def build_quant_cfg(load_in_4bit: bool):
    if not load_in_4bit:
        return None
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

def main():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")  # TF32 fast path on A100
    
    args = parse_args()
    
    cfg = DPOTrainCfg(
        sft_model_dir=args.sft_model_dir,
        train_file=args.train_file,
        eval_file=args.eval_file,
        output_dir=args.output_dir,
        beta=args.beta,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
        max_prompt_length=args.max_prompt_length,
        load_in_4bit=args.load_in_4bit,
        per_device_train_batch_size = args.per_device_train_batch_size,
        gradient_accumulation_steps = args.gradient_accumulation_steps,
        warmup_steps = args.warmup_steps,
        lr_scheduler_type = args.lr_scheduler_type,
        max_grad_norm = args.max_grad_norm,
        save_steps = args.save_steps,
        save_total_limit = args.save_total_limit,
        logging_steps = args.logging_steps,
        # precompute_ref_log_probs=cfg.precompute_ref_log_probs,
        # precompute_ref_batch_size=cfg.precompute_ref_batch_size,

        seed = args.seed,
    )
    
    set_seed(cfg.seed)

    os.makedirs(cfg.output_dir, exist_ok=True)

    quant_cfg = build_quant_cfg(cfg.load_in_4bit)

    # Policy model: start from SFT checkpoint
 
    base_name, model, ref_model = load_policy_and_ref_from_sft(cfg.sft_model_dir, quant_cfg)
    tokenizer = build_tokenizer(base_name)
    model.config.use_cache = False
    
    # Dataset: JSON/JSONL with fields prompt, chosen, rejected
    data_files = {"train": cfg.train_file}
    if cfg.eval_file:
        data_files["eval"] = cfg.eval_file
    ds = load_dataset("json", data_files=data_files)

    train_ds = ds["train"]
    eval_ds = ds["eval"] if "eval" in ds else None

    # Apply chat template
    def clean(ex):
        if getattr(tokenizer, "chat_template", None):
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": ex["prompt"]}],
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            prompt = ex["prompt"]

        return {
            "prompt": prompt,
            "chosen": ex["chosen"].strip(),
            "rejected": ex["rejected"].strip(),
        }

    
    train_ds = train_ds.map(clean, remove_columns=train_ds.column_names,num_proc=cfg.dataset_num_proc)
    if eval_ds is not None:
        eval_ds = eval_ds.map(clean,remove_columns=eval_ds.column_names)
    
    peft_config = None

    dpo_args = DPOConfig(
        output_dir=cfg.output_dir,
        beta=cfg.beta,
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        warmup_steps = cfg.warmup_steps,
        max_grad_norm=cfg.max_grad_norm,
        weight_decay=cfg.weight_decay,
        logging_steps=cfg.logging_steps,
        disable_dropout = cfg.disable_dropout,
        eval_strategy="no", #cfg.eval_strategy if eval_ds is not None else "no",,
        save_steps=cfg.save_steps,
        save_total_limit=cfg.save_total_limit,
        fp16=cfg.fp16,
        bf16=cfg.bf16 and torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        report_to="none",
        precompute_ref_log_probs=True,
        precompute_ref_batch_size=16,
        use_logits_to_keep=True,
        remove_unused_columns=False,  
        max_length=cfg.max_length,
        max_prompt_length=cfg.max_prompt_length,
        group_by_length=False,
        optim="paged_adamw_32bit",
        lr_scheduler_type=cfg.lr_scheduler_type,
        save_safetensors=True,
    ) 

    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=dpo_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        peft_config=None, #because model is already a PEFT model
        # Note: policy model already includes SFT LoRA adapters
        # DPO will further update these adapters directly.
    )

    from pathlib import Path
    from transformers.trainer_utils import get_last_checkpoint
    
    out_dir = Path(cfg.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    last_ckpt = get_last_checkpoint(str(out_dir))

    print("\n==== DPO RUN SUMMARY ====")
    print("cwd:", Path.cwd().resolve())
    print("output_dir:", out_dir)
    print("last_ckpt:", last_ckpt)
    print("save_steps:", cfg.save_steps, "| save_total_limit:", cfg.save_total_limit)
    print("max_grad_norm:", cfg.max_grad_norm)
    print("lr:", cfg.learning_rate, "| beta:", cfg.beta)
    print("bs:", cfg.per_device_train_batch_size,
          "| grad_acc:", cfg.gradient_accumulation_steps,
          "| effective_bs:", cfg.per_device_train_batch_size * cfg.gradient_accumulation_steps)
    print("max_length:", cfg.max_length, "| max_prompt_length:", cfg.max_prompt_length)
    print("bf16:", cfg.bf16 and torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
          "| fp16:", cfg.fp16)
    # print("precompute_ref_log_probs:", cfg.precompute_ref_log_probs,
          # "| precompute_ref_batch_size:", cfg.precompute_ref_batch_size)
    print("existing checkpoints:", sorted([p.name for p in out_dir.glob("checkpoint-*")])[-3:])
    print("=========================\n")

    trainer.train(resume_from_checkpoint=last_ckpt)
   

    #Save final model
    trainer.save_model(cfg.output_dir)

    print(f"\n DPO complete. Saved to {cfg.output_dir}")

if __name__ == "__main__":
    main()
