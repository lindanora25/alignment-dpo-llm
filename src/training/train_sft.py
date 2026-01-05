
# src/training/train_sft.py
import os
import json
import argparse
from dataclasses import dataclass
from typing import Optional, Dict, Any

import torch
from datasets import load_dataset

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

# -------------------------
# Config helpers
# -------------------------
@dataclass
class SFTTrainCfg:
    base_model: str="mistralai/Mistral-7B-Instruct-v0.2"
    train_file: str =  "data/processed/sft_dataset.json"   
    eval_file: Optional[str] = None

    output_dir: str="models/sft"
    max_seq_length: int = 512 #1024

    # training
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float=2e-4
    warmup_steps: int = 50
    logging_steps: int = 10
    eval_strategy: str="steps"
    eval_steps: int = 200
    save_steps: int = 200
    save_total_limit: int = 2

    fp16: bool = False
    bf16: bool = True

    # LoRA
    use_lora: bool = True
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]

    # quantization
    load_in_4bit: bool = True
    
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--base_model", type=str, default="mistralai/Mistral-7B-Instruct-v0.2")
    p.add_argument("--train_file", type=str, default="data/processed/sft_dataset.json")
    p.add_argument("--eval_file", type=str, default=None)
    p.add_argument("--output_dir", type=str, default="models/sft")
    p.add_argument("--max_seq_length", type=int, default=512)

    p.add_argument("--num_train_epochs", type=int, default=1)
    p.add_argument("--per_device_train_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=8)
    p.add_argument("--learning_rate", type=float, default=2e-4)

    p.add_argument("--use_lora", action="store_true")
    p.add_argument("--no_lora", dest="use_lora", action="store_false")
    p.set_defaults(use_lora=True)

    p.add_argument("--load_in_4bit", action="store_true")
    p.add_argument("--no_4bit", dest="load_in_4bit", action="store_false")
    p.set_defaults(load_in_4bit=True)

    return p.parse_args()

def build_tokenizer(model_name: str):
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    # pad_token=oes and padding_side="right" to avoid FP16 overflow issues
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    return tok

def build_model(model_name: str, load_in_4bit: bool):
    quant_cfg = None
    if load_in_4bit:
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit = True,
            bnb_4bit_use_double_quant = True,
            bnb_4bit_quant_type = "nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quant_cfg,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else None,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False #Helps training stability 
    return model

def main():
    args = parse_args()

    cfg = SFTTrainCfg(
      base_model=args.base_model,
      train_file=args.train_file,
      eval_file=args.eval_file,
      output_dir=args.output_dir,
      max_seq_length=args.max_seq_length,
      num_train_epochs=args.num_train_epochs,
      per_device_train_batch_size=args.per_device_train_batch_size,
      gradient_accumulation_steps=args.gradient_accumulation_steps,
      learning_rate=args.learning_rate,
      use_lora=args.use_lora,
      load_in_4bit=args.load_in_4bit,
    )

    os.makedirs(cfg.output_dir, exist_ok=True)

    tokenizer = build_tokenizer(cfg.base_model)
    model = build_model(cfg.base_model, cfg.load_in_4bit)

    #Expect JSON list or JSONL with fields prompt, response
    data_files: Dict[str, str] = {"train": cfg.train_file}
    if cfg.eval_file is not None:
        data_files["eval"] = cfg.eval_file
    
    ds = load_dataset("json", data_files=data_files)
    train_ds = ds["train"]
    eval_ds = ds["eval"] if "eval" in ds else None

    
    def to_text(ex):
        messages = [
            {"role": "user", "content": ex["prompt"]},
            {"role": "assistant", "content": ex["response"]},
        ]

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

        return {"text": text}

    train_ds = train_ds.map(to_text,remove_columns=train_ds.column_names)
    if eval_ds is not None:
        eval_ds = eval_ds.map(to_text,remove_columns=eval_ds.column_names)

    peft_config = None
    if cfg.use_lora:
        peft_config = LoraConfig(
            r = cfg.lora_r,
            lora_alpha = cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=list(cfg.target_modules),
        )
    
    sft_args = SFTConfig(
        output_dir= cfg.output_dir,
        max_length=cfg.max_seq_length,
        packing=True,
        group_by_length=True,
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        warmup_steps=cfg.warmup_steps,
        logging_steps=cfg.logging_steps,
        eval_strategy=cfg.eval_strategy if eval_ds is not None else "no",
        eval_steps=cfg.eval_steps,
        save_steps=cfg.save_steps,
        save_total_limit = cfg.save_total_limit,
        fp16=cfg.fp16,
        bf16=cfg.bf16 and torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        report_to="none",
        dataset_text_field="text",
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=sft_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        peft_config=peft_config,
    )

    from transformers.trainer_utils import get_last_checkpoint

    last_ckpt = get_last_checkpoint(cfg.output_dir)
    trainer.train(resume_from_checkpoint=last_ckpt)

    trainer.save_model(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)

    print(f"\n SFT complete. Save to {cfg.output_dir}")

if __name__ == "__main__":
    main()
