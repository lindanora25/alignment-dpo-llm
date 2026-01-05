
# src/data/prepare_dpo_data.py

import json
import argparse
from datasets import load_dataset
from tqdm import tqdm

def extract_text(msg):
    if isinstance(msg, list):
        return msg[-1]["content"].strip()
    return msg.strip()

def main(args):
    ds = load_dataset(args.dataset)

    output = []

    for split in ds:
        for row in tqdm(ds[split], desc=f"Processing {split}"):
            prompt = row["prompt"].strip()
            chosen = extract_text(row["chosen"])
            rejected = extract_text(row["rejected"])

            output.append({
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected
            })
    
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    
    print(f" DPO dataset to {args.output}")
    print(f"Total samples: {len(output)}")

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default= "argilla/ultrafeedback-binarized-preferences-cleaned"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/dpo_dataset.json"
    )
    args = parser.parse_args()

    main(args)
