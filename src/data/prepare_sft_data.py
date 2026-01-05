
# src/data/prepare_sft_data.py

import json
import argparse
from datasets import load_dataset
from tqdm import tqdm

def extract_text(msg):
    """
    UltraFeedback stores responses as message lists.
    We take the final assistant message.
    """
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

            output.append({
                 "prompt": prompt,
                 "response": chosen   
        })
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    
    print(f" SFT saved to {args.output}")
    print(f"Total samples: {len(output)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default= "argilla/ultrafeedback-binarized-preferences-cleaned"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="data/preprocessed/sft_dataset.json"
    )
    args = parser.parse_args()

    main(args)
