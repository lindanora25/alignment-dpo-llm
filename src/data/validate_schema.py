
import json
import sys

def main(path, required_keys):
    with open(path) as f:
        data = json.load(f)
    
    for i, row in enumerate(data[:100]):
        for key in required_keys:
            assert key in row, f"Missing key '{key}' in row {i}"
    
    print(f" Schema validated for {path}")

if __name__=="__main__":
    path = sys.argv[1]
    keys = sys.argv[2:]
    main(path, keys)

#Usage:
#python validate_schema.py data/processed/sft_dataset.json prompt response
#python validate_schema.py data/processed/dpo_dataset.json prompt chosen rejected

