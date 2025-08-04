# train/prepare_data.py

import os
from datasets import load_dataset

def main():
    # 1. Load GSM8K (Grade School Math 8K) from HF
    #    config "main" provides train/test splits :contentReference[oaicite:0]{index=0}
    gsm8k = load_dataset("openai/gsm8k", "main")

    # 2. Ensure data directory exists
    os.makedirs("data", exist_ok=True)

    # 3. Export each split to JSONL for easy downstream loading
    print("Saving train split to data/gsm8k_train.jsonl...")
    gsm8k["train"].to_json("data/gsm8k_train.jsonl", orient="records", lines=True)

    if "test" in gsm8k:
        print("Saving test split to data/gsm8k_test.jsonl...")
        gsm8k["test"].to_json("data/gsm8k_test.jsonl", orient="records", lines=True)
    elif "validation" in gsm8k:
        print("Saving validation split to data/gsm8k_valid.jsonl...")
        gsm8k["validation"].to_json("data/gsm8k_valid.jsonl", orient="records", lines=True)
    else:
        print("Warning: no 'test' or 'validation' split found in GSM8K.")

    print("Done.")

if __name__ == "__main__":
    main()
