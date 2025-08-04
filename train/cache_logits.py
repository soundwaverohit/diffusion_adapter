# train/cache_logits.py
"""
Batches through GSM8K prefixes from local JSONL and caches next-token logits for fast critic training.
Usage:
    python train/cache_logits.py \
        --model_name gpt2 \
        --split train \
        --batch_size 128 \
        --output_path data/gsm8k_train_logits.pt
"""
import os
import json
import random
import argparse
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_prefixes(split: str, max_samples=None, seed=42):
    """
    Load questions from local JSONL: data/gsm8k_{split}.jsonl
    """
    path_map = {
        "train": "data/gsm8k_train.jsonl",
        "validation": "data/gsm8k_valid.jsonl",
        "test": "data/gsm8k_test.jsonl",
    }
    if split not in path_map:
        raise ValueError(f"Unknown split '{split}', expected one of {list(path_map.keys())}")
    file_path = path_map[split]
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(file_path, "r") as f:
        lines = f.readlines()
    if max_samples:
        lines = lines[:max_samples]
    prefixes = [json.loads(l)["question"] for l in lines]
    return prefixes


def main():
    parser = argparse.ArgumentParser(description="Cache GSM8K next-token logits from local JSONL.")
    parser.add_argument("--model_name", type=str, default="gpt2",
                        help="HuggingFace model name for base LLM")
    parser.add_argument("--split", type=str, default="train",
                        choices=["train", "validation", "test"],
                        help="GSM8K split: train/validation/test")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Limit to first N examples for quick runs")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for LLM inference")
    parser.add_argument("--output_path", type=str, default="data/gsm8k_train_logits.pt",
                        help="Where to save the [N, V] logits tensor")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    args = parser.parse_args()

    set_seed(args.seed)
    print("starting")
    prefixes = load_prefixes(args.split, args.max_samples, args.seed)
    print(f"Loaded {len(prefixes)} prefixes for split '{args.split}'")

    # Prepare tokenizer & model on CPU
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_name).cpu()
    model.eval()

    # DataLoader to batch prefixes
    def collate_fn(batch):
        return tokenizer(batch, return_tensors="pt", padding=True)

    loader = DataLoader(
        prefixes,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )

    all_logits = []
    with torch.no_grad():
        for batch in loader:
            inputs = {k: v.cpu() for k, v in batch.items()}
            outputs = model(**inputs)
            # [batch_size, seq_len, vocab_size] -> take last token logits
            next_logits = outputs.logits[..., -1, :].cpu()
            all_logits.append(next_logits)

    all_logits = torch.cat(all_logits, dim=0)
    os.makedirs(os.path.dirname(args.output_path) or '.', exist_ok=True)
    torch.save(all_logits, args.output_path)
    print(f"Saved cached logits tensor with shape {all_logits.shape} to {args.output_path}")

if __name__ == "__main__":
    main()
