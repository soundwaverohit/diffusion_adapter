# infer/run_inference.py
"""
End-to-end inference script: generates answers on GSM8K using MCTS + diffusion critic,
loading questions directly from local JSONL files under data/.
"""
import argparse
import json
import os
import torch

from models.base_llm import BaseLLM
from models.diffusion_critic import DiffusionCritic
from models.mcts_explorer import MCTSExplorer
from infer.utils import load_config, setup_logging


def generate_text(prefix: str,
                  explorer: MCTSExplorer,
                  tokenizer,
                  max_len: int,
                  eos_token_id: int) -> str:
    """
    Generate text one token at a time via MCTSExplorer until EOS or max_len.
    """
    output = prefix
    for _ in range(max_len):
        next_id = explorer.search(output)
        if next_id is None:
            break
        output += tokenizer.decode([next_id])
        if next_id == eos_token_id:
            break
    return output


def load_local_split(split: str):
    """
    Load data/gsm8k_{split}.jsonl and return a list of dicts with
    'question' and 'answer' fields.
    """
    path_map = {
        "train": "data/gsm8k_train.jsonl",
        "validation": "data/gsm8k_valid.jsonl",
        "test": "data/gsm8k_test.jsonl",
    }
    if split not in path_map:
        raise ValueError(f"Unknown split '{split}', choose from {list(path_map)}")
    file_path = path_map[split]
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Expected {file_path} to exist")
    examples = []
    with open(file_path, "r") as f:
        for line in f:
            rec = json.loads(line)
            examples.append({
                "question": rec["question"],
                "answer": rec.get("answer", "")
            })
    return examples


def main():
    parser = argparse.ArgumentParser("Diffusion + MCTS Inference on GSM8K")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                        help="Path to YAML config file")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained diffusion critic checkpoint")
    parser.add_argument("--model_name", type=str, default="gpt2",
                        help="HuggingFace model name for base LLM")
    parser.add_argument("--split", type=str, default="test",
                        choices=["train", "validation", "test"],
                        help="Which local split to run (train/validation/test)")
    parser.add_argument("--max_len", type=int, default=128,
                        help="Maximum generation length per example")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="If set, only run inference on the first N examples")
    parser.add_argument("--output_path", type=str, default="results.jsonl",
                        help="File to write predictions (JSONL)")
    args = parser.parse_args()

    # Load config & logger
    cfg = load_config(args.config)
    logger = setup_logging(cfg.get("log_file", None))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # Load examples from local JSONL
    examples = load_local_split(args.split)
    if args.max_samples is not None:
        examples = examples[: args.max_samples]
    logger.info(f"Loaded {len(examples)} examples from split '{args.split}'")

    # Initialize Base LLM
    base_llm = BaseLLM(model_name=args.model_name, device=device)
    tokenizer = base_llm.tokenizer
    eos_token_id = tokenizer.eos_token_id

    # Load diffusion critic
    vocab_size = base_llm.model.config.vocab_size
    critic = DiffusionCritic(vocab_size,
                              hidden_dim=cfg["critic"]["hidden_dim"]).to(device)
    critic.load_state_dict(torch.load(args.checkpoint, map_location=device))
    critic.eval()

    # Initialize MCTS Explorer
    explorer = MCTSExplorer(base_llm, critic, cfg["mcts"])

    # Run inference
    results = []
    for ex in examples:
        question = ex["question"]
        answer_ref = ex["answer"]
        prompt = question + "\nA: "

        logger.info(f"Prompt: {prompt}")
        pred = generate_text(prompt, explorer, tokenizer, args.max_len, eos_token_id)
        logger.info(f"Pred : {pred}\nRef  : {answer_ref}\n")

        results.append({
            "question": question,
            "prediction": pred,
            "reference": answer_ref
        })

    # Save JSONL
    with open(args.output_path, "w") as fout:
        for rec in results:
            fout.write(json.dumps(rec) + "\n")
    logger.info(f"Saved {len(results)} results to {args.output_path}")


if __name__ == "__main__":
    main()
