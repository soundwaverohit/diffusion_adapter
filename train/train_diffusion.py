# train/train_diffusion.py
"""
Script to train the DiffusionCritic on GSM8K prefixes by denoising base LLM logits,
loading questions directly from local JSONL files under data/.
"""
import argparse
import random
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

from models.base_llm import BaseLLM
from models.diffusion_critic import DiffusionCritic


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class GSM8LLocalDataset(Dataset):
    """
    Reads `data/gsm8k_{split}.jsonl` and exposes the 'question' field.
    """
    def __init__(self, split: str, max_samples: int = None, seed: int = 42):
        path_map = {
            "train": "data/gsm8k_train.jsonl",
            "validation": "data/gsm8k_valid.jsonl",
            "test": "data/gsm8k_test.jsonl",
        }
        if split not in path_map:
            raise ValueError(f"Unknown split '{split}', expected one of {list(path_map)}")
        file_path = path_map[split]
        # load all lines
        with open(file_path, "r") as f:
            lines = f.readlines()
        # optionally limit
        if max_samples:
            lines = lines[:max_samples]
        # parse JSON and extract questions
        self.prefixes = [json.loads(l)["question"] for l in lines]
        set_seed(seed)

    def __len__(self):
        return len(self.prefixes)

    def __getitem__(self, idx):
        return self.prefixes[idx]


def collate_fn(batch):
    # batch: list of prefix strings
    return batch


def train(args):
    # Setup device & seeds
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    # Instantiate models
    base_llm = BaseLLM(model_name=args.model_name, device=device)
    vocab_size = base_llm.model.config.vocab_size
    critic = DiffusionCritic(vocab_size, hidden_dim=args.hidden_dim).to(device)

    optimizer = torch.optim.Adam(critic.parameters(), lr=args.lr)
    mse_loss = torch.nn.MSELoss()

    # Prepare data from local JSONL
    dataset = GSM8LLocalDataset(
        split=args.split,
        max_samples=args.max_samples,
        seed=args.seed
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2
    )

    # Training loop
    for epoch in range(1, args.epochs + 1):
        total_loss = 0.0
        for batch in loader:
            # batch is a list of prefix strings
            for prefix in batch:
                # 1) Get original logits for next-token
                orig_logits = base_llm.full_logits(prefix).to(device)
                # 2) Add noise
                noise = torch.randn_like(orig_logits) * args.noise_level
                noisy_logits = orig_logits + noise
                # 3) Predict noise
                pred_noise = critic(noisy_logits)
                # 4) Compute loss: MSE between true and predicted noise
                loss = mse_loss(pred_noise, noise)
                # 5) Backprop
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

        avg_loss = total_loss / len(dataset)
        print(f"Epoch {epoch}/{args.epochs} — Avg Loss: {avg_loss:.4f}")

    # Save model
    torch.save(critic.state_dict(), args.output_path)
    print(f"Saved diffusion critic checkpoint to {args.output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train diffusion critic on GSM8K prefixes.")
    parser.add_argument("--model_name", type=str, default="gpt2",
                        help="HuggingFace model name for base LLM (e.g. gpt2)")
    parser.add_argument("--split", type=str, default="train",
                        choices=["train", "validation", "test"],
                        help="Which local JSONL split to use")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="If set, limit to first N samples for quick debugging")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Number of prefixes per batch")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate for Adam optimizer")
    parser.add_argument("--hidden_dim", type=int, default=512,
                        help="Hidden dimension for diffusion critic MLP")
    parser.add_argument("--noise_level", type=float, default=0.1,
                        help="Std dev of Gaussian noise added to logits")
    parser.add_argument("--output_path", type=str, default="models/diffusion_critic.pt",
                        help="Path to save trained critic checkpoint")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--device", type=str, default=None,
                        help="Device (cpu or cuda) for training")
    args = parser.parse_args()
    train(args)
