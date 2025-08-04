import argparse
import os
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.cuda.amp import GradScaler, autocast

from models.diffusion_critic import DiffusionCritic


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class CachedLogitsDataset(Dataset):
    def __init__(self, logits_path, indices):
        # logits: [N, V]
        self.logits = torch.load(logits_path)
        self.indices = indices
    def __len__(self):
        return len(self.indices)
    def __getitem__(self, idx):
        orig = self.logits[self.indices[idx]]  # [V]
        return orig


def get_scheduler(optimizer, num_warmup_steps, num_training_steps):
    # linear warmup + cosine decay
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return LambdaLR(optimizer, lr_lambda)


def train(args):
    device = args.device
    set_seed(args.seed)

    # Load cached logits
    all_logits = torch.load(args.logits_path)
    N = all_logits.size(0)

    # Train/val split
    val_size = int(N * args.val_ratio)
    train_size = N - val_size
    train_idx, val_idx = random_split(list(range(N)), [train_size, val_size], generator=torch.Generator().manual_seed(args.seed))

    train_ds = CachedLogitsDataset(args.logits_path, train_idx)
    val_ds   = CachedLogitsDataset(args.logits_path, val_idx)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # Model
    vocab_size = all_logits.size(1)
    critic = DiffusionCritic(vocab_size, hidden_dim=args.hidden_dim).to(device)

    optimizer = AdamW(critic.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = args.epochs * len(train_loader)
    scheduler = LambdaLR(optimizer, lambda step: min((step+1)/args.warmup_steps, 1.0))

    scaler = GradScaler()
    mse = nn.MSELoss()

    best_val = float('inf')
    patience = 0

    for epoch in range(1, args.epochs + 1):
        critic.train()
        total_loss = 0.0
        for step, orig in enumerate(train_loader):
            orig = orig.to(device)
            noise = torch.randn_like(orig) * args.noise_level
            noisy = orig + noise

            optimizer.zero_grad()
            with autocast():
                pred_noise = critic(noisy)
                loss = mse(pred_noise, noise)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_loss += loss.item()

        avg_train = total_loss / len(train_loader)

        # Validation
        critic.eval()
        val_loss = 0.0
        with torch.no_grad(), autocast():
            for orig in val_loader:
                orig = orig.to(device)
                noise = torch.randn_like(orig) * args.noise_level
                noisy = orig + noise
                pred = critic(noisy)
                val_loss += mse(pred, noise).item()
        avg_val = val_loss / len(val_loader)

        print(f"Epoch {epoch}/{args.epochs}  Train Loss: {avg_train:.4f}  Val Loss: {avg_val:.4f}")

        # Early stopping
        if avg_val < best_val - args.min_delta:
            best_val = avg_val
            patience = 0
            torch.save(critic.state_dict(), args.output_path)
            print(f"Saved best model checkpoint at epoch {epoch}")
        else:
            patience += 1
            if patience >= args.patience:
                print(f"Early stopping at epoch {epoch}")
                break

    print("Training complete.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--logits_path', type=str, default='data/gsm8k_train_logits.pt')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--hidden_dim', type=int, default=1024)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-2)
    parser.add_argument('--warmup_steps', type=int, default=500)
    parser.add_argument('--noise_level', type=float, default=0.1)
    parser.add_argument('--val_ratio', type=float, default=0.1)
    parser.add_argument('--min_delta', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=2)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--output_path', type=str, default='models/diffusion_critic_best.pt')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    train(args)
