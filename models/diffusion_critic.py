"""
A lightweight diffusion-style critic that predicts added noise and scores candidate distributions.
"""
import torch
import torch.nn as nn

def add_noise(logits: torch.Tensor, noise_level: float = 0.1) -> torch.Tensor:
    """
    Adds Gaussian noise to a logit vector.
    """
    return logits + noise_level * torch.randn_like(logits)

class DiffusionCritic(nn.Module):
    def __init__(self, vocab_size: int, hidden_dim: int = 512):
        super().__init__()
        # A simple denoiser: MLP from [vocab_size] -> [vocab_size]
        self.denoiser = nn.Sequential(
            nn.Linear(vocab_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, vocab_size)
        )

    def forward(self, noisy_logits: torch.Tensor) -> torch.Tensor:
        """
        Predicts the noise residual given noisy logits.
        Input: noisy_logits [vocab_size]
        Output: predicted_noise [vocab_size]
        """
        return self.denoiser(noisy_logits)

    def compute_score(self, orig_logits: torch.Tensor, noisy_logits: torch.Tensor) -> float:
        """
        Computes a score based on how well the denoiser reconstructs the original logits.
        Returns a scalar: negative L2 distance (higher means better).
        """
        with torch.no_grad():
            pred_noise = self.forward(noisy_logits)
            denoised = noisy_logits - pred_noise
            dist = torch.norm(denoised - orig_logits, p=2)
        return (-dist).item()
