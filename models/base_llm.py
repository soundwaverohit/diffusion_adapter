"""
Wrapper around a Hugging Face causal LLM that exposes top-K and full logits for the next token.
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class BaseLLM:
    def __init__(self, model_name: str = "gpt2", device: str = None):
        # Choose device automatically if not provided
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Ensure a pad token exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def top_k_logits(self, prefix: str, K: int):
        """
        Given a text prefix, returns the top-K logits and token IDs for the next token.
        Returns:
          - topk_values: torch.FloatTensor of shape [K]
          - topk_indices: torch.LongTensor of shape [K]
        """
        inputs = self.tokenizer(prefix, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits[..., -1, :]  # [1, vocab_size]
        topk_vals, topk_idx = torch.topk(logits, K, dim=-1)  # [1, K]
        return topk_vals.squeeze(0), topk_idx.squeeze(0)

    def full_logits(self, prefix: str):
        """
        Returns the full unnormalized logits vector for the next token.
        """
        inputs = self.tokenizer(prefix, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            return outputs.logits[..., -1, :].squeeze(0)  # [vocab_size]
