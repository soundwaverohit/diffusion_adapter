# Diffusion Adapter: Improving Reasoning Without Fine-tuning

## The Problem
Traditional language models often struggle with complex, multi-step reasoning tasks like mathematical problem solving. While fine-tuning can help, it's expensive and can degrade the model's general capabilities.

## The Solution: Diffusion as a "Reasoning Critic"
Instead of modifying the base model, we add a lightweight **diffusion critic** that acts as a "reasoning guide" during inference.

## How It Works

### 1. **Training the Critic** (Not the Base Model)
- Take the base model's token predictions (logits)
- Add random noise to create "corrupted" predictions
- Train a small neural network to predict this noise
- **Result**: The critic learns to recognize which token distributions are "natural" vs "unnatural"

### 2. **Inference with Guidance**
- During text generation, the base model suggests next tokens
- The diffusion critic scores each suggestion based on how "natural" it feels
- **MCTS exploration** combines base model scores with critic guidance
- **Result**: Better token choices that maintain reasoning coherence

## Why This Works

### **Preserves Base Model Knowledge**
- The original model's weights never change
- All learned knowledge and capabilities remain intact
- No risk of catastrophic forgetting

### **Adds Reasoning Awareness**
- The critic learns patterns of coherent reasoning from training data
- It can identify when the base model is about to make a reasoning error
- Guides generation toward more logical, step-by-step solutions

### **Efficient and Lightweight**
- Only trains a small critic network (~1MB vs full model fine-tuning)
- Can be trained on domain-specific reasoning tasks
- Easily swapped between different reasoning domains

## Example: Math Problem Solving
- **Before**: Base model might jump to conclusions or skip steps
- **After**: Diffusion critic guides toward step-by-step reasoning
- **Result**: More accurate mathematical solutions without changing the base model

## Key Insight
The diffusion critic doesn't teach new facts—it teaches **how to reason better** by learning what coherent, logical token sequences look like in the base model's own representation space.
