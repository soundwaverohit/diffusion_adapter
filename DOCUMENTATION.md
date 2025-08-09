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

#### **Detailed Training Process**
The diffusion critic training follows a specific pattern:

1. **Input Preparation**: For each training example (e.g., math problem), get the base model's logits for the next token
2. **Noise Addition**: Add Gaussian noise with controlled variance (e.g., noise_level = 0.1) to corrupt the logits
3. **Noise Prediction**: The critic network takes the noisy logits and tries to predict the exact noise that was added
4. **Loss Calculation**: Use Mean Squared Error (MSE) between predicted noise and actual added noise
5. **Backpropagation**: Update only the critic network weights, leaving the base model completely unchanged

**Why This Works**: By learning to predict noise, the critic develops an intuitive sense of what "clean" vs "corrupted" token distributions look like. This allows it to identify when the base model is about to make reasoning errors.

### 2. **Inference with Guidance**
- During text generation, the base model suggests next tokens
- The diffusion critic scores each suggestion based on how "natural" it feels
- **MCTS exploration** combines base model scores with critic guidance
- **Result**: Better token choices that maintain reasoning coherence

#### **MCTS: Smart Token Exploration**
The Monte Carlo Tree Search component works like a "smart planner" that:

1. **Explores Options**: Considers top-K token candidates (default: 10 tokens)
2. **Simulates Outcomes**: Runs multiple simulations (default: 20) to see how each choice affects reasoning
3. **Combines Scores**: Merges base model logits with diffusion critic guidance
4. **Balances Exploration vs Exploitation**: Uses UCB (Upper Confidence Bound) to try new paths while favoring promising ones

**Key Insight**: Instead of just picking the highest-scoring token, MCTS explores multiple reasoning paths and uses the diffusion critic to evaluate which paths lead to more coherent solutions.

#### **Inference: Step-by-Step Generation**
During text generation, the system works as follows:

1. **Token-by-Token Generation**: For each step, the base model suggests the next token
2. **Critic Evaluation**: The diffusion critic scores how "natural" each token choice feels
3. **MCTS Planning**: Explores multiple reasoning paths using both scores
4. **Best Path Selection**: Chooses the token that leads to the most coherent reasoning
5. **Repeat**: Continues until the complete solution is generated

**Result**: More logical, step-by-step solutions that maintain reasoning coherence throughout the entire generation process.

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

## Technical Training Details

### **Architecture**
- **Input**: Full vocabulary logits (typically 50K+ dimensions)
- **Hidden Layers**: Simple MLP with configurable hidden dimensions (default: 1024)
- **Output**: Predicted noise vector (same dimensions as input)
- **Activation**: ReLU between layers

### **Training Hyperparameters**
- **Learning Rate**: Typically 1e-4 (much lower than full model training)
- **Batch Size**: 16-32 examples per batch
- **Epochs**: 5-10 epochs (converges quickly due to simple task)
- **Noise Level**: 0.1 (standard deviation of Gaussian noise)
- **Loss Function**: MSE between predicted and actual noise

### **Data Flow**
```
Base Model Logits → Add Noise → Critic Predicts Noise → MSE Loss → Update Critic Only
```

### **Why Noise Prediction Works**
The key insight is that **clean, coherent reasoning** produces logit distributions that are "smooth" and follow predictable patterns. When noise is added, the critic learns to distinguish between:
- **Natural variations** (part of coherent reasoning)
- **Artificial corruption** (added noise that breaks reasoning flow)

This learned distinction allows the critic to guide the base model toward more coherent token sequences during inference.
