# diffusion_adapter
A model alignment adapter to align tokens at inference level using diffusion models 


## Commands:



srun -p griz256 python3 -m train.train_diffusion   --model_name gpt2   --split 
train   --epochs 5   --batch_size 16   --lr 1e-4   --hidden_dim 1024   --noise_level 0.1   --device cuda   --output_path models/diffusion_critic.pt

srun -p griz256 python3 -m infer.run_inference    --config configs/default.yaml    --checkpoint models/diffusion_critic.pt    --model_name gpt2    --split test    --max_len 128    --max_samples 20    --output_path results_small.jsonl

srun -p griz256 python3 evaluation/evaluate_results.py   --baseline baseline.jsonl   --enhanced results_small.jsonl   --top_n 20
