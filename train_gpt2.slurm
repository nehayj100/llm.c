#!/bin/bash
#SBATCH --job-name=gpt2_train
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1  
#SBATCH --cpus-per-task=32
#SBATCH --time=26:00:00              #Request 26 hours (2 extra hours)
#SBATCH --mem=128GB                  #Request 128GB per node
#SBATCH --partition=gpu              #Request the GPU partition/queue
#SBATCH --gres=gpu:a100:1            #Request one A100 GPU to use

#SBATCH --output=gpt2_train.%j.log            #Redirect stdout/err to file

# Run the training script
python train_gpt2.py --input_bin="dev/data/fineweb10B/fineweb_train_*.bin" \
                      --input_val_bin="dev/data/fineweb10B/fineweb_val_*.bin" \
                      --output_dir="log124M" \
                      --batch_size=2 \
                      --write_tensors=0 \
                      --sequence_length=1024 \
                      --total_batch_size=32768 \
                      --zero_stage=1 \
                      --weight_decay=0.1 \
                      --learning_rate=0.0006 \
                      --learning_rate_decay_frac=0.0 \
                      --warmup_iters=700 \
                      --val_loss_every=250 \
                      --sample_every=20000 \
                      --device="cuda" \
                      --num_iterations=5000 \
                      --compile=1 \
                      --model="d12" \
                      --dtype=bfloat16 \
                      --tensorcores=1 