#!/bin/bash
#SBATCH --job-name=my-dlcluster-job
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=01:00:00
#SBATCH --output=output-%j.txt
#SBATCH --error=error-%j.txt

set -euo pipefail

echo "Job ID: ${SLURM_JOB_ID} | Node: $(hostname) | Partition: ${SLURM_JOB_PARTITION}"
nvidia-smi

python3 -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU: {torch.cuda.get_device_name(0)}')
x = torch.randn(1000, 1000, device='cuda')
y = torch.matmul(x, x)
print(f'Matrix multiply test: OK (shape: {y.shape})')
"