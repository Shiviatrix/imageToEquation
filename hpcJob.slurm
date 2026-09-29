#!/bin/bash
#SBATCH --job-name=imageToEquation
#SBATCH --output=logs/hpc_%j.log
#SBATCH --error=logs/hpc_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=48:00:00

mkdir -p logs

if [ -d "venv" ]; then
    source venv/bin/activate
fi

python3 -u runHpcTrainer.py \
    --mode curated \
    --targetPsnr 26.0 \
    --targetLoss 0.015 \
    --maxLayers 36 \
    --patienceSteps 35 \
    --outputDir hpc_transcriptions
