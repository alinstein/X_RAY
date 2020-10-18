#!/bin/bash
#SBATCH --gres=gpu:1       # Request GPU "generic resources"
#SBATCH --cpus-per-task=6  # Cores proportional to GPUs: 6 on Cedar, 16 on Graham.
#SBATCH --mem=32000M       # Memory proportional to GPUs: 32000 Cedar, 64000 Graham.
#SBATCH --time=0-1:25
#SBATCH --output=%N-%j.out

module load python/3.6
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip3 install --upgrade pip
pip3 install  -r /home/alinjose/scratch/alinjose/final_project/covid_19/requirements.txt

cd 
# Prepare data
cd $SLURM_TMPDIR/

mkdir public
cd public
mkdir CheXpert
cd CheXpert

cp /scratch/alinjose/alinjose/final_project/covid_19/datasets/chestXpert/CheXpert-v1.0-small.zip  $SLURM_TMPDIR/public/CheXpert                                                                                               
unzip -qq CheXpert-v1.0-small.zip

mv CheXpert-v1.0-small CheXpert-v1.0

cd $SLURM_TMPDIR
python /scratch/alinjose/alinjose/final_project/covid_19/Chexpert/bin/train.py /scratch/alinjose/alinjose/final_project/covid_19/Chexpert/config/example_PCAM.json logdir --num_workers 8 --device_ids "0"