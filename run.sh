#!/bin/bash
#SBATCH --gres=gpu:2       # Request GPU "generic resources"
#SBATCH --cpus-per-task=6  # Cores proportional to GPUs: 6 on Cedar, 16 on Graham.
#SBATCH --mem=32000M       # Memory proportional to GPUs: 32000 Cedar, 64000 Graham.
#SBATCH --time=0-10:00
#SBATCH --output=%N-%j.out

module load python/3.6
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip3 install --upgrade pip
pip3 install  -r /home/alinjose/scratch/alinjose/final_project/covid_19/requirements.txt


# Prepare data
cd $SLURM_TMPDIR/
mkdir temp
cd temp

cp /scratch/alinjose/alinjose/final_project/covid_19/datasets/NIH.zip  $SLURM_TMPDIR/temp                                                                                               
unzip -qq NIH.zip

cd $SLURM_TMPDIR/
mkdir  dataset
cd dataset
mkdir images
cd $SLURM_TMPDIR/temp
find . -name '*.png' -exec mv {} $SLURM_TMPDIR/dataset/images \;

cp /scratch/alinjose/alinjose/final_project/covid_19/cxr8/dataset/* $SLURM_TMPDIR/dataset/ 
cd $SLURM_TMPDIR
mkdir outputs

mkdir savedModels
cp /scratch/alinjose/alinjose/final_project/covid_19/cxr8/savedModels/* $SLURM_TMPDIR/savedModels


python /scratch/alinjose/alinjose/final_project/covid_19/cxr8/train.py 

cp $SLURM_TMPDIR/savedModels/* /scratch/alinjose/alinjose/final_project/covid_19/cxr8/savedModels/
cp $SLURM_TMPDIR/outputs/* /scratch/alinjose/alinjose/final_project/covid_19/cxr8/outputs/