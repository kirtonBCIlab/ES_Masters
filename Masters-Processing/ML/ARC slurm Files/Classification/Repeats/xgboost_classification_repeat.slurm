#!/bin/bash
#SBATCH --mail-user=emily.schrag@ucalgary.ca
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=24:0:0

export PATH=~/miniconda3/bin/envs/ml/bin:$PATH
echo $(which python)
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export VECLIB_MAXIMUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

DATA="/home/emily.schrag/testing_ml/features-Master.csv"
OUT="/home/emily.schrag/testing_ml/repeats/xgboost_results_repeat.csv"

python xgboost_classification_repeat.py $DATA $OUT
