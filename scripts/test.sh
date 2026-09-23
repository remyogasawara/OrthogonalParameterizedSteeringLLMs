#!/bin/bash
#SBATCH --account=utahdb-gpu-np
#SBATCH --partition=utahdb-gpu-np
#SBATCH --qos=utahdb-gpu-np
#SBATCH --time=02:00:00
#SBATCH --ntasks=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1              # request 1 GPU (adjust if needed)
#SBATCH -o slurmjob-%j.out-%N
#SBATCH -e slurmjob-%j.err-%N

# set up scratch directory
SCRDIR=/scratch/general/vast/$USER/$SLURM_JOB_ID
mkdir -p $SCRDIR

# copy input files and move over to the scratch directory
cp inputfile.csv myscript.r $SCRDIR
cd $SCRDIR

# load modules (adjust depending on GPU framework you use)
module load R/4.4.0
# Optional: load CUDA if needed
# module load cuda

# run your script
Rscript myscript.r inputfile.csv

# copy output to your home directory and clean up
cp outputfile.csv $HOME
cd $HOME
rm -rf $SCRDIR