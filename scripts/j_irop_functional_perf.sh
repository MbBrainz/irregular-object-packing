#!/bin/bash
#Set job requirements
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --exclusive
#SBATCH --time=20:00:00
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=maurits.bos@gmail.com

module load 2022
module load Anaconda3/2022.05
PACKAGE_DIR="$HOME"/code/irregular-object-packing

source "$HOME"/.bashrc
conda activate irop
# Run the data collector script
cd $PACKAGE_DIR

srun python irregular_object_packing/performance_analysis/collect_profile_optimizer.py\
 --output-dir "$HOME"/results/\
 --input-dir "$HOME"/data/mesh/  
 