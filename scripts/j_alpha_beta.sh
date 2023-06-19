#!/bin/bash
#Set job requirements
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --partition=thin
#SBATCH --time=05:00:00
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=maurits.bos@gmail.com
#SBATCH --verbose

module load 2022
module load Anaconda3/2022.05
PACKAGE_DIR="$HOME"/code/irregular-object-packing

source "$HOME"/.bashrc
conda activate irop

# Run the data collector script
srun python "$PACKAGE_DIR"/irregular_object_packing/performance_analysis/data_collector.py\
 -N 1 \
 --description alpha_beta \
 --case alpha_beta \
 --output-dir "$HOME"/results/ \
 --input-dir "$HOME"/data/ 
