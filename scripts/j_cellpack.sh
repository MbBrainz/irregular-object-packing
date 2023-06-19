#!/bin/bash
#Set job requirements
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --partition=exclusive
#SBATCH --time=08:00:00
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=maurits.bos@gmail.com

module load 2022
module load Anaconda3/2022.05
PACKAGE_DIR="$HOME"/code/irregular-object-packing

source "$HOME"/.bashrc
conda activate irop

# Run the data collector script
srun python "$PACKAGE_DIR"/irregular_object_packing/performance_analysis/data_collector.py\
 -N 1 \
 --description bc_max_cellpack \
 --case bloodcell_max_cellpack \
 --output-dir "$HOME"/results/ \
 --input-dir "$HOME"/data/ 