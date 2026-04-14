#!/bin/bash
# specify a partition
#SBATCH -p {partition}
# specify number of nodes
#SBATCH -N 1
# specify the job name
#SBATCH -J nuke_{partition}
# specify the filename to be used for writing output
#SBATCH -o {slurm_logs_folder}/nuke_{partition}.%N.%j.out
# specify the filename for stderr
#SBATCH -e {slurm_logs_folder}/nuke_{partition}.%N.%j.err

# Nuclear option to clean up ghost processes. Will kill every process belonging to user
pkill -9 -u $USER