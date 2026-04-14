#!/bin/bash
# specify a partition
#SBATCH -p {partition}
# specify number of nodes
#SBATCH -N 1
# specify the job name
#SBATCH -J {job_name}
# specify the filename to be used for writing output
#SBATCH -o {slurm_logs_folder}/slurm_{job_name}.%N.%j.out
# specify the filename for stderr
#SBATCH -e {slurm_logs_folder}/slurm_{job_name}.%N.%j.err
# exclude nodes
#SBATCH --exclude=mscluster[{node_excludes}]
echo ------------------------------------------------------
echo -n 'Job is running on node ' $SLURM_JOB_NODELIST
echo ------------------------------------------------------
echo SLURM: sbatch is running on $SLURM_SUBMIT_HOST
echo SLURM: job ID is $SLURM_JOB_ID
echo SLURM: submit directory is $SLURM_SUBMIT_DIR
echo SLURM: number of nodes allocated is $SLURM_JOB_NUM_NODES
echo SLURM: number of cores is $SLURM_NTASKS
echo SLURM: job name is $SLURM_JOB_NAME
echo ------------------------------------------------------
cd $SLURM_SUBMIT_DIR
pwd

# Most extreme clean up method - will kill every process of mine on that node
cleanup() {
    echo "Nuclear option triggered by termination signal. Killing all processes for user $USER on this node..."
    pkill -9 -u $USER
}

trap cleanup TERM INT XCPU EXIT

# Required if PyGame is called to stop it from crashing
export SDL_VIDEODRIVER="dummy"

{setup_commands}

source activate {conda_env}
{run_commands}
wait

echo All processes finished

{finish_commands}

echo Cleaned up and done :P
