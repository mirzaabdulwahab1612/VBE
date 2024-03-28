#!/bin/bash
#SBATCH --job-name=gravitar_bdqn
#SBATCH --output=output_log/%A%a.out
#SBATCH --error=output_log/%A%a.err
#SBATCH --array=0-2:1
#SBATCH --time=6-23:59
#SBATCH --account=rrg-whitem
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=36000M
#SBATCH --mail-user=wahab1@ualberta.ca
#SBATCH --mail-type=ALL

echo Running..$SLURM_ARRAY_TASK_ID
python atari_experiment_eval_repeats.py --env gravitar --agent_type 'bdqn' --bonus_scale 10 --num_rvecs 10 --runs $SLURM_ARRAY_TASK_ID