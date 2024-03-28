#!/bin/bash
#SBATCH --job-name=opi_30
#SBATCH --output=output_log/%A%a.out
#SBATCH --error=output_log/%A%a.err
#SBATCH --array=0-11:1
#SBATCH --time=48:00:00
#SBATCH --account=def-whitem
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=12000M
#SBATCH --mail-user=awahab.bscs16seecs@seecs.edu.pk
#SBATCH --mail-type=ALL

echo Running..$SLURM_ARRAY_TASK_ID

python online_experiment_eval_repeats.py  --run_num 0 --json_file parameters/opi_NN/deep_sea30x30_opi.json --id 0 --config_num $SLURM_ARRAY_TASK_ID