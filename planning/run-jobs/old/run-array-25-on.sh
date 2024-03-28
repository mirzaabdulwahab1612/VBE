#!/bin/bash
#SBATCH --job-name=on_25x25
#SBATCH --output=output_log/%A%a.out
#SBATCH --error=output_log/%A%a.err
#SBATCH --array=0-8:1
#SBATCH --time=12:00:00
#SBATCH --account=def-whitem
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8000M
#SBATCH --mail-user=awahab.bscs16seecs@seecs.edu.pk
#SBATCH --mail-type=ALL

echo Running..$SLURM_ARRAY_TASK_ID

python online_experiment_eval_repeats.py  --run_num 0 --json_file parameters/deep_sea25x25_opi_evecest_multi_on_policy.json --id 0 --config_num $SLURM_ARRAY_TASK_ID