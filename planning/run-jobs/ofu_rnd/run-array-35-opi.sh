#!/bin/bash
#SBATCH --job-name=ofu_rnd_l_35
#SBATCH --output=output_log/%A%a.out
#SBATCH --error=output_log/%A%a.err
#SBATCH --array=0-5:1
#SBATCH --time=23:59:00
#SBATCH --account=rrg-whitem
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=12000M
#SBATCH --mail-user=awahab.bscs16seecs@seecs.edu.pk
#SBATCH --mail-type=FAIL

echo Running..$SLURM_ARRAY_TASK_ID

python online_experiment_eval_repeats.py  --run_num 0 --json_file parameters/ofu_rnd/deep_sea35.json --id 0 --config_num $SLURM_ARRAY_TASK_ID