# VBE

## Running Atari experiments
python atari_experiment_eval_repeats.py --env breakout --agent_type 'opi' --bonus_scale 10 --num_rvecs 1 --runs 0

python atari_experiment_eval_repeats.py --env breakout --agent_type 'bdqn' --bonus_scale 10 --num_rvecs 10 --runs 0

python atari_experiment_eval_repeats.py --env breakout --agent_type 'vbeacb' --bonus_scale 10 --num_rvecs 128 --runs 0

python atari_experiment_eval_repeats.py --env breakout --agent_type 'vbernd' --bonus_scale 10 --runs 0

agent_type = ['opi', 'bdqn', 'vbeacb', 'vbernd']
env = [breakout, pong, qbert, pitfall, privateeye, gravitar]
bonus_scale = [1, 3, 10] for bonus scale
runs = [0, 1, 2, ...] for random repeats