from agent.atari_agents.OPI_outofsample_NN import *
from agent.atari_agents.bdqnp_updated import *
from agent.atari_agents.VBE_RND_Atari import *
from agent.atari_agents.VBE_ACB_Atari import *

from environment.atari import * 
import configparser
from torch.multiprocessing import Pipe
import pdb
import argparse
from tensorboardX import SummaryWriter
import numpy as np
import hashlib
import gym
import os

config = configparser.ConfigParser()
config.read('config.conf')
default = 'DEFAULT'
default_config = config[default]

def main():
    if args.env == 'montezuma': default_config['EnvID'] = 'MontezumaRevengeNoFrameskip-v4'
    if args.env == 'asterix': default_config['EnvID'] = 'AsterixNoFrameskip-v4'
    if args.env == 'breakout': default_config['EnvID'] = 'BreakoutNoFrameskip-v4'
    if args.env == 'pong': default_config['EnvID'] = 'PongNoFrameskip-v4'
    if args.env == 'beamrider': default_config['EnvID'] = 'BeamRiderNoFrameskip-v4'
    if args.env == 'qbert': default_config['EnvID'] = 'QbertNoFrameskip-v4'
    if args.env == 'riverraid': default_config['EnvID'] = 'RiverraidNoFrameskip-v4'
    if args.env == 'seaquest': default_config['EnvID'] = 'SeaquestNoFrameskip-v4'
    if args.env == 'spaceinvaders': default_config['EnvID'] = 'SpaceInvadersNoFrameskip-v4'
    if args.env == 'pitfall': default_config['EnvID'] = 'PitfallNoFrameskip-v4'
    if args.env == 'gravitar': default_config['EnvID'] = 'GravitarNoFrameskip-v4'
    if args.env == 'solaris': default_config['EnvID'] = 'SolarisNoFrameskip-v4'
    if args.env == 'privateeye': default_config['EnvID'] = 'PrivateEyeNoFrameskip-v4'
    if args.env == 'venture': default_config['EnvID'] = 'VentureNoFrameskip-v4'


    env_id = default_config['EnvID']
    env_type = default_config['EnvType']

    print({section: dict(config[section]) for section in config.sections()})

    if env_type == 'atari':
        env = gym.make(env_id)
    else:
        raise NotImplementedError
    input_size = env.observation_space.shape  # 4
    output_size = env.action_space.n  # 2
    print(f"input_size: {input_size}, output_size: {output_size}")
    env.close()
    sz = 84

    if 'Breakout' in env_id:
        output_size -= 1

    is_load_model = False
    is_render = False
    model_path = 'models/{}.model'.format(env_id)
    predictor_path = 'models/{}.pred'.format(env_id)
    target_path = 'models/{}.target'.format(env_id)

    use_cuda = default_config.getboolean('UseGPU')
    mini_batch = int(default_config['MiniBatch'])
    learning_rate = float(default_config['LearningRate'])
    gamma = float(default_config['Gamma'])
    sticky_action = default_config.getboolean('StickyAction')
    action_prob = float(default_config['ActionProb'])
    life_done = default_config.getboolean('LifeDone')
    pre_obs_norm_step = int(default_config['ObsNormStep'])
    bonus_scale = args.bonus_scale
    num_rvecs = args.num_rvecs

    agent_type = args.agent_type
    runs = args.runs
    # new change: save results in a folder with bonus scale
    writer = SummaryWriter(f'runs/{agent_type}/{env_id}/{bonus_scale}/{runs}/')

    torch.manual_seed(seed=runs)
    torch.backends.cudnn.deterministic = True
    base_save_path = f"results/{agent_type}/{args.env}/{bonus_scale}/"
    if agent_type == "opi":
        agent = OPI(
            input_size,
            output_size,
            gamma,
            learning_rate,
            np_random = np.random.RandomState(runs),
            target_update_freq = 10000,
            update_network_freq = 4,
            num_updates = 1,
            mini_batch_size = mini_batch,
            device = torch.device('cuda' if use_cuda else 'cpu'),
            num_rvecs = num_rvecs,
            p = bonus_scale,
            target_polic_type = "greedy",
            # new change: not updating all rvfs
            rvfs_update_all = False,
            model_save_path = f"{base_save_path}/{runs}/models/"
        )
    elif agent_type == "vbeacb":
        agent = VBEACB(
            input_size,
            output_size,
            gamma,
            learning_rate,
            np_random = np.random.RandomState(runs),
            target_update_freq = 10000,
            update_network_freq = 4,
            num_updates = 1,
            mini_batch_size = mini_batch,
            device = torch.device('cuda' if use_cuda else 'cpu'),
            num_rvecs = num_rvecs,
            p = bonus_scale,
            target_polic_type = "greedy",
            model_save_path = f"{base_save_path}/{runs}/models/"
        )
    elif agent_type == "vbernd":
        agent = VBERND(
            input_size,
            output_size,
            gamma,
            learning_rate,
            np_random = np.random.RandomState(runs),
            target_update_freq = 10000,
            update_network_freq = 4,
            num_updates = 1,
            mini_batch_size = mini_batch,
            device = torch.device('cuda' if use_cuda else 'cpu'),
            p = bonus_scale,
            target_polic_type = "greedy",
            model_save_path = f"{base_save_path}/{runs}/models/"
        )
    elif agent_type == "bdqn":
        agent = BootstrappedDqn(
            input_size,
            output_size,
            gamma,
            learning_rate,
            np_random_seed = runs,
            np_random = np.random.RandomState(runs),
            target_update_period = 10000,
            sgd_period = 4,
            mini_batch_size = mini_batch,
            device = torch.device('cuda' if use_cuda else 'cpu'),
            mask_prob = 0.5,
            noise_scale = 0,
            num_heads = num_rvecs,
            prior_scale = bonus_scale,
            buffer_size = 100000,
            model_save_path = f"{base_save_path}/{runs}/models/"
            )

    if default_config['EnvType'] == 'atari':
        env_type = AtariEnvironment
    else:
        raise NotImplementedError

    if is_load_model:
        print('load model...')
        if use_cuda:
            agent.model.load_state_dict(torch.load(model_path))
            agent.rnd.predictor.load_state_dict(torch.load(predictor_path))
            agent.rnd.target.load_state_dict(torch.load(target_path))
        else:
            agent.model.load_state_dict(torch.load(model_path, map_location='cpu'))
            agent.rnd.predictor.load_state_dict(torch.load(predictor_path, map_location='cpu'))
            agent.rnd.target.load_state_dict(torch.load(target_path, map_location='cpu'))
        print('load finished!')

    environment = AtariEnvironment(env_id, is_render, sticky_action=sticky_action, p=action_prob,
                        life_done=life_done)

    states = np.zeros([4, sz, sz])

    sample_episode = 0
    sample_rall = 0
    sample_step = 0
    sample_i_rall = 0
    global_update = 0
    global_step = 0

    allStates = set()
    allStatesFull = set()
    #allStates = set()
    #allStatesFull = set()

    # initialize the environment and agent
    observation = environment.reset()
    action = agent.start(np.float32(observation) / 255.)

    step_reward = []
    episode_reward = []
    steps_per_episode = []
    # unique_states_count = []
    # unique_full_states_count = []
    # unique_states_flag = []
    # unique_full_states_flag = []
    # val_data = [[0] * output_size]
    # unc_data = [[0] * output_size]

    # new change: save results in a folder with bonus scale
    base_save_path = f"results/{agent_type}/{args.env}/{bonus_scale}/"
    if not os.path.exists(base_save_path):
        try:
            os.makedirs(base_save_path)
        except:
            assert (os.path.exists(base_save_path))

    np.save(f"{base_save_path}{runs}_step_reward.npy", step_reward)
    np.save(f"{base_save_path}{runs}_episode_reward.npy", episode_reward)
    np.save(f"{base_save_path}{runs}_steps_per_episode.npy", steps_per_episode)
    # np.save(f"{base_save_path}{runs}_unique_states_count.npy", unique_states_count)
    # np.save(f"{base_save_path}{runs}_unique_full_states_count.npy", unique_full_states_count)
    # np.save(f"{base_save_path}{runs}_unique_states_flag.npy", unique_states_flag)
    # np.save(f"{base_save_path}{runs}_unique_full_states_flag.npy", unique_full_states_flag)
    # np.save(f"{base_save_path}{runs}_val_data.npy", val_data)
    # np.save(f"{base_save_path}{runs}_unc_data.npy", unc_data)

    while True:
        global_update += 1

        states, rewards, dones, real_dones, log_rewards, full_states = environment.step(action)
        # l1 = len(allStates)
        # l2 = len(allStatesFull)
        allStates.add(hashlib.sha1(states[3,:,:]).hexdigest())
        allStatesFull.add(hashlib.sha1(states).hexdigest())
        # l1_ = len(allStates)
        # l2_ = len(allStatesFull)

        # if l1_ > l1:
        #     newStateFlag = 1
        # else:
        #     newStateFlag = 0

        # if l2_ > l2:
        #     newFullStateFlag = 1
        # else:
        #     newFullStateFlag = 0

        # unique_states_count.append(len(allStates))
        # unique_full_states_count.append(len(allStatesFull))
        # unique_states_flag.append(newStateFlag)
        # unique_full_states_flag.append(newFullStateFlag)


        if agent_type == "opi" or agent_type == "vbernd" or agent_type == "vbeacb":
            action, val, uncertainty_bonus, value = agent.step(np.float32(states) / 255., rewards, dones)
        else:
            action = agent.step(np.float32(states) / 255., rewards, dones)

        sample_rall += log_rewards
        step_reward.append(rewards)

        # if global_update > 1001:
        #     val_data.append(val.data.cpu().numpy())
        #     unc_data.append(uncertainty_bonus.data.cpu().numpy())
        # else:
        #     val_data.append(np.array(val))
        #     unc_data.append(np.array(uncertainty_bonus))

        sample_step += 1
        if real_dones:
            sample_episode += 1

            episode_reward.append(sample_rall)
            steps_per_episode.append(sample_step)

            writer.add_scalar('data/reward_per_epi', sample_rall, sample_episode)
            writer.add_scalar('data/reward_per_rollout', sample_rall, global_update)
            writer.add_scalar('data/step', sample_step, sample_episode)
            sample_rall = 0
            sample_step = 0
            sample_i_rall = 0

        if global_update % 1000 == 0:
            print('stats', global_update, np.mean(rewards), flush=True)
            print('progress', global_update, sample_episode, sample_step, len(allStates), len(allStatesFull), flush=True)

        if global_update % 1000 == 0:
            temp_ = np.load(f"{base_save_path}{runs}_step_reward.npy")
            temp_ = np.append(temp_, step_reward)
            np.save(f"{base_save_path}{runs}_step_reward.npy", temp_)
            del temp_

            # temp_ = np.load(f"{base_save_path}{runs}_unique_states_count.npy")
            # temp_ = np.append(temp_, unique_states_count)
            # np.save(f"{base_save_path}{runs}_unique_states_count.npy", temp_)

            # temp_ = np.load(f"{base_save_path}{runs}_unique_full_states_count.npy")
            # temp_ = np.append(temp_, unique_full_states_count)
            # np.save(f"{base_save_path}{runs}_unique_full_states_count.npy", temp_)

            # temp_ = np.load(f"{base_save_path}{runs}_val_data.npy")
            # temp_ = np.vstack((temp_, val_data))
            # np.save(f"{base_save_path}{runs}_val_data.npy", temp_)

            # temp_ = np.load(f"{base_save_path}{runs}_unc_data.npy")
            # temp_ = np.vstack((temp_, unc_data))
            # np.save(f"{base_save_path}{runs}_unc_data.npy", temp_)

            # temp_ = np.load(f"{base_save_path}{runs}_unique_states_flag.npy")
            # temp_ = np.append(temp_, unique_states_flag)
            # np.save(f"{base_save_path}{runs}_unique_states_flag.npy", temp_)

            # temp_ = np.load(f"{base_save_path}{runs}_unique_full_states_flag.npy")
            # temp_ = np.append(temp_, unique_full_states_flag)
            # np.save(f"{base_save_path}{runs}_unique_full_states_flag.npy", temp_)

            # unique_states_count = []
            # unique_full_states_count = []
            # val_data = []
            # unc_data = []
            # unique_states_flag = []
            # unique_full_states_flag = []
            step_reward = []

        if sample_episode % 100 == 0:
            temp_ = np.load(f"{base_save_path}{runs}_episode_reward.npy")
            temp_ = np.append(temp_, episode_reward)
            np.save(f"{base_save_path}{runs}_episode_reward.npy", temp_)
            episode_reward = []
            del temp_

        if sample_episode % 100 == 0:
            temp_ = np.load(f"{base_save_path}{runs}_steps_per_episode.npy")
            temp_ = np.append(temp_, steps_per_episode)
            np.save(f"{base_save_path}{runs}_steps_per_episode.npy", temp_)
            steps_per_episode = []
            del temp_           

        # if global_update % 1000 == 0 and agent_type == "opi":

        #     uncertainty_bonus_ = {}
        #     for index, element in enumerate(uncertainty_bonus):
        #         uncertainty_bonus_[str(index)] = element
        #     writer.add_scalars('data/uncertainty_bonus', uncertainty_bonus_, global_update)
        
        #     value_ = {}
        #     for index, element in enumerate(value):
        #         value_[str(index)] = element
        #     writer.add_scalars('data/value', value_, global_update)

        #     val_ = {}
        #     for index, element in enumerate(val):
        #         val_[str(index)] = element
        #     writer.add_scalars('data/val', val_, global_update)

        # if global_step % 100 == 0:
        #     print('Now Global Step :{}'.format(global_step))
        #     torch.save(agent.model.state_dict(), model_path)
        #     torch.save(agent.rnd.predictor.state_dict(), predictor_path)
        #     torch.save(agent.rnd.target.state_dict(), target_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', help='', type=str, default='breakout')
    parser.add_argument('--numAux', help='ensemble size', type=int, default=128)
    parser.add_argument('--agent_type', help='ddqn-egreedy or opi', type=str, default='opi')
    parser.add_argument('--runs', help='number of runs in parallel', type=int, default=0)
    parser.add_argument('--bonus_scale', help='Scale for VBE', type=int, default=10)
    parser.add_argument('--num_rvecs', help='Scale for VBE', type=int, default=1)

    args = parser.parse_args()

    main()
