from importlib import import_module
import numpy as np
import torch
import logging
import itertools
import datetime
from logging import handlers
import copy

from utils.parser import *
from utils.misc import *
from utils.dummy import *
from utils.csvLogger import csvLogger
import tensorflow as tf


if __name__ == '__main__':

    #parse args
    args = parse_args()

    #load json
    params = DummyObject(file_name=args.json_file)
    params_extra = []

    # datetime.datetime.now().strftime("%d%B%Y")
    params.basic.path = params.basic.path+"/"+datetime.datetime.now().strftime("%H:%M:%S-%d%B%Y")+"/"+params.basic.prefix+"/"+str(params.basic.agent)+"/"+str(args.config_num)
    make_directory(params.basic.path)

    params.basic.prefix = str(args.run_num)

    #--logger
    logger = logging.getLogger('experiment')
    params.logger = logger
    params_extra.append("logger")
    logfile = params.basic.path+"/"+params.basic.prefix+"_log.txt"
    with open(logfile, 'w'):
        pass
    fh = logging.FileHandler(logfile)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter('%(levelname)-8s %(message)s'))
    logger.addHandler(fh)
    # ch = logging.handlers.logging.StreamHandler()
    # ch.setLevel(logging.DEBUG)
    # ch.setFormatter(logging.Formatter('%(levelname)-8s %(message)s'))
    # logger.addHandler(ch)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    logger.info("Start time: {},{}:{}".format(str(datetime.datetime.now().strftime("%d%B%Y")),str(datetime.datetime.now().strftime("%H")),str(datetime.datetime.now().strftime("%M"))))

    #--create and seed numpy random object
    params.np_random = np.random.RandomState(args.run_num+1)
    params_extra.append("np_random")
    # eval_random = np.random.RandomState(args.run_num+1)

    params.np_random_seed = args.run_num+1

    if params.basic.pytorch:
        tf.random.set_seed(seed=args.run_num+1)
        torch.manual_seed(seed=args.run_num+1)
        torch.set_num_threads(params.basic.num_threads)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        use_cuda = torch.cuda.is_available()
        params.dtype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
        params.dtype_long = torch.cuda.LongTensor if use_cuda else torch.LongTensor
        params.device = torch.device('cuda:'+str(args.id%params.basic.n_gpus) if use_cuda else 'cpu')
        logger.info("CUDA: {}".format(str(params.device)))
        params_extra.append("dtype")
        params_extra.append("dtype_long")
        params_extra.append("device")

    #--create environment
    environment_code = import_module("environment.{}".format(params.basic.environment))
    params.environment = environment_code.init(params)
    params_extra.append("environment")

    # temp_random = params.np_random
    # params.np_random = eval_random
    # params.eval_environment = environment_code.init(params)
    # params.eval_environment.print_mode = False
    # params_extra.append("eval_environment")
    # params.np_random = temp_random

    params.environment_code = environment_code
    params_extra.append("environment_code")

    #--figure out parameters
    feature_constructor_code = import_module("feature_constructor.{}".format(params.basic.feature_constructor))
    agent_code = import_module("agent.{}".format(params.basic.agent))

    eligible_feature_constructor_params = feature_constructor_code.get_params()
    eligible_agent_params = agent_code.get_params()

    lists = []
    get_lists(params.feature_constructor_params_orig,lists,eligible_feature_constructor_params)
    get_lists(params.agent_params_orig,lists,eligible_agent_params)
    combinations = [p for p in itertools.product(*lists)]

    #--Setting parameters based on config_num, give array id/number as config number for array jobs
    params.feature_constructor_params = copy.deepcopy(params.feature_constructor_params_orig)
    params.agent_params = copy.deepcopy(params.agent_params_orig)

    start_pos = 0
    set_params(combinations[args.config_num],start_pos,eligible_feature_constructor_params,params.feature_constructor_params)
    start_pos += len(eligible_feature_constructor_params)
    set_params(combinations[args.config_num],start_pos,eligible_agent_params,params.agent_params)

    #--create feature_constructor
    params.feature_constructor = feature_constructor_code.init(params)
    params_extra.append("feature_constructor")

    #--create agent
    params.agent = agent_code.init(params)
    params_extra.append("agent")

    if args.run_num == 0:
        params_temp = DummyObject()
        write_json(params.basic.path + "/metadata.json", params, DummyEncoder, params_extra, params_temp)

    # log files for results
    if params.environment.episodic:
        step_reward = []
        episode_reward = []
        episode_steps = []
    else:
        step_reward = []
        accumulated_reward = []

    step = 0
    num_episodes = 0
    current_step = 0
    current_reward = 0
    cumulative_reward = 0

    # csv logger
    csv_logger = csvLogger(str(params.basic.path+"/"+params.basic.prefix+"_agent.csv"))

    observation = params.environment.reset()
    action = params.agent.start(observation)

    logger.info("Run:{}".format(args.run_num))

    right_action_count = 0
    while True:
        after_step = params.environment.step(action)
        cumulative_reward += after_step[1]
        current_reward = after_step[1]
        step_reward.append(current_reward)

        # print(f"current_step: {current_step} , action: {action} , current_reward: {current_reward}")

        step += 1
        current_step += 1
        action_right = after_step[3]
        

        action = params.agent.step(after_step[0], after_step[1], after_step[2])
        if action is None:
            logger.info("Numerical error, invalid action")
            exit()
        observation = after_step[0]

        if(action or action_right):
            right_action_count += 1

        if params.environment.episodic:
            if after_step[2] or (params.basic.max_steps_episode!= -1 and current_step == params.basic.max_steps_episode):
                
                # updating weights using samples from replay buffer at the end of each episode
                print(f"Num episodes: {num_episodes+1}, number of steps: {current_step} current_reward: {cumulative_reward}")
                # print(f"Num episodes: {num_episodes+1}, number of steps remaining: {params.basic.max_steps_episode - current_step}")
                # params.agent.learn(batch_size=16)

                episode_reward.append(cumulative_reward)
                episode_steps.append(current_step)
                logger.info("Episode:{}, Steps:{}, Reward:{}, Return:{}, Remaining: {}".format(num_episodes,current_step,current_reward,cumulative_reward,(params.basic.num_steps-step)))
                num_episodes += 1
                if not after_step[2]:
                    observation = params.environment.reset()
                    action = params.agent.start(observation)
                cumulative_reward = 0
                current_step = 0
                right_action_count = 0
                
                # writing data for bsuite analysis
                # csv_logger.write(params.environment.bsuite_info())

        else:
            accumulated_reward.append(cumulative_reward)
            if step%1000 == 0:
                print(f"Num episodes: {step}, number of right_action_count: {right_action_count} , cumulative_reward: {cumulative_reward} observation: {after_step[0]}")
                logger.info("Step:{}, Reward:{}, Accurmulated reward:{}".format(current_step,current_reward, cumulative_reward))
                right_action_count = 0

        if params.environment.episodic:
            if params.basic.num_episodes != -1 and num_episodes == params.basic.num_episodes:
                break

        if params.basic.num_steps !=-1 and step == params.basic.num_steps:
            break

    # if params.eval_environment.episodic:
    #     if len(episode_reward) == 0:
    #         episode_reward.append(cumulative_reward)
    #         episode_steps.append(current_step)

    if params.environment.episodic:
        logger.info("Episode:{}, Steps:{}, Reward:{}, Accumulated reward:{}, Remaining: {}".format(num_episodes,current_step,current_reward,cumulative_reward,(params.basic.num_steps-step)))

    
    if params.environment.episodic:
        np.save(params.basic.path+"/"+params.basic.prefix+"_step_reward",np.array(step_reward))
        np.save(params.basic.path+"/"+params.basic.prefix+"_episode_reward",np.array(episode_reward))
        np.save(params.basic.path+"/"+params.basic.prefix+"_episode_steps",np.array(episode_steps))
    else:
        np.save(params.basic.path+"/"+params.basic.prefix+"_step_reward",np.array(step_reward))
        np.save(params.basic.path+"/"+params.basic.prefix+"_acc_reward",np.array(accumulated_reward))

    logger.info("End time: {},{}:{}".format(str(datetime.datetime.now().strftime("%d%B%Y")),str(datetime.datetime.now().strftime("%H")),str(datetime.datetime.now().strftime("%M"))))