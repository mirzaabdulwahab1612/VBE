import copy
from importlib import import_module
import numpy as np
import torch
import logging
import itertools
import datetime
from logging import handlers
import tensorflow as tf

from utils.parser import *
from utils.misc import *
from utils.dummy import *

from offline_eval import *

if __name__ == '__main__':

    total_episode_reward = []
    total_step_reward = []
    total_episode_steps = []
    date_time = datetime.datetime.now().strftime("%H:%M:%S-%d%B%Y")

    #parse args
    args = parse_args()
    #load json
    params = DummyObject(file_name=args.json_file)
    

    # grid_size_str = str(params.environment_params.grid_size) + "x" + str(params.environment_params.grid_size)

    params.basic.path = params.basic.path+"/"+str(params.basic.agent)+"/"+params.basic.prefix+"/"+str(args.config_num)
    # params.basic.path = params.basic.path+"/"+grid_size_str+"/"+update_type+"/"+date_time+"/"+params.basic.prefix+"/"+str(params.basic.agent)+"/"+str(args.config_num)
    make_directory(params.basic.path)
    params_extra = []
    params.basic.prefix = str(args.run_num)
    num_repeats = params.basic.num_repeats

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

    mean_episode_reward_time_repeat = []
    number_of_completed_episodes_time_repeat = []
    eval_cumulative_reward_time_repeat = []

    for repeat in range(num_repeats):
        print(f"Repeat: {repeat}")
        logger.info("Repeapt Number: {} , Start time: {},{}:{}".format(repeat, str(datetime.datetime.now().strftime("%d%B%Y")), str(datetime.datetime.now().strftime("%H")),str(datetime.datetime.now().strftime("%M"))))
        params.repeat = repeat
        #--create and seed numpy random object
        params.np_random = np.random.RandomState(args.run_num+1+repeat)
        params_extra.append("np_random")
        params.np_random_seed = args.run_num + repeat + 1

        if params.basic.pytorch:
            tf.random.set_seed(seed=args.run_num+1+repeat)
            torch.manual_seed(seed=args.run_num+1+repeat)
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

        if args.run_num + repeat == 0:
            params_temp = DummyObject()
            if args.config_num == 0:
                write_json(params.basic.path+"/../metadata.json", params, DummyEncoder, params_extra, params_temp)
            write_json(params.basic.path+"/metadata.json", params, DummyEncoder, params_extra, params_temp)
            params_extra = []

       
        step_reward = []
        if params.environment.episodic:
            episode_reward = []
            episode_steps = []

        if params.agent_params.offline_eval:
            eval_cumulative_reward_time = []
            if params.environment.episodic:
                mean_episode_reward_time = []
                number_of_completed_episodes_time = []

        step = 0
        num_episodes = 0
        current_step = 0
        current_reward = 0
        cumulative_reward = 0

        observation = params.environment.reset()
        action = params.agent.start(observation)

        logger.info("Run:{}".format(args.run_num))
        eval_count = 0
        while True:
            after_step = params.environment.step(action)
            cumulative_reward += after_step[1]
            current_reward = after_step[1]
            step_reward.append(current_reward)
            step += 1
            current_step += 1

            action = params.agent.step(after_step[0], after_step[1], after_step[2])
            if action is None:
                logger.info("Numerical error, invalid action")
                exit()
            observation = after_step[0]

            if params.environment.episodic:
                if after_step[2] or ((params.basic.max_steps_episode != -1) and (current_step == params.basic.max_steps_episode)):
                    episode_reward.append(cumulative_reward)
                    episode_steps.append(current_step)
                    logger.info("Episode:{}, Steps:{}, Reward:{}, Return:{}, Remaining: {}".format(num_episodes,current_step,current_reward,cumulative_reward,(params.basic.num_steps-step)))
                    num_episodes += 1
                    if not after_step[2]:
                        observation = params.environment.reset()
                        action = params.agent.start(observation)
                    cumulative_reward = 0
                    current_step = 0

            else:
                if step%1000 == 0:
                    logger.info("Step:{}, Reward:{}, Accurmulated reward:{}".format(current_step,current_reward, cumulative_reward))


            if params.environment.episodic:
                if params.basic.num_episodes != -1 and num_episodes == params.basic.num_episodes:
                    # print(f"save model at step: {step}")
                    # model_save_path = params.basic.path+"/"+str(params.repeat)+"_"
                    # params.agent.save_stats(model_save_path)
                    break

            if params.basic.num_steps !=-1 and step == params.basic.num_steps:
                break
            
            # if params.agent_params.model_save_freq > 0 and (params.agent.time_step % params.agent_params.model_save_freq == 0):
            #     print(f"save model at step: {step}")
            #     model_save_path = params.basic.path+"/models/" + str(params.repeat) + "/"
            #     make_directory(model_save_path)
            #     params.agent.save_policy(model_save_path)

            if params.agent_params.offline_eval and (params.agent.time_step % params.agent_params.model_save_freq == 0):
                eval_count += 1
                if params.environment.episodic:
                    mean_episode_reward, number_of_completed_episodes = eval_policy(params.agent, params, eval_count)
                    mean_episode_reward_time.append(mean_episode_reward)
                    number_of_completed_episodes_time.append(number_of_completed_episodes)
                else:
                    eval_cumulative_reward = eval_policy(params.agent, params, eval_count)
                    eval_cumulative_reward_time.append(eval_cumulative_reward)

        if params.agent_params.offline_eval:
            eval_cumulative_reward_time = np.array(eval_cumulative_reward_time)
            eval_cumulative_reward_time_repeat.append(eval_cumulative_reward_time)
            if params.environment.episodic:
                mean_episode_reward_time = np.array(mean_episode_reward_time)
                number_of_completed_episodes_time = np.array(number_of_completed_episodes_time)
                mean_episode_reward_time_repeat.append(mean_episode_reward_time)
                number_of_completed_episodes_time_repeat.append(number_of_completed_episodes_time)
        
        if params.environment.episodic:
            logger.info("Episode:{}, Steps:{}, Reward:{}, Accumulated reward:{}, Remaining: {}".format(num_episodes,current_step,current_reward,cumulative_reward,(params.basic.num_steps-step)))

        np.save(params.basic.path+"/"+params.basic.prefix+str(repeat)+"_step_reward",np.array(step_reward))
        if params.environment.episodic:
            np.save(params.basic.path+"/"+params.basic.prefix+str(repeat)+"_episode_reward",np.array(episode_reward))
            np.save(params.basic.path+"/"+params.basic.prefix+str(repeat)+"_episode_steps",np.array(episode_steps))


        logger.info("End time: {},{}:{}".format(str(datetime.datetime.now().strftime("%d%B%Y")),str(datetime.datetime.now().strftime("%H")),str(datetime.datetime.now().strftime("%M"))))

        if params.environment.episodic:
            total_episode_reward.append(episode_reward)
            total_episode_steps.append(episode_steps)
            total_step_reward.append(step_reward)
        else:
            total_step_reward.append(step_reward)

    if params.environment.episodic:
        total_episode_reward = np.array(total_episode_reward, dtype=object)
        total_episode_steps = np.array(total_episode_steps, dtype=object)
        total_step_reward = np.array(total_step_reward, dtype=object)
    else:
        total_step_reward = np.array(total_step_reward, dtype=object)

    if params.agent_params.offline_eval:
        if params.environment.episodic:
            eval_cumulative_reward_time_repeat = np.array(eval_cumulative_reward_time_repeat, dtype=object)
            mean_episode_reward_time_repeat = np.array(mean_episode_reward_time_repeat, dtype=object)
            number_of_completed_episodes_time_repeat = np.array(number_of_completed_episodes_time_repeat, dtype=object)
        else:
            eval_cumulative_reward_time_repeat = np.array(eval_cumulative_reward_time_repeat, dtype=object)
            
    if params.environment.episodic:
        np.save(params.basic.path+"/"+params.basic.prefix+"_total_step_reward",np.array(total_step_reward))
        np.save(params.basic.path+"/"+params.basic.prefix+"_total_episode_reward",np.array(total_episode_reward))
        np.save(params.basic.path+"/"+params.basic.prefix+"_total_episode_steps",np.array(total_episode_steps))
    else:
        np.save(params.basic.path+"/"+params.basic.prefix+"_total_step_reward",np.array(total_step_reward))

    if params.agent_params.offline_eval:
        if params.environment.episodic:
            np.save(params.basic.path+"/"+params.basic.prefix+"_total_offline_cumulative_reward",np.array(eval_cumulative_reward_time_repeat))
            np.save(params.basic.path+"/"+params.basic.prefix+"_total_offline_mean_episode_reward",np.array(mean_episode_reward_time_repeat))
            np.save(params.basic.path+"/"+params.basic.prefix+"_total_offline_number_of_completed_episodes",np.array(number_of_completed_episodes_time_repeat))
        else:
            np.save(params.basic.path+"/"+params.basic.prefix+"_total_offline_cumulative_reward",np.array(eval_cumulative_reward_time_repeat))