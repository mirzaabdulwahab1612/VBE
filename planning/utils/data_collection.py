class DataCollector():

    def __init__(self, params):
        self.environment = params.environment
        self.agent = params.agent

        self.num_data = params.data_collector_params.num_data
        self.max_data_size = params.data_collector_params.max_data_size
        self.other_policy = params.data_collector_params.other_policy
        self.on_policy = params.data_collector_params.on_policy

        self.max_data_met = False
        self.current_point = 0

        self.current_observation = self.environment.reset()

    def collect_data(self, data, other_policy_current=None, agent=None, agent2=None):
        current_observation = self.current_observation
        if self.other_policy:
            action = self.agent.policy_obs_other(current_observation,other_policy_current)
        else:
            action = self.agent.policy_obs(current_observation)
        if not self.max_data_met:
            for i in range(self.num_data):
                after_step = self.environment.step(action)
                if self.other_policy:
                    next_action = self.agent.policy_obs_other(after_step[0],other_policy_current)
                else:
                    next_action = self.agent.policy_obs(after_step[0])
                data.current_observation.append(current_observation)
                data.current_action.append(action)
                data.next_observation.append(after_step[0])
                data.next_reward.append(after_step[1])
                data.next_terminal.append(after_step[2])
                data.next_action.append(next_action)

                #update uncertainty
                if agent is not None:
                    if self.on_policy:
                        next_action_target = next_action
                    else:
                        next_action_target = agent2.get_action_target(after_step[0])
                    # next_action_target = next_action
                    # agent.evaluate_incremental_uncertainty_update(current_observation,action,after_step[0],next_action_target,after_step[2],after_step[1],agent2.get_td_error(current_observation,action,after_step[0],next_action_target,after_step[2],after_step[1]))
                    err1,err2 = agent2.get_td_error(current_observation,action,after_step[0],next_action_target,after_step[2],after_step[1])
                    agent.evaluate_incremental_uncertainty_update(current_observation,action,after_step[0],next_action_target,after_step[2],after_step[1],err1,err2)

                current_observation = after_step[0]
                action = next_action
                self.current_point += 1
            self.current_observation = current_observation
        else:
            for i in range(self.num_data):
                after_step = self.environment.step(action)
                if self.other_policy:
                    next_action = self.agent.policy_obs_other(after_step[0],other_policy_current)
                else:
                    next_action = self.agent.policy_obs(after_step[0])
                data.current_observation[self.current_point] = current_observation
                data.current_action[self.current_point] = action
                data.next_observation[self.current_point] = after_step[0]
                data.next_reward[self.current_point] = after_step[1]
                data.next_terminal[self.current_point] = after_step[2]
                data.next_action[self.current_point] = next_action

                #update uncertainty
                if agent is not None:
                    if self.on_policy:
                        next_action_target = next_action
                    else:
                        next_action_target = agent2.get_action_target(after_step[0])
                    # next_action_target = next_action
                    # agent.evaluate_incremental_uncertainty_update(current_observation,action,after_step[0],next_action_target,after_step[2],after_step[1],agent2.get_td_error(current_observation,action,after_step[0],next_action_target,after_step[2],after_step[1]))
                    err1,err2 = agent2.get_td_error(current_observation,action,after_step[0],next_action_target,after_step[2],after_step[1])
                    agent.evaluate_incremental_uncertainty_update(current_observation,action,after_step[0],next_action_target,after_step[2],after_step[1],err1,err2)

                current_observation = after_step[0]
                action = next_action
                self.current_point += 1
            self.current_observation = current_observation

        if self.current_point == self.max_data_size:
            self.max_data_met = True
            self.current_point = 0

    def collect_data2(self, data, other_policy_current=None, agent=None, agent2=None):
        current_observation = self.current_observation
        if self.other_policy:
            action = self.agent.policy_obs_other(current_observation,other_policy_current)
        else:
            action = self.agent.policy_obs(current_observation)

        self.start_point = self.current_point

        for i in range(self.num_data):
            after_step = self.environment.step(action)
            if self.other_policy:
                next_action = self.agent.policy_obs_other(after_step[0],other_policy_current)
            else:
                next_action = self.agent.policy_obs(after_step[0])
            data.current_observation[self.current_point] = current_observation
            data.current_action[self.current_point] = action
            data.next_observation[self.current_point] = after_step[0]
            data.next_reward[self.current_point] = after_step[1]
            data.next_terminal[self.current_point] = after_step[2]
            data.next_action[self.current_point] = next_action

            #update uncertainty
            if agent is not None:
                if self.on_policy:
                    next_action_target = next_action
                else:
                    next_action_target = agent2.get_action_target(after_step[0])
                # next_action_target = next_action
                agent.evaluate_incremental_uncertainty_update(current_observation,action,after_step[0],next_action_target,after_step[2],after_step[1],agent2.get_td_error(current_observation,action,after_step[0],next_action_target,after_step[2],after_step[1]))

            current_observation = after_step[0]
            action = next_action
            self.current_point += 1
        self.current_observation = current_observation

        self.end_point = self.current_point
        if self.current_point == self.max_data_size:
            self.max_data_met = True
            self.current_point = 0

    def return_current_size(self):
        if self.max_data_met:
            return self.max_data_size
        else:
            return self.current_point
