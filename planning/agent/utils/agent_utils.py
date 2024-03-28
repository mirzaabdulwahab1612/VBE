import numpy as np
import copy
import torch

def get_action_values_linear(features,wvec,values):
    length = features.size
    for i in range(values.size):
        values[i] = np.dot(wvec[int(i*length):int((i+1)*length)],features)

def get_action_values_linear_sparse(features,wvec,values,feature_value):
    length = int((wvec.size)/values.size)
    for i in range(values.size):
        offset = i*length
        values[i] = 0
        for j in features:
            values[i] += (wvec[int(offset+j)]*feature_value)

def get_greedy_action(values, np_random, target_action=False):
    if target_action:
        # return torch.argmax(values, axis=-1)
        return np.argmax(values, axis=-1)

    maxPos = np.where(values>=np.max(values))[0]
    # maxPos = torch.where(values>=torch.max(values))[0]
    if len(maxPos) > 1:
        return maxPos[np_random.randint(len(maxPos))]
    else:
        try:
            return maxPos[0]
        except:
            print(f"Something went wrong: values: {values}, maxPos: {maxPos}")
            return None

def get_action(values, epsilon, np_random):
    if np_random.uniform(0,1) < epsilon:
        return np_random.randint(values.size)
    else:
        return get_greedy_action(values, np_random)

def get_softmax_distribution(values, eta):
    e_x = np.exp(eta*(values - np.max(values)))
    scores = e_x / e_x.sum()
    return scores

def get_action_softmax(values, eta, np_random):
    scores = get_softmax_distribution(values, eta)
    return np_random.choice(values.size,p=scores)

def expected_value_softmax(values_p, eta, np_random, values):
    scores = get_softmax_distribution(values, eta)
    return np.dot(scores,values)

def get_values_state(feature_constructor, features, wvec, values, matrix_style=False):
    if feature_constructor.real_mode:
        if not matrix_style:
            get_action_values_linear(features,wvec,values)
        else:
            features_new = features*features
            get_action_values_linear(features_new,wvec,values)
    else:
        get_action_values_linear_sparse(features,wvec,values,feature_constructor.feature_value)

def get_values(features, wvec, num_actions, multi=False):
    val = []
    length = int((wvec.shape[-1])/num_actions)
    for i in range(num_actions):
        offset = int(i*length)
        if(multi):
            val.append(np.dot(wvec[:, offset:offset+length], features))
        else:
            val.append(np.dot(features, wvec[offset:offset+length]))

    return np.array(val).T

def get_values_state_weighted(feature_constructor, features, wvec, values, matrix):
    if feature_constructor.real_mode:
        length = features.size
        for i in range(values.size):
            start = int(i*length)
            end = int((i+1)*length)
            values[i] = np.dot(features,matrix[start:end,start:end].dot(features))
    else:
        assert not "implemented"

# def get_features(observation, feature_constructor):
#     temp = np.zeros(feature_constructor.sparse_feature_size)
#     feature_constructor.get_features_sparse(observation,temp)
#     features_vec = np.zeros(feature_constructor.feature_dim)

#     for i in range(temp.size):
#         features_vec[int(temp[i])] = feature_constructor.feature_value
    
#     return features_vec

def get_features(observation, feature_constructor, features, features_vec=None, action=None):
    if feature_constructor.obs_mode:
        temp = np.zeros(feature_constructor.sparse_feature_size)
        feature_constructor.get_features_sparse(observation,temp)
        features_vec[:] = temp

    elif feature_constructor.real_mode:
        if feature_constructor.input_action:
            if action is None:
                assert not "implemented"
            else:
                feature_constructor.get_features(observation,features,action=action)
                if features_vec is not None:
                    features_vec[:] = features
        else:
            feature_constructor.get_features(observation,features)
            if features_vec is not None:
                if action is not None:
                    offset = int(feature_constructor.feature_dim*action)
                    features_vec.fill(0)
                    features_vec[offset:offset+feature_constructor.feature_dim] = features
                else:
                    features_vec[:] = features
    else:
        temp = np.zeros(feature_constructor.sparse_feature_size)
        feature_constructor.get_features_sparse(observation,temp)
        # print(f"features_vec: {features_vec} features: {temp}")
        if features_vec is not None:
            if action is not None:
                offset = int(feature_constructor.feature_dim*action)
            else:
                offset = 0
            features_vec.fill(0)
            for i in range(temp.size):
                features_vec[int(offset+temp[i])] = feature_constructor.feature_value

    
def get_state(observation, feature_constructor, features):
    features.fill(0)
    temp = np.zeros(feature_constructor.sparse_feature_size)
    feature_constructor.get_features_sparse(observation,temp)
    features[int(temp[0])] = feature_constructor.feature_value

def get_features_waction(observation, action, feature_constructor, features):
    feature_constructor.get_features(observation,features,action)

def get_values_obs(observation, feature_constructor, features, wvec, values, matrix_style=False):
    if feature_constructor.input_action:
        get_values_obs_waction(observation, feature_constructor, features, wvec, values)
    else:
        get_features(observation, feature_constructor, features)
        get_values_state(feature_constructor, features, wvec, values, matrix_style)

def get_values_obs_waction(observation, feature_constructor, features, wvec, values):
    for i in range(values.size):
        feature_constructor.get_features(observation,features,i)
        values[i] = np.dot(wvec,features)

def get_temporal_difference_vector_obs(yvec, feature_constructor, features, gamma, current_state, current_action, next_state, next_action, next_terminal, weighted_yvec=False, num_action=None, greedy_wt=None, all_wt=None, greedy_action=None):

    yvec.fill(0)

    offset = int(current_action*feature_constructor.feature_dim)
    if feature_constructor.real_mode:
        feature_constructor.get_features(current_state,features,action=current_action)
        if feature_constructor.input_action:
            yvec[:] = features
        else:
            yvec[offset:offset+feature_constructor.feature_dim] = features
    else:
        feature_constructor.get_features_sparse(current_state,features)
        for i in set(features):
            pos = int(i)
            yvec[pos+offset] = 1

    if not next_terminal:
        if weighted_yvec:
            if feature_constructor.real_mode:
                if feature_constructor.input_action:
                    for a in range(num_action):
                        feature_constructor.get_features(next_state,features,action=a)
                        weight = all_wt
                        if a == greedy_action:
                            weight += greedy_wt
                        yvec -= (weight*gamma*features)
                else:
                    feature_constructor.get_features(next_state,features)
                    for a in range(num_action):
                        offset = int(a*feature_constructor.feature_dim)
                        weight = all_wt
                        if a == greedy_action:
                            weight += greedy_wt
                        yvec[offset:offset+feature_constructor.feature_dim] -= (weight*gamma*features)
            else:
                feature_constructor.get_features_sparse(next_state,features)
                for a in range(num_action):
                    offset = int(a*feature_constructor.feature_dim)
                    weight = all_wt
                    if a == greedy_action:
                        weight += greedy_wt
                    done_list = []
                    for i in set(features):
                        pos = int(i)+offset
                        if pos in done_list:
                            continue
                        done_list.append(pos)
                        yvec[pos] -= (weight*gamma)
        else:
            offset = int(next_action*feature_constructor.feature_dim)
            if feature_constructor.real_mode:
                feature_constructor.get_features(next_state,features,action=next_action)
                if feature_constructor.input_action:
                    yvec -= (gamma*features)
                else:
                    yvec[offset:offset+feature_constructor.feature_dim] -= (gamma*features)
            else:
                feature_constructor.get_features_sparse(next_state,features)
                done_list = []
                for i in set(features):
                    pos = int(i)+offset
                    if pos in done_list:
                        continue
                    done_list.append(pos)
                    yvec[pos] -= gamma


def get_temporal_difference_vector_state_action_features(yvec, gamma, current_state, current_action, next_state, next_action, next_terminal, feature_dim):

    # yvec.fill(0)
    yvec[:] = current_state

    if not next_terminal:
        offset = int(next_action*feature_dim)
        yvec[offset+int(next_state)] -= gamma


def get_td_vector(yvec, gamma, current_state, current_action, next_state, next_action, next_terminal, feature_dim):
    terminals =  np.invert(np.tile(next_terminal, (current_state.shape[-1],1)).T)
    yvec[:] = current_state - gamma*(np.multiply(terminals, next_state))

    # return td_vector
    # yvec[:] = current_state

    # if not next_terminal:
    #     offset = int(next_action*feature_dim)
    #     yvec[offset+int(next_state)] -= gamma


def get_temporal_difference_vector_state(yvec, gamma, current_state, current_action, next_state, next_action, next_terminal, feature_dim):

    yvec.fill(0)

    offset = int(current_action*feature_dim)
    yvec[offset:offset+feature_dim] = current_state[:]

    if not next_terminal:
        offset = int(next_action*feature_dim)
        yvec[offset:offset+feature_dim] -= next_state

def get_features_state(features_vec, current_state, current_action, feature_dim):
    features_vec.fill(0)
    offset = int(current_action*feature_dim)
    features_vec[offset:offset+feature_dim] = current_state[:]


def get_features_state_tabular(features_vec, current_state, current_action, feature_dim, feature_value):
    features_vec.fill(0)
    offset = int(current_action*feature_dim)
    features_vec[offset+int(current_state)] = feature_value


def get_state_features_tabular(features, feature_dim):
    state_features = []
    # temp = np.zeros((len(features), feature_dim))
    # temp[np.arange(len(features)), features] = 1
    for i in features:
        temp = np.zeros(feature_dim)
        temp[int(i)] = 1
        state_features.append(temp)

    return np.array(state_features)

def get_state_action_features_tabular(features, actions, num_actions, feature_dim):
    state_action_features = []

    for feat, act in zip(features, actions):
        temp = np.zeros(feature_dim*num_actions)
        offset = int(act*feature_dim)
        temp[offset:offset+feature_dim] = feat
        state_action_features.append(temp)

    return np.array(state_action_features)


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def normalize_reps(reps):
  for i in range(reps.shape[0]):
    reps[i,:] /= np.linalg.norm(reps[i,:])