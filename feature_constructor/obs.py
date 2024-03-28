import pickle
import numpy as np
import math

from .utils.tiles3 import *

class AggAgent():
    def __init__(self, params):

        self.dynamic = False
        self.np_random = params.np_random
        # control settings
        self.num_action = params.environment_params.num_action
        self.obs_dim = params.environment_params.obs_dim
        self.obs_limits = params.environment_params.obs_limits

        self.use_bias = params.feature_constructor_params.use_bias
        if self.use_bias:
            self.feature_dim = params.feature_constructor_params.feature_dim+1
        else:
            self.feature_dim = params.feature_constructor_params.feature_dim
            
        self.pure_tabular = params.feature_constructor_params.pure_tabular
        self.rotate_dense = params.feature_constructor_params.rotate_dense
        self.input_action = params.feature_constructor_params.input_action

        if self.pure_tabular:
            self.sparse_feature_size = 1
            self.features = np.zeros(self.obs_dim)
            self.num_bins = params.feature_constructor_params.num_bins
            assert (np.power(self.num_bins,self.obs_dim) == self.feature_dim), "Number of features not equal to n ^ observation dimension"
        else:
            self.sparse_feature_size = self.obs_dim
            self.num_bins = params.feature_constructor_params.feature_dim/self.obs_dim
            assert (params.feature_constructor_params.feature_dim%self.obs_dim == 0.), "Number of features not divisible by dimension"

        if self.rotate_dense:
            self.real_mode = True
            if self.rotate_dense:
                if self.pure_tabular:
                    self.features_sparse = np.zeros(1)
                else:
                    self.features_sparse = np.zeros(self.obs_dim)
            temp_mat = self.np_random.normal(size=(self.feature_dim, self.feature_dim))
            self.rotation_matrix, _ = np.linalg.qr(temp_mat)
        else:
            self.real_mode = False
            self.feature_value = 1.0

        self.obs_mode = True

    def save_feature_constructor(self, path):
        return

    def get_features(self,current_observation,features,action=None):
        features = current_observation[:]

    def get_features_sparse(self,current_obs,features):
        features[:] = current_obs

    def load_feature_constructor(self, path):
        with open(path+'.pkl', 'rb') as f:
            self.iht = pickle.load(f)

    def feature_length(self, features):
        return self.sparse_feature_size


def init(params):
    return AggAgent(params)

def get_params():
    return ["feature_dim"]

if __name__ == "__main__":
    tc = AggAgent(None)
