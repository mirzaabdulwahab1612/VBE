import pickle
import numpy as np

from .utils.tiles3 import *

class TCAgent():
    def __init__(self, params):

        self.dynamic = False
        self.full = False

        # control settings
        self.num_action = params.environment_params.num_action
        self.obs_dim = params.environment_params.obs_dim
        self.obs_limits = params.environment_params.obs_limits
        self.obs_mode = False

        self.environment = params.basic.environment

        # tiling settings
        self.num_tiles = params.feature_constructor_params.num_tiles
        self.num_tilings = params.feature_constructor_params.num_tilings
        self.use_bias = params.feature_constructor_params.use_bias
        if self.use_bias:
            self.feature_dim = params.feature_constructor_params.feature_dim+1
        else:
            self.feature_dim = params.feature_constructor_params.feature_dim
        self.tile_independently = params.feature_constructor_params.tile_independently
        # self.input_action = params.feature_constructor_params.input_action
        self.normalized = params.feature_constructor_params.normalized
        self.use_hash = params.feature_constructor_params.use_hash

        self.real_mode = False
        if self.use_bias:
            self.sparse_feature_size = self.num_tilings+1
        else:
            self.sparse_feature_size = self.num_tilings
        if self.normalized:
            # self.feature_value = 1.0/np.sqrt(self.sparse_feature_size)
            self.feature_value = 1.0/self.sparse_feature_size

        else:
            self.feature_value = 1.0

        self.np_random = params.np_random

        self.scaled_obs = []
        for i in range(self.obs_dim):
            self.scaled_obs.append(0.0)

        if self.environment == "ac":
            self.extra_obs = []
            for i in range(3):
                self.extra_obs.append(0.0)

        if self.use_hash:
            if self.use_bias:
                self.iht = IHT(self.feature_dim-1)
            else:
                self.iht = IHT(self.feature_dim)
        else:
            if self.use_bias:
                self.size = (self.feature_dim-1)/self.obs_dim
            else:
                self.size = self.feature_dim/self.obs_dim


    def save_feature_constructor(self, path):
        if self.use_hash:
            with open(path+'.pkl', 'wb') as f:
                pickle.dump(self.iht, f)

    def get_features_sparse(self,current_state,features):
        if self.tile_independently:
            assert not "implemented"
        elif self.environment == "ac":
            tiles_all = []
            numcall = 0

            self.scaled_obs[0] = (current_state[0]+self.obs_limits[0][1])*6/self.obs_limits[0][2]
            self.scaled_obs[1] = (current_state[1]+self.obs_limits[1][1])*6/self.obs_limits[1][2]
            self.scaled_obs[2] = (current_state[2]+self.obs_limits[2][1])*7/self.obs_limits[2][2]
            self.scaled_obs[3] = (current_state[3]+self.obs_limits[3][1])*7/self.obs_limits[3][2]
            tiles_all.extend(tiles(self.feature_dim, 12, self.scaled_obs, [numcall]))
            numcall += 1

            for i in range(4):
                count = 0
                for j in range(4):
                    if j != i:
                        self.extra_obs[count] = self.scaled_obs[j]
                        count += 1
                tiles_all.extend(tiles(self.feature_dim, 3, self.extra_obs, [numcall]))
                numcall += 1

            obs_here = self.extra_obs[:2]
            for i in range(3):
                for j in range(i+1,4):
                    obs_here[0] = self.scaled_obs[i]
                    obs_here[1] = self.scaled_obs[j]
                    tiles_all.extend(tiles(self.feature_dim, 2, obs_here, [numcall]))
                    numcall += 1

            obs_here = self.extra_obs[:1]
            for i in range(4):
                obs_here[0] = self.scaled_obs[i]
                tiles_all.extend(tiles(self.feature_dim, 3, obs_here, [numcall]))
                numcall += 1

            features[:] = tiles_all

        else:
            for i in range(self.obs_dim):
                # self.scaled_obs[i] = current_state[i]*self.num_tiles
                self.scaled_obs[i] = ((current_state[i]-self.obs_limits[i][0])/self.obs_limits[i][2])*self.num_tiles
            if self.use_hash:
                if self.use_bias:
                    features[:-1] = tiles(self.iht, self.num_tilings, self.scaled_obs)
                    features[-1] = self.feature_dim-1
                else:
                    features[:] = tiles(self.iht, self.num_tilings, self.scaled_obs)
                self.full = self.iht.full
            else:
                if self.use_bias:
                    features[:-1] = tiles_nohash(self.size, self.num_tilings, self.num_tiles, self.obs_dim, self.scaled_obs)
                    features[-1] = self.feature_dim-1
                else:
                    features[:] = tiles_nohash(self.size, self.num_tilings, self.num_tiles, self.obs_dim, self.scaled_obs)


    def load_feature_constructor(self, path):
        if self.use_hash:
            with open(path, 'rb') as f:
                self.iht = pickle.load(f)

    def feature_length(self, features):
        return self.sparse_feature_size


def init(params):
    return TCAgent(params)

def get_params():
    return ["feature_dim","num_tiles","num_tilings"]

if __name__ == "__main__":
    tc = TCAgent(None)