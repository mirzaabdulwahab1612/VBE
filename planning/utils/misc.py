import os
import json
import numpy as np

def get_lists(params, lists, items_eligible):
    for item in items_eligible:
        item_value = getattr(params, item)
        assert type(item_value) == list
        lists.append(item_value)
    return lists

def set_params(config, pos, items_eligible, params):
    for item in items_eligible:
        setattr(params, item+"_range", getattr(params, item))
        setattr(params, item, config[pos])
        pos += 1

def print_object(params):
    for item in dir(params):
        if not callable(getattr(params, item)) and not item.startswith("__"):
            print(item, getattr(params, item))

def make_directory(path):
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except:
            assert (os.path.exists(path))

def del_params(params,items_eligible,params_extra):
    for item in items_eligible:
        setattr(params_extra, item, getattr(params, item))
        delattr(params, item)

def replace_params(params,items_eligible,params_extra):
    for item in items_eligible:
        setattr(params, item, getattr(params_extra, item))

def write_json(file, params, encoder, items_eligible=None, params_extra=None):
    if items_eligible:
        del_params(params, items_eligible, params_extra)
    param_string = json.dumps(params, indent=4, cls=encoder)
    with open(file, 'w') as f:
        f.write(param_string)
    if items_eligible:
        replace_params(params, items_eligible, params_extra)
    # if not os.path.exists(file):
    #     try:
    #         with open(file, 'w') as f:
    #             f.write(param_string)
    #     except:
    #         pass


def sparsity_stat(encoding_data, logger):

    pop_sparse = np.zeros([encoding_data.shape[0]])
    pop_max = np.zeros([encoding_data.shape[0]])
    pop_mean = np.zeros([encoding_data.shape[0]])
    pop_min = np.zeros([encoding_data.shape[0]])

    life_sparse = np.zeros([encoding_data.shape[1]])
    life_max = np.zeros([encoding_data.shape[1]])
    life_mean = np.zeros([encoding_data.shape[1]])
    life_min = np.zeros([encoding_data.shape[1]])

    for i in range(0, encoding_data.shape[1]):
        number_non_zero = np.count_nonzero(encoding_data.T[i])
        life_sparse[i] = float(number_non_zero) / encoding_data.shape[0]
        if number_non_zero != 0:
            life_mean[i] = np.sum(encoding_data.T[i]) / number_non_zero
        else:
            life_mean[i] = 0

        life_max[i] = encoding_data.T[i].max()

    number_active_units = np.count_nonzero(life_sparse)
    for i in range(0, encoding_data.shape[0]):
        if number_active_units != 0:
            pop_sparse[i] = float(np.count_nonzero(encoding_data[i])) / number_active_units
        pop_max[i] = encoding_data[i].max()
        pop_mean[i] = np.mean(encoding_data[i])

    half_n = int(number_active_units/2)
    life_sparse[::-1].sort()
    life_mean[::-1].sort()
    life_max[::-1].sort()

    logger.info('                                   max  median   min  (over one episode)')
    logger.info('Instance Active Percentage (%)   {:.4f} {:.4f} {:.4f}'.format(pop_sparse.max(), np.median(pop_sparse), pop_sparse.min()))
    logger.info('Instance Max Magnitude           {:.4f} {:.4f} {:.4f}'.format(pop_max.max(),pop_max.mean(),pop_max.min()))
    logger.info('Instance Mean Magnitude          {:.4f} {:.4f} {:.4f}'.format(pop_mean.max(),pop_mean.mean(),pop_mean.min()))
    logger.info('Instance Min Magnitude           {:.4f} {:.4f} {:.4f}'.format(pop_min.max(),pop_min.mean(),pop_min.min()))

    logger.info('                                   max  median   min  (over all hidden units)')
    logger.info('Lifetime Active Percentage (%)   {:.4f} {:.4f} {:.4f}'.format(life_sparse.max(), life_sparse[half_n], life_sparse[number_active_units-1]))
    logger.info('Lifetime Max Magnitude           {:.4f} {:.4f} {:.4f}'.format(life_max.max(), life_max[half_n], life_max[number_active_units-1]))
    logger.info('Lifetime Mean Magnitude          {:.4f} {:.4f} {:.4f}'.format(life_mean.max(), life_mean[half_n], life_mean[number_active_units-1]))
    logger.info('Lifetime Min Magnitude           {:.4f} {:.4f} {:.4f}'.format(life_min.max(), life_min[half_n], life_min[number_active_units-1]))

    logger.info('Active units / number of units   {}/{}'.format(number_active_units, len(life_sparse)))

    return pop_sparse, number_active_units
