import numpy as np
import config

t_size = config.t_size
r_size = config.r_size

def init_val():
    global dict_val
    dict_val = {'dis_type_emb': np.zeros((t_size, t_dimension, t_dimension)), 'gen_type_emb': np.zeros((t_size, t_dimension, t_dimension)), 'dis_relation_emb':np.zeros((r_size, t_dimension, t_dimension)),\
                'gen_relation_emb':  np.zeros((r_size, t_dimension, t_dimension)), 'st': [], 'ed':[], 'pos_st': [], 'pos_st': [], 'neg_st': [], 'neg_ed': []}

def set_value(key, value):
    # global dict_val
    dict_val[key] = value


def get_value(key):
    # global dict_val
    return dict_val[key]

if __name__ == '__main__':
    init_val()
    print(get_value('ed'))

