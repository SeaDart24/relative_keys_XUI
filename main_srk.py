# -*- coding: utf-8 -*-

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import storge_tree as st
import srk as sc
import pickle
import os
import time
import csv
from collections import defaultdict
import tracemalloc
from utils import alg_config_parse, check_exp, compute_con_acc

def SRK(res_dict, data_df, epsilon, sample_num):
    
    # starting the monitoring
    tracemalloc.start()

    tree_set_dict = res_dict['tree_set_dict']
    complement_index_dict = res_dict['complement_index_dict']
    same_set_dict = res_dict['same_set_dict']
    res_dict.clear()

    data_df.reset_index(drop=True, inplace=True)

    print(data_df)

    #data_df = data_df.drop('Target', axis=1)

    columns_name_list = data_df.columns.values.tolist()

    X = columns_name_list[0:-1]
    Y = columns_name_list[-1]

    print(columns_name_list)


    # record the explanation of each sample to be explained
    res_dict = {}
    s_time = np.zeros(sample_num, dtype='float')
    exp_s = np.zeros(sample_num, dtype='int')
    consistency_s = np.zeros(sample_num, dtype='float')
    acc_s = np.zeros(sample_num, dtype='bool')

    # shuffle
    data_df = data_df.sample(frac=1).reset_index(drop=True)

    for instance_id in range(data_df.shape[0]):
        start_time = time.time() * 1000
        print("instance_id", instance_id)
        if instance_id >= sample_num:
            break
        instance_value = data_df.loc[instance_id]
        subsets = {}
        # Calculate complement on the spot
        diff_set_dict_time = time.time()
        diff_set_dict = st.get_completary(tree_set_dict, complement_index_dict, same_set_dict, columns_name_list, instance_value)
        universe = set(diff_set_dict[Y][instance_value[Y]])
        # sub_set_dict_time = time.time()
        for x in X:
            subsets[x] = set(diff_set_dict[x][instance_value[x]])   
        cover_sets = sc.greedy_set_cover(universe, subsets, epsilon)

        res_dict[instance_id] = cover_sets
        exp_s[instance_id] = len(cover_sets)  
        s_time[instance_id] = time.time() * 1000 - start_time

        consistency, acc = compute_con_acc(data_df, instance_value, cover_sets)
        consistency_s[instance_id] = consistency
        acc_s[instance_id] = acc
        if epsilon==0:
            check_exp(data_df.loc[0:data_df.shape[0], :], instance_id, cover_sets)

        key_df = pd.DataFrame.from_dict(res_dict, orient='index')
        key_df.columns = ['feature' + str(i+1) for i in range(key_df.shape[1])]
        res_df = pd.concat([data_df.loc[0:sample_num-1, :], key_df], axis=1)
            
        
    return({'method':'SRK', 'results':len(res_dict), 'min_size': np.min(exp_s),
    "max_size" : np.max(exp_s),
    "mean_size" : round(np.mean(exp_s),2),
    "min_time" : round(np.min(s_time), 2),
    "max_time" : round(np.max(s_time), 2),
    "mean_time" : round(np.mean(s_time), 2),
    "mean_precision" : round(np.mean(acc_s), 3),
    "mean_conformity" : round(np.mean(consistency_s), 3),
    "relative_keys" : res_dict,
    "res_df" : res_df
    })



