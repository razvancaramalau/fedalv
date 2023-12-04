import os
import pickle
import torch
import torch.nn.functional as F
from models import get_model
from args import args_parser

from util import set_result_dir, set_dict_user_path
from query_strategies import random_query_samples, algo_query_samples
from datasets import *
from train_test import train_test

if __name__ == '__main__':
    args = args_parser()

    # Set up GPU/CPU compute
    args.device = torch.device('cuda:{}'.format(args.gpu) 
                               if torch.cuda.is_available() and 
                               args.gpu != -1 else 'cpu')   
    # for init the current_ratio with each env
    if not args.resume_ratio:
        args.current_ratio = args.query_ratio
    else:
        args.current_ratio = args.resume_ratio
    # Running experiments while targeting every environment
    for test_env in range(num_environments(args.dataset)):
        # To init the current_ratio with each env
        if not args.resume_ratio:
            args.current_ratio = args.query_ratio
        else:
            args.current_ratio = args.resume_ratio
        print('------------------------')
        print('[--Target domain %s--]'%get_dataset_class(args.dataset).ENVIRONMENTS[test_env])
        # Set up the results directory and model directory
        args = set_result_dir(args) 
        args = set_dict_user_path(args) 
        
        # Get the dataset separation train, test, query 
        args.test_env = test_env
        args, dict_users_train_total, dict_users_test_total = get_dataset(args)
        dict_users_train_label = None
        args.results_save_path = os.path.join(args.result_dir, 
                                              'results_%s_%s_%s_%d.csv'%(args.fl_algo, args.al_method, args.custom_name, test_env))
        if os.path.isfile(args.results_save_path):
            os.remove(args.results_save_path)
        while round(args.current_ratio, 2) <= args.end_ratio:
            print('[Current data ratio] %.3f' % args.current_ratio)

            net_glob = get_model(args)
    
            if args.query_ratio == args.current_ratio:                
                # Initial labelled set is selected randomly for each domain
                dict_users_train_label, args = random_query_samples(dict_users_train_total, dict_users_test_total, args)
                
            else:
                if dict_users_train_label is None:
                    path = os.path.join(args.dict_user_path, 
                                        'dict_users_train_label_%.2f_%s_%s.pkl'%(args.current_ratio - args.query_ratio, args.al_method))
                    with open(path, 'rb') as f:
                        dict_users_train_label = pickle.load(f)
                    args.dict_users_total_path = os.path.join(args.dict_user_path, 
                                                              'dict_users_train_test_total.pkl'.format(args.seed))
                    
                    last_ckpt = torch.load(args.query_model)
                                
                print("Load Total Data Idxs from {}".format(args.dict_users_total_path))
                with open(args.dict_users_total_path, 'rb') as f:
                    dict_users_train_total, dict_users_test_total = pickle.load(f) 
                    
                # Run the AL selection algorithm with the latest model
                dict_users_train_label = algo_query_samples(args, dict_users_train_total)
                            
            if args.reset == 'continue' and args.query_model:
                query_net_state_dict = torch.load(args.query_model)
                net_glob.load_state_dict(query_net_state_dict)
                
            # Train test FEDA with each cycle
            last_ckpt = train_test(net_glob, dict_users_train_label, args)
            
            args.current_ratio += args.query_ratio
            
            # update path
            args = set_result_dir(args) 
            args = set_dict_user_path(args) 