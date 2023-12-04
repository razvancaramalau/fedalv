import os
import sys
import copy
import pickle
import random
import datetime
import numpy as np

import torch

from models import get_model
from .random_sampling import RandomSampling
from .least_confidence import LeastConfidence
from .margin_sampling import MarginSampling
from .entropy_sampling import EntropySampling
from .core_set import CoreSet
from .bv2b_core import bv2b_core
from .badge_sampling  import BadgeSampling
from .adversial_deepfool import AdversarialDeepFool
from .dbal import DBAL
from .eada import EADA
from .fedal import FEDAL, compute_fea
from .egl import EGL
from .gcnal import GCNAL, GCNAL2, GCNAL3
from .alfa_mix import ALFAMix
from .fal import EnsLogitEntropy, EnsLogitBadge
from .fal import EnsRankEntropy, EnsRankBadge
from .fal import FTEntropy, FTBadge
from .fal import LoGo
from torch.utils.data import DataLoader, Dataset


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        x = np.argwhere(self.dataset.indices==self.idxs[item])
        x = x[0][0]
        image, label = self.dataset[x]
        return image, label, x

def random_query_samples(dict_users_train_total, dict_users_test_total, args):
    """ randomly select the labeled samples at the first round
    """
    args.dict_users_total_path = os.path.join(args.dict_user_path, 'dict_users_train_test_total.pkl'.format(args.seed))
            
    with open(args.dict_users_total_path, 'wb') as handle:
        pickle.dump((dict_users_train_total, dict_users_test_total), handle)
    if os.path.isfile(os.path.join(args.dict_user_path, 
                                   'dict_users_train_label_%.2f_%s_%s.pkl'
                                   %(args.current_ratio, 'random', args.dataset))):
        print("Loaded init set")
        dict_users_train_label_path = os.path.join(args.dict_user_path, 
                                                   'dict_users_train_label_%.2f_%s_%s.pkl'
                                                   %(args.current_ratio, 'random', args.dataset))
        with open(dict_users_train_label_path, 'rb') as f:
            dict_users_train_label = pickle.load(f)
    else:
        dict_users_train_label_path = os.path.join(args.dict_user_path, 
                                                   'dict_users_train_label_%.2f_%s_%s.pkl'
                                                   %(args.current_ratio, args.al_method, args.dataset))
        # if args.domain_per_client:
        #     dict_users_train_label = {0: dict_users_train_total}
        # else:
        dict_users_train_label = {user_idx: [] for user_idx in dict_users_train_total.keys()}

        # sample n_start example on each client
        for idx in dict_users_train_total.keys():
            if args.domain_per_client:
                dict_users_train_label[idx] = np.random.choice(np.array(list(dict_users_train_total[idx])), 
                                                            int(args.n_data[idx]), replace=False)
            else:
                dict_users_train_label[idx] = np.random.choice(np.array(list(dict_users_train_total[idx])), 
                                                            int(args.n_data / args.num_users), replace=False)
            
        with open(dict_users_train_label_path, 'wb') as handle:
            pickle.dump(dict_users_train_label, handle)    
    
    return dict_users_train_label, args
    
    
def algo_query_samples(args, dict_users_train_total):
    """ query samples from the unlabeled pool
    """
    previous_ratio = args.current_ratio - args.query_ratio
    if previous_ratio==args.query_ratio and os.path.isfile('dict_users_train_label_%.2f_%s_%s.pkl'
                            %(previous_ratio, 'random', args.dataset)):
        path = os.path.join(args.dict_user_path, 'dict_users_train_label_%.2f_%s_%s.pkl'
                            %(previous_ratio, 'random', args.dataset))
    else:
        path = os.path.join(args.dict_user_path, 'dict_users_train_label_%.2f_%s_%s.pkl'
                            %(previous_ratio, args.al_method, args.dataset))
    with open(path, 'rb') as f:
        dict_users_train_label = pickle.load(f) 

        print("Before Querying")
        total_data_cnt = 0
        for user_idx in range(args.num_users):
            print(user_idx, len(dict_users_train_label[user_idx]))
            total_data_cnt += len(dict_users_train_label[user_idx])

        print(total_data_cnt)
        print("-" * 20)

    # Build model
    query_net = get_model(args)
    args.raw_ckpt = copy.deepcopy(query_net.state_dict())

    query_net_state_dict = torch.load(args.query_model)
    query_net.load_state_dict(query_net_state_dict)
                
    dataset_query = args.dataset_query
    dataset_train = args.dataset_train
    # AL baselines
    if args.al_method == "random":
        strategy = RandomSampling(dataset_query, dataset_train, query_net, args)
    elif args.al_method == "conf":
        strategy = LeastConfidence(dataset_query, dataset_train, query_net, args)
    elif args.al_method == "margin":
        strategy = MarginSampling(dataset_query, dataset_train, query_net, args)
    elif args.al_method == "entropy":
        strategy = EntropySampling(dataset_query, dataset_train, query_net, args)
    elif args.al_method == "coreset":
        strategy = CoreSet(dataset_query, dataset_train, query_net, args)
    elif args.al_method == "badge":
        strategy = BadgeSampling(dataset_query, dataset_train, query_net, args)
    elif args.al_method == "gcnal":
        strategy = GCNAL(dataset_query, dataset_train, query_net, args)
    elif args.al_method == "gcnal2":
        strategy = GCNAL2(dataset_query, dataset_train, query_net, args)
    elif args.al_method == "gcnal3":
        strategy = GCNAL3(dataset_query, dataset_train, query_net, args)
    elif args.al_method == "alfa_mix":
        strategy = ALFAMix(dataset_query, dataset_train, query_net, args)
    elif args.al_method == "bv2b2_core":
        strategy = bv2b_core(dataset_query, dataset_train, query_net, args)
    elif args.al_method == "eada":
        strategy = EADA(dataset_query, dataset_train, query_net, args)
    elif args.al_method == "fedal":
        strategy = FEDAL(dataset_query, dataset_train, query_net, args)    
    elif args.al_method == "fedalv":
        strategy = FEDAL(dataset_query, dataset_train, query_net, args)    
       
    # FAL baselines
    elif args.al_method == "ens_logit_entropy":
        strategy = EnsLogitEntropy(dataset_query, dataset_train, query_net, args)
    elif args.al_method == "ens_logit_badge":
        strategy = EnsLogitBadge(dataset_query, dataset_train, query_net, args) 
    elif args.al_method == "ens_rank_entropy":
        strategy = EnsRankEntropy(dataset_query, dataset_train, query_net, args)
    elif args.al_method == "ens_rank_badge":
        strategy = EnsRankBadge(dataset_query, dataset_train, query_net, args)
    elif args.al_method == "ft_entropy":
        strategy = FTEntropy(dataset_query, dataset_train, query_net, args)
    elif args.al_method == "ft_badge":
        strategy = FTBadge(dataset_query, dataset_train, query_net, args)
    elif args.al_method == "logo":
        strategy = LoGo(dataset_query, dataset_train, query_net, args)
        
    else:
        exit('There is no al methods')    
    
    time = datetime.timedelta()
    
    idxs_users = list(range(args.num_users))
    idxs_users.pop(args.test_env)
    
    if args.al_method == "bv2b2_core":
        # Pre-compute the target top bv2b
        top_bv2b     = strategy.compute_target_bv2b(args.test_env, 
                                                    dict_users_train_total[args.test_env], 
                                                    query_net, int(args.n_query[args.test_env]))
    # for user_idx in dict_users_train_total.keys():
    target_dict = dict_users_train_total[args.test_env]
    if args.al_method == "fedalv":
        # compute target FE and mvsm_uncertainty
        target_fea_samples, first_stats_samples = compute_fea(args, query_net, target_dict)
        unlabel_idxs = []
        for user_idx in idxs_users:
            total_idxs = dict_users_train_total[user_idx]
            label_idxs = dict_users_train_label[user_idx]
            unlabel_idx = [x for x in total_idxs if x not in label_idxs]
            unlabel_idxs.append(unlabel_idx)
        new_data = strategy.query2(idxs_users, unlabel_idxs, 
                                  args.n_query, target_dict, 
                                  target_fea_samples, first_stats_samples)
        for idx, user_idx in enumerate(idxs_users):
            dict_users_train_label[user_idx] = np.array(list(new_data[idx]) + 
                                                        list(dict_users_train_label[user_idx]))   
    else:
        if args.al_method == "feda":
            target_fea_samples, first_stats_samples = compute_fea(args, query_net, target_dict)
        for user_idx in idxs_users:                 
            total_idxs = dict_users_train_total[user_idx]
            label_idxs = dict_users_train_label[user_idx]
            # unlabel_idxs = list(set(total_idxs) - set(label_idxs))
            unlabel_idxs = [x for x in total_idxs if x not in label_idxs]
            if args.al_method == "bv2b2_core":
                start = datetime.datetime.now()
                # print(top_bv2b)
                new_data = strategy.query(user_idx, label_idxs, unlabel_idxs, top_bv2b, int(args.n_query[user_idx]))
                time += datetime.datetime.now() - start
            elif args.al_method == "feda":
                start = datetime.datetime.now()
                new_data = strategy.query(user_idx, unlabel_idxs, int(args.n_query[user_idx]), target_dict, target_fea_samples)
                time += datetime.datetime.now() - start
            else:
                start = datetime.datetime.now()
                new_data = strategy.query(user_idx, label_idxs, unlabel_idxs, int(args.n_query[user_idx]))
                time += datetime.datetime.now() - start
 
            print(args.al_method, user_idx)
            print("(Before) Label examples: {}".format(len(label_idxs)))
            if len(new_data) < int(args.n_query[user_idx]):
                sys.exit("too few remaining examples to query")
            dict_users_train_label[user_idx] = np.array(list(new_data) + list(label_idxs))   
            print("(After) Label examples: {}".format(len(list(new_data)) + len(label_idxs))) 
    
    source_embedding, source_labels = strategy.get_embedding2(args.test_env, dict_users_train_total[args.test_env], strategy.net)
    # np.save("target_%d_feat%f.npy"%(user_idx, args.current_ratio), np.concatenate((source_embedding, np.expand_dims(source_labels, 1)), 1))        
    
    time /= len(dict_users_train_total)     
    print('Querying instances takes {}'.format(time))           

    # Save dict_users for next round
    # path = os.path.join(args.dict_user_path, 'dict_users_train_label_{:.3f}.pkl'.format(args.current_ratio))
    path = os.path.join(args.dict_user_path, 
                        'dict_users_train_label_%.2f_%s_%s.pkl'
                        %(args.current_ratio, args.al_method, args.dataset)) 
    with open(path, 'wb') as handle:
        pickle.dump(dict_users_train_label, handle)

    return dict_users_train_label
    