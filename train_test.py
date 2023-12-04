import os
import gc
import copy
import pandas as pd
import numpy as np
import torch
from util import AverageMeter, all_gather, DatasetSplit, momentum_update
from torch.utils.data import DataLoader, random_split
from datetime import datetime
from fl_methods import get_fl_method_class
from fl_methods.base import compute_w_ga
import time
from query_strategies.eada import FreeEnergyAlignmentLoss, NLLLoss

def adjust_learning_rate(args, r):
    lr = args.lr
    iterations = [int(args.rounds * 3 / 4)]
    
    lr_decay_epochs = []
    for it in iterations:
        lr_decay_epochs.append(int(it))
        
    steps = np.sum(r > np.asarray(lr_decay_epochs))
    if steps > 0:
        lr = lr * (args.lr_decay ** steps)
        
    return lr

# def train_test(net_glob, dataset_train, dataset_test, dict_users_train_label, args):
def train_test(net_glob, dict_users_train_label, args):
    

    fl_method = get_fl_method_class(args.fl_algo)(args, dict_users_train_label)
    if args.fl_algo == 'scaffold':
        fl_method.init_c_nets(net_glob)

    results = []   
    best_target, best_source = 0.0, 0.0
    start_time = time.time()
    w_ga = 1.0
    step_size = 0.2
    step_size_decay = step_size / (args.rounds)
    if args.fl_algo == "feda":
        ENERGY_ALIGN_WEIGHT = 0.01
    else:
        ENERGY_ALIGN_WEIGHT = 0.0
    test_output = None
    data_loaders = []
    idxs_users = list(range(args.num_users))
    idxs_users.pop(args.test_env)
    for idx in idxs_users:
        data_idx = dict_users_train_label[idx]
        data_loaders.append(DataLoader(DatasetSplit(args.dataset_train[idx], data_idx), 
                                        batch_size=args.local_bs, shuffle=True))
    target_loader =  DataLoader(args.dataset_train[args.test_env], 
                    batch_size=args.local_bs, shuffle=True)
    
    uns_criterion = FreeEnergyAlignmentLoss(1.0)
    
    for round in range(args.rounds):
        w_glob = None
        loss_locals = []
        w_locals = []     
        lr = adjust_learning_rate(args, round)
        total_data_num = sum([len(dict_users_train_label[idx]) for idx in idxs_users])
            

        fl_method.on_round_start(net_glob=net_glob)

        if ENERGY_ALIGN_WEIGHT>0.0 and round>0:
            target_loader =  DataLoader(args.dataset_train[args.test_env], 
                    batch_size=256, shuffle=True)
            net_glob.train()
            optimizer = torch.optim.SGD(net_glob.parameters(), lr=lr, 
                            momentum=args.momentum, 
                            weight_decay=args.weight_decay)
            optimizer.zero_grad()

            for i, idx in enumerate(idxs_users):
                for step in range(args.global_rounds):             
                    for images, labels in data_loaders[i]:             
                        images, labels = images.to(args.device), labels.to(args.device)                 
                        optimizer.zero_grad()

                        if args.fl_algo == "feda":
                            z, (z_mu,z_sigma) = net_glob.featurize(images, return_dist=True)
                            output = net_glob.cls(z)
                        else:
                            output, emb = net_glob(images)

                        test_images, _ = next(iter(target_loader))
                        test_images = test_images.to(args.device)
                        if args.fl_algo == "feda":
                            z, (z_mu,z_sigma) = net_glob.featurize(test_images, return_dist=True)
                            test_output = net_glob.cls(z)
                        else:
                            test_output = net_glob(test_images)[0]

                        if output.shape[0] == 1:
                            labels = labels.reshape(1,)

                        with torch.no_grad():

                            # free energy of samples
                            output_div_t = -1.0 * output
                            output_logsumexp = torch.logsumexp(output_div_t, dim=1, keepdim=False)
                            free_energy = -1.0 * output_logsumexp 

                            src_batch_free_energy = free_energy.mean().detach()

                            # init global mean free energy
                            if round == 1:
                                global_mean = src_batch_free_energy
                            # update global mean free energy
                            global_mean = momentum_update(global_mean, src_batch_free_energy)
                        fea_loss = uns_criterion(test_output, global_mean)
                        obj = ENERGY_ALIGN_WEIGHT * fea_loss
                        obj.backward()
                        optimizer.step()

        for i, idx in enumerate(idxs_users):
            fl_method.on_user_iter_start(args.dataset_train[idx], idx)

            net_local = copy.deepcopy(net_glob)
            # source_images, source_labels = next(iter(data_loaders[i]))
            w_local, loss = fl_method.train(net=net_local.to(args.device), 
                                            user_idx=idx,
                                            lr=lr,
                                            momentum=args.momentum,
                                            weight_decay=args.weight_decay)
                                            # round=round,
                                            # test_output=test_output)            
            loss_locals.append(copy.deepcopy(loss))
            w_locals.append(copy.deepcopy(w_local))
            fl_method.on_user_iter_end()
            
        w_ga = [1.0 / 3.0] * len(idxs_users) if (args.fl_algo != 'fedaga') or (round == 0) \
                                     else compute_w_ga(p_loss_locals, loss_locals, step_size - (round-1) * step_size_decay, w_ga)                
        for k, idx in enumerate(idxs_users):
            if (args.fl_algo == 'fedaga'):
                w_glob = fl_method.aggregate2(w_glob=w_glob, w_local=w_locals[k], 
                                            idx_user=idx, total_data_num=total_data_num, 
                                            total_users=len(idxs_users), w_ga=w_ga[k])
            else:
                w_glob = fl_method.aggregate(w_glob=w_glob, w_local=w_locals[k], 
                                            idx_user=idx, total_data_num=total_data_num, 
                                            total_users=len(idxs_users))
        
            # print("Aggre User %d --- %s seconds ---" % (idx, time.time() - start_time))
        p_loss_locals = loss_locals
        fl_method.on_round_end(idxs_users)
        net_glob.load_state_dict(w_glob, strict=False)
        # Test on each source domain
        sacc_test = []
        # Test on  target domain
        if  (round % 3 == 0):
            tacc_test, tloss_test = fl_method.test(net_glob, args.dataset_test[args.test_env])
            for idx in idxs_users:
                acc_test, _ = fl_method.test(net_glob, args.dataset_test[idx])
                sacc_test.append(acc_test)
            acc_test = sum(sacc_test) /  len(idxs_users)
            
            if (best_source < acc_test) :
                best_source = acc_test
            if (best_target < tacc_test) :  
                best_target = tacc_test
                results = np.array([round, tloss_test, best_source, best_target])
        
                last_save_path = os.path.join(args.result_dir, 
                                            '{:.3f}'.format(args.current_ratio) + 
                                            "_%s_%s_last.pt"%(args.fl_algo, args.al_method))
                torch.save(net_glob.state_dict(), last_save_path)

                
            print('Round {:3d}, Source Avg Acc {:.3f}, Target loss {:.3f}, Target Acc: {:.2f}, Best Acc: {:.2f}'.format(
                            round+1, acc_test, tloss_test, tacc_test, best_target))    

        loss_avg = sum(loss_locals) / len(loss_locals)
        print('Loss Avg Sources {:.4f}'.format(loss_avg))
        # print("Test Target --- %s seconds ---" % (time.time() - start_time))
        
    final_results = np.array(results)
    final_results = pd.DataFrame(final_results, 
                                 columns=['%s'%args.current_ratio])
    if os.path.isfile(args.results_save_path):
        prev_results = pd.read_csv(args.results_save_path)#, sep=',\s+', quotechar='"')
        prev_results = pd.DataFrame(prev_results)
        prev_results.insert(prev_results.shape[1], '%s'%args.current_ratio, final_results)
        prev_results.to_csv(args.results_save_path, index=False, sep=" ")
    else:
        final_results.to_csv(args.results_save_path, index=False, sep=" ")
            
    return net_glob.state_dict()
