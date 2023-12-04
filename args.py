#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.9

import argparse
# import medmnist
from medmnist import INFO

def args_parser():
    parser = argparse.ArgumentParser()
    
    # basic arguments
    parser.add_argument('--gpu', type=int, default=3, help="GPU ID, -1 for CPU")
    parser.add_argument('--custom_name', type=str, default=None)
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--save_dir', type=str, default='.', help='when to start saving models')    
    parser.add_argument('--test_env', type=int, default=0)
    parser.add_argument('--saved_folder', type=str, default='None')
    parser.add_argument('--rounds_per_eval', type=int, default=1)
    parser.add_argument('--num_samples', type=int, default=20)
    # federated learning arguments
    parser.add_argument('--rounds', type=int, default=1, help="rounds of training")
    parser.add_argument('--global_rounds', type=int, default=3, help="intra-server rounds of training")
    parser.add_argument('--num_users', type=int, default=3, help="number of users: K")
    parser.add_argument('--frac', type=float, default=1.0, help="the fraction of clients: C")
    parser.add_argument('--local_ep', type=int, default=1, help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=128, help="local batch size: B")
    parser.add_argument('--test_bs', type=int, default=128, help="test batch size")
    parser.add_argument('--lr', type=float, default=0.01, help="learning rate")
    parser.add_argument('--momentum', type=float, default=0.9, help="momentum")
    parser.add_argument('--weight_decay', type=float, default=5e-4, help="weight decay (default: 0.00001)")
    parser.add_argument('--lr_decay', type=float, default=0.1, help="learning rate decay ratio")
    parser.add_argument('--reset', type=str, default='random_init', help='doing FL with queried dataset or not')
    
    parser.add_argument('--fl_algo', type=str, default='feda', help='federated learning algorithm')
    
    parser.add_argument('--mu', type=float, default=0.01, help='weight of FedProx regularization term')
    parser.add_argument('--z_dim', type=int, default=512)
    parser.add_argument('--save_checkpoint', type=str, default='True',choices=['True','False'])

    parser.add_argument('--optim', type=str, default='SGD')
  # dataset arguments
    parser.add_argument('--data_dir', type=str, default='./falda/data', help='data path')
    parser.add_argument('--dataset', type=str, default='PACS', help="name of dataset")
    parser.add_argument('--partition', type=str, default="dir_balance", help="methods for Non-IID")
    parser.add_argument('--dd_beta', type=float, default=0.1, help="beta for dirichlet distribution")               
    parser.add_argument('--num_classes_per_user', type=int, default=2, help="classes per user")
    parser.add_argument('--imb_ratio', type=float, default=1.0, help="imbalance ratio for long tail dataset")
    parser.add_argument('--dataset_folder', type=str, default='/home/razvan/falda/data/')
    # model arguments
    parser.add_argument('--model', type=str, default='resnet18', help='model name')
    
    # active learning arguments
    parser.add_argument('--resume_ratio', type=float, default=0., help="ratio of data examples for resume")
    parser.add_argument('--query_ratio', type=float, default=0.01, help="ratio of data examples per one query") 
    parser.add_argument('--end_ratio', type=float, default=0.05, help="ratio for stopping query")
    
    parser.add_argument('--query_model_mode', type=str, default="global")
    parser.add_argument('--al_method', type=str, default='fedalv')
    
    parser.add_argument('--train_split', type=float, default=0.9)
    parser.add_argument('--domain_per_client', type=bool, default=True)
    
    parser.add_argument('--L2R_coeff', type=float, default=1e-2) #1e-2
    parser.add_argument('--CMI_coeff', type=float, default=5e-4)

    args = parser.parse_args()
    


    # popular benchmark
    if args.dataset == 'PACS':
        args.num_classes = 7
        args.in_channels = 3
        args.img_size = 224
        if not args.end_ratio: args.end_ratio = 1.0
    elif args.dataset == 'DomainNet':
        args.num_classes = 345
        args.in_channels = 3
        args.img_size = 128
    elif args.dataset == 'OfficeHome':
        args.num_classes = 65
        args.in_channels = 3
        args.img_size = 128
        if not args.end_ratio: args.end_ratio = 1.0
    elif args.dataset == 'mnist':
        args.num_classes = 10
        args.in_channels = 1
        args.img_size = 28
        if not args.end_ratio: args.end_ratio = 0.6
    elif args.dataset == 'fmnist':
        args.num_classes = 10
        args.in_channels = 1
        args.img_size = 28
        if not args.end_ratio: args.end_ratio = 0.6
    elif args.dataset == 'emnist': # 814,255 samples
        args.num_classes = 62
        args.in_channels = 1
        args.img_size = 28
        args.query_ratio = 0.005
        if not args.end_ratio: args.end_ratio = 0.05
    elif args.dataset == 'svhn':
        args.num_classes = 10
        args.in_channels = 3
        args.img_size = 32
        if not args.end_ratio: args.end_ratio = 0.6
    elif args.dataset in ['cifar10', 'cifar10_lt']:
        args.num_classes = 10
        args.in_channels = 3
        args.img_size = 32
        if not args.end_ratio: args.end_ratio = 0.8
    elif args.dataset in ['cifar100', 'cifar100_lt']:
        args.num_classes = 100
        args.in_channels = 3
        args.img_size = 32
        if not args.end_ratio: args.end_ratio = 0.4
    elif args.dataset == 'tinyimagenet':
        args.num_classes = 200
        args.in_channels = 3
        args.img_size = 64
        if not args.end_ratio: args.end_ratio = 0.6
    
    # medical benchmark
    elif args.dataset == 'pathmnist': # 107,180 samples, 89,996 train
        info = INFO[args.dataset]
        args.num_classes = len(info['label']) # 9
        args.in_channels = info['n_channels'] # 3
        args.img_size = 28
        if not args.end_ratio: args.end_ratio = 0.6
    elif args.dataset == 'octmnist': # 109,309 samples, 97,477 train
        info = INFO[args.dataset]
        args.num_classes = len(info['label']) # 4
        args.in_channels = info['n_channels'] # 1
        args.img_size = 28
        if not args.end_ratio: args.end_ratio = 0.6
    elif args.dataset == 'organamnist': # 58,850 samples, 34,581 train
        info = INFO[args.dataset]
        args.num_classes = len(info['label']) # 11
        args.in_channels = info['n_channels'] # 1
        args.img_size = 28
        if not args.end_ratio: args.end_ratio = 0.6
    elif args.dataset == 'dermamnist': # 10,015 samples, 7,007 train
        info = INFO[args.dataset]
        args.num_classes = len(info['label']) # 7
        args.in_channels = info['n_channels'] # 3
        args.img_size = 28
        if not args.end_ratio: args.end_ratio = 0.6
    elif args.dataset == 'bloodmnist': # 17,092 samples, 11,959 train
        info = INFO[args.dataset]
        args.num_classes = len(info['label']) # 8
        args.in_channels = info['n_channels'] # 3
        args.img_size = 28
        if not args.end_ratio: args.end_ratio = 0.6
        
        
    if args.dataset in ['cifar10_lt', 'cifar100_lt']:
        args.data_dir += '/{}_{}/'.format(args.dataset, args.imb_ratio)
    else:
        args.data_dir += '/{}/'.format(args.dataset)
        
    return args
