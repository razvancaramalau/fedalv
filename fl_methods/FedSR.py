import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch.distributions as distributions

import torch
from .base import FederatedLearning
import numpy as np
from collections import defaultdict, OrderedDict

from fl_methods.base import *
from util import *
from query_strategies.eada import NLLLoss, FreeEnergyAlignmentLoss

class FedSR(FederatedLearning):
    def __init__(self, args, dict_users_train_label=None):
        super().__init__(args, dict_users_train_label)
        self.r_mu = nn.Parameter(torch.zeros(args.num_classes,args.z_dim)).to(args.device)
        self.r_sigma = nn.Parameter(torch.ones(args.num_classes,args.z_dim)).to(args.device)
        self.C = nn.Parameter(torch.ones([])).to(args.device)
        self.nll_criterion = NLLLoss(1.0)
        # unsupervised energy alignment bound loss
        self.uns_criterion = FreeEnergyAlignmentLoss(1.0)
    def train(self, net, user_idx=None, lr=0.01, momentum=0.9, weight_decay=0.00001, 
              round=1, test_output=None, images=None, labels=None):
        net.train()

        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        # optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)

        epoch_loss = [] 
        
        for epoch in range(self.args.local_ep):
            my_dict = {}
            batch_loss = []
            for images, labels in self.data_loader:
                if self.args.dataset in ['pathmnist', 'octmnist', 'organamnist', 'dermamnist', 'bloodmnist']:
                        labels = labels.squeeze().long()
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                if epoch==0 and round == 0:
                    target = labels.squeeze().long().cpu()
                    for item in target.numpy():
                        if item in my_dict:
                            my_dict[item] += 1
                        else:
                            my_dict[item] = 1                    
                optimizer.zero_grad()
                # output, emb = net(images)
                output = net(images)
                z, (z_mu,z_sigma) = net.featurize(images, return_dist=True)
                output = net.cls(z)
                if output.shape[0] == 1:
                    labels = labels.reshape(1,)

                # loss = self.loss_func(output, labels)
                loss = self.nll_criterion(output, labels)
                obj = loss
                regL2R = torch.zeros_like(obj)
                regCMI = torch.zeros_like(obj)
                if self.args.L2R_coeff != 0.0:
                    regL2R = z.norm(dim=1).mean()
                    obj = obj + self.args.L2R_coeff*regL2R
                if self.args.CMI_coeff != 0.0:
                    r_sigma_softplus = F.softplus(self.r_sigma)
                    r_mu = self.r_mu[labels]
                    r_sigma = r_sigma_softplus[labels]
                    z_mu_scaled = z_mu*self.C
                    z_sigma_scaled = z_sigma*self.C
                    regCMI = torch.log(r_sigma) - torch.log(z_sigma_scaled) + \
                            (z_sigma_scaled**2+(z_mu_scaled-r_mu)**2)/(2*r_sigma**2) - 0.5
                    regCMI = regCMI.sum(1).mean()
                    obj = obj + self.args.CMI_coeff*regCMI            
               # obj.backward(retain_graph=True)
                obj.backward()
                optimizer.step()
                batch_loss.append(obj.item())
            if epoch==0 and round == 0:
                print("Domain %d class distrib: "%user_idx, dict(sorted(my_dict.items())))
            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)

# class Model(FederatedLearning):
#     def __init__(self, args):
#         self.probabilistic = True
#         super(Model, self).__init__(args)
#         self.r_mu = nn.Parameter(torch.zeros(args.num_classes,args.z_dim))
#         self.r_sigma = nn.Parameter(torch.ones(args.num_classes,args.z_dim))
#         self.C = nn.Parameter(torch.ones([]))
#         self.optim.add_param_group({'params':[self.r_mu,self.r_sigma,self.C],'lr':self.lr,'momentum':0.9})

#     def train_client(self,loader,steps=1):
#         self.train()
#         lossMeter = AverageMeter()
#         accMeter = AverageMeter()
#         regL2RMeter = AverageMeter()
#         regCMIMeter = AverageMeter()
#         regNegEntMeter = AverageMeter()
#         for step in range(steps):
#             x, y = next(iter(loader))
#             x, y = x.to(self.device), y.to(self.device)
#             z, (z_mu,z_sigma) = self.featurize(x,return_dist=True)
#             logits = self.cls(z)
#             loss = F.cross_entropy(logits,y)

#             obj = loss
#             regL2R = torch.zeros_like(obj)
#             regCMI = torch.zeros_like(obj)
#             regNegEnt = torch.zeros_like(obj)
#             if self.L2R_coeff != 0.0:
#                 regL2R = z.norm(dim=1).mean()
#                 obj = obj + self.L2R_coeff*regL2R
#             if self.CMI_coeff != 0.0:
#                 r_sigma_softplus = F.softplus(self.r_sigma)
#                 r_mu = self.r_mu[y]
#                 r_sigma = r_sigma_softplus[y]
#                 z_mu_scaled = z_mu*self.C
#                 z_sigma_scaled = z_sigma*self.C
#                 regCMI = torch.log(r_sigma) - torch.log(z_sigma_scaled) + \
#                         (z_sigma_scaled**2+(z_mu_scaled-r_mu)**2)/(2*r_sigma**2) - 0.5
#                 regCMI = regCMI.sum(1).mean()
#                 obj = obj + self.CMI_coeff*regCMI

#             z_dist = distributions.Independent(distributions.normal.Normal(z_mu,z_sigma),1)
#             mix_coeff = distributions.categorical.Categorical(x.new_ones(x.shape[0]))
#             mixture = distributions.mixture_same_family.MixtureSameFamily(mix_coeff,z_dist)
#             log_prob = mixture.log_prob(z)
#             regNegEnt = log_prob.mean()


#             self.optim.zero_grad()
#             obj.backward()
#             self.optim.step()

#             acc = (logits.argmax(1)==y).float().mean()
#             lossMeter.update(loss.data,x.shape[0])
#             accMeter.update(acc.data,x.shape[0])
#             regL2RMeter.update(regL2R.data,x.shape[0])
#             regCMIMeter.update(regCMI.data,x.shape[0])
#             regNegEntMeter.update(regNegEnt.data,x.shape[0])

#         return {'acc': accMeter.average(), 'loss': lossMeter.average(), 'regL2R': regL2RMeter.average(), 'regCMI': regCMIMeter.average(), 'regNegEnt': regNegEntMeter.average()}

