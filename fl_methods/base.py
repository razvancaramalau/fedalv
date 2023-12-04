import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
from util import DatasetSplit, weight_clip

def compute_w_ga(p_loss_locals, loss_locals, step_size, p_w_ga):
    value = []
    for x, y in zip(p_loss_locals, loss_locals):
        value.append(y - x)
    value = np.array(value)
    if np.max(np.absolute(value)) == 0:
        norm_gap_list = value
    else:
        norm_gap_list = value / np.max(np.absolute(value))
    w_ga = p_w_ga +  1.0 * norm_gap_list * step_size
    w_ga = weight_clip(w_ga)
    return w_ga

class FederatedLearning:
    def __init__(self, args, dict_users_train_label=None):
        self.args = args
        self.dict_users_train_label = dict_users_train_label
        self.loss_func = nn.CrossEntropyLoss()

    def train(self):
        pass  

    def aggregate(self, w_glob, w_local, idx_user, total_data_num, total_users):
        if w_glob is None:
            w_glob = copy.deepcopy(w_local)
            for k in w_glob.keys():
                w_glob[k] = w_local[k] *  \
                            len(self.dict_users_train_label[idx_user]) / total_data_num
        else:
            for k in w_glob.keys():
                w_glob[k] += w_local[k] *  \
                            len(self.dict_users_train_label[idx_user]) / total_data_num

        return w_glob

    def aggregate2(self, w_glob, w_local, idx_user, total_data_num, total_users, w_ga=1.0):
        if w_glob is None:
            w_glob = copy.deepcopy(w_local)
            for k in w_glob.keys():
                w_glob[k] = w_local[k] * w_ga #* \
                            #len(self.dict_users_train_label[idx_user]) / total_data_num
        else:
            for k in w_glob.keys():
                w_glob[k] += w_local[k] * w_ga #* \
                            #len(self.dict_users_train_label[idx_user]) / total_data_num

        return w_glob

    def test(self, net_g, dataset):
        data_loader = DataLoader(dataset, batch_size=self.args.test_bs)
        data_nums = len(data_loader.dataset)

        net_g.eval()
        
        test_loss, correct = 0, 0
        probs = []
        for idx, (data, target) in enumerate(data_loader):
            # if self.args.dataset in ['pathmnist', 'octmnist', 'organamnist', 'dermamnist', 'bloodmnist']:
            #     target = target.squeeze().long()

            # if self.args.gpu != -1:
            data, target = data.to(self.args.device), target.to(self.args.device)
            output, emb = net_g(data)
            # output = net_g(data)

            # sum up batch loss
            test_loss += self.loss_func(output, target).item()
            # get the index of the max log-probability
            y_pred = output.data.min(1, keepdim=True)[1]
            correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()

        test_loss /= data_nums
        accuracy = 100.00 * float(correct) / data_nums

        return accuracy, test_loss

    def test2(self, net_g, dataset):
        data_loader = DataLoader(dataset, batch_size=128)
        data_nums = len(data_loader.dataset)

        net_g.eval()
        
        test_loss, correct = 0, 0
        probs = []
        correct = []
        my_dict = {}
        accuracy = np.zeros(7)
        divider  = np.ones(7)
        for idx, (data, target) in enumerate(data_loader):
            # if self.args.dataset in ['pathmnist', 'octmnist', 'organamnist', 'dermamnist', 'bloodmnist']:
            #     target = target.squeeze().long()

            # if self.args.gpu != -1:
            data, target = data.to(self.args.device), target.to(self.args.device)
            output, emb = net_g(data)
            # output = net_g(data)

            # sum up batch loss
            # test_loss += self.loss_func(output, target).item()
            # get the index of the max log-probability
            y_pred = output.data.min(1, keepdim=True)[1]
            target = target.data.view_as(y_pred)
            target = target.squeeze().long().cpu()
            # temp_acc = Accuracy(task="multiclass",average=None, num_classes=7)(y_pred.squeeze().long().cpu(), target).numpy()
            # nan_val = np.isnan(temp_acc)
            # if idx > 0:
            #     for i, x in enumerate(nan_val):
            #         if not x:
            #             divider[i] += 1 
            # temp_acc = np.nan_to_num(temp_acc, nan=0)
            # accuracy += temp_acc
            # correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()
        # test_loss /= data_nums
        # accuracy = 100.00 * float(correct) / data_nums
        accuracy = np.divide(accuracy, divider)
        
        return accuracy

    def on_round_start(self, net_glob=None):
        pass

    def on_user_iter_start(self, dataset, user_idx):
        data_idx = self.dict_users_train_label[user_idx]
        self.data_loader = DataLoader(DatasetSplit(dataset, data_idx), batch_size=self.args.local_bs, shuffle=True)
        self.target_loader = DataLoader(self.args.dataset_train[self.args.test_env], 
                                        batch_size=self.args.local_bs, shuffle=True)

    def on_round_end(self, idxs_users=None):
        pass

    def on_user_iter_end(self):
        pass
