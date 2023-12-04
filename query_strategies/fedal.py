import numpy as np
import torch
import pdb
import torch.nn as nn
from sklearn.metrics import pairwise_distances
import math
from .strategy import Strategy
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable
from scipy import stats

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        new_idx = np.argwhere(self.dataset.indices==np.array(self.idxs[item]))
        new_idx = new_idx[0][0]
        image, label = self.dataset[new_idx]
        return image, label, self.idxs[item]

def bound_max_loss(energy, bound):
    """
    return the loss value of max(0, \mathcal{F}(x) - \Delta )
    """
    energy_minus_bound = energy - bound
    energy_minus_bound = torch.unsqueeze(energy_minus_bound, dim=1)
    zeros = torch.zeros_like(energy_minus_bound)
    for_select = torch.cat((energy_minus_bound, zeros), dim=1)
    selected = torch.max(for_select, dim=1).values

    return selected.mean()


class FreeEnergyAlignmentLoss(nn.Module):
    """
    free energy alignment loss
    """

    def __init__(self, TRAINER_ENERGY_BETA):
        super(FreeEnergyAlignmentLoss, self).__init__()
        assert TRAINER_ENERGY_BETA > 0, "beta for energy calculate must be larger than 0"
        self.beta = TRAINER_ENERGY_BETA

        self.type = 'max'

        if self.type == 'l1':
            self.loss = nn.L1Loss()
        elif self.type == 'mse':
            self.loss = nn.MSELoss()
        elif self.type == 'max':
            self.loss = bound_max_loss

    def forward(self, inputs, bound):
        mul_neg_beta = -1.0 * self.beta * inputs
        log_sum_exp = torch.logsumexp(mul_neg_beta, dim=1)
        free_energies = -1.0 * log_sum_exp / self.beta

        bound = torch.ones_like(free_energies) * bound
        loss = self.loss(free_energies, bound)

        return loss


class NLLLoss(nn.Module):
    """
    NLL loss for energy based model
    """

    def __init__(self, TRAINER_ENERGY_BETA):
        super(NLLLoss, self).__init__()
        assert TRAINER_ENERGY_BETA > 0, "beta for energy calculate must be larger than 0"
        self.beta = TRAINER_ENERGY_BETA

    def forward(self, inputs, targets):
        indices = torch.unsqueeze(targets, dim=1)
        energy_c = torch.gather(inputs, dim=1, index=indices)

        all_energy = -1.0 * self.beta * inputs
        free_energy = -1.0 * torch.logsumexp(all_energy, dim=1, keepdim=True) / self.beta

        nLL = energy_c - free_energy

        return nLL.mean()


def compute_fea(args, net, target_dict):
    TRAINER_ENERGY_BETA = 1.0
    net.eval()
    loader_target = DataLoader(DatasetSplit(args.dataset_train[args.test_env], target_dict), 
                                batch_size=args.test_bs, shuffle=False)
    first_stat = list()
    with torch.no_grad():
        for x, y, idxs in loader_target:
            
            x, y = x.to(args.device), y.to(args.device)
            tgt_out, _ = net(x)

            # MvSM of each sample
            # minimal energy - second minimal energy
            min2 = torch.topk(tgt_out, k=2, dim=1, largest=False).values
            mvsm_uncertainty = min2[:, 0] - min2[:, 1]

            # free energy of each sample
            output_div_t = -1.0 * tgt_out / TRAINER_ENERGY_BETA
            output_logsumexp = torch.logsumexp(output_div_t, dim=1, keepdim=False)
            free_energy = -1.0 * TRAINER_ENERGY_BETA * output_logsumexp

            for i in range(len(free_energy)):
                first_stat.append([y[i].item(), idxs[i].long(),
                                mvsm_uncertainty[i].item(), free_energy[i].item()])

    second_sample_num = 1000
    second_stat = sorted(first_stat, key=lambda x: x[3], reverse=True)
    second_stat = np.array(second_stat)
    active_target_samples = second_stat[:second_sample_num, 1, ...]
    fe_stat_samples = second_stat[:second_sample_num, 3, ...]
    mvsm_stat_samples = second_stat[:second_sample_num, 2, ...]
    
    target_labels = second_stat[:second_sample_num, 0, ...]

    return active_target_samples, [target_labels, fe_stat_samples, mvsm_stat_samples]
    
class FEDAL(Strategy):
    def closest_first(self, X, X_set, n):
        m = np.shape(X)[0]
        if np.shape(X_set)[0] == 0:
            min_dist = np.tile(float("inf"), m)
        else:
            dist_ctr = pairwise_distances(X, X_set)
            min_dist = np.amin(dist_ctr, axis=1)

        idxs = []

        for i in range(n):
            idx = min_dist.argmin()
            idxs.append(idx)
            dist_new_ctr = pairwise_distances(X, X[[idx], :])
            for j in range(m):
                min_dist[j] = max(min_dist[j], dist_new_ctr[j, 0])

        return idxs

    def furthest_first(self, X, X_set, n):
        m = np.shape(X)[0]
        if np.shape(X_set)[0] == 0:
            min_dist = np.tile(float("inf"), m)
        else:
            dist_ctr = pairwise_distances(X, X_set)
            min_dist = np.amax(dist_ctr, axis=1)

        idxs = []

        for i in range(n):
            idx = min_dist.argmax()
            idxs.append(idx)
            dist_new_ctr = pairwise_distances(X, X[[idx], :])
            for j in range(m):
                min_dist[j] = max(min_dist[j], dist_new_ctr[j, 0])

        return idxs

    def init_centers(self, X, K):
        ind = np.argmax([np.linalg.norm(s, 2) for s in X])
        mu = [X[ind]]
        indsAll = [ind]
        centInds = [0.] * len(X)
        cent = 0
        while len(mu) < K:
            if len(mu) == 1:
                D2 = pairwise_distances(X, mu).ravel().astype(float)
            else:
                newD = pairwise_distances(X, [mu[-1]]).ravel().astype(float)
                for i in range(len(X)):
                    if D2[i] >  newD[i]:
                        centInds[i] = cent
                        D2[i] = newD[i]
            if sum(D2) == 0.0: pdb.set_trace()
            D2 = D2.ravel().astype(float)
            Ddist = (D2 ** 2)/ sum(D2 ** 2)
            customDist = stats.rv_discrete(name='custm', values=(np.arange(len(D2)), Ddist))
            ind = customDist.rvs(size=1)[0]
            while ind in indsAll: ind = customDist.rvs(size=1)[0]
            mu.append(X[ind])
            indsAll.append(ind)
            cent += 1
        return indsAll

    def compute_local_fea(self, user_idx, target_dict, n_query=None):
        TRAINER_ENERGY_BETA = 1.0
        self.net.eval()
        loader_target = DataLoader(DatasetSplit(self.args.dataset_train[user_idx], target_dict), 
                                    batch_size=self.args.test_bs, shuffle=False)
        first_stat = list()
        with torch.no_grad():
            for x, y, idxs in loader_target:
                
                x, y = x.to(self.args.device), y.to(self.args.device)
                tgt_out, _ = self.net(x)

                # MvSM of each sample
                # minimal energy - second minimal energy
                min2 = torch.topk(tgt_out, k=2, dim=1, largest=False).values
                mvsm_uncertainty = min2[:, 0] - min2[:, 1]

                # free energy of each sample
                output_div_t = -1.0 * tgt_out / TRAINER_ENERGY_BETA
                output_logsumexp = torch.logsumexp(output_div_t, dim=1, keepdim=False)
                free_energy = -1.0 * TRAINER_ENERGY_BETA * output_logsumexp
                for i in range(len(free_energy)):
                    first_stat.append([y[i].item(), idxs[i].long(),
                                    mvsm_uncertainty[i].item(), free_energy[i].item()])
        second_stat = np.array(first_stat)
        first_stat_samples = second_stat[:, 2, ...]
        second_stat_samples = second_stat[:, 3, ...]
        return first_stat_samples, second_stat_samples
    
    def closest_to_target(self, X, X_set, n):
        dist_ctr = pairwise_distances(X_set, X)
        argmin_dist = np.argsort(dist_ctr, axis=1)
        idxs = []
        for i in range(n):
            # idx = min_dist.argmax()
            cnt = 0
            idx = argmin_dist[i, cnt]
            while idx in idxs:
                cnt += 1
                idx = argmin_dist[i, cnt]
            idxs.append(idx)
        
        return idxs

    def query(self, user_idxs, unlabel_idxs, n_query=100, 
              target_dict=None, active_target_samples=None,
              first_stat_samples=None): 
        self.net.eval()


        active_target_samples = np.array(active_target_samples)

        unlabel_idxs = np.array(unlabel_idxs)
        source_embedding = self.get_embedding(user_idxs, unlabel_idxs, self.net)
        target_embedding = self.get_embedding(self.args.test_env, active_target_samples, self.net)
        chosen_closest = self.closest_to_target(source_embedding, target_embedding[:n_query], n_query)
        return unlabel_idxs[chosen_closest]
    
    
    def query2(self, user_idxs, unlabel_idxs, n_query=100, 
              target_dict=None, active_target_samples=None,
              first_stat_samples=None): 
        self.net.eval()


        active_target_samples = np.array(active_target_samples)
        target_embedding = self.get_embedding(self.args.test_env, active_target_samples, self.net)
        # chosen_closest = self.closest_to_target(source_embedding, target_embedding[:n_query], n_query)
        interval = [0]
        for idx, user_idx in enumerate(user_idxs):
            
            source_embedding, source_labels = self.get_embedding2(user_idx, unlabel_idxs[idx], self.net)
            # np.save("source_fedalv_%d_feat%f.npy"%(user_idx, self.args.current_ratio), 
            #             np.concatenate((source_embedding, np.expand_dims(source_labels, 1)), 1))
            if idx == 0:
                dist_ctr = pairwise_distances(target_embedding, source_embedding)
                interval.append(dist_ctr.shape[1])
            else:
                dist_ctr = np.concatenate((dist_ctr, pairwise_distances(target_embedding, source_embedding)), axis=1)
                interval.append(dist_ctr.shape[1])
        idxs = []
        n_total = int(sum(n_query))
        argmin_dist = np.argsort(dist_ctr, axis=1)
        for i in range(n_total):
            # idx = min_dist.argmax()
            cnt = 0
            idx = argmin_dist[i, cnt]
            while idx in idxs:
                cnt += 1
                idx = argmin_dist[i, cnt]
            idxs.append(idx)
        sel_idx = [] 
        
        for idx, user_idx in enumerate(user_idxs):          
            intervals = [(interval[idx], interval[idx+1])]
            # Create a list comprehension to select values within the intervals
            selected_values = [value for value in idxs if any(start <= value < end for start, end in intervals)]
            unlab_source = np.array(unlabel_idxs[idx])
            if selected_values == []:
                source_idx = []
            else:
                selected_values = [x - interval[idx] for x in selected_values] 
                source_idx = unlab_source[selected_values]
            sel_idx.append(source_idx)
       

        return sel_idx