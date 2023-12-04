import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import pairwise_distances
import math
from .strategy import Strategy
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable

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

class EADA(Strategy):
    # tgt_unlabeled_loader_full, tgt_unlabeled_ds, tgt_selected_ds, active_ratio, totality, model, cfg):
    #
    def query(self, user_idx, label_idxs, unlabel_idxs, n_query=100): 
        self.net.eval()
        TRAINER_ENERGY_BETA = 1.0
        TRAINER_FIRST_SAMPLE_RATIO = 0.5
        loader_te = DataLoader(DatasetSplit(self.dataset_train[user_idx], 
                                            unlabel_idxs), shuffle=False)
        
        first_stat = list()
        with torch.no_grad():
            for x, y, idxs in loader_te:
                
                x, y = Variable(x.to(self.args.device)), Variable(y.to(self.args.device))
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

        first_sample_ratio = TRAINER_FIRST_SAMPLE_RATIO
        first_sample_num = math.ceil(len(unlabel_idxs) * first_sample_ratio)
        # print(first_sample_num)
        # second_sample_ratio = n_query * TRAINER_FIRST_SAMPLE_RATIO
        # second_sample_num = math.ceil(first_sample_num * second_sample_ratio)
        # print(second_sample_num)
        # the first sample using \mathca{F}, higher value, higher consideration
        first_stat = sorted(first_stat, key=lambda x: x[3], reverse=True)
        second_stat = first_stat[:first_sample_num]

        # the second sample using \mathca{U}, higher value, higher consideration
        second_stat = sorted(second_stat, key=lambda x: x[2], reverse=True)
        second_stat = np.array(second_stat)

        # active_samples = second_stat[:second_sample_num, 0:2, ...]
        candidate_ds_index = second_stat[:n_query, 1, ...]
        candidate_ds_index = np.array(candidate_ds_index, dtype=np.int32)

        return candidate_ds_index