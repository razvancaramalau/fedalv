import copy
import torch
import numpy as np
from sklearn.metrics import pairwise_distances

from .strategy import Strategy


# def compute_target_bv2b(net, data_loader, n_query=100):
#         loader_te = DataLoader(DatasetSplit(self.dataset_query[user_idx], unlabel_idxs), shuffle=False)
    
#     if net is None:
#         net = self.net
        
#     net.eval()
#     probs = torch.zeros([len(unlabel_idxs), self.args.num_classes])
#     with torch.no_grad():
#         for x, y, idxs in loader_te:
#             x, y = Variable(x.to(self.args.device)), Variable(y.to(self.args.device))
#             output, emb = net(x)
#             probs[idxs] = torch.nn.functional.softmax(output, dim=1).cpu().data
#     return probs

class bv2b_core(Strategy):
    def furthest_first(self, X, X_set, n):
        m = np.shape(X)[0]
        if np.shape(X_set)[0] == 0:
            min_dist = np.tile(float("inf"), m)
        else:
            dist_ctr = pairwise_distances(X, X_set)
            min_dist = np.amin(dist_ctr, axis=1)

        idxs = []

        for i in range(n):
            idx = min_dist.argmax()
            idxs.append(idx)
            dist_new_ctr = pairwise_distances(X, X[[idx], :])
            for j in range(m):
                min_dist[j] = min(min_dist[j], dist_new_ctr[j, 0])

        return idxs
    
    def compute_target_bv2b(self, user_idx, idxs, net, budget):
        # print(len(idxs))
        probs = self.predict_prob(user_idx, idxs, net)
        # print(probs, probs.shape)
        sorted_probs, _ = torch.sort(probs, 1, descending=True)
        # print(sorted_probs)
        bv2b = torch.absolute(sorted_probs[:,0] - sorted_probs[:,1])
        _, bv2b_idx = torch.sort(bv2b, descending=False)
        return bv2b_idx[:budget]
    
    def query(self, user_idx, unlabel_idxs, label_idxs, target_data_idx, n_query=100):
        data_idxs = list(unlabel_idxs) 
        
        unlabel_idxs = np.array(unlabel_idxs)
        label_idxs = np.array(label_idxs)
        # print(target_data_idx)
        if self.args.query_model_mode == "global":
            target_embeddings = self.get_embedding(self.args.test_env, target_data_idx, self.net)
            embedding = self.get_embedding(user_idx, data_idxs, self.net)
        elif self.args.query_model_mode == "local_only":
            local_net = self.training_local_only(user_idx, label_idxs)
            embedding = self.get_embedding(user_idx, data_idxs, local_net)
            target_embeddings = self.get_embedding(self.args.test_env, target_data_idx, local_net)
        
        # embedding = embedding.numpy()

        chosen = self.furthest_first(embedding[:len(unlabel_idxs), :], target_embeddings, n_query)

        return unlabel_idxs[chosen]