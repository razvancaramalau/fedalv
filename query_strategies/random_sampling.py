import random
import numpy as np
from .strategy import Strategy


class RandomSampling(Strategy):
    def query(self, user_idx, label_idxs, unlabel_idxs, n_query=100):
                # for idx, user_idx in enumerate(user_idxs):
        self.net.eval() 
        source_embedding, source_labels = self.get_embedding2(user_idx, unlabel_idxs, self.net)
        np.save("source_%d_feat%f.npy"%(user_idx, self.args.current_ratio), np.concatenate((source_embedding, np.expand_dims(source_labels, 1)), 1))
        return random.sample(unlabel_idxs, n_query)