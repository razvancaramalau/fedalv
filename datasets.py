
import os
import torch
from PIL import Image, ImageFile
from torchvision import transforms
import torchvision.datasets.folder
from torch.utils.data import Subset, random_split
from torchvision.datasets import ImageFolder
import numpy as np
from util import shard_balance, dir_balance
# from data.meta_dataset import MetaDataset, GetDataLoaderDict
# from configs.default import pacs_path

#from wilds.datasets.camelyon17_dataset import Camelyon17Dataset
#from wilds.datasets.fmow_dataset import FMoWDataset

ImageFile.LOAD_TRUNCATED_IMAGES = True

DATASETS = [
    "PACS",
    "OfficeHome",
    "DomainNet",
]

def get_dataset_class(dataset_name):
    """Return the dataset class with the given name."""
    if dataset_name not in globals():
        raise NotImplementedError("Dataset not found: {}".format(dataset_name))
    return globals()[dataset_name]


def num_environments(dataset_name):
    return len(get_dataset_class(dataset_name).ENVIRONMENTS)


class MultipleDomainDataset:
    EPOCHS = 100             # Default, if train with epochs, check performance every epoch.
    N_WORKERS = 4            # Default, subclasses may override
    ENVIRONMENTS = None      # Subclasses should override
    INPUT_SHAPE = None       # Subclasses should override

    def __getitem__(self, index):
        return self.datasets[index]

    def __len__(self):
        return len(self.datasets)



class MultipleEnvironmentImageFolder(MultipleDomainDataset):
    def __init__(self, root, test_envs, augment):
        super().__init__()
        environments = [f.name for f in os.scandir(root) if f.is_dir()]
        environments = sorted(environments)

        self.transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.augment_transform = transforms.Compose([
            # transforms.Resize((224,224)),
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
            transforms.RandomGrayscale(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.datasets = []
        for i, environment in enumerate(environments):

            if augment and (i not in test_envs):
                env_transform = self.augment_transform
            else:
                env_transform = self.transform

            path = os.path.join(root, environment)
            env_dataset = ImageFolder(path,
                transform=env_transform)

            self.datasets.append(env_dataset)

        self.input_shape = (3, 224, 224,)
        self.num_classes = len(self.datasets[-1].classes)


class PACS(MultipleEnvironmentImageFolder):
    ENVIRONMENTS = ["A", "C", "P", "S"]
    def __init__(self, root, test_envs, augment=True):
        self.dir = os.path.join(root, "PACS/")
        super().__init__(self.dir, test_envs, augment)

class DomainNet(MultipleEnvironmentImageFolder):
    ENVIRONMENTS = ["clip", "info", "paint", "quick", "real", "sketch"]
    def __init__(self, root, test_envs, augment=True):
        self.dir = os.path.join(root, "domainnet/")
        super().__init__(self.dir, test_envs, augment)

class OfficeHome(MultipleEnvironmentImageFolder):
    ENVIRONMENTS = ["A", "C", "P", "R"]
    def __init__(self, root, test_envs, augment=True):
        self.dir = os.path.join(root, "officehome/")
        super().__init__(self.dir, test_envs, augment)

def get_dataset(args):
    
    data = eval(args.dataset)(root=args.dataset_folder,test_envs=[args.test_env])
    source_domains = list(range(num_environments(args.dataset)))
    # source_domains.pop(args.test_env)
    
    if args.domain_per_client:
        args.num_users = len(source_domains)
    datasets = [data.datasets[rank] for rank in source_domains]
    args.dataset_train, args.dataset_test, args.total_data = [], [], []
    args.n_query, args.n_data = [], []
    client_idx = 0
    dict_users_train_total, dict_users_test_total = {}, {}
    
    if args.domain_per_client:
        # Generate the set of clients dataset.
        args.num_users = num_environments(args.dataset)
        for i in range(args.num_users):
            dict_users_train_total[i] = []
            dict_users_test_total[i] = []
    for dataset in datasets:
        train_len = int(len(dataset) * args.train_split)
        test_len = len(dataset) - train_len
        datasettrain, datasettest = random_split(dataset, [train_len,test_len], 
                                    generator=torch.Generator().manual_seed(args.seed))
        # change the transform of test split 
        if hasattr(datasettest.dataset,'transform'):
            import copy
            datasettest.dataset = copy.copy(datasettest.dataset)
            datasettest.dataset.transform = data.transform
    
        args.dataset_train.append(datasettrain)
        args.dataset_test.append(datasettest)
        args.total_data.append(len(datasettrain))

        # No initial partition to multiple clients
        if args.domain_per_client:
            # Generate the set of clients dataset.
            # args.num_users = len(data.ENVIRONMENTS)
            
            dict_users_train_total[client_idx] = list(datasettrain.indices)
            dict_users_test_total[client_idx] = list(datasettest.indices)
            client_idx += 1
            # for cls, data in total_data.items():
            #     cum = list(cumsum[int(cls)].numpy())
            #     tmp = np.split(np.array(data), cum)

            # for client_idx in clients_data.keys():
            #     clients_data[client_idx] += list(tmp[client_idx])
                # clients_data_num[client_idx][int(cls)] += len(list(tmp[client_idx]))
            
            # dict_users_train_total.append(datasettrain.indices)
            # dict_users_test_total.append(datasettest.indices)
        else:
            if args.partition == "shard_balance":
                dict_users_train_total = shard_balance(datasettrain, args)
                dict_users_test_total = shard_balance(datasettest, args)
            elif args.partition == "dir_balance":
                dict_users_train_total, sample = dir_balance(datasettrain, args)
                dict_users_test_total, _ = dir_balance(datasettest, args, sample)
    
        args.n_query.append(round(len(datasettrain) -2) * args.query_ratio)
        args.n_data.append(round(len(datasettrain), -2) * args.current_ratio)
    args.dataset_query = args.dataset_train
    
    return args, dict_users_train_total, dict_users_test_total 