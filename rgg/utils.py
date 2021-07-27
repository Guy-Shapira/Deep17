import numpy as np
from classes.basic_classes import DatasetType, DataSet
from typing import NamedTuple
import torch
import os.path as osp
from torch_geometric.datasets import Planetoid


import sys
sys.path.append("../code")
from helpers.getGitPath import getGitPath


class Masks(NamedTuple):
    train: torch.tensor
    val: torch.tensor
    test: torch.tensor

def loadDataset(dataset, device):
        dataset_path = osp.join(getGitPath(), 'datasets')

        if type(dataset) == type(""):
            if dataset == "pubmed":
                dataset = DataSet.PUBMED
            else:
                raise(Exception("Unrecognized dataset name: {}".format(dataset)))

        if dataset is DataSet.PUBMED or dataset is DataSet.CORA or dataset is DataSet.CITESEER:
            dataset = Planetoid(dataset_path, dataset.string())
        elif dataset is DataSet.TWITTER:
            twitter_glove_path = osp.join(dataset_path, 'twitter', 'glove.pkl')
            if not osp.exists(twitter_glove_path):
                quit("Go to README and follow the download instructions to the TWITTER dataset")
            else:
                dataset = TwitterDataset(osp.dirname(twitter_glove_path))
                with open(twitter_glove_path, 'rb') as file:
                    glove_matrix = pickle.load(file)
                self.glove_matrix = torch.tensor(glove_matrix, dtype=torch.float32).to(device)


        data = dataset[0].to(device)
        setattr(data, 'num_classes', dataset.num_classes)


        return data, dataset.num_classes, dataset.num_features



def Get_Masks(dataset):
    if dataset in [DataSet.PUBMED, DataSet.CORA, DataSet.CITESEER]:
        assert hasattr(data, 'train_mask') and hasattr(data, 'val_mask') and hasattr(data, 'test_mask')
        masks = Masks(data.train_mask, data.val_mask, data.test_mask)

    else:
        raise NotImplementedError("The current datadet doesn't have default masking")
