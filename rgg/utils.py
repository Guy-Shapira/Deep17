import numpy as np
from classes.basic_classes import DatasetType, DataSet
from typing import NamedTuple
import torch
import os.path as osp
from torch_geometric.datasets import Planetoid

import sys
sys.path.append("../")
from rgnn.utils import sparse_tensor

# run command for 2nd paper:
"""
python experiment_train.py with "dataset=pubmed" "seed=5" "model_params={\"label\": \"Soft Medoid GDC (T=1.0)\", \"model\": \"RGNN\", \"do_cache_adj_prep\": True, \"n_filters\": 64, \"dropout\": 0.5, \"mean\": \"soft_k_medoid\", \"mean_kwargs\": {\"k\": 32, \"temperature\": 1.0}, \"svd_params\": None, \"jaccard_params\": None, \"gdc_params\": {\"alpha\": 0.15, \"k\": 32}}" "artifact_dir=cache" "binary_attr=False"
"""


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

def GetPrepGraph(dataset_name, device):
    """
    Calculates the properties returned by prep_graph from Ben-style dataset 
    """
    dataset_path = osp.join(getGitPath(), 'datasets')
    dataset_string = {"pubmed" : "PubMed"}[dataset_name]

    dataset = Planetoid(dataset_path, dataset_string)

    n_vertices = dataset.data.to_dict()['y'].shape[0]

    adj_matrix = torch.zeros((n_vertices, n_vertices))
    edges = dataset.data.to_dict()['edge_index']
    for e in edges.t():
        adj_matrix[e[0], e[1]] = 1

    ### HAVE n_vertices, adj_matrix

    attr = dataset.data.to_dict()['x']
    labels = dataset.data.to_dict()['y']

    # adj_sparse = sparse_tensor(adj_matrix.tocoo())

    adj_sparse = torch.sparse_coo_tensor(edges, torch.ones(edges.shape[1]),
     size=(n_vertices,n_vertices), dtype=torch.uint8)

    return attr.to(device), adj_sparse.coalesce().to(device), labels.to(device)

def Get_Masks(dataset):
    if hasattr(dataset, 'train_mask') and hasattr(dataset, 'val_mask') and hasattr(dataset, 'test_mask'):
        masks = Masks(dataset.train_mask, dataset.val_mask, dataset.test_mask)
        train_mask = [idx for idx in range(masks.train.shape[0]) if masks.train[idx]]
        val_mask = [idx for idx in range(masks.val.shape[0]) if masks.val[idx]]
        test_mask = [idx for idx in range(masks.test.shape[0]) if masks.test[idx]]
        # print(train_mask)

        # print(val_mask)

        # print(test_mask)
        # input("wait Get_Masks")


        return masks
    else:
        raise NotImplementedError("The current datadet doesn't have default masking")
