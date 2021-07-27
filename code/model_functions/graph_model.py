from classes.basic_classes import DataSet
from model_functions.basicTrainer import basicTrainer
from helpers.fileNamer import fileNamer
from adversarial_attack.adversarialTrainer import adversarialTrainer
from helpers.getGitPath import getGitPath

import os.path as osp
import torch
from torch import nn
from torch.nn import functional as F
import copy
from torch_geometric.data import Data

class Model(torch.nn.Module):
    def __init__(self, gnn_type, num_layers, dataset, device):
        super(Model, self).__init__()
        self.attack = False
        self.layers = nn.ModuleList().to(device)

        if dataset is DataSet.TWITTER:
            self.glove_matrix = dataset.glove_matrix.to(device)
        else:
            self.glove_matrix = torch.eye(dataset.data.x.shape[1]).to(device)

        num_initial_features = dataset.num_features
        num_final_features = dataset.num_classes
        hidden_dims = [32] * (num_layers - 1)
        all_channels = [num_initial_features] + hidden_dims + [num_final_features]

        # gcn layers
        for in_channel, out_channel in zip(all_channels[:-1], all_channels[1:]):
            self.layers.append(gnn_type.get_layer(in_dim=in_channel, out_dim=out_channel).to(device))

        self.name = gnn_type.string()
        self.num_layers = num_layers
        self.device = device
        self.edge_index = dataset.data.edge_index.to(device)
        self.edge_weight = None

    def forward(self, x=None):
        if x is None:
            x = self.getInput().to(self.device)

        x = torch.matmul(x, self.glove_matrix).to(self.device)
        for layer in self.layers[:-1]:
            x = F.relu(layer(x=x, edge_index=self.edge_index, edge_weight=self.edge_weight).to(self.device))\
                .to(self.device)
            x = F.dropout(x, training=self.training and not self.attack).to(self.device)

        x = self.layers[-1](x=x, edge_index=self.edge_index, edge_weight=self.edge_weight).to(self.device)
        return F.log_softmax(x, dim=1).to(self.device)

    def getInput(self):
        raise NotImplementedError

# class CustomNodeModel(Model): # RGG
#     def __init__(self, gnn_type, num_layers, dataset, device):
#         super(CustomNodeModel, self).__init__(gnn_type, num_layers, dataset, device)
#         self.attack = False
#         self.layers = None
#         # Load their model
    
#         import sys 
#         import os
#         sys.path.append("../reliable_gnn_via_robust_aggregation/rgnn")
#         from models import create_model
#         print("This is our model")
#         self.model = create_model({
#             "model": "RGNN",
#             "n_features": 500, # for pubmed
#             "n_classes": 3 
#         })
     
#         self.name = gnn_type.string()
#         self.num_layers = num_layers
#         self.device = device
#         self.edge_index = dataset.data.edge_index.to(device)
#         self.edge_weight = None
#         self.model.layers.to(self.device)

#         data = dataset.data
#         node_attribute_list = []
#         for idx in range(data.x.shape[0]):
#             node_attribute_list += [torch.nn.Parameter(data.x[idx].unsqueeze(0), requires_grad=False).to(device)]
#         self.node_attribute_list = node_attribute_list

#     def getInput(self):
#         return torch.cat(self.node_attribute_list, dim=0)

#     def setNodesAttribute(self, idx_node, idx_attribute, value):
#         self.node_attribute_list[idx_node].data[0][idx_attribute] = value

#     def setNodesAttributes(self, idx_node, values):
#         self.node_attribute_list[idx_node].data[0] = values

#     def forward(self, x=None):
#         if x is None:
#             x = self.getInput().to(self.device)
        
#         # print("Ben ", x.shape)
#         # print(f" Edge_index {self.edge_index}")
#         y = self.model.forward(Data(x=x, edge_index=self.edge_index))

#         # RGG
#         # return F.log_softmax(y, dim=1).to(self.device)
#         return y

class NodeModel(Model):
    def __init__(self, gnn_type, num_layers, dataset, device):
        super(NodeModel, self).__init__(gnn_type, num_layers, dataset, device)
        data = dataset.data
        node_attribute_list = []
        for idx in range(data.x.shape[0]):
            node_attribute_list += [torch.nn.Parameter(data.x[idx].unsqueeze(0), requires_grad=False).to(device)]
        self.node_attribute_list = node_attribute_list

    def getInput(self):
        return torch.cat(self.node_attribute_list, dim=0)

    def setNodesAttribute(self, idx_node, idx_attribute, value):
        self.node_attribute_list[idx_node].data[0][idx_attribute] = value

    def setNodesAttributes(self, idx_node, values):
        self.node_attribute_list[idx_node].data[0] = values


class EdgeModel(Model):
    def __init__(self, gnn_type, num_layers, dataset, device):
        super(EdgeModel, self).__init__(gnn_type, num_layers, dataset, device)
        data = dataset.data
        self.x = data.x.to(device)
        self.edge_weight = torch.nn.Parameter(torch.ones(data.edge_index.shape[1]), requires_grad=False).to(device)

    def getInput(self):
        return self.x

    # a helper function which adds the new possible edges in 2 modes
    # full = True : adds all edges from the whole graph to the neighbourhood
    # full = False : adds all edges from malicious index to the neighbourhood
    @torch.no_grad()
    def expandEdges(self, dataset, attacked_node, neighbours, device, expansion_mode):
        data = dataset.data
        clique = torch.cat((attacked_node, neighbours))
        n = data.num_nodes

        zero_dim_edge_index = []
        first_dim_edge_index = []
        for neighbour_num, neighbour in enumerate(clique):
            ignore = dataset.reversed_arr_list[neighbour]  # edges which already exist

            tmp_zero_dim_edge_index = []
            if expansion_mode["full"]:
                # adds all edges from the whole graph to the neighbourhood
                tmp_zero_dim_edge_index = [idx for idx in range(n) if idx not in ignore]
            else:
                # adds all edges from malicious index to the neighbourhood
                if expansion_mode["malicious_index"] not in ignore:
                    tmp_zero_dim_edge_index = [expansion_mode["malicious_index"]]

            zero_dim_edge_index += tmp_zero_dim_edge_index
            first_dim_edge_index += [neighbour.item()] * len(tmp_zero_dim_edge_index)

        if zero_dim_edge_index:
            model_edge_index = torch.tensor([zero_dim_edge_index, first_dim_edge_index]).to(device)
            model_edge_weight = torch.zeros(len(zero_dim_edge_index)).to(device)

            self.edge_index = torch.cat((self.edge_index, model_edge_index), dim=1)
            self.edge_weight.data = torch.cat((self.edge_weight.data, model_edge_weight))

# class AdversarialModelWrapper(ModelWrapper):
#     def __init__(self, node_model, gnn_type, num_layers, dataset, patience, device, seed):
#         super(AdversarialModelWrapper, self).__init__(node_model, gnn_type, num_layers, dataset, patience, device, seed)

#     # override
#     def _setLR(self):
#         self.lr = 0.005

#     def useTrainer(self, data, attack=None):
#         return adversarialTrainer(attack=attack)
