import os.path as osp
import torch
from torch import nn
from torch.nn import functional as F
import copy
from torch_geometric.data import Data
import sys 
import os

import sys
sys.path.append("../code/")
from helpers.getGitPath import getGitPath
from helpers.fileNamer import fileNamer
from model_functions.graph_model import NodeModel, EdgeModel, Model
from model_functions.basicTrainer import basicTrainer

sys.path.append("../reliable_gnn_via_robust_aggregation/")
from rgnn.models import create_model

class CustomNodeModel(NodeModel): # RGG
    def __init__(self, gnn_type, num_layers, dataset, device):
        super(CustomNodeModel, self).__init__(gnn_type, num_layers, dataset, device)
        self.attack = False
        self.layers = None
        # Load their model
    
        sys.path.append("../reliable_gnn_via_robust_aggregation/rgnn")
        print("This is our model")
        self.model = create_model({
            "model": "RGNN",
            "n_features": 500, # for pubmed
            "n_classes": 3 
        })
     
        self.name = gnn_type.string()
        self.num_layers = num_layers
        self.device = device
        self.edge_index = dataset.data.edge_index.to(device)
        self.edge_weight = None
        self.model.layers.to(self.device)

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

    def forward(self, x=None):
        if x is None:
            x = self.getInput().to(self.device)
        
        # print("Ben ", x.shape)
        # print(f" Edge_index {self.edge_index}")
        y = self.model.forward(Data(x=x, edge_index=self.edge_index))

        # RGG
        # return F.log_softmax(y, dim=1).to(self.device)
        return y


class ModelWrapper(object):
    def __init__(self, node_model, gnn_type, num_layers, dataset, patience, device, seed):
        self.gnn_type = gnn_type
        self.num_layers = num_layers
        print("ModelWrapper init")
        if node_model:
            self.model = NodeModel(gnn_type, num_layers, dataset, device)
        else:
            self.model = EdgeModel(gnn_type, num_layers, dataset, device)
        # self.model = CustomNodeModel(gnn_type, num_layers, dataset, device)
        print("self.model is our model")
        # input("wait")
        self.node_model = node_model
        self.patience = patience
        self.device = device
        self.seed = seed
        self._setOptimizer()

        self.basic_log = None
        self.clean = None

    def _setOptimizer(self):
        model = self.model
        layers_obj = model.layers
        if model.layers is None:
            layers_obj = model.model.layers # RGG bad patch again

        list_dict_param = [dict(params=layers_obj[0].parameters(), weight_decay=5e-4)]
        for layer in layers_obj[1:]:
            list_dict_param += [dict(params=layer.parameters(), weight_decay=0)]
        self._setLR()
        self.optimizer = torch.optim.Adam(list_dict_param, lr=self.lr)  # Only perform weight-decay on first convolution.

    def _setLR(self):
        self.lr = 0.01

    def setModel(self, model):
        self.model = copy.deepcopy(model)

    def train(self, dataset, attack=None):
        model = self.model
        folder_name = osp.join(getGitPath(), 'models')
        if attack is None:
            folder_name = osp.join(folder_name, 'basic_models')
            targeted, attack_epochs = None, None
        else:
            folder_name = osp.join(folder_name, 'adversarial_models')
            targeted, attack_epochs = attack.targeted, attack.attack_epochs

        file_name = fileNamer(node_model=self.node_model, dataset_name=dataset.name, model_name=model.name,
                              num_layers=model.num_layers, patience=self.patience, seed=self.seed, targeted=targeted,
                              attack_epochs=attack_epochs, end='.pt')

        file_name = "pretrained_118.pt"
        file_name = "fuck_you_ron's_mom"
        model_path = osp.join(folder_name, file_name)


        # load model and optimizer
        if not osp.exists(model_path):
            # train model
            model, model_log, test_acc = self.useTrainer(data=dataset.data, attack=attack)
            #RGG - to remove!
            if file_name != "fuck_you_ron's_mom":
                torch.save((model.model.state_dict(), model_log, test_acc), model_path)
        else:
            print("Loading model from {}".format(model_path))
            model_state_dict = torch.load(model_path)
            # model_state_dict, model_log, test_acc = torch.load(model_path)
            model.model.load_state_dict(model_state_dict)
            # print(model_log + '\n')
        # self.basic_log = model_log
        # self.clean = test_acc

    def useTrainer(self, data, attack=None):
        return basicTrainer(self.model, self.optimizer, data, self.patience)


# class AdversarialModelWrapper(ModelWrapper):
#     """
#         a wrapper which includes an adversarial model
#         more information at ModelWrapper
#     """
#     def __init__(self, node_model, gnn_type, num_layers, dataset, patience, device, seed):
#         super(AdversarialModelWrapper, self).__init__(node_model, gnn_type, num_layers, dataset, patience, device, seed)

#     # override
#     def _setLR(self):
#         """
#             information at the base class ModelWrapper
#         """
#         self.lr = 0.005

#     def useTrainer(self, dataset: GraphDataset, attack=None) -> Tuple[Model, str, torch.Tensor]:
#         """
#             information at the base class ModelWrapper
#         """
#         return adversarialTrainer(attack=attack)
