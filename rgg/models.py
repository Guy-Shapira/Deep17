import os.path as osp
import torch
from torch import nn
from torch.nn import functional as F
import copy
from torch_geometric.data import Data
import sys 
from typing import NamedTuple
import os
from enum import Enum, auto
from functools import reduce
import torch_geometric
# import sys
# sys.path.append("../code/")
# from model_functions.graph_model import Model

sys.path.append("../reliable_gnn_via_robust_aggregation/")
sys.path.append("../reliable_gnn_via_robust_aggregation/rgnn")

from models import create_model

class DatasetType(Enum):
    """
        an object for the different types of datasets
    """
    CONTINUOUS = auto()
    DISCRETE = auto()



class DataSet(Enum):
    """
        an object for the different datasets
    """
    PUBMED = auto()
    CORA = auto()
    CITESEER = auto()
    TWITTER = auto()

    @staticmethod
    def from_string(s):
        try:
            return DataSet[s]
        except KeyError:
            raise ValueError()

    def get_type(self) -> DatasetType:
        """
            gets the dataset type for each dataset

            Returns
            -------
            DatasetType
        """
        if self is DataSet.PUBMED or self is DataSet.TWITTER:
            return DatasetType.CONTINUOUS
        elif self is DataSet.CORA or self is DataSet.CITESEER:
            return DatasetType.DISCRETE

    def get_l_inf(self) -> float:
        """
            Get the default l_inf

            Returns
            -------
            l_inf: float
        """
        if self.get_type() is DatasetType.DISCRETE:
            return 1
        if self is DataSet.PUBMED:
            return 0.04
        if self is DataSet.TWITTER:
            return 0.001

    def get_l_0(self) -> float:
        """
            Get the default l_0

            Returns
            -------
            l_0: float
        """
        if self.get_type() is DatasetType.DISCRETE:
            return 0.01
        if self is DataSet.PUBMED:
            return 0.05
        if self is DataSet.TWITTER:
            return 0.05

    def string(self) -> str:
        """
            converts dataset to string

            Returns
            -------
            dataset_name: str
        """
        if self is DataSet.PUBMED:
            return "PubMed"
        elif self is DataSet.CORA:
            return "Cora"
        elif self is DataSet.CITESEER:
            return "CiteSeer"
        elif self is DataSet.TWITTER:
            return "twitter"



class Masks(NamedTuple):
    """
        a Mask object with the following fields:
    """
    train: torch.tensor
    val: torch.tensor
    test: torch.tensor


class GraphDataset(object):
    """
        a base class for datasets

        Parameters
        ----------
        dataset: DataSet
        device: torch.device
    """
    def __init__(self, dataset: DataSet, device: torch.device):
        super(GraphDataset, self).__init__()
        name = dataset.string()
        self.name = name
        self.device = device

        data = self._loadDataset(dataset, device)

        if dataset is DataSet.TWITTER:
            train_mask, val_mask, test_mask = torch.load('../masks/twitter.dat')
            setattr(data, 'train_mask', train_mask)
            setattr(data, 'val_mask', val_mask)
            setattr(data, 'test_mask', test_mask)
            setattr(data, 'test_mask', test_mask)
            self.num_features = 200# after multiplying x by the golve matrix the new feature dim is 200
            self.num_classes = data.num_classes
        else:
            self._setMasks(data, name)

        self._setReversedArrayList(data)

        self.data = data
        self.type = dataset.get_type()

    def _loadDataset(self, dataset: DataSet, device: torch.device) -> torch_geometric.data.Data:
        """
            a loader function for the requested dataset
        """
        dataset_path = osp.join(getGitPath(), 'datasets')
        if dataset is DataSet.PUBMED or dataset is DataSet.CORA or dataset is DataSet.CITESEER:
            dataset = Planetoid(dataset_path, dataset.string())
        elif dataset is DataSet.TWITTER:
            twitter_glove_path = osp.join(dataset_path, 'twitter', 'glove.pkl')
            if not osp.exists(twitter_glove_path):
                exit("Go to README and follow the download instructions to the TWITTER dataset")
            else:
                dataset = TwitterDataset(osp.dirname(twitter_glove_path))
                with open(twitter_glove_path, 'rb') as file:
                    glove_matrix = pickle.load(file)
                self.glove_matrix = torch.tensor(glove_matrix, dtype=torch.float32).to(device)

        data = dataset[0].to(self.device)
        setattr(data, 'num_classes', dataset.num_classes)

        self.num_features = data.num_features
        self.num_classes = dataset.num_classes
        return data

    def _setMasks(self, data: torch_geometric.data.Data, name: str):
        """
            sets train,val and test mask for the data

            Parameters
            ----------
            data: torch_geometric.data.Data
            name: str
        """
        if not hasattr(data, 'train_mask') or not hasattr(data, 'val_mask') or not hasattr(data, 'test_mask'):
            self.train_percent = train_percent = 0.1
            self.val_percent = val_percent = 0.3
            masks = self._generateMasks(data, name, train_percent, val_percent)
        else:
            masks = Masks(data.train_mask, data.val_mask, data.test_mask)

        setattr(data, 'train_mask', masks.train)
        setattr(data, 'val_mask', masks.val)
        setattr(data, 'test_mask', masks.test)

    def _generateMasks(self, data: torch_geometric.data.Data, name: str, train_percent: float, val_percent: float)\
            -> Masks:
        """
            generates train,val and test mask for the data

            Parameters
            ----------
            data: torch_geometric.data.Data
            name: str
            train_percent: float
            val_percent: float

            Returns
            -------
            masks: Masks
        """
        train_mask = torch.zeros(data.num_nodes).type(torch.bool)
        val_mask = torch.zeros(data.num_nodes).type(torch.bool)
        test_mask = torch.zeros(data.num_nodes).type(torch.bool)

        # taken from Planetoid
        for c in range(data.num_classes):
            idx = (data.y == c).nonzero(as_tuple=False).view(-1)
            num_train_per_class = round(idx.size(0) * train_percent)
            num_val_per_class = round(idx.size(0) * val_percent)

            idx_permuted = idx[torch.randperm(idx.size(0))]
            train_idx = idx_permuted[:num_train_per_class]
            val_idx = idx_permuted[num_train_per_class:num_train_per_class + num_val_per_class]
            test_idx = idx_permuted[num_train_per_class + num_val_per_class:]

            train_mask[train_idx] = True
            val_mask[val_idx] = True
            test_mask[test_idx] = True
            print(f'Class {c}: training: {train_idx.size(0)}, val: {val_idx.size(0)}, test: {test_idx.size(0)}')

        masks = Masks(train_mask, val_mask, test_mask)
        torch.save(masks, osp.join(getGitPath(), 'masks', name + '.dat'))
        return masks

    # converting graph edge index representation to graph array list representation
    def _setReversedArrayList(self, data: torch_geometric.data.Data):
        """
            creates a reversed array list from the edges

            Parameters
            ----------
            data: torch_geometric.data.Data
        """
        edge_index = data.edge_index
        num_nodes = data.num_nodes
        reversed_arr_list = [[] for _ in range(num_nodes)]

        for idx, column in enumerate(edge_index.T):
            edge_from = edge_index[0, idx].item()
            edge_to = edge_index[1, idx].item()

            # swapping positions to find all the neighbors that can go to the root
            reversed_arr_list[edge_to].append(edge_from)

        self.reversed_arr_list = reversed_arr_list




class Model(torch.nn.Module):
    """
        Generic model class
        each gnn sets a different model

        Parameters
        ----------
        gnn_type: GNN_TYPE
        num_layers: int
        dataset: GraphDataset
        device: torch.cuda
    """
    def __init__(self, gnn_type, num_layers: int, dataset: GraphDataset, device: torch.cuda):
        super(Model, self).__init__()
        self.attack = False
        self.layers = nn.ModuleList().to(device)
        data = dataset.data

        if hasattr(dataset, 'glove_matrix'):
            self.glove_matrix = dataset.glove_matrix.to(device)
        else:
            self.glove_matrix = torch.eye(data.x.shape[1]).to(device)

        num_initial_features = dataset.num_features
        num_final_features = dataset.num_classes
        hidden_dims = [32] * (num_layers - 1)
        all_channels = [num_initial_features] + hidden_dims + [num_final_features]


        # for in_channel, out_channel in zip(all_channels[:-1], all_channels[1:]):
        #     self.layers.append(gnn_type.get_layer(in_dim=in_channel, out_dim=out_channel).to(device))

        self.name = "RGNN_RGG"
        self.num_layers = num_layers
        self.device = device
        self.edge_index = data.edge_index.to(device)
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

    def getInput(self) -> torch.Tensor:
        """
            a get function for the models input

            Returns
            ----------
            model_input: torch.Tensor
        """
        raise NotImplementedError

    def injectNode(self, dataset: GraphDataset, attacked_node: torch.Tensor) -> torch.Tensor:
        """
            injects a node to the model

            Parameters
            ----------
            dataset: GraphDataset
            attacked_node: torch.Tensor - the victim/attacked node

            Returns
            -------
            malicious_node: torch.Tensor - the injected/attacker/malicious node
            dataset: GraphDataset - the injected dataset
        """
        raise NotImplementedError

    def removeInjectedNode(self, attack):
        """
            removes the injected node from the model

            Parameters
            ----------
            attack: oneGNNAttack
        """
        raise NotImplementedError

class CustomNodeModel(Model): # RGG
    def __init__(self, gnn_type, num_layers, dataset, device):
        super(CustomNodeModel, self).__init__(gnn_type, num_layers, dataset, device)
        self.attack = False
        self.layers = None
        # Load their model
    
        sys.path.append("../reliable_gnn_via_robust_aggregation/rgnn")
        self.model = create_model({
            "model": "RGNN",
            "n_features": 500, # for pubmed
            "n_classes": 3 
        })
     
        self.layers = self.model.layers
        
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

    def is_zero_grad(self) -> bool:
        nodes_with_gradient = filter(lambda node: node.grad is not None, self.node_attribute_list)
        abs_gradients = map(lambda node: node.grad.abs().sum().item(), nodes_with_gradient)
        if reduce(lambda x, y: x + y, abs_gradients) == 0:
            return True
        else:
            return False

    def forward(self, x=None):
        # print("our forward")

        if x is None:
            x = self.getInput().to(self.device)
        
        # print("Ben ", x.shape)
        # print(f" Edge_index {self.edge_index}")
        y = self.model.forward(Data(x=x, edge_index=self.edge_index))

        # RGG
        # return F.log_softmax(y, dim=1).to(self.device)
        return y


# class ModelWrapper(object):
#     def __init__(self, node_model, gnn_type, num_layers, dataset, patience, device, seed):
#         self.gnn_type = gnn_type
#         self.num_layers = num_layers
#         print("ModelWrapper init")
#         # if node_model:
#         #     self.model = NodeModel(gnn_type, num_layers, dataset, device)
#         # else:
#         #     self.model = EdgeModel(gnn_type, num_layers, dataset, device)
#         self.model = CustomNodeModel(gnn_type, num_layers, dataset, device)

#         self.node_model = node_model
#         self.patience = patience
#         self.device = device
#         self.seed = seed
#         self._setOptimizer()

#         self.basic_log = None
#         self.clean = None

#     def _setOptimizer(self):
#         model = self.model
#         list_dict_param = [dict(params=model.model.layers[0].parameters(), weight_decay=5e-4)]
#         for layer in model.model.layers[1:]:
#             list_dict_param += [dict(params=layer.parameters(), weight_decay=0)]
#         self._setLR()
#         self.optimizer = torch.optim.Adam(list_dict_param, lr=self.lr)  # Only perform weight-decay on first convolution.

#     def _setLR(self):
#         self.lr = 0.01

#     def setModel(self, model):
#         self.model = copy.deepcopy(model)

#     def train(self, dataset, attack=None):
#         model = self.model
#         folder_name = osp.join(getGitPath(), 'models')
#         if attack is None:
#             folder_name = osp.join(folder_name, 'basic_models')
#             targeted, attack_epochs = None, None
#         else:
#             folder_name = osp.join(folder_name, 'adversarial_models')
#             targeted, attack_epochs = attack.targeted, attack.attack_epochs

#         file_name = fileNamer(node_model=self.node_model, dataset_name=dataset.name, model_name=model.name,
#                               num_layers=model.num_layers, patience=self.patience, seed=self.seed, targeted=targeted,
#                               attack_epochs=attack_epochs, end='.pt')

#         file_name = "pretrained_118.pt"
#         file_name = "fuck_you_ron's_mom"
#         model_path = osp.join(folder_name, file_name)


#         # load model and optimizer
#         if not osp.exists(model_path):
#             # train model
#             model, model_log, test_acc = self.useTrainer(data=dataset.data, attack=attack)
#             torch.save((model.state_dict(), model_log, test_acc), model_path)
#         else:
#             print("Loading model from {}".format(model_path))
#             model_state_dict = torch.load(model_path)
#             # model_state_dict, model_log, test_acc = torch.load(model_path)
#             model.model.load_state_dict(model_state_dict)
#             # print(model_log + '\n')
#         # self.basic_log = model_log
#         # self.clean = test_acc

#     def useTrainer(self, data, attack=None):
#         return basicTrainer(self.model, self.optimizer, data, self.patience)
