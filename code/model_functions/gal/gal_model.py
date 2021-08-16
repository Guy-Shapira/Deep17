import torch
from torch_geometric.nn import GCNConv
import torch.nn.functional as F

from torch_geometric.utils import train_test_split_edges

torch.autograd.set_detect_anomaly(True)

class GradReverse(torch.autograd.Function):
    """
    Implement the gradient reversal layer for the convenience of domain adaptation neural network.
    The forward part is the identity function while the backward part is the negative function.
    """
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg()

class GradientReversalLayer(torch.nn.Module):
    def __init__(self):
        super(GradientReversalLayer, self).__init__()

    def forward(self, inputs):
        return GradReverse.apply(inputs)


class GalModel(torch.nn.Module):
    def __init__(self, dataset, device):
        super(GalModel, self).__init__()

        self.conv1 = GCNConv(dataset.num_features, 64).to(device)
        self.conv2 = GCNConv(64, 64).to(device)
        self.conv3 = GCNConv(64, 64).to(device)

        # for _setOptimizer, but not important what's inside
        # since we ignore _setOptimizer's actions and override it
        self.layers = torch.nn.ModuleList([self.conv1, self.conv2, self.conv3])
        self.num_layers = len(self.layers)


        self.attr = GCNConv(64, dataset.num_classes, cached=True,
                                normalize=True).to(device)
        self.attk = GCNConv(64, dataset.num_classes, cached=True,
                            normalize=True).to(device)
        self.reverse = GradientReversalLayer().to(device)


        # Ben's attack att
        self.attack = True

        # start of changes XXXXX
        
        data = dataset.data
        self.data = data

        if hasattr(dataset, 'glove_matrix'):
            self.glove_matrix = dataset.glove_matrix.to(device)
        else:
            self.glove_matrix = torch.eye(data.x.shape[1]).to(device)

        self.name = 'GAL'
        self.device = device
        self.edge_index = data.edge_index.to(device)
        # self.edge_weight = data.edge_attr.to(device)

        node_attribute_list = []
        for idx in range(data.x.shape[0]):
            node_attribute_list += [torch.nn.Parameter(data.x[idx].unsqueeze(0), requires_grad=False).to(device)]
        self.node_attribute_list = node_attribute_list

        self.labels = data.y.to(self.device)


    def forward(self, pos_edge_index=None, neg_edge_index=None, input=None):

        if input is None:
            input = self.getInput().to(self.device)
        x = torch.matmul(input, self.glove_matrix).to(self.device)

        x = F.relu(self.conv1(x, self.edge_index))
        x = self.conv2(x, self.edge_index)
        x = self.conv3(x, self.edge_index)

        feat = x
        attr = self.attr(x, self.edge_index)

        if pos_edge_index is None and neg_edge_index is None:
            return F.log_softmax(attr, dim=1)

        else:
            attack = self.reverse(x)
            att = self.attk(attack, self.edge_index)    
            total_edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)

            x_j = torch.index_select(x, 0, total_edge_index[0])
            x_i = torch.index_select(x, 0, total_edge_index[1])
            res = torch.einsum("ef,ef->e", x_i, x_j)

            return res, F.log_softmax(attr, dim=1), att, feat        



    # def forward(self, pos_edge_index, neg_edge_index, input=None):

    #     if input is None:
    #         input = self.getInput().to(self.device)
    #     x = torch.matmul(input, self.glove_matrix).to(self.device)

    #     x = F.relu(self.conv1(x, self.edge_index))
    #     x = self.conv2(x, self.edge_index)
    #     x = self.conv3(x, self.edge_index)

    #     feat = x
    #     attr = self.attr(x, self.edge_index)

    #     attack = self.reverse(x)
    #     att = self.attk(attack, self.edge_index)

    #     total_edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)

    #     x_j = torch.index_select(x, 0, total_edge_index[0])
    #     x_i = torch.index_select(x, 0, total_edge_index[1])
    #     res = torch.einsum("ef,ef->e", x_i, x_j)

    #     return res, F.log_softmax(attr, dim=1), att, feat

    def getInput(self):
        return torch.cat(self.node_attribute_list, dim=0)

    def setNodesAttribute(self, idx_node, idx_attribute, value):
        self.node_attribute_list[idx_node][0][idx_attribute] = value

    def setNodesAttributes(self, idx_node, values):
        self.node_attribute_list[idx_node][0] = values
