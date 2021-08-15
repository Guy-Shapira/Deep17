import os.path as osp

import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from torch_geometric.utils import (negative_sampling, remove_self_loops,
                                   add_self_loops)
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, ChebConv, GINConv, GATConv  # noqa
from torch_geometric.utils import train_test_split_edges
import argparse
import numpy as np
import random
import os
from sklearn.metrics import roc_auc_score, f1_score
import json
from torch.nn import Sequential, ReLU, Linear
from tqdm import tqdm

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

def sim(lambda_reg, seed):
    class Net(torch.nn.Module):
        def __init__(self, name='GCNConv'):
            super(Net, self).__init__()
            self.name = name
            if (name == 'GCNConv'):
                self.conv1 = GCNConv(dataset.num_features, 64)
                self.conv2 = GCNConv(64, 64)
                self.conv3 = GCNConv(64, 64)
            elif (name == 'ChebConv'):
                self.conv1 = ChebConv(dataset.num_features, 64, K=2)
                self.conv2 = ChebConv(64, 64, K=2)
                self.conv3 = ChebConv(64, 64, K=2)
            elif (name == 'GATConv'):
                self.conv1 = GATConv(dataset.num_features, 64)
                self.conv2 = GATConv(64, 64)
                self.conv3 = GATConv(64, 64)

            self.attr = GCNConv(64, dataset.num_classes, cached=True,
                                    normalize=not gdc)

            self.attack = GCNConv(64, dataset.num_classes, cached=True,
                                normalize=not gdc)
            self.reverse = GradientReversalLayer()

        def forward(self, pos_edge_index, neg_edge_index):

            print(pos_edge_index.shape) # torch.Size([2, 8976])
            print(neg_edge_index.shape) # torch.Size([2, 8970])
            input("wait")
            
            
            if (self.name == 'GINConv'):
                x = F.relu(self.conv1(data.x, data.train_pos_edge_index))
                x = self.bn1(x)
                x = F.relu(self.conv2(x, data.train_pos_edge_index))
                x = self.bn2(x)
                x = F.relu(self.conv3(x, data.train_pos_edge_index))
                x = self.bn3(x)
            else:
                x = F.relu(self.conv1(data.x, data.train_pos_edge_index))
                x = self.conv2(x, data.train_pos_edge_index)
                x = self.conv3(x, data.train_pos_edge_index)

            feat = x
            attr = self.attr(x, edge_index, edge_weight)

            #print(feat.size())
            attack = self.reverse(x)
            att = self.attack(attack, edge_index, edge_weight)

            total_edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)
            
            print(total_edge_index.shape) # torch.Size([2, 17946])
            print(x.shape) # torch.Size([2708, 64])
            
            x_j = torch.index_select(x, 0, total_edge_index[0])
            x_i = torch.index_select(x, 0, total_edge_index[1])
            res = torch.einsum("ef,ef->e", x_i, x_j)
            
            
            print(x_j.shape) # torch.Size([17946, 64])
            print(x_i.shape) # torch.Size([17946, 64])
            print(res.shape) # orch.Size([17946])
            input("wait")

            #print(res.size())
            return res, F.log_softmax(attr, dim=1), att, feat
    
    m = 'GCNConv' 
    lr = 0.01
    num_epochs = 200
    finetune_epochs = 100
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

    dataset = "Cora"
    path = osp.join('..', 'data', dataset)
    dataset = Planetoid(path, dataset, 'public', T.NormalizeFeatures())
    data = dataset[0]
    gdc = False

    labels = data.y.cuda()
    edge_index, edge_weight = data.edge_index.cuda(), data.edge_attr

    #print(labels.size())
    # Train/validation/test
    data = train_test_split_edges(data)

    #print(labels)

    device = torch.device('cuda')
    model, data = Net(m).cuda(), data.to("cuda")

    #if (m=='GINConv'):
    #    optimizer = torch.optim.Adam([
    #        dict(params=model.conv1.parameters(), weight_decay=0),
    #        dict(params=model.bn1.parameters(), weight_decay=0),
    #        dict(params=model.conv2.parameters(), weight_decay=0),
    #        dict(params=model.bn2.parameters(), weight_decay=0),
    #    ], lr=lr)
    #else:
    optimizer = torch.optim.Adam([
        dict(params=model.conv1.parameters(), weight_decay=0),
        dict(params=model.conv2.parameters(), weight_decay=0),
        dict(params=model.conv3.parameters(), weight_decay=0),
        dict(params=model.attr.parameters(), weight_decay=0)
    ], lr=lr)

    #if (m=='GINConv'):
    #    optimizer_att = torch.optim.Adam([
    #        dict(params=model.conv2.parameters(), weight_decay=5e-4), 
    #        dict(params=model.bn2.parameters(), weight_decay=0),  
    #        dict(params=model.attack.parameters(), weight_decay=5e-4),
    #    ], lr=lr * lambda_reg)
    #else:
    optimizer_att = torch.optim.Adam([
        dict(params=model.conv2.parameters(), weight_decay=5e-4),   
        dict(params=model.conv3.parameters(), weight_decay=5e-4),   
        dict(params=model.attack.parameters(), weight_decay=5e-4),
    ], lr=lr * lambda_reg)

    def get_link_labels(pos_edge_index, neg_edge_index):
        link_labels = torch.zeros(pos_edge_index.size(1) +
                                neg_edge_index.size(1)).float().to(device)
        link_labels[:pos_edge_index.size(1)] = 1.
        return link_labels

    global switch
    switch = True
    
    def train():
        global switch
        model.train()

        x, pos_edge_index = data.x, data.train_pos_edge_index

        _edge_index, _ = remove_self_loops(pos_edge_index)
        pos_edge_index_with_self_loops, _ = add_self_loops(_edge_index,
                                                        num_nodes=x.size(0))

        neg_edge_index = negative_sampling(
            edge_index=pos_edge_index_with_self_loops, num_nodes=x.size(0),
            num_neg_samples=pos_edge_index.size(1))

        link_logits, attr_prediction, attack_prediction,_ = model(pos_edge_index, neg_edge_index)
        link_labels = get_link_labels(pos_edge_index, neg_edge_index)

        loss = F.binary_cross_entropy_with_logits(link_logits, link_labels)
        one_hot = torch.cuda.FloatTensor(attack_prediction.size(0), attack_prediction.size(1)).zero_()
        mask = one_hot.scatter_(1, labels.view(-1,1), 1)
    
        nonzero = mask * attack_prediction
        avg = torch.mean(nonzero, dim = 0)
        loss2 = torch.abs(torch.max(avg) - torch.min(avg))
        
        if switch:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            switch = False
        else:
            optimizer_att.zero_grad()
            loss2.backward()
            optimizer_att.step()
            switch = True
            
            for p in model.attack.parameters():
                p.data.clamp_(-1, 1)
                
        return loss


    def test():
        model.eval()
        perfs = []
        for prefix in ["val", "test"]:
            pos_edge_index, neg_edge_index = [
                index for _, index in data("{}_pos_edge_index".format(prefix),
                                        "{}_neg_edge_index".format(prefix))
            ]
            link_probs = torch.sigmoid(model(pos_edge_index, neg_edge_index)[0])
            link_labels = get_link_labels(pos_edge_index, neg_edge_index)
            link_probs = link_probs.detach().cpu().numpy()
            link_labels = link_labels.detach().cpu().numpy()
            perfs.append(roc_auc_score(link_labels, link_probs))
        return perfs


    best_val_perf = test_perf = 0
    for epoch in range(1, num_epochs+1):
        train_loss = train()
        val_perf, tmp_test_perf = test()
        if val_perf > best_val_perf:
            best_val_perf = val_perf
            test_perf = tmp_test_perf
        log = 'Epoch: {:03d}, Loss: {:.4f}, Val: {:.4f}, Test: {:.4f}'
        #print(log.format(epoch, train_loss, val_perf, tmp_test_perf))


    optimizer_attr = torch.optim.Adam([
        dict(params=model.attr.parameters(), weight_decay=5e-4),
    ], lr=lr)

    def train_attr():
        model.train()
        optimizer_attr.zero_grad()

        x, pos_edge_index = data.x, data.train_pos_edge_index

        _edge_index, _ = remove_self_loops(pos_edge_index)
        pos_edge_index_with_self_loops, _ = add_self_loops(_edge_index,
                                                        num_nodes=x.size(0))

        neg_edge_index = negative_sampling(
            edge_index=pos_edge_index_with_self_loops, num_nodes=x.size(0),
            num_neg_samples=pos_edge_index.size(1))

        F.nll_loss(model(pos_edge_index, neg_edge_index)[1][data.train_mask], labels[data.train_mask]).backward()
        optimizer_attr.step()


    @torch.no_grad()
    def test_attr():
        model.eval()
        accs = []
        m = ['train_mask', 'val_mask', 'test_mask']
        i = 0
        for _, mask in data('train_mask', 'val_mask', 'test_mask'):

            if (m[i] == 'train_mask') :
                x, pos_edge_index = data.x, data.train_pos_edge_index

                _edge_index, _ = remove_self_loops(pos_edge_index)
                pos_edge_index_with_self_loops, _ = add_self_loops(_edge_index,
                                                                num_nodes=x.size(0))

                neg_edge_index = negative_sampling(
                    edge_index=pos_edge_index_with_self_loops, num_nodes=x.size(0),
                    num_neg_samples=pos_edge_index.size(1))
            else:
                pos_edge_index, neg_edge_index = [
                index for _, index in data("{}_pos_edge_index".format(m[i].split("_")[0]),
                                        "{}_neg_edge_index".format(m[i].split("_")[0]))
                ]
            _, logits, _, _ = model(pos_edge_index, neg_edge_index)

            pred = logits[mask].max(1)[1]
            #acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
            #accs.append(acc)

            macro = f1_score((data.y[mask]).cpu().numpy(), pred.cpu().numpy(),average='macro')
            accs.append(macro)

            i+=1
        return accs

    if True:
        best_val_acc = test_acc = 0
        for epoch in range(1, finetune_epochs+1):
            train_attr()
            train_acc, val_acc, tmp_test_acc = test_attr()
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                test_acc = tmp_test_acc
            log = 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
            #print(log.format(epoch, train_acc, val_acc, tmp_test_acc))
    return train_loss, test_acc


res = {}
res_train = {}
for strength in tqdm(np.arange(0, 1, 0.05)):
    l = []
    lt = []
    for seed in [1,2,3,4,5]:
        loss, acc = sim(strength, seed)
        l.append(acc)
        lt.append(loss)
    res[strength] = l
    res_train[strength] = lt



"""
A,B,C,D are gnns

Loop1:
input -> A() -> emb
attacker: emb -> B() -> node_class_attack
net: emb -> C() -> link_pred


Loop2: 
emb -> D() -> node_class_net


"""