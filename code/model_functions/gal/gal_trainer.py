import torch
import torch_geometric
from torch_geometric.utils import (negative_sampling, remove_self_loops,
                                   add_self_loops)
import torch.nn.functional as F


def create_gal_optimizer(model, lr=0.01, lambda_reg=0.05):
    optimizer = torch.optim.Adam([
        dict(params=model.conv1.parameters(), weight_decay=0),
        dict(params=model.conv2.parameters(), weight_decay=0),
        dict(params=model.conv3.parameters(), weight_decay=0),
        dict(params=model.attr.parameters(), weight_decay=0)
    ], lr=lr)

    optimizer_attack = torch.optim.Adam([
        dict(params=model.conv2.parameters(), weight_decay=5e-4),   
        dict(params=model.conv3.parameters(), weight_decay=5e-4),   
        dict(params=model.attack.parameters(), weight_decay=5e-4),
    ], lr=lr * lambda_reg)


    optimizer_fine_tune = torch.optim.Adam([
        dict(params=model.attr.parameters(), weight_decay=5e-4),
    ], lr=lr)

    return (optimizer, optimizer_attack, optimizer_fine_tune)

def galTrainer(model, data: torch_geometric.data.Data, patience: int):
    """
        trains the model according to the required epochs/patience

        Parameters
        ----------
        model: Model
        data: torch_geometric.data.Data
        patience:

        Returns
        -------
        model: Model
        model_log: str
        test_accuracy: torch.Tensor
    """
    (optimizer, optimizer_attack, optimizer_fine_tune) = create_gal_optimizer(model)

    train_epochs = 200
    fine_tune_epochs = 20

    patience_counter = 0
    best_val_accuracy = test_accuracy = 0
    switch = True
    for epoch in range(1, train_epochs+1):

        train(model, (optimizer, optimizer_attack),data, switch)
        switch = not switch

        # Ben's logging
        log_template = 'Regular Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
        train_accuracy, val_acc, tmp_test_acc = test(model, data)

        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            test_accuracy = tmp_test_acc
            # patience
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= patience:
            break
        print(log_template.format(epoch, train_accuracy, best_val_accuracy, test_accuracy), flush=True)

    best_val_acc = test_acc = 0
    for epoch in range(1, fine_tune_epochs+1):
        train_attr(model, optimizer_fine_tune,data)
        train_acc, val_acc, tmp_test_acc = test_attr(model, data)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            test_acc = tmp_test_acc
        log = 'Finetune Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
        print(log.format(epoch, train_acc, val_acc, tmp_test_acc))

    model_log = 'Basic Model - Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'\
        .format(train_acc, best_val_acc, test_acc)

    return model, model_log, test_accuracy


def _get_link_labels(pos_edge_index, neg_edge_index):
    link_labels = torch.zeros(pos_edge_index.size(1) +
                            neg_edge_index.size(1)).float()
    link_labels[:pos_edge_index.size(1)] = 1.
    return link_labels
    
def _get_edge_mask(edge_index, vertex_mask):
    # leave only edges which have both ends in vertex mask

    idx1 = torch.index_select(vertex_mask, 0, edge_index[0, :])
    idx2 = torch.index_select(vertex_mask, 0, edge_index[1, :])

    # print(idx1.shape)
    # print(idx2.shape)

    # print(edge_index)

    # print(idx1.int().sum())
    # print(idx2.int().sum())

    # print(idx1)
    # print(idx2)

    good_edges_mask = torch.logical_and(idx1, idx2)

    # print(good_edges_mask.shape)
    # print(good_edges_mask.int().sum())
    # input("wait")


    return good_edges_mask

def _prepare_edge_index(edge_index, num_nodes, device):
    """
    Processes a single edge index object into sampled pos+neg edges with link labels
    """
    pos_edge_index = edge_index

    _edge_index, _ = remove_self_loops(pos_edge_index)

    pos_edge_index_with_self_loops, _ = add_self_loops(_edge_index,
                                                    num_nodes=num_nodes)

    neg_edge_index = negative_sampling(
        edge_index=pos_edge_index_with_self_loops, num_nodes=num_nodes,
        num_neg_samples=pos_edge_index.size(1))

    link_labels = _get_link_labels(pos_edge_index, neg_edge_index)

    return pos_edge_index.to(device), neg_edge_index.to(device), link_labels.to(device)

# training the current model
def train(model, optimizers: torch.optim, data: torch_geometric.data.Data, switch):
    """
        trains the model for one epoch

        Parameters
        ----------
        model: Model
        optimizer: torch.optim
        data: torch_geometric.data.Data
    """
    model.train()

    ##### edge masking stuff
    edge_mask = _get_edge_mask(model.edge_index, data.train_mask)
    full_edge_mask = edge_mask.repeat(2,1)
    edge_index = torch.masked_select(model.edge_index, full_edge_mask).reshape((2, -1))
    #####

    pos_edge_index, neg_edge_index, link_labels = _prepare_edge_index(
        edge_index, model.data.x.size(0), model.device)

    link_logits, attr_prediction, attack_prediction,_ = model(pos_edge_index, neg_edge_index)

    attr_prediction = attr_prediction[data.train_mask]
    attack_prediction = attack_prediction[data.train_mask]

    true_labels = data.y[data.train_mask]

    loss = F.binary_cross_entropy_with_logits(link_logits, link_labels)

    # loss 2
    one_hot = torch.cuda.FloatTensor(attack_prediction.size(0), attack_prediction.size(1)).zero_()
    mask = one_hot.scatter_(1, true_labels.view(-1,1), 1)

    nonzero = mask * attack_prediction
    avg = torch.mean(nonzero, dim = 0)
    loss2 = torch.abs(torch.max(avg) - torch.min(avg))

    
    (optimizer, optimizer_attack) = optimizers
    if switch:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    else:
        optimizer_attack.zero_grad()
        loss2.backward()
        optimizer_attack.step()
        
        for p in model.attack.parameters():
            p.data.clamp_(-1, 1)
        
    model.eval()


# testing the current model
@torch.no_grad()
def test(model, data: torch_geometric.data.Data) -> torch.Tensor:
    """
        tests the model according to the train/val/test masks

        Parameters
        ----------
        model: Model
        data: torch_geometric.data.Data

        Returns
        -------
        accuracies: torch.Tensor - 3d-tensor that includes
                                    1st-d - the train accuracy
                                    2nd-d - the val accuracy
                                    3rd-d - the test accuracy
    """
    model.eval()
    accuracies = []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        ##### edge masking stuff
        edge_mask = _get_edge_mask(model.edge_index, mask)
        full_edge_mask = edge_mask.repeat(2,1)
        edge_index = torch.masked_select(model.edge_index, full_edge_mask).reshape((2, -1))
        #####

        pos_edge_index, neg_edge_index, link_labels = _prepare_edge_index(
            edge_index, model.data.x.size(0), model.device)

        link_logits, _, _,_ = model(pos_edge_index, neg_edge_index)
        logits = link_logits

        accuracy = logits.eq(link_labels).sum().item() / link_labels.sum().item()
        accuracies.append(accuracy)

    return accuracies



def train_attr(model, optimizer: torch.optim, data: torch_geometric.data.Data):
    model.train()
    optimizer.zero_grad()

    pos_edge_index, neg_edge_index, _ = _prepare_edge_index(
        model.edge_index, model.data.x.size(0), model.device)

    F.nll_loss(model(pos_edge_index, neg_edge_index)[1][data.train_mask], model.labels[data.train_mask]).backward()
    optimizer.step()
    model.eval()


@torch.no_grad()
def test_attr(model, data):
    raise Exception("edge masking not fixed here")
    model.eval()
    pos_edge_index, neg_edge_index, link_labels = _prepare_edge_index(
        model.edge_index,model.data.x.size(0), model.device)

    link_logits, attr_prediction, attack_prediction,_ = model(pos_edge_index, neg_edge_index)
    logits, accuracies = attr_prediction, []

    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        pred = logits[mask].max(1)[1]
        accuracy = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accuracies.append(accuracy)
    return accuracies