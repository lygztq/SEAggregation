from models.networks import SEAggrNet
import os
import json
import datetime

import dgl
from dgl.data import CitationGraphDataset
import torch
import torch.nn.functional as F
from typing_extensions import final

from utils import get_stats
from models import HighOrderGCN, SEAggrNet

d_name = "pubmed".lower()
default_hyperparams = {
    "dataset": "{}".format(d_name),
    "data_dir": "./dataset/citation/{}".format(d_name),
    "output_dir": "./output/citation/{}".format(d_name),

    "num_layers": 2,
    "dropout": 0.8,
    "hidden": 64,
    "K": 16,
    "res_connect": True,
    "res_scale": 0.1,
    "gc_type": "se-aggregation-m",
    "excitation_rate": 4.0,
    "layernorm": False,
    "bias": True,
    "mode": "att",
    "init_weight": 0.1,
    
    "weight_decay": 5e-3,
    "lr": 1e-2,
    "patience": 100,
    "num_epochs": 400,
    "trial_times": 10,
    "print_every": 20
}
if not os.path.exists(default_hyperparams["output_dir"]):
    os.makedirs(default_hyperparams["output_dir"])
default_hyperparams["name"] = "{}-{}-{}".format(
    d_name, default_hyperparams["num_layers"], default_hyperparams["K"])


def one_trial(params):
    dataset = CitationGraphDataset(params["dataset"].lower(), raw_dir=params["data_dir"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    graph = dataset[0].to(device)
    graph = dgl.add_self_loop(graph)
    node_features = graph.ndata['feat']
    node_labels = graph.ndata['label']
    train_mask = graph.ndata['train_mask']
    valid_mask = graph.ndata['val_mask']
    test_mask = graph.ndata['test_mask']
    num_features = node_features.shape[1]
    num_labels = int(node_labels.max().item() + 1)
    model = SEAggrNet(num_features, num_labels, params["hidden"], params["K"],
                  dropout=params["dropout"], bias=params["bias"],
                  excitation_rate=params["excitation_rate"],
                  mode=params["mode"], init_weight=params["init_weight"]).to(device)
    optimizer = torch.optim.Adam(params=model.parameters(),
                                 weight_decay=params["weight_decay"],
                                 lr=params["lr"])
    
    def train():
        model.train()
        optimizer.zero_grad()
        loss = F.nll_loss(model(graph, node_features)[train_mask], node_labels[train_mask])
        loss.backward()
        optimizer.step()
        return loss.item()

    @torch.no_grad()
    def eval(mask):
        model.eval()
        logits = model(graph, node_features)
        loss = F.nll_loss(logits[mask], node_labels[mask]).item()
        pred = logits[mask].max(1)[1]
        acc = pred.eq(node_labels[mask]).sum().item() / mask.sum().item()
        return loss, acc

    best_val_loss = float("inf")
    final_acc = 0.
    bad_count = 0
    best_epoch = 0
    for e in range(params["num_epochs"]):
        train_loss = train()
        _, train_acc = eval(train_mask)
        val_loss, val_acc = eval(valid_mask)
        _, test_acc = eval(test_mask)
        if val_loss <= best_val_loss:
            best_val_loss = val_loss
            final_acc = test_acc
            bad_count = 0
            best_epoch = e
        else:
            bad_count += 1
        if e % 20 == 0 and params["print_every"] > 0:
            log = 'Epoch: {:03d}, Train loss: {:.4f}, Val loss: {:.4f}, Train acc: {:.4f}, Val acc: {:.4f}, Test acc: {:.4f}'
            print(log.format(e, train_loss, val_loss, train_acc, val_acc, final_acc))
        if bad_count == params["patience"]:
            break
    
    return final_acc

result_dict = {
    "hyperparams": default_hyperparams
}
result_dict["mean"], result_dict["err_bound"] = 0, 0
result_dict["detail"] = []

for i in range(default_hyperparams["trial_times"]):
    print("trial {}/{}".format(i + 1, default_hyperparams["trial_times"]))
    result_dict["detail"].append(one_trial(default_hyperparams))

result_dict["mean"], result_dict["err_bound"] = get_stats(result_dict["detail"], conf_interval=True)
curr_time = str(datetime.datetime.now()).replace(':', "-")

with open(os.path.join(default_hyperparams["output_dir"], "{}.log".format(curr_time)), 'w') as f:
    json.dump(result_dict, f, indent=4)
