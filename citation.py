from models.networks import SEAggrNet
import os
import json
import datetime

import dgl
from dgl.data import CitationGraphDataset
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score

from utils import get_stats
from models import HighOrderGCN, SEAggrNet, TransAggrNet

# m_name = "trans"
m_name = "se"
d_name = "cora".lower()
default_hyperparams = {
    "dataset": "{}".format(d_name),
    "data_dir": "./dataset/citation/{}".format(d_name),
    "output_dir": "./output/citation/{}".format(d_name),
    "model_type": m_name,

    "num_layers": 2,
    "dropout": 0.8,
    "hidden": 64,
    "K": 31,
    "res_connect": True,
    "res_scale": 0.1,
    "gc_type": "se-aggregation-m",
    "excitation_rate": 6.0,
    "layernorm": True,
    "bias": True,
    "mode": "att",
    "init_weight": 0.1,
    
    "weight_decay": 5e-3,
    "lr": 1e-2,
    "patience": 80,
    "num_epochs": 400,
    "trial_times": 10,
    "print_every": 20,
    "num_heads": 1,

    "ablation_type": "None",
    "ablation_attr": {
        "base": 1.0,
        "scale": 0.9
    },
    "double_update": False
}
if not os.path.exists(default_hyperparams["output_dir"]):
    os.makedirs(default_hyperparams["output_dir"])
default_hyperparams["name"] = "{}-{}-{}".format(
    d_name, default_hyperparams["num_layers"], default_hyperparams["K"])

default_hyperparams["logname"] = "{}-{}".format(
    default_hyperparams["model_type"], default_hyperparams["ablation_type"])


def one_trial(params):
    dataset = CitationGraphDataset(params["dataset"].lower(), raw_dir=params["data_dir"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    graph = dataset[0].to(device)
    graph = dgl.add_self_loop(graph)
    node_features = graph.ndata['feat']
    # node_features = F.normalize(node_features)
    node_labels = graph.ndata['label']
    train_mask = graph.ndata['train_mask']
    valid_mask = graph.ndata['val_mask']
    test_mask = graph.ndata['test_mask']
    num_features = node_features.shape[1]
    num_labels = int(node_labels.max().item() + 1)

    if m_name == "se":
        model = SEAggrNet(num_features, num_labels, params["hidden"], params["K"],
                    dropout=params["dropout"], bias=params["bias"],
                    excitation_rate=params["excitation_rate"],
                    mode=params["mode"], init_weight=params["init_weight"]).to(device)
    else:
        model = TransAggrNet(num_features, num_labels, params["hidden"], params["K"],
                             dropout=params["dropout"], bias=params["bias"],
                             num_heads=params["num_heads"], init_weight=params["init_weight"]).to(device)
    optimizer = torch.optim.Adam(params=model.parameters(),
                                 weight_decay=params["weight_decay"],
                                 lr=params["lr"])
    
    def train():
        model.train()
        optimizer.zero_grad()
        loss = F.nll_loss(model(graph, node_features, ablation_type=params["ablation_type"], ablation_attr=params["ablation_attr"])[train_mask], node_labels[train_mask])
        loss.backward()
        optimizer.step()
        return loss.item()

    @torch.no_grad()
    def eval(mask, cal_f1_score=False):
        model.eval()
        logits = model(graph, node_features, ablation_type=params["ablation_type"], ablation_attr=params["ablation_attr"])
        loss = F.nll_loss(logits[mask], node_labels[mask]).item()
        pred = logits[mask].max(1)[1]
        acc = pred.eq(node_labels[mask]).sum().item() / mask.sum().item()

        if cal_f1_score:
          pred_onehot = F.one_hot(pred, num_classes=num_labels).cpu().numpy()
          label_onehot = F.one_hot(node_labels[mask], num_classes=num_labels).cpu().numpy()
          f1 = f1_score(label_onehot, pred_onehot, average="macro")
          return loss, acc, f1

        return loss, acc

    best_val_loss = float("inf")
    final_acc = 0.
    final_f1 = 0.
    double_best = False
    bad_count = 0
    best_epoch = 0
    for e in range(params["num_epochs"]):
        train_loss = train()
        _, train_acc = eval(train_mask)
        val_loss, val_acc = eval(valid_mask)
        _, test_acc, test_f1 = eval(test_mask, True)
        if val_loss <= best_val_loss:
            if double_best or not params["double_update"]: # or e > 3 * params["num_epochs"] // 4:
                best_val_loss = val_loss
                final_acc = test_acc
                final_f1 = test_f1
                bad_count = 0
                best_epoch = e
                double_best = False
            else:
                double_best = True
        else:
            bad_count += 1
        if e % 20 == 0 and params["print_every"] > 0:
            log = 'Epoch: {:03d}, Train loss: {:.4f}, Val loss: {:.4f}, best loss: {:.4f}, Train acc: {:.4f}, Val acc: {:.4f}, Test acc: {:.4f}'
            print(log.format(e, train_loss, val_loss, best_val_loss, train_acc, val_acc, final_acc))
        if bad_count == params["patience"]:
            break
    
    return final_acc, final_f1

result_dict = {
    "hyperparams": default_hyperparams
}

result_dict["detail"] = {
  "acc": [],
  "f1": []
}

for i in range(default_hyperparams["trial_times"]):
    print("trial {}/{}".format(i + 1, default_hyperparams["trial_times"]))
    acc, f1 = one_trial(default_hyperparams)
    result_dict["detail"]["acc"].append(acc)
    result_dict["detail"]["f1"].append(f1)

percent_fn = lambda x : "{:.3f}".format(x * 100)
result_dict["acc_mean"], result_dict["acc_err_bound"] = get_stats(result_dict["detail"]["acc"], conf_interval=True)
result_dict["f1_mean"], result_dict["f1_err_bound"] = get_stats(result_dict["detail"]["f1"], conf_interval=True)
result_dict["acc_mean"], result_dict["acc_err_bound"] = percent_fn(result_dict["acc_mean"]), percent_fn(result_dict["acc_err_bound"])
result_dict["f1_mean"], result_dict["f1_err_bound"] = percent_fn(result_dict["f1_mean"]), percent_fn(result_dict["f1_err_bound"])
curr_time = str(datetime.datetime.now()).replace(':', "-")

with open(os.path.join(default_hyperparams["output_dir"], "{}-{}.log".format(default_hyperparams["logname"], curr_time)), 'w') as f:
    json.dump(result_dict, f, indent=4)
