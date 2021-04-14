import os
import torch
import torch.nn.functional as F
import dgl
import torch.nn as nn
from .layers import *

__all__ = ["HighOrderGCN", "SEAggrNet"]

class SEAggrNet(nn.Module):
    def __init__(self, in_dim, out_dim, hidden, K,
                 mode="pool", init_weight=0.9, batchnorm=False,
                 dropout=.6, bias=False, excitation_rate=1.0):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden = hidden
        self.K = K

        self.mlp = nn.ModuleList()
        self.mlp.append(MLPLayer(in_dim, hidden, bias=bias, activation=F.relu, batchnorm=batchnorm, dropout=dropout))
        self.mlp.append(MLPLayer(hidden, out_dim, bias=bias, activation=None, dropout=dropout))
        self.gc = SEAggregation(
            out_dim, K, excitation_rate=excitation_rate, bias=bias,
            init_weight=init_weight, mode=mode)
    def forward(self, graph, n_feat, softmax=True):
        for i, layer in enumerate(self.mlp):
            n_feat = layer(n_feat)
        feats = self.gc(graph, n_feat)
        return feats if not softmax else F.log_softmax(feats, dim=1)


class HighOrderGCN(nn.Module):
    r"""Graph convolutional network with high-order graph
    convolution schemes

    Args:
        in_dim (int): Size of each input sample.
        out_dim (int): Size of each output sample.
        hidden (int): Size of hidden layers.
        num_layer (int): Number of layers in this network.
        K (int): Number of hops :math:`K`.
        dropout (float, optional): Dropout rate. (default: 0.6)
        res_connect (bool, optional): Whether attach residual connection
            at each hidden layer if possible. (default: :obj:`True`)
        res_scale (float, optional): scale value for resconnect
            (default: 1.0)
        layernorm (bool, optional): Whether perform layer-normalization
            at each hidden layer. (default: :obj:`True`)
        bias (bool, optional): use bias
            (default: :obj:`True`)
        gc_type (str, optional): which graph convolution scheme to be used
            (default: :obj:`"se-aggregation"`)
        **kwargs: parameters for gc layers
            - for "se-aggregation":
                * excitation_rate (= 1.0)
    """
    def __init__(self, in_dim, out_dim, hidden, num_layer, K, dropout=.6,
                 res_connect=False, res_scale=1., layernorm=True,
                 bias=True, gc_type="se-aggregation", **kwargs):
        super(HighOrderGCN, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden = hidden
        self.num_layers = num_layer
        self.K = K
        self.use_lnorm = layernorm
        self.res_connect = res_connect
        self.res_scale = res_scale
        self.dropout = dropout
        if gc_type == "se-aggregation" or gc_type == "se-aggregation-m":
            gc_op = _SEAggregation
        else:
            raise ValueError("invalid graph convolution type")
        
        self.layers = nn.ModuleList()
        self.layernorms = nn.ModuleList()
        L = num_layer
        for i in range(L):
            in_c = in_dim if i == 0 else hidden
            out_c = hidden if i < L - 1 else out_dim
            self.layers.append(gc_op(
                in_c, out_c, K, res_connect=res_connect,
                res_scale=res_scale, bias=bias, **kwargs))
            if i < L - 1 and layernorm:
                self.layernorms.append(nn.LayerNorm(out_c, elementwise_affine=True))

    def forward(self, graph, n_feat, logsoftmax=True, add_self_loop=False):
        # add self loop, this can be done outside if only one
        # graph
        if add_self_loop:
            graph = dgl.add_self_loop(graph)

        for i in range(self.num_layers - 1):
            n_feat = F.dropout(n_feat, self.dropout, training=self.training)
            n_feat = self.layers[i](graph, n_feat)
            if self.use_lnorm:
                n_feat = self.layernorms[i](n_feat)
            n_feat= F.relu6(n_feat)
        
        n_feat = F.dropout(n_feat, self.dropout, training=self.training)
        n_feat = self.layers[-1](graph, n_feat)
        if logsoftmax:
            return F.log_softmax(n_feat, dim=1)
        else:
            return n_feat

    def __repr__(self):
        return '{}({}, {}, {})'.format(self.__class__.__name__,
            self.in_dim, self.hidden, self.out_dim)

    def save(self, savedir, name=None):
        if name is None:
            name = self.__repr__()
        if not name.endswith(".pt") and not name.endswith(".pth"):
            name = "{}.pt".format(name)
        if not os.path.exists(savedir):
            os.makedirs(savedir)
        torch.save(self.state_dict(), os.path.join(savedir, name))
    
    def load(self, savedir, name=None):
        if name is None:
            name = self.__repr__()
        if not name.endswith(".pt") and not name.endswith(".pth"):
            name = "{}.pt".format(name)
        if not os.path.exists(savedir):
            raise RuntimeError("no such save dir: {}".format(savedir))
        self.load_state_dict(torch.load(os.path.join(savedir, name)))

