import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter, init
# from dgl.nn.conv import GraphConv
from dgl.nn.pytorch.conv import GraphConv

__all__ = ["_SEAggregation", "SEAggregation", "MLPLayer", "TransAggregation"]


class _SEAggregation(nn.Module):
    r"""The basic graph convolutional layer (squeeze and excitation version)

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        K (int, optional): Number of hops :math:`K`. (default: 1)
        excitation_rate (float, optional): The squeeze rate of the excitation
            layer. Given `squeeze_rate = r`, we have the dimension of the excitation
            layer `int((K + 1) * r) + 1`. (default: 1.0)
        res_connect (bool, optional): Whether add GNN version residual connection
            to the result. (default: :obj:`False`)
        res_scale (float, optional): scale value for resconnect
            (default: 1.0)
        normalize (str, optional): Whether to add self-loops and apply
            symmetric normalization. (default: :obj:`"both"`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`False`)
    """
    def __init__(self, in_dim, out_dim, K, excitation_rate=1.,
                 res_connect=False, res_scale=0.1,
                 normalize="both", bias=False):
        super(_SEAggregation, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.K = K
        self.res_connect = res_connect
        self.res_scale = res_scale

        self.weight = Parameter(torch.Tensor(in_dim, out_dim))
        e_chs = int((K + 1) * excitation_rate - 1) + 1  # excitation chennel
        self.e_weight1 = Parameter(torch.Tensor(K + 1, e_chs))
        self.e_weight2 = Parameter(torch.Tensor(e_chs, K + 1))
        self.e_att = Parameter(torch.Tensor(out_dim, 1))
        self.gc = GraphConv(out_dim, out_dim, norm=normalize, weight=False, bias=False)

        if bias:
            self.bias = Parameter(torch.Tensor(out_dim))
            self.e_bias1 = Parameter(torch.Tensor(e_chs))
            self.e_bias2 = Parameter(torch.Tensor(K + 1))
        else:
            self.register_parameter("bias", None)
            self.register_parameter("e_bias1", None)
            self.register_parameter("e_bias2", None)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        init.xavier_uniform_(self.weight)
        init.xavier_uniform_(self.e_weight1)
        init.xavier_uniform_(self.e_weight2)
        init.xavier_uniform_(self.e_att)
        if self.bias is not None:
            init.zeros_(self.bias)
            init.zeros_(self.e_bias1)
            init.zeros_(self.e_bias2)
    
    def forward(self, graph, n_feat):
        # aggregation
        if self.res_connect:
            res_n_feat = n_feat
        n_feat = torch.matmul(n_feat, self.weight)
        if self.bias is not None:
            n_feat = n_feat + self.bias

        aggr_results = [n_feat]
        for _ in range(self.K):
            aggr_results.append(self.gc(graph, aggr_results[-1]))
        stack_result = torch.stack(aggr_results, dim=-1) # N x in_dim x (K+1)

        # squeeze and excitation
        # squeeze_result = stack_result.sum(dim=-2).squeeze() # N x (K+1)
        squeeze_result = (stack_result * self.e_att).sum(dim=-2).squeeze()
        squeeze_result = F.normalize(squeeze_result, dim=-1)
        excitation_result = torch.matmul(squeeze_result, self.e_weight1)
        if self.e_bias1 is not None:
            excitation_result = excitation_result + self.e_bias1
        excitation_result = F.relu6(excitation_result)

        excitation_result = torch.matmul(excitation_result, self.e_weight2)
        if self.e_bias2 is not None:
            excitation_result = excitation_result + self.e_bias2
        excitation_result = torch.tanh(excitation_result).view(-1, 1, self.K + 1)
        out = (stack_result * excitation_result).sum(dim=-1)

        # update
        # out = torch.matmul(excitation_result, self.weight)
        # if self.bias is not None:
        #     out = out + self.bias
        if self.res_connect and self.in_dim == self.out_dim:
            out = self.res_scale * out + res_n_feat
        
        return out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_dim,
                                   self.out_dim)


class SEAggregation(nn.Module):
    r"""The basic graph convolutional layer (squeeze and excitation version)

    Args:
        in_dim (int): Size of each input sample.
        K (int): Number of hops :math:`K`.
        excitation_rate (float, optional): The squeeze rate of the excitation
            layer. Given `squeeze_rate = r`, we have the dimension of the excitation
            layer `int((K + 1) * r) + 1`. (default: 1.0)
        mode (str, optional): Could be 'pool' or 'att'. If 'pool', use mean pooling,
            else, use attention vector
            (default: 'pool')
        init_weight (float, optional): Weigth for add init feature
            (default: 0.9)
        normalize (str, optional): Whether to add self-loops and apply
            symmetric normalization. (default: :obj:`"both"`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`False`)
    """
    def __init__(self, in_dim, K, excitation_rate=1.,
                 mode="pool", init_weight=0.9,
                 normalize="both", bias=False):
        super(SEAggregation, self).__init__()
        self.in_dim = in_dim
        self.K = K
        self.bias = bias
        self.mode = mode
        self.init_weight = init_weight

        e_chs = int((K + 1) * excitation_rate - 1) + 1  # excitation chennel
        self.e_weight1 = Parameter(torch.Tensor(K + 1, e_chs))
        self.e_weight2 = Parameter(torch.Tensor(e_chs, K + 1))
        if self.mode == "att":
            self.e_att = Parameter(torch.Tensor(in_dim, 1))
        self.gc = GraphConv(in_dim, in_dim, norm=normalize, weight=False, bias=False)

        if bias:
            self.e_bias1 = Parameter(torch.Tensor(e_chs))
            self.e_bias2 = Parameter(torch.Tensor(K + 1))
        else:
            self.register_parameter("e_bias1", None)
            self.register_parameter("e_bias2", None)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        init.xavier_uniform_(self.e_weight1)
        init.xavier_uniform_(self.e_weight2)
        if self.mode == "att":
            init.xavier_uniform_(self.e_att)
        if self.bias:
            init.zeros_(self.e_bias1)
            init.zeros_(self.e_bias2)

    def forward(self, graph, n_feat, ablation_type=None, ablation_attr=None):
        # aggregation
        aggr_results = [n_feat]
        for _ in range(self.K):
            aggr_results.append( (1 - self.init_weight) * self.gc(graph, aggr_results[-1]) + self.init_weight * n_feat)
        stack_result = torch.stack(aggr_results, dim=-1) # N x in_dim x (K+1)

        if ablation_type == "poly":
            base = ablation_attr["base"]
            scale = ablation_attr["scale"]
            weights = torch.ones(self.K + 1, dtype=n_feat.dtype, device=n_feat.device)
            weights[1:] *= scale
            weights = torch.cumprod(weights, 0)
            weights = weights * base
            out = (stack_result * weights).sum(dim=-1)
            return out
        elif ablation_type == "const":
            return stack_result.mean(dim=-1)
        elif ablation_type == "random":
            weights = torch.rand(self.K + 1, dtype=n_feat.dtype, device=n_feat.device)
            return (stack_result * weights).sum(dim=-1)
        # squeeze and excitation
        if self.mode == "att":
            squeeze_result = torch.matmul(stack_result.transpose(-1, -2), self.e_att).squeeze() # N x (k+1)
        else:
            squeeze_result = stack_result.sum(dim=-2).squeeze() # N x (K+1)
        squeeze_result = F.normalize(squeeze_result, dim=-1)
        excitation_result = torch.matmul(squeeze_result, self.e_weight1)
        if self.e_bias1 is not None:
            excitation_result = excitation_result + self.e_bias1
        excitation_result = F.relu6(excitation_result)

        excitation_result = torch.matmul(excitation_result, self.e_weight2)
        if self.e_bias2 is not None:
            excitation_result = excitation_result + self.e_bias2
        excitation_result = F.normalize(excitation_result, dim=-1).view(-1, 1, self.K + 1)
        out = (stack_result * excitation_result).sum(dim=-1)
        
        return out


class TransAggregation(nn.Module):
    r"""The basic graph convolutional layer (Transformer Encoder version)

    Args:
        in_dim (int): Size of each input sample.
        K (int): Number of hops :math:`K`.
        num_heads (int, optional): Number of heads of the encoder layer of transformer.
            (default: 1)
        mode (str, optional): Could be 'pool' or 'att'. If 'pool', use mean pooling,
            else, use attention vector
            (default: 'pool')
        init_weight (float, optional): Weigth for add init feature
            (default: 0.9)
        normalize (str, optional): Whether to add self-loops and apply
            symmetric normalization. (default: :obj:`"both"`)
        dropout (float, optional): Dropout rate for the transformer encoder.
            (default: 0.1)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`False`)
    """
    def __init__(self, in_dim, K, num_heads=1,
                 init_weight=0.9, normalize="both",
                 dropout=0.1, bias=False):
        super(TransAggregation, self).__init__()
        self.in_dim = in_dim
        self.K = K
        self.bias = bias
        self.num_heads = num_heads
        self.init_weight = init_weight

        self.weight = Parameter(torch.Tensor(K + 1))
        self.gc = GraphConv(in_dim, in_dim, norm=normalize, weight=False, bias=False)
        # self.trans = nn.TransformerEncoderLayer(in_dim, num_heads, dim_feedforward=in_dim, dropout=dropout)
        self.att = nn.MultiheadAttention(in_dim, num_heads, dropout=dropout, bias=bias)

        self.init_parameter()

    def init_parameter(self):
        init.uniform_(self.weight)

    def forward(self, graph, n_feat):
        # aggregation
        aggr_results = [n_feat]
        for _ in range(self.K):
            aggr_results.append( (1 - self.init_weight) * self.gc(graph, aggr_results[-1]) + self.init_weight * n_feat)
        stack_result = torch.stack(aggr_results) # (K+1) x N x in_dim
        S, N, D = stack_result.shape

        # trans_result = self.trans(stack_result) # (K+1) x N x in_dim
        att = self.att(torch.ones_like(stack_result), stack_result, stack_result)[0] # (K+1) x N x in_dim
        att = att.sum(dim=-1).transpose(0, 1) # N x (K+1)
        att = F.normalize(att, dim=-1) # N x (K+1)

        # out = (torch.reshape(trans_result, (N, D, S)) * self.weight).sum(dim=-1)
        # out = (att * stack_result).sum(dim=0)
        out = (att.unsqueeze(-1) * stack_result.transpose(0, 1)).sum(dim=1)
        return out


class MLPLayer(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 bias=True,
                 batchnorm=False,
                 activation=None,
                 dropout=0):
        super(MLPLayer, self).__init__()

        self.linear = nn.Linear(in_dim, out_dim, bias=bias)
        self.activation = activation
        self.dropout = nn.Dropout(dropout)
        if batchnorm: self.bn = nn.BatchNorm1d(out_dim)
        else: self.bn = None
        self.reset_parameters()

    def reset_parameters(self):
        gain = 1.
        if self.activation is F.relu:
            gain = nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(self.linear.weight, gain=gain)
        if self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)

    def forward(self, feats):

        feats = self.dropout(feats)
        feats = self.linear(feats)
        if self.bn is not None:
            feats = self.bn(feats)
        if self.activation:
            feats = self.activation(feats)

        return feats
