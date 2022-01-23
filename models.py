import torch
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch_geometric.nn import GCNConv, Linear
from torch_geometric.nn import dense_diff_pool

from layers import GCN, HGPSLPool
from torch_geometric.nn import global_mean_pool
from hgcn_conv import HGCNConv, HypAct

from ultils import edge_2_adj_tf, adj_2_edge, adj_2_edge_tf, edge_2_adj

import manifolds

class HGP(torch.nn.Module):
    def __init__(self, args):
        super(HGP, self).__init__()
        self.args = args
        self.num_features = args.num_features
        self.nhid = args.nhid
        self.num_classes = args.num_classes
        self.pooling_ratio = args.pooling_ratio
        self.dropout_ratio = args.dropout_ratio
        self.sample = args.sample_neighbor
        self.sparse = args.sparse_attention
        self.sl = args.structure_learning
        self.lamb = args.lamb

        self.conv1 = GCNConv(self.num_features, self.nhid)
        self.conv2 = GCN(self.nhid, self.nhid)
        self.conv3 = GCN(self.nhid, self.nhid)

        self.pool1 = HGPSLPool(self.nhid, self.pooling_ratio, self.sample, self.sparse, self.sl, self.lamb)
        self.pool2 = HGPSLPool(self.nhid, self.pooling_ratio, self.sample, self.sparse, self.sl, self.lamb)

        self.lin1 = torch.nn.Linear(self.nhid * 2, self.nhid)
        self.lin2 = torch.nn.Linear(self.nhid, self.nhid // 2)
        self.lin3 = torch.nn.Linear(self.nhid // 2, self.num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_attr = None

        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x, edge_index, edge_attr, batch = self.pool1(x, edge_index, edge_attr, batch)
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv2(x, edge_index, edge_attr))
        x, edge_index, edge_attr, batch = self.pool2(x, edge_index, edge_attr, batch)
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv3(x, edge_index, edge_attr))
        x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(x1) + F.relu(x2) + F.relu(x3)

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        x = F.relu(self.lin2(x))
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        x = F.log_softmax(self.lin3(x), dim=-1)

        return x


class GCN_ori(torch.nn.Module):
    def __init__(self, args):
        super(GCN_ori, self).__init__()
        torch.manual_seed(1234567)

        self.args = args

        self.channel_in = args.num_features
        self.channel_out = args.num_classes
        self.hidden_channels = args.nhid

        self.conv1 = GCNConv(self.channel_in, self.hidden_channels)
        self.conv2 = GCNConv(self.hidden_channels, self.hidden_channels)
        self.lin1 = Linear(self.hidden_channels, self.channel_out)

        self.ln = torch.nn.LayerNorm(self.hidden_channels)

    def forward(self, data):
        # 1. Obtain node embeddings
        x, edge, batch = data.x, data.edge_index, data.batch

        x = self.conv1(x, edge)

        x = x.relu()

        x = self.conv2(x, edge)

        x = global_mean_pool(x, batch)

        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin1(x)

        return F.log_softmax(x)


class HGCN_pyg(torch.nn.Module):
    """
    Hyperbolic-GCN.
    """

    def __init__(self, args):
        super(HGCN_pyg, self).__init__()
        torch.manual_seed(1234567)

        self.args = args

        self.channel_in = args.num_features
        self.channel_out = args.num_classes
        self.hidden_channels = args.nhid

        self.c = args.c

        # self.manifold = getattr(manifolds, 'Euclidean')()
        self.manifold = getattr(manifolds, 'Hyperboloid')()
        # self.manifold = getattr(manifolds, 'PoincareBall')()
        act = getattr(F, 'relu')
        self.hconv1 = HGCNConv(self.manifold, self.channel_in, self.hidden_channels, self.c)
        self.hconv2 = HGCNConv(self.manifold, self.hidden_channels, self.hidden_channels, self.c)
        self.hyp_act = HypAct(self.manifold, self.c, act)

        self.ln = torch.nn.LayerNorm(self.hidden_channels)
        self.lin4 = Linear(self.hidden_channels, self.channel_out)

    def forward(self, data):
        x, edge, batch = data.x, data.edge_index, data.batch

        x = self.hconv1(x, edge)
        x = self.hyp_act(x)

        x = self.hconv2(x, edge)
        x = self.hyp_act(x)

        x = self.manifold.logmap0(x, c=self.c)
        x = self.manifold.proj_tan0(x, c=self.c)

        x = global_mean_pool(x, batch)

        x = F.dropout(x, p=0.5, training=self.training)

        x = self.lin4(x)

        return F.log_softmax(x)

class GCN_adj(torch.nn.Module):
    def __init__(self, args):
        super(GCN_adj, self).__init__()
        torch.manual_seed(1234567)

        self.args = args

        self.channel_in = args.num_features
        self.channel_out = args.num_classes
        self.hidden_channels = args.nhid

        self.conv1 = GCNConv(self.channel_in, self.hidden_channels)
        self.conv2 = GCNConv(self.hidden_channels, self.hidden_channels)
        self.lin1 = Linear(self.hidden_channels, self.channel_out)
        self.lin_adj = Linear(400, 400)

        self.ln = torch.nn.LayerNorm(self.hidden_channels)

    def forward(self, data):
        # 1. Obtain node embeddings

        # x, edge, batch, identity = data.x, data.edge_index, data.batch, data.identity
        x, edge, batch = data.x, data.edge_index, data.batch
        # print(edge)

        # mask = self.lin_adj(identity)
        # mask = torch.sigmoid(mask)
        # mask = torch.triu(mask)
        # mask += mask.T - torch.diag(torch.diag(mask, 0), 0)
        #
        # ori_edge = edge_2_adj_tf(edge)
        #
        # adj = ori_edge.mul(mask)
        # print(adj)

        x = self.conv1(x, edge)

        x = x.relu()

        x = self.conv2(x, edge)

        x = global_mean_pool(x, batch)

        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin1(x)

        return F.log_softmax(x)
