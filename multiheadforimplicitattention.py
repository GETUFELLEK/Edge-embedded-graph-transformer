import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import math

'''I am making this modification to a code by kool et al to introduce the 
       implicit attention and explicit attention as claimed by graph transformer network,we dont
        need positional encoding and edge embedding this time'''

''' Today saturday 28/08/2021 by reconsidering my base code of kool,
 I am hereby changing all input_dim to hidden node dim'''
# class SkipConnection(nn.Module):
#
#     def __init__(self, module):
#         super(SkipConnection, self).__init__()
#         self.module = module
#
#     def forward(self, input):
#         return input + self.module(input)


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_node_dim,hidden_edge_dim,n_nodes,n_heads=4,  attn_dropout=0.1, dropout=0):
        super(MultiHeadAttention, self).__init__()


        self.hidden_node_dim = hidden_node_dim
        self.hidden_edge_dim =hidden_edge_dim
        self.n_nodes = n_nodes
        self.n_heads = n_heads
        self.head_dim = self.hidden_node_dim / self.n_heads
        self.norm = 1 / math.sqrt(self.head_dim)

        self.q = nn.Linear(hidden_node_dim, hidden_node_dim, bias=False)
        self.k = nn.Linear(hidden_node_dim, hidden_node_dim, bias=False)
        self.v = nn.Linear(hidden_node_dim, hidden_node_dim, bias=False)
        self.proj_e = nn.Linear(hidden_edge_dim,n_nodes, bias=False)
        self.fc = nn.Linear(hidden_node_dim, hidden_node_dim, bias=False)
        # if INIT:
        #     for name, p in self.named_parameters():
        #         if 'weight' in name:
        #             if len(p.size()) >= 2:
        #                 nn.init.orthogonal_(p, gain=1)
        #         elif 'bias' in name:
        #             nn.init.constant_(p, 0)

    def forward(self,h,e,mask=None):
        '''
        :param e
        :param h: （batch_size,n_nodes,input_dim）
        :param edge_index
        :return:
        '''
        batch_size,n_nodes, hidden_node_dim = h.size()
        # batch_size, n_nodes,n_nodes, hidden_edge_dim  = e.size()
        Q = self.q(h).view(batch_size, n_nodes, self.n_heads, -1)
        K = self.k(h).view(batch_size, n_nodes, self.n_heads, -1)
        V = self.v(h).view(batch_size, n_nodes, self.n_heads, -1)
        # Proj_e = self.proj_e(e).view(-1,n_nodes,self.n_heads,self.n_heads)#self.proj_e(e).view(batch_size,n_nodes,n_nodes, -1)
        Proj_e = self.proj_e(e).view(batch_size,n_nodes,n_nodes,-1)

        # .view(batch_size,self.n_heads,n_nodes,n_nodes, -1)
        # Q, K, V,proj_e = Q.transpose(1, 2), K.transpose(1, 2), V.transpose(1, 2),proj_e.transpose(2,3)

        compatibility = self.norm * torch.matmul(Q.reshape(batch_size,self.n_heads,n_nodes,-1), K.reshape(batch_size,self.n_heads,-1,n_nodes))  # (batch_size,n_heads,1,hidden_dim)*(batch_size,n_heads,hidden_dim,n_nodes)
        # new edge based compatibility is computed ..(batchsize,nheads,1,nnodes)*(batch_size,n_heads,n_nodes,hidden dim
        # compatibility = compatibility.squeeze(2)
        # compatibility = compatibility.reshape(batch_size,self.n_heads,n_nodes,n_nodes,-1)
        # proj_e = proj_e.reshape(batch_size,self.n_heads,n_nodes,-1,n_nodes)
        #
        new_compatibility = torch.matmul(compatibility, Proj_e.transpose(1,3))
        # new_compatibility = new_compatibility.squeeze(3)  # (batch_size,n_heads,n_nodes)
        # mask = mask.unsqueeze(1).expand_as(new_compatibility)
        # u_i = new_compatibility.masked_fill(mask.bool(), float("-inf"))

        scores = F.softmax(new_compatibility, dim=-1)  # (batch_size,n_heads,n_nodes)
        # scores = scores.unsqueeze(2)
        out_put = torch.matmul(scores, V.view(batch_size,self.n_heads,n_nodes,-1))  # (batch_size,n_heads,1,n_nodes )*(batch_size,n_heads,n_nodes,head_dim)
        out_put = out_put.view(-1, self.hidden_node_dim)  # （batch_size,n_heads,hidden_dim）
        out_put = self.fc(out_put)

        return out_put  # (batch_size,hidden_dim)

#



class MultiHeadAttentionLayer(nn.Module):
    """
        Param:
    """

    def __init__(self,hidden_node_dim, hidden_edge_dim,n_heads, dropout=0.0, batch_norm=True, residual=True,
                 use_bias=False):
        super().__init__()

        # self.input_node_dim = input_node_dim
        self.hidden_node_dim = hidden_node_dim
        # self.input_edge_dim = input_edge_dim
        self.hidden_edge_dim = hidden_edge_dim
        self.n_heads = n_heads
        # self.edge_index = edge_index
        self.dropout = dropout
        self.residual = residual
        # self.layer_norm = layer_norm
        self.batch_norm = batch_norm
        feed_forward_hidden = 512

        # self.laysers = nn.Sequential(*(
        #     MultiHeadAttentionLayer(n_heads, embed_dim, feed_forward_hidden, normalization)
        #     for _ in range(n_laysers)
        self.attentions = MultiHeadAttention(hidden_node_dim, hidden_edge_dim,n_heads)


        # self.attention = MultiHeadAttentionLayer(in_dim, out_dim // num_heads, num_heads, use_bias)

        self.O_h = nn.Linear(hidden_node_dim, hidden_node_dim)
        if self.batch_norm:
            self.batch_norm1_h = nn.BatchNorm1d(hidden_node_dim)
        # FFN for h
        self.FFN_h_layer1 = nn.Linear(hidden_node_dim, feed_forward_hidden)
        self.FFN_h_layer2 = nn.Linear(feed_forward_hidden, hidden_node_dim)

        if self.batch_norm:
            self.batch_norm2_h = nn.BatchNorm1d(hidden_node_dim)


    def forward(self, h, e):
        h_in1 = h.reshape(-1,self.hidden_node_dim) # for first residual connection
        # multi-head attention out
        h_attn_out = self.attentions(h,e)

        h = h_attn_out.view(-1, self.hidden_node_dim)


        h = F.dropout(h, self.dropout, training=self.training)

        h = self.O_h(h)

        if self.residual:
            h = h_in1 + h  # residual connection
        if self.batch_norm:
            h = self.batch_norm1_h(h)

        h_in2 = h  # for second residual connection


        # FFN for h
        h = self.FFN_h_layer1(h)
        h = F.relu(h)
        h = F.dropout(h, self.dropout, training=self.training)
        h = self.FFN_h_layer2(h)

        if self.residual:
            h = h_in2 + h  # residual connection

        if self.batch_norm:
            h = self.batch_norm2_h(h)


        return h
#
# class Normalization(nn.Module):
#
#     def __init__(self,hidden_node_dim, batch_norm=True,layer_norm=False):
#         super(Normalization, self).__init__()
#         self.layer_norm = layer_norm
#         self.batch_norm = batch_norm
#
#     def forward(self, h):
#
#         if self.layer_norm:
#             h = self.layer_norm1_h(h)
# #             e = self.layer_norm1_e(e)
#
#         if self.batch_norm:
#             h = self.batch_norm1_h(h)
# #             e = self.batch_norm1_e(e)
        

#     def __init__(self, embed_dim, normalization='batch'):
#         super(Normalization, self).__init__()

#         normalizer_class = {
#             'batch': nn.BatchNorm1d,
#             'instance': nn.InstanceNorm1d
#         }.get(normalization, None)

#         self.normalizer = normalizer_class(embed_dim, affine=True)

#         # Normalization by default initializes affine parameters with bias 0 and weight unif(0,1) which is too large!
#         # self.init_parameters()

#     def init_parameters(self):

#         for name, param in self.named_parameters():
#             stdv = 1. / math.sqrt(param.size(-1))
#             param.data.uniform_(-stdv, stdv)

#     def forward(self, input):

#         if isinstance(self.normalizer, nn.BatchNorm1d):
#             return self.normalizer(input.view(-1, input.size(-1))).view(*input.size())
#         elif isinstance(self.normalizer, nn.InstanceNorm1d):
#             return self.normalizer(input.permute(0, 2, 1)).permute(0, 2, 1)
#         else:
#             assert self.normalizer is None, "Unknown normalizer type"
#             return h


# class MultiHeadAttentionLayer(nn.Sequential):
#
#     def __init__(
#             self,
#             n_heads,
#             #embed_dim,
#             hidden_node_dim,
#             hidden_edge_dim,
#             feed_forward_hidden= 512,
#             normalization = 'batch',
#
#     ):
#      super(MultiHeadAttentionLayer, self).__init__(
#         SkipConnection(
#             MultiHeadAttention(
#                 n_heads,
#                  # input_dim=embed_dim,
#                 # input_dim=hidden_node_dim,
#     #           embed_dim = embed_dim,
#                 hidden_node_dim = hidden_node_dim,
#                 hidden_edge_dim = hidden_edge_dim
#       )
#     ),
#     Normalization(hidden_node_dim, normalization),
#     SkipConnection(
#         nn.Sequential(
#             nn.Linear(hidden_node_dim, feed_forward_hidden),
#             nn.ReLU(),
#             nn.Linear(feed_forward_hidden, hidden_node_dim)
#         ) if feed_forward_hidden > 0 else nn.Linear(hidden_node_dim, hidden_node_dim)
#     ),
#     Normalization(hidden_node_dim, normalization))

# class GraphtransformerEncoder(nn.Module):
#     def __init__(
#             self,
#             n_heads,
#             embed_dim,
#             edge_dim,
#             n_layers,
#             node_dim=None,
#             normalization='batch',
#             feed_forward_hidden=512
#     ):
#         super(GraphtransformerEncoder, self).__init__()
#
#         # To map input to embedding space
#         self.init_embed = nn.Linear(node_dim, embed_dim) if node_dim is not None else None
#         self.init_embed_e =nn.Linear(edge_dim,embed_dim)
#
#         self.layers = nn.Sequential(*(
#             MultiHeadAttentionLayer(n_heads, embed_dim, feed_forward_hidden, normalization)
#             for _ in range(n_layers)
#         ))
#
#     def forward(self, data,mask=None):
#
#         assert mask is None, "TODO mask not yet supported!"
#
#         # Batch multiply to get initial embeddings of nodes
#         h=torch.cat[data.x,data.demand]
#         # h = self.init_embed(x.view(-1, x.size(-1))).view(*x.size()[:2], -1) if self.init_embed is not None else x
#         # e = self.init_embed_e(edge_attr)
#         e=data.egde_attr
#         h,e = self.layers(h,e)  
#         # we have obtained a separate node and edge embedding,how shall we use it,shall we pass it
#         # into another layer mean pooled,concatenated or simply added? ?
#
#         output_embedding=h.mean+e.mean
#         return (
#            output_embedding, # (batch_size, graph_size, embed_dim)
#         h.mean(dim=1),)#  average to get embedding of graph, (batch_size, embed_dim))