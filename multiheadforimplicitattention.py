import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import math




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

        new_compatibility = torch.matmul(compatibility, Proj_e.transpose(1,3))


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
#