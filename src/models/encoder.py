import math

import torch
import torch.nn as nn

from models.neural import MultiHeadedAttention, PositionwiseFeedForward
from models.rnn import LayerNormLSTM


class Classifier(nn.Module):
    def __init__(self, hidden_size):
        super(Classifier, self).__init__()
        self.linear1 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, mask_cls):
        h = self.linear1(x).squeeze(-1)
        sent_scores = self.sigmoid(h) * mask_cls.float()
        return sent_scores


class PositionalEncoding(nn.Module):

    def __init__(self, dropout, dim, max_len=5000):
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, dim, 2, dtype=torch.float) *
                              -(math.log(10000.0) / dim)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(0)
        super(PositionalEncoding, self).__init__()
        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(p=dropout)
        self.dim = dim

    def forward(self, emb, step=None):
        emb = emb * math.sqrt(self.dim)
        if (step):
            emb = emb + self.pe[:, step][:, None, :]

        else:
            emb = emb + self.pe[:, :emb.size(1)]
        emb = self.dropout(emb)
        return emb

    def get_emb(self, emb):
        return self.pe[:, :emb.size(1)]


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, heads, d_ff, dropout):
        super(TransformerEncoderLayer, self).__init__()

        self.self_attn = MultiHeadedAttention(
            heads, d_model, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, iter, query, inputs, mask):
        if (iter != 0):
            input_norm = self.layer_norm(inputs)
        else:
            input_norm = inputs

        mask = mask.unsqueeze(1)
        context = self.self_attn(input_norm, input_norm, input_norm,
                                 mask=mask)
        out = self.dropout(context) + inputs
        return self.feed_forward(out)


class TransformerInterEncoder(nn.Module):
    def __init__(self, d_model, d_ff, heads, dropout, num_inter_layers=0):
        super(TransformerInterEncoder, self).__init__()
        self.d_model = d_model
        self.num_inter_layers = num_inter_layers
        self.pos_emb = PositionalEncoding(dropout, d_model)
        self.transformer_inter = nn.ModuleList(
            [TransformerEncoderLayer(d_model, heads, d_ff, dropout)
             for _ in range(num_inter_layers)])
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.wo = nn.Linear(d_model, 1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, top_vecs, mask):
        """ See :obj:`EncoderBase.forward()`"""

        batch_size, n_sents = top_vecs.size(0), top_vecs.size(1)
        pos_emb = self.pos_emb.pe[:, :n_sents]
        x = top_vecs * mask[:, :, None].float()
        x = x + pos_emb

        for i in range(self.num_inter_layers):
            x = self.transformer_inter[i](i, x, x, ~mask)  # all_sents * max_tokens * dim

        x = self.layer_norm(x)
        sent_scores = self.sigmoid(self.wo(x))
        sent_scores = sent_scores.squeeze(-1) * mask.float()

        return sent_scores

class GRUEncoder_attn(nn.Module):

    def __init__(self,bidirectional, num_layers, input_size, hidden_size,dropout=0.0):
        super(GRUEncoder_attn,self).__init__()


class RNNEncoder_attn(nn.Module):

    def __init__(self, bidirectional, num_layers, input_size,
                 hidden_size, dropout=0.0):
        super(RNNEncoder_attn, self).__init__()
        num_directions = 2 if bidirectional else 1
        assert hidden_size % num_directions == 0
        hidden_size = hidden_size // num_directions
        self.relu = nn.ReLU()

        self.rnn = LayerNormLSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional)

        self.wo = nn.Linear(num_directions * hidden_size, 1, bias=True)

        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax()
        print('this is dropout',dropout)

    def forward(self, x, mask):
        """See :func:`EncoderBase.forward()`"""
        batch, layer, seq, hidden = x.size()
        x1=x.contiguous().view(batch * layer, -1, hidden)
        x1 = torch.transpose(x1, 1, 0)
        memory_bank, _ = self.rnn(x1)
        memory_bank = self.dropout(memory_bank) + x1
        memory_bank = torch.transpose(memory_bank, 1, 0)
        # sent_scores = self.softmax(self.relu(self.wo(memory_bank)).squeeze(dim=-1)).unsqueeze(-1)

        sent_scores = self.softmax(self.relu(self.wo(memory_bank[:,-1,:])).squeeze(dim=-1).view(-1,layer)).unsqueeze(-1)
        x=x.transpose(1,2)
        sent_vec = torch.matmul(sent_scores.transpose(1,2).unsqueeze(dim = 1).expand(batch,seq,1,layer),x)

        return sent_vec.squeeze(dim = 2)

class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, heads, d_ff, dropout):
        super(TransformerDecoderLayer, self).__init__()

        self.self_attn = MultiHeadedAttention(
            heads, d_model, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, iter, ent_enc, inputs, self_attn_mask=None,context_attn_mask=None):
        context = self.self_attn(inputs, inputs, inputs,
                                 mask=self_attn_mask)
        dec_output = self.self_attn(
            ent_enc, ent_enc, context, mask=context_attn_mask)
        dec_output = self.feed_forward(dec_output)
        return dec_output

class TransformerInterDecoder(nn.Module):
    def __init__(self, d_model, d_ff, heads, dropout, d_hidden, num_inter_layers=0):
        super(TransformerInterDecoder, self).__init__()
        self.d_model = d_model
        self.num_inter_layers = num_inter_layers
        self.pos_emb = PositionalEncoding(dropout, d_model)
        self.transformer_inter = nn.ModuleList(
            [TransformerDecoderLayer(d_model, heads, d_ff, dropout)
             for _ in range(num_inter_layers)])
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.wo = nn.Linear(d_model, d_hidden , bias=True)
        self.wi = nn.Linear(d_model, d_hidden, bias=True)
        self.v = nn.Linear(d_hidden, 1, bias=True)
        self.LR = nn.LeakyReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, top_vecs, inputs, mask, label_mask=None):
        """ See :obj:`EncoderBase.forward()`"""
        n_out = inputs.size(1)
        pos_emb = self.pos_emb.pe[:, :n_out]
        seq_mask=subsequent_mask(inputs)
        self_attn_mask = torch.gt((~label_mask.unsqueeze(1).expand(-1, n_out, -1) + seq_mask), 0)
        inputs=inputs+pos_emb
        for i in range(self.num_inter_layers):
            inputs = self.transformer_inter[i](i, top_vecs, inputs,self_attn_mask,~ mask.unsqueeze(1).expand(-1, n_out,-1))
        scores=self.v(self.LR(
            self.wo(inputs.unsqueeze(2)).expand(-1, -1, top_vecs.size(1), -1) + self.wi(top_vecs).unsqueeze(
                1))).squeeze(-1)
        sent_scores = self.softmax(scores)
        return sent_scores


class RNNEncoder(nn.Module):

    def __init__(self, bidirectional, num_layers, input_size,
                 hidden_size, dropout=0.0):
        super(RNNEncoder, self).__init__()
        num_directions = 2 if bidirectional else 1
        assert hidden_size % num_directions == 0
        hidden_size = hidden_size // num_directions

        self.rnn = LayerNormLSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional)

        self.wo = nn.Linear(num_directions * hidden_size, 1, bias=True)
        self.dropout = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, mask):
        """See :func:`EncoderBase.forward()`"""
        x = torch.transpose(x, 1, 0)
        memory_bank, _ = self.rnn(x)
        memory_bank = self.dropout(memory_bank) + x
        memory_bank = torch.transpose(memory_bank, 1, 0)

        sent_scores = self.sigmoid(self.wo(memory_bank))
        sent_scores = sent_scores.squeeze(-1) * mask.float()
        return sent_scores
class GCN(nn.Module):
    def __init__(self,in_channel,out_channel,hidden_dim,drop):
        super(GCN, self).__init__()
        self.in_channel=in_channel
        self.out_channel=out_channel
        self.hidden_dim=hidden_dim
        self.dropout = nn.Dropout(p=drop)
        self.gcn_x_11=GCNConv(self.in_channel,self.hidden_dim)
        self.gcn_x_12=GCNConv(self.hidden_dim,self.out_channel)#No.1-*2*2
        # self.gcn_x_21=GCNConv(self.in_channel,self.hidden_dim)
        # self.gcn_x_22=GCNConv(self.hidden_dim,self.out_channel)#No.2-*2
        # self.gcn_mix=GCNConv(self.hidden_dim*2,self.hidden_dim)#No.2-*2
        self.relu=nn.ReLU(inplace=True)
    def forward(self, x_1, edge_index_1,  edge_index_2=None,edge_weight_1=None,edge_weight_2=None):
        syn=self.gcn_x_11(x_1, edge_index_1, edge_weight_1)
        syn=self.relu(syn)
        syn=self.dropout(syn)
        syn = self.gcn_x_12(syn, edge_index_1, edge_weight_1)
        syn = self.relu(syn)
        syn = self.dropout(syn)
        # x2 = self.gcn_x_21(x_1, edge_index_2, edge_weight_2)
        # x2 = self.relu(x2)
        # x2 = self.dropout(x2)
        # mix = self.gcn_mix(torch.cat((syn,x2),-1), edge_index_2, edge_weight_2)
        # x2 = self.gcn_x_22(mix, edge_index_2, edge_weight_2)
        # syn=self.gcn_x_12(mix, edge_index_1, edge_weight_1)
        # syn=self.relu(syn)
        # syn=self.dropout(syn)
        # x2 = self.relu(x2)
        # x2 = self.dropout(x2)
        return syn

