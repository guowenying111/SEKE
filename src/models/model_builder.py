import copy
from collections import OrderedDict
import torch
from models.layer_attention import Layer_Attention
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import torch.nn as nn
from transformers import BertModel, BertConfig
from torch.nn.init import xavier_uniform_,uniform_
from transformers import AlbertModel,AlbertTokenizer,AlbertConfig,ElectraModel,ElectraConfig, RobertaModel,RobertaConfig
from torch_geometric.data import Batch
import sys
sys.path.append('../')
torch.backends.cudnn.enabled = False
from models.encoder import TransformerInterEncoder, Classifier, RNNEncoder,GCN,TransformerInterDecoder,RNNEncoder_attn
from models.optimizers import Optimizer

def build_optim(args, model, checkpoint):
    """ Build optimizer """
    saved_optimizer_state_dict = None

    if args.train_from != '':
        optim = checkpoint['optim']
        saved_optimizer_state_dict = optim.optimizer.state_dict()
    else:
        optim = Optimizer(
            args.optim, args.lr, args.max_grad_norm,
            beta1=args.beta1, beta2=args.beta2,
            decay_method=args.decay_method,
            warmup_steps=args.warmup_steps)

    optim.set_parameters(list(model.named_parameters()))

    if args.train_from != '':
        optim.optimizer.load_state_dict(saved_optimizer_state_dict)
        if args.visible_gpus != '-1':
            for state in optim.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()

        if (optim.method == 'adam') and (len(optim.optimizer.state) < 1):
            raise RuntimeError(
                "Error: loaded Adam optimizer from existing model" +
                " but optimizer state is empty")

    return optim


class Bert(nn.Module):
    def __init__(self, temp_dir, load_pretrained_bert, bert_config):
        super(Bert, self).__init__()
        if(load_pretrained_bert):
            self.model = BertModel.from_pretrained('bert-base-uncased', cache_dir=temp_dir)# ,config=bert_config)
        else:
            bert_config= BertConfig.from_pretrained('bert-base-uncased')
            self.model = BertModel(bert_config)

    def forward(self, x, segs, mask):
        output = self.model(input_ids=x, token_type_ids=segs, attention_mask=mask)
        return output[0]


class Albert_unshared_attn(nn.Module):
    def  __init__(self, temp_dir, load_pretrained_bert, Albert_config):
        super(Albert_unshared_attn, self).__init__()
        if (load_pretrained_bert):
            self.config = AlbertConfig.from_pretrained('albert-base-v2')
            self.config.num_hidden_groups = self.config.num_hidden_layers
            self.config.output_hidden_states = True
            self.model = AlbertModel(self.config)
            loaded_dict = torch.load('../../PreSumm/src/albert-base-v2/pytorch_model.bin',map_location='cpu')
            new_dict = OrderedDict()
            for k, v in loaded_dict.items():
                if k.startswith('albert.'):
                    if not k.startswith('albert.encoder.albert'):
                        new_dict[k[7:]] = copy.deepcopy(v)
                    else:
                        for i in range(0, self.model.config.num_hidden_groups):
                            new_dict[k[7:35] + str(i) + k[36:]] = copy.deepcopy(v)
            self.model.load_state_dict(new_dict, strict=False)
        else:
            self.config = AlbertConfig.from_pretrained('albert-base-v2')
            self.config.output_hidden_states = True
            self.config.num_hidden_groups = self.config.num_hidden_layers
            self.model = AlbertModel(self.config)
    def forward(self, x, segs, mask):
        output = self.model(input_ids=x, token_type_ids=segs, attention_mask=mask)

        all_layer = torch.cat((
                               output[2][1].unsqueeze(dim=2), output[2][2].unsqueeze(dim=2),
                               output[2][3].unsqueeze(dim=2), output[2][4].unsqueeze(dim=2),
                               output[2][5].unsqueeze(dim=2), output[2][6].unsqueeze(dim=2),
                               output[2][7].unsqueeze(dim=2), output[2][8].unsqueeze(dim=2),
                               output[2][9].unsqueeze(dim=2), output[2][10].unsqueeze(dim=2),
                               output[2][11].unsqueeze(dim=2), output[2][12].unsqueeze(dim=2)), dim=-2)
        return output[0],all_layer
        # output = self.model(input_ids=x, token_type_ids=segs, attention_mask=mask)
        # return output[0]


class Albert_unshared(nn.Module):
    def  __init__(self, temp_dir, load_pretrained_bert, Albert_config):
        super(Albert_unshared, self).__init__()
        if (load_pretrained_bert):
            self.config = AlbertConfig.from_pretrained('albert-base-v2')
            # self.config.num_hidden_layers=6
            # self.config.hidden_dropout_prob=0.1
            # self.config.attention_probs_dropout_prob=0.1
            self.config.num_hidden_groups = self.config.num_hidden_layers
            self.model = AlbertModel(self.config)
            loaded_dict = torch.load('../../PreSumm/src/albert-base-v2/pytorch_model.bin',map_location='cpu')
            new_dict = OrderedDict()
            for k, v in loaded_dict.items():
                if k.startswith('albert.'):
                    if not k.startswith('albert.encoder.albert'):
                        new_dict[k[7:]] = copy.deepcopy(v)
                    else:
                        for i in range(0, self.model.config.num_hidden_groups):
                            new_dict[k[7:35] + str(i) + k[36:]] = copy.deepcopy(v)
            self.model.load_state_dict(new_dict, strict=False)
        else:
            print('thisi is i exec')
            self.config = AlbertConfig.from_pretrained('albert-base-v2')
            # self.config.num_hidden_layers = 6
            # self.config.hidden_dropout_prob = 0.1
            # self.config.attention_probs_dropout_prob = 0.1
            self.config.num_hidden_groups = self.config.num_hidden_layers
            self.model = AlbertModel(self.config)
    def forward(self, x, segs, mask):
        output = self.model(input_ids=x, token_type_ids=segs, attention_mask=mask)
        return output[0]


class Albert(nn.Module):
    def __init__(self, temp_dir, load_pretrained_bert, Albert_config):
        super(Albert, self).__init__()
        if(load_pretrained_bert):
            self.model = AlbertModel.from_pretrained('albert-base-v2', cache_dir=temp_dir)
            self.model.config.output_hidden_states=True
        else:
            Albert_config=AlbertConfig.from_pretrained('albert-base-v2')
            self.model = AlbertModel(Albert_config)
            self.model.config.output_hidden_states = True

    def forward(self, x, segs, mask):
        output = self.model(input_ids=x, token_type_ids=segs, attention_mask=mask)

        return output[0]

class Albert_mean(nn.Module):
    def __init__(self, temp_dir, load_pretrained_bert, Albert_config):
        super(Albert_mean, self).__init__()
        if(load_pretrained_bert):
            self.model = AlbertModel.from_pretrained('albert-base-v2', cache_dir=temp_dir)
            self.model.config.output_hidden_states=True
        else:
            Albert_config=AlbertConfig.from_pretrained('albert-base-v2')
            self.model = AlbertModel(Albert_config)
            self.model.config.output_hidden_states = True

    def forward(self, x, segs, mask):
        output = self.model(input_ids=x, token_type_ids=segs, attention_mask=mask)
        all_layer = torch.cat((
                               output[2][1].unsqueeze(dim=2), output[2][2].unsqueeze(dim=2),
                               output[2][3].unsqueeze(dim=2), output[2][4].unsqueeze(dim=2),
                               output[2][5].unsqueeze(dim=2), output[2][6].unsqueeze(dim=2),
                               output[2][7].unsqueeze(dim=2), output[2][8].unsqueeze(dim=2),
                               output[2][9].unsqueeze(dim=2), output[2][10].unsqueeze(dim=2),
                               output[2][11].unsqueeze(dim=2), output[2][12].unsqueeze(dim=2)), dim=2)
        return torch.mean(all_layer,dim=-2)

class Albert_attn(nn.Module):
    def __init__(self, temp_dir, load_pretrained_bert, Albert_config):
        super(Albert_attn, self).__init__()
        if(load_pretrained_bert):
            self.model = AlbertModel.from_pretrained('albert-base-v2', cache_dir=temp_dir)
            self.model.config.output_hidden_states=True
        else:
            Albert_config=AlbertConfig.from_pretrained('albert-base-v2')
            self.model = AlbertModel(Albert_config)
            self.model.config.output_hidden_states = True

    def forward(self, x, segs, mask):
        output = self.model(input_ids=x, token_type_ids=segs, attention_mask=mask)
        all_layer = torch.cat((
                             output[2][1].unsqueeze(dim=0), output[2][2].unsqueeze(dim=0),
                             output[2][3].unsqueeze(dim=0), output[2][4].unsqueeze(dim=0),
                             output[2][5].unsqueeze(dim=0), output[2][6].unsqueeze(dim=0),
                             output[2][7].unsqueeze(dim=0), output[2][8].unsqueeze(dim=0),
                             output[2][9].unsqueeze(dim=0), output[2][10].unsqueeze(dim=0),
                             output[2][11].unsqueeze(dim=0), output[2][12].unsqueeze(dim=0)
        ), dim=0).transpose(0,1)
        return output[0], all_layer


class AlbertSummarizer_mean(nn.Module):
    def __init__(self, args, device, load_pretrained_bert = False, Albert_config = None):
        super(AlbertSummarizer_mean, self).__init__()
        self.args = args
        self.device = device
        self.bert = Albert_attn(args.temp_dir, load_pretrained_bert, Albert_config)
        # self.bert = Albert(args.temp_dir, load_pretrained_bert, Albert_config)
        if (args.encoder == 'classifier'):
            self.encoder = Classifier(self.bert.model.config.hidden_size)
        elif(args.encoder=='transformer'):
            self.encoder = TransformerInterEncoder(self.bert.model.config.hidden_size, args.ff_size, args.heads,
                                                   args.dropout, args.inter_layers)

        elif(args.encoder=='rnn'):
            self.encoder = RNNEncoder_attn(bidirectional=True, num_layers=1,
                                      input_size=self.bert.model.config.hidden_size, hidden_size=args.rnn_size,
                                      dropout=args.dropout)
            self.score = nn.Linear(self.bert.model.config.hidden_size*2,1,bias=True)
            self.sigmoid = nn.Sigmoid()

        elif (args.encoder == 'baseline'):
            bert_config = AlbertConfig.from_pretrained('albert-base-v2')
            self.bert.model = AlbertModel(bert_config)
            self.encoder = Classifier(self.bert.model.config.hidden_size)

        if args.param_init != 0.0:
            for p in self.encoder.parameters():
                p.data.uniform_(-args.param_init, args.param_init)
        if args.param_init_glorot:
            for p in self.encoder.parameters():
                if p.dim() > 1:
                    xavier_uniform_(p)
        self.to(device)
    def load_cp(self, pt):
        self.load_state_dict(pt['model'], strict=True)

    def forward(self, x, segs, clss, mask, mask_cls, sentence_range=None):
        top_vec = self.bert(x, segs, mask)
        sents_vec = top_vec[torch.arange(top_vec.size(0)).unsqueeze(1), clss]
        sents_vec = sents_vec * mask_cls[:, :, None].float()
        sent_scores = self.encoder(sents_vec, mask_cls).squeeze(-1)
        return sent_scores, mask_cls


class Summarizer(nn.Module):
    def __init__(self, args, device, load_pretrained_bert = False, bert_config = None):
        super(Summarizer, self).__init__()
        self.args = args
        self.device = device
        self.bert = Bert(args.temp_dir, load_pretrained_bert, bert_config)
        if (args.encoder == 'classifier'):
            self.encoder = Classifier(self.bert.model.config.hidden_size)
        elif(args.encoder=='transformer'):
            self.encoder = TransformerInterEncoder(self.bert.model.config.hidden_size, args.ff_size, args.heads,
                                                   args.dropout, args.inter_layers)
        elif(args.encoder=='rnn'):
            self.encoder = RNNEncoder(bidirectional=True, num_layers=1,
                                      input_size=self.bert.model.config.hidden_size, hidden_size=args.rnn_size,
                                      dropout=args.dropout)
        elif (args.encoder == 'baseline'):
            bert_config = BertConfig.from_pretrained('bert-base-uncased')
            self.bert.model = BertModel(bert_config)
            self.encoder = Classifier(self.bert.model.config.hidden_size)

        if args.param_init != 0.0:
            for p in self.encoder.parameters():
                p.data.uniform_(-args.param_init, args.param_init)
        if args.param_init_glorot:
            for p in self.encoder.parameters():
                if p.dim() > 1:
                    xavier_uniform_(p)
        self.to(device)
    def load_cp(self, pt):
        self.load_state_dict(pt['model'], strict=True)

    def forward(self, x, segs, clss, mask, mask_cls, sentence_range=None):

        top_vec = self.bert(x, segs, mask)
        sents_vec = top_vec[torch.arange(top_vec.size(0)).unsqueeze(1), clss]
        sents_vec = sents_vec * mask_cls[:, :, None].float()
        sent_scores = self.encoder(sents_vec, mask_cls).squeeze(-1)
        return sent_scores, mask_cls

class GCNSummarizer(nn.Module):
    def __init__(self, args, device, load_pretrained_bert = False, Albert_config = None):
        super(GCNSummarizer, self).__init__()
        self.args = args
        self.device = device
        self.sigmoid = nn.Sigmoid()
        self.bert = Albert(args.temp_dir, load_pretrained_bert, Albert_config)
        self.encoder = GCN(in_channel=self.bert.model.config.hidden_size,hidden_dim=args.hidden_dim,out_channel=args.out_channel,drop=args.dropout)
        self.classifier=nn.Linear(args.out_channel,1)
        if args.param_init != 0.0:
            for p in self.encoder.parameters():
                p.data.uniform_(-args.param_init, args.param_init)
            for p in self.classifier.parameters():
                p.data.uniform_(-args.param_init, args.param_init)
        if args.param_init_glorot:
            for p in self.encoder.parameters():
                if p.dim() > 1:
                    xavier_uniform_(p)
            for p in self.classifier.parameters():
                if p.dim() > 1:
                    xavier_uniform_(p)

        self.to(device)
    def load_cp(self, pt):
        print(pt['opt'])
        self.load_state_dict(pt['model'], strict=True)

    def forward(self, x, segs, clss, mask, mask_cls, graph, sentence_range=None):

        top_vec = self.bert(x, segs, mask)
        for i,gr in enumerate(graph):
            gr.x_s=top_vec[i]
        batch = Batch.from_data_list(graph)
        top_ve=self.encoder(x_1=batch.x_s, edge_index_1=batch.edge_index_s,edge_weight_1=batch.edge_weight_s)#,edge_weight_2=batch.edge_weight_t,  edge_index_2=batch.edge_index_t,x_1, edge_index_1, x_2, edge_index_2,edge_weight_1=None,edge_weight_2=None
        top_ve=top_ve.view(top_vec.size(0),top_vec.size(1),self.args.out_channel)

        sents_vec = top_ve[torch.arange(top_vec.size(0)).unsqueeze(1), clss]
        sents_vec = sents_vec * mask_cls[:, :, None].float()
        sent_scores = self.classifier(sents_vec).squeeze(-1)
        sent_scores = self.sigmoid(sent_scores) * mask_cls.float()
        torch.cuda.empty_cache()
        return sent_scores, mask_cls
class AlbertSummarizer(nn.Module):
    def __init__(self, args, device, load_pretrained_bert = False, Albert_config = None):
        super(AlbertSummarizer, self).__init__()
        self.args = args
        self.device = device
        # self.bert = Albert_unshared(args.temp_dir, load_pretrained_bert, Albert_config)
        self.bert = Albert_attn(args.temp_dir, load_pretrained_bert, Albert_config)
        if (args.encoder == 'classifier'):
            self.encoder = Classifier(self.bert.model.config.hidden_size)
        elif(args.encoder=='transformer'):
            self.encoder = TransformerInterEncoder(self.bert.model.config.hidden_size, args.ff_size, args.heads,
                                                   args.dropout, args.inter_layers)

        elif(args.encoder=='rnn'):
            self.encoder = RNNEncoder_attn(bidirectional=True, num_layers=1,
                                      input_size=self.bert.model.config.hidden_size, hidden_size=args.rnn_size,
                                      dropout=args.dropout)
            self.score = nn.Linear(self.bert.model.config.hidden_size * 2, 1, bias=True)
            self.sigmoid = nn.Sigmoid()

        elif (args.encoder == 'baseline'):
            bert_config = AlbertConfig.from_pretrained('albert-base-v2')
            self.bert.model = AlbertModel(bert_config)
            self.encoder = Classifier(self.bert.model.config.hidden_size)

        if args.param_init != 0.0:
            for p in self.encoder.parameters():
                p.data.uniform_(-args.param_init, args.param_init)
        if args.param_init_glorot:
            for p in self.encoder.parameters():
                if p.dim() > 1:
                    xavier_uniform_(p)
        self.to(device)
    def load_cp(self, pt):
        self.load_state_dict(pt['model'], strict=True)

    def forward(self, x, segs, clss, mask, mask_cls, sentence_range=None):
        # top_vec = self.bert(x, segs, mask)
        # sents_vec = top_vec[torch.arange(top_vec.size(0)).unsqueeze(1), clss]
        # sents_vec = sents_vec * mask_cls[:, :, None].float()
        # sent_scores = self.encoder(sents_vec, mask_cls).squeeze(-1)
        # return sent_scores, mask_cls
        last,top_vec = self.bert(x, segs, mask)
        sents_vec = top_vec.transpose(1,2)[torch.arange(top_vec.size(0)).unsqueeze(1), clss].transpose(1,2)
        last = last[torch.arange(last.size(0)).unsqueeze(1), clss]
        batch,layer,seq,hidden=top_vec.size()


        sents_vec = self.encoder(sents_vec, mask_cls)#encoder应该加入mask_cls
        sents_vec = sents_vec.view(batch,-1,hidden)


        sents_vec = torch.cat((sents_vec, last), dim=-1)#第一次效果有提升

        sents_vec = sents_vec * mask_cls[:, :, None].float()
        score = self.score(sents_vec)
        sent_scores = self.sigmoid(score)* mask_cls[:, :, None].float()
        return sent_scores.squeeze(-1), mask_cls
        # last, top_vec = self.bert(x, segs, mask)
        # sents_vec = top_vec#.transpose(1, 2)[torch.arange(top_vec.size(0)).unsqueeze(1), clss].transpose(1, 2)
        # # last = last[torch.arange(last.size(0)).unsqueeze(1), clss]
        # batch, layer, seq, hidden = top_vec.size()
        #
        # sents_vec = self.encoder(sents_vec, mask_cls)  # encoder应该加入mask_cls
        # sents_vec = sents_vec.view(batch, -1, hidden)
        #
        # sents_vec = torch.cat((sents_vec, last), dim=-1)  # 第一次效果有提升
        # sents_vec = sents_vec[torch.arange(sents_vec.size(0)).unsqueeze(1), clss]
        # sents_vec = sents_vec * mask_cls[:, :, None].float()
        #
        # score = self.score(sents_vec)
        # sent_scores = self.sigmoid(score) * mask_cls[:, :, None].float()
        # return sent_scores.squeeze(-1), mask_cls


class PointerSummarizer(nn.Module):
    def __init__(self, args, device, load_pretrained_bert = False, Albert_config = None):
        super(PointerSummarizer, self).__init__()
        self.args = args
        self.device = device
        self.bert = Albert(args.temp_dir, load_pretrained_bert, Albert_config)
        self.init = nn.Parameter(torch.Tensor(self.bert.model.config.hidden_size))
        self.encoder = TransformerInterDecoder(self.bert.model.config.hidden_size, args.ff_size, args.heads,
                                                   args.dropout, args.hidden_dim, args.inter_layers)
        if args.param_init != 0.0:
            for p in self.encoder.parameters():
                p.data.uniform_(-args.param_init, args.param_init)
        if args.param_init_glorot:
            for p in self.encoder.parameters():
                if p.dim() > 1:
                    xavier_uniform_(p)
        uniform_(self.init)
        self.to(device)
    def load_cp(self, pt):
        print('this is the parameters: ',pt['opt'])
        self.load_state_dict(pt['model'], strict=True)

    def forward(self, x, segs, clss, mask, mask_cls, mask_out_cls, out_cls, mask_target_cls, sentence_range=None):
        if self.args.train:
            top_vec = self.bert(x, segs, mask)
            sents_vec = top_vec[torch.arange(top_vec.size(0)).unsqueeze(1), clss]
            if out_cls.size(1)!=0:
                output = top_vec[torch.arange(top_vec.size(0)).unsqueeze(1), out_cls]
                output = output * mask_out_cls[:, :, None].float()
            init_i = self.init.unsqueeze(0).unsqueeze(1).expand(x.size(0), 1, top_vec.size(-1))
            sents_vec = sents_vec * mask_cls[:, :, None].float()
            if out_cls.size(1) != 0:
                output=torch.cat([init_i,output],dim=1)
            else:
                output=init_i
            sent_scores = self.encoder(sents_vec, output, mask_cls, mask_target_cls).squeeze(-1)
            return sent_scores,mask_target_cls
        else:
            preds = None
            top_vec = self.bert(x, segs, mask)
            sents_vec = top_vec[torch.arange(top_vec.size(0)).unsqueeze(1), clss]
            init_i = self.init.unsqueeze(0).unsqueeze(1).expand(x.size(0), 1, top_vec.size(-1))
            sents_vec = sents_vec * mask_cls[:, :, None].float()
            output = init_i
            mask_out_cls = (torch.ones([x.size(0), 1])==1).to(init_i.device)
            scores=None
            for j in range(3):
                sent_scores = self.encoder(sents_vec, output, mask_cls, mask_out_cls)
                sent_scores=sent_scores[:,-1,:]*mask_cls.float()
                indices=torch.max(sent_scores, dim=-1)[1]
                if j==0:
                    mask_out_cls = torch.cat([mask_out_cls, mask_out_cls],dim=-1)
                    scores=sent_scores.unsqueeze(dim=1)
                else:
                    mask_out_cls = torch.cat([mask_out_cls,(torch.ones([x.size(0), 1])==1).to(mask_out_cls.device)],dim=-1)#~((( mask_out_cls[:,-1] == 0).unsqueeze(-1)) + ((torch.sum(indices.unsqueeze(-1) == preds,dim=-1)>0).unsqueeze(-1)))
                    scores = torch.cat([scores,sent_scores.unsqueeze(dim=1)],dim=1)#
                if preds is None:
                    preds = indices.unsqueeze(-1)
                else:
                    indices[~(mask_out_cls[:,-1])] = -1
                    preds = torch.cat([preds,indices.unsqueeze(-1)],dim = -1)
                output = torch.cat([output, sents_vec[torch.arange(top_vec.size(0)).unsqueeze(1), indices.unsqueeze(-1)]], dim=1)
            return scores,preds

class ComSummarizer(nn.Module):
    def __init__(self, args, device, load_pretrained_bert = False, Albert_config = None):
        super(ComSummarizer, self).__init__()
        self.args = args
        self.device = device
        self.bert = Albert_unshared(args.temp_dir, load_pretrained_bert, Albert_config)
        if (args.encoder == 'classifier'):
            self.encoder = Classifier(self.bert.model.config.hidden_size)
        elif(args.encoder=='transformer'):
            self.encoder = TransformerInterEncoder(self.bert.model.config.hidden_size, args.ff_size, args.heads,
                                                   args.dropout, args.inter_layers)
        elif(args.encoder=='rnn'):
            self.encoder = RNNEncoder(bidirectional=True, num_layers=1,
                                      input_size=self.bert.model.config.hidden_size, hidden_size=args.rnn_size,
                                      dropout=args.dropout)
        elif (args.encoder == 'baseline'):
            bert_config = AlbertConfig.from_pretrained('albert-base-v2')
            self.bert.model = AlbertModel(bert_config)
            self.encoder = Classifier(self.bert.model.config.hidden_size)

        if args.param_init != 0.0:
            for p in self.encoder.parameters():
                p.data.uniform_(-args.param_init, args.param_init)
        if args.param_init_glorot:
            for p in self.encoder.parameters():
                if p.dim() > 1:
                    xavier_uniform_(p)
        self.to(device)
    def load_cp(self, pt):
        self.load_state_dict(pt['model'], strict=True)

    def forward(self, x, segs, clss, mask, mask_cls, com_mask, com_ids, com_segs, com_cls, sentence_range=None):

        top_vec = self.bert(x, segs, mask)
        with torch.no_grad():
            com_vec = self.bert(com_ids, com_segs, com_mask)
        sents_vec = top_vec[torch.arange(top_vec.size(0)).unsqueeze(1), clss]
        coms_vec = com_vec[torch.arange(com_vec.size(0)).unsqueeze(1), com_cls]
        sents_vec = sents_vec * mask_cls[:, :, None].float()
        coms_vec = coms_vec * mask_cls[:, :, None].float()
        sents_vec = (1-self.args.rate)*sents_vec + self.args.rate * coms_vec
        sent_scores = self.encoder(sents_vec, mask_cls).squeeze(-1)

        return sent_scores, mask_cls

class ClfSummarizer(nn.Module):
    def __init__(self, args, device, load_pretrained_bert = False, Albert_config = None):
        super(ClfSummarizer, self).__init__()
        self.args = args
        self.device = device
        self.bert = Bert(args.temp_dir, load_pretrained_bert, Albert_config)
        self.to(device)
    def load_cp(self, pt):
        self.load_state_dict(pt['model'], strict=True)

    def forward(self, tgt_ids, tgt_segs, mask, pred_sent, pred_segs,pred_mask,pred_labels,lables_mask, sentence_range=None):#  tgt_src, tgt_segs, tgt_mask
        top_vec = self.bert(tgt_ids, tgt_segs, mask)
        sent_vec=self.bert(pred_sent.view(-1, pred_sent.size(-1)),pred_segs.view(-1, pred_sent.size(-1)) , pred_mask.view(-1, pred_sent.size(-1)))#
        sent_vec = sent_vec.view(tgt_segs.size(0), pred_sent.size(1), self.bert.model.config.hidden_size)#[:, 0, :]
        top_vec = top_vec.unsqueeze(1).expand_as(sent_vec)
        sent_scores = torch.cosine_similarity(sent_vec, top_vec, dim=-1)
        return sent_scores, lables_mask#

