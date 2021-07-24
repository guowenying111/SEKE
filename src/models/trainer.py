import os
import torch
from torch import nn
from tensorboardX import SummaryWriter
import multiprocessing
import sys
from os.path import join, exists
from pyrouge import Rouge155
from transformers import AlbertTokenizer
import tempfile
import logging
from pyrouge.utils import log
sys.path.append('../')
import distributed
from torch.nn import functional as F
import subprocess as sp
_ROUGE_PATH = '/home/gwy/ROUGE/RELEASE-1.5.5'
from models.reporter import ReportMgr
from models.stats import Statistics
from others.logging import logger
from others.utils import test_rouge, rouge_results_to_str
import numpy as np
from itertools import combinations
import jsonlines

temp_path = '../temp/train_low'


class MarginLoss(nn.Module):

    def __init__(self, margin):
        super(MarginLoss, self).__init__()
        self.margin = margin
        self.loss_func = torch.nn.MarginRankingLoss(margin)

    def forward(self, score):
        # equivalent to initializing TotalLoss to 0
        # here is to avoid that some special samples will not go into the following for loop
        ones = torch.ones(score.size()).cuda()
        loss_func = torch.nn.MarginRankingLoss(0.0)
        TotalLoss = loss_func(score, score, ones)

        # candidate loss
        n = score.size(1)
        for i in range(1, n - 1):
            pos_score = score[:, :-i]
            neg_score = score[:, i:]
            pos_score = pos_score.contiguous().view(-1)
            neg_score = neg_score.contiguous().view(-1)
            ones = torch.ones(pos_score.size()).cuda(score.device)
            loss_func = torch.nn.MarginRankingLoss(self.margin * i)
            TotalLoss += loss_func(pos_score, neg_score, ones)

        return TotalLoss

class Margin2Loss(nn.Module):

    def __init__(self, margin):
        super(Margin2Loss, self).__init__()
        self.margin = margin
        self.loss_func = torch.nn.MarginRankingLoss(margin)

    def forward(self, sum_score, score):
        # equivalent to initializing TotalLoss to 0
        # here is to avoid that some special samples will not go into the following for loop
        ones = torch.ones(score.size()).cuda()
        loss_func = torch.nn.MarginRankingLoss(0.0)
        TotalLoss = loss_func(score, score, ones)

        # candidate loss
        n = score.size(1)
        for i in range(1, n - 1):
            pos_score = score[:, :-i]
            neg_score = score[:, i:]
            pos_score = pos_score.contiguous().view(-1)
            neg_score = neg_score.contiguous().view(-1)
            ones = torch.ones(pos_score.size()).cuda(score.device)
            loss_func = torch.nn.MarginRankingLoss(self.margin * i)
            TotalLoss += loss_func(pos_score, neg_score, ones)
        pos_score = sum_score.unsqueeze(-1).expand_as(score)
        neg_score = score
        pos_score = pos_score.contiguous().view(-1)
        neg_score = neg_score.contiguous().view(-1)
        ones = torch.ones(pos_score.size()).cuda(score.device)
        loss_func = torch.nn.MarginRankingLoss(0.0)

        TotalLoss += loss_func(pos_score, neg_score, ones)

        return TotalLoss

class Label_smoothing(nn.Module):
    def  __init__(self,label_smoothing):
        super(Label_smoothing,self).__init__()
        self.label_smoothing = label_smoothing
    def forward(self, output, target):
        target = target.float()*(1-self.label_smoothing) + 0.5 * self.label_smoothing
        loss = F.binary_cross_entropy(output, target.type_as(output), reduction='none')
        return loss
def _tally_parameters(model):
    n_params = sum([p.nelement() for p in model.parameters()])
    return n_params

count=0

def build_trainer(args, device_id, model,
                  optim):
    """
    Simplify `Trainer` creation based on user `opt`s*
    Args:
        opt (:obj:`Namespace`): user options (usually from argument parsing)
        model (:obj:`onmt.models.NMTModel`): the model to train
        fields (dict): dict of fields
        optim (:obj:`onmt.utils.Optimizer`): optimizer used during training
        data_type (str): string describing the type of data
            e.g. "text", "img", "audio"
        model_saver(:obj:`onmt.models.ModelSaverBase`): the utility object
            used to save the model
    """
    device = "cpu" if args.visible_gpus == '-1' else "cuda"


    grad_accum_count = args.accum_count
    n_gpu = args.world_size

    if device_id >= 0:
        gpu_rank = int(args.gpu_ranks[device_id])
    else:
        gpu_rank = 0
        n_gpu = 0

    print('gpu_rank %d' % gpu_rank)

    tensorboard_log_dir = args.model_path

    writer = SummaryWriter(tensorboard_log_dir, comment="Unmt")

    report_manager = ReportMgr(args.report_every, start_time=-1, tensorboard_writer=writer)

    trainer = Trainer(args, model, optim, grad_accum_count, n_gpu, gpu_rank, report_manager)

    # print(tr)
    if (model):
        n_params = _tally_parameters(model)
        logger.info('* number of parameters: %d' % n_params)

    return trainer


class Trainer(object):
    """
    Class that controls the training process.

    Args:
            model(:py:class:`onmt.models.model.NMTModel`): translation model
                to train
            train_loss(:obj:`onmt.utils.loss.LossComputeBase`):
               training loss computation
            valid_loss(:obj:`onmt.utils.loss.LossComputeBase`):
               training loss computation
            optim(:obj:`onmt.utils.optimizers.Optimizer`):
               the optimizer responsible for update
            trunc_size(int): length of truncated back propagation through time
            shard_size(int): compute loss in shards of this size for efficiency
            data_type(string): type of the source input: [text|img|audio]
            norm_method(string): normalization methods: [sents|tokens]
            grad_accum_count(int): accumulate gradients this many times.
            report_manager(:obj:`onmt.utils.ReportMgrBase`):
                the object that creates reports, or None
            model_saver(:obj:`onmt.models.ModelSaverBase`): the saver is
                used to save a checkpoint.
                Thus nothing will be saved if this parameter is None
    """

    def __init__(self,  args, model,  optim,
                  grad_accum_count=1, n_gpu=1, gpu_rank=1,
                  report_manager=None):
        # Basic attributes.
        self.args = args
        self.save_checkpoint_steps = args.save_checkpoint_steps
        self.model = model
        self.optim = optim
        self.grad_accum_count = grad_accum_count
        self.n_gpu = n_gpu
        self.gpu_rank = gpu_rank
        self.report_manager = report_manager
        self.loss = torch.nn.BCELoss(reduction='none')#Label_smoothing(0) #torch.nn.NLLLoss(reduction='none')# Margin2Loss(margin=0.01) Margin2Loss(margin=0.01)
        assert grad_accum_count > 0
        # Set model in training mode.
        if (model):
            self.model.train()

    def train(self, train_iter_fct, train_steps, valid_iter_fct=None, valid_steps=-1):
        """
        The main training loops.
        by iterating over training data (i.e. `train_iter_fct`)
        and running validation (i.e. iterating over `valid_iter_fct`

        Args:
            train_iter_fct(function): a function that returns the train
                iterator. e.g. something like
                train_iter_fct = lambda: generator(*args, **kwargs)
            valid_iter_fct(function): same as train_iter_fct, for valid data
            train_steps(int):
            valid_steps(int):
            save_checkpoint_steps(int):

        Return:
            None
        """
        logger.info('Start training...')

        step =  self.optim._step + 1
        true_batchs = []
        accum = 0
        normalization = 0
        train_iter = train_iter_fct()

        total_stats = Statistics()
        report_stats = Statistics()
        self._start_report_manager(start_time=total_stats.start_time)

        while step <= train_steps:

            reduce_counter = 0
            for i, batch in enumerate(train_iter):
                if self.n_gpu == 0 or (i % self.n_gpu == self.gpu_rank):
                    true_batchs.append(batch)
                    normalization += batch.batch_size
                    accum += 1
                    if accum == self.grad_accum_count:
                        reduce_counter += 1
                        if self.n_gpu > 1:
                            normalization = sum(distributed
                                                .all_gather_list
                                                (normalization))

                        self._gradient_accumulation(
                            true_batchs, normalization, total_stats,
                            report_stats,step)

                        report_stats = self._maybe_report_training(
                            step, train_steps,
                            self.optim.learning_rate,
                            report_stats)

                        true_batchs = []
                        accum = 0
                        normalization = 0
                        if (step % self.save_checkpoint_steps == 0 and self.gpu_rank == 0):
                            self._save(step)

                        step += 1
                        if step > train_steps:
                            break
            train_iter = train_iter_fct()

        return total_stats

    def validate(self, valid_iter, step=0):
        """ Validate model.
            valid_iter: validate data iterator
        Returns:
            :obj:`nmt.Statistics`: validation loss statistics
        """
        # Set model in validating mode.
        self.model.eval()
        stats = Statistics()

        with torch.no_grad():
            for batch in valid_iter:

                if self.args.model == 'clf_albert':
                    mask = batch.mask
                    tgt_src = batch.tgt_src
                    labels = batch.labels
                    tgt_mask = batch.tgt_mask
                    tgt_segs = batch.tgt_segs
                    pred_segs = batch.pred_segs
                    pred_sent = batch.pred_sent
                    pred_sent_mask = batch.pred_sent_mask
                    if self.args.reproduce:
                        sum_id = batch.sum_src
                        sum_mask = batch.sum_mask
                        sum_segs = batch.sum_segs
                elif self.args.model == 'pointer':
                    clss = batch.clss
                    src = batch.src
                    labels = batch.labels
                    segs = batch.segs
                    mask = batch.mask
                    mask_cls = batch.mask_cls
                    out_clss = batch.out_clss
                    mask_out_clss = batch.mask_out_cls
                    mask_target_clss = batch.mask_target_cls
                    mask_label = batch.mask_label
                else:
                    src = batch.src
                    labels = batch.labels
                    segs = batch.segs
                    clss = batch.clss
                    mask = batch.mask
                    mask_cls = batch.mask_cls
                    if self.args.model == 'gcn':
                        graph = batch.graph
                    elif self.args.model == 'albert_com':
                        com_mask = batch.com_mask
                        com_ids = batch.com_ids
                        com_segs = batch.com_segs
                        com_cls = batch.com_clss



                if self.args.model == 'gcn':
                    sent_scores, mask = self.model(src, segs, clss, mask, mask_cls, graph)
                    loss = self.loss(sent_scores, labels.float())
                    loss = (loss * mask.float()).sum()
                elif self.args.model == 'pointer':
                    sent_scores, _ = self.model(src, segs, clss, mask, mask_cls, mask_out_clss, out_clss,
                                                mask_target_clss)
                    loss = self.loss(torch.log(sent_scores[:, :labels.size(-1), :]).view(-1, sent_scores.size(-1)),
                                     labels.view(-1).long())
                    loss = (loss.view(labels.size()) * mask_label.float()).sum()
                elif self.args.model == 'clf_albert':
                    if self.args.reproduce:
                        sum_score, sent_scores, mask = self.model(tgt_src, tgt_segs, tgt_mask, pred_sent, pred_segs,
                                                                  pred_sent_mask, labels, mask, sum_id, sum_mask,
                                                                  sum_segs)  #
                        loss = self.loss(sum_score, sent_scores)
                    else:
                        sent_scores, mask = self.model(tgt_src, tgt_segs, tgt_mask, pred_sent, pred_segs,
                                                       pred_sent_mask,
                                                       labels, mask)  # sum_score,, sum_id, sum_mask
                        loss = self.loss(sent_scores)
                elif self.args.model=='albert_com':
                    sent_scores, mask = self.model(src, segs, clss, mask, mask_cls, com_mask, com_ids, com_segs,com_cls)
                    loss = self.loss(sent_scores, labels.float())
                    loss = (loss * mask.float()).sum()
                else:
                    sent_scores, mask = self.model(src, segs, clss, mask, mask_cls)
                    loss = self.loss(sent_scores, labels.float())
                    loss = (loss * mask.float()).sum()
                batch_stats = Statistics(float(loss.cpu().data.numpy()), len(labels))
                stats.update(batch_stats)
            self._report_step(0, step, valid_stats=stats)
            return stats

    def test(self, test_iter, step, cal_lead=False, cal_oracle=False):
        """ Validate model.
            valid_iter: validate data iterator
        Returns:
            :obj:`nmt.Statistics`: validation loss statistics
        """
        # Set model in validating mode.
        def _get_ngrams(n, text):
            ngram_set = set()
            text_length = len(text)
            max_index_ngram_start = text_length - n
            for i in range(max_index_ngram_start + 1):
                ngram_set.add(tuple(text[i:i + n]))
            return ngram_set

        def _block_tri(c, p):
            tri_c = _get_ngrams(3, c.split())
            for s in p:
                tri_s = _get_ngrams(3, s.split())
                if len(tri_c.intersection(tri_s))>0:
                    return True
            return False

        if (not cal_lead and not cal_oracle):
            self.model.eval()
        stats = Statistics()

        can_path = '%s_step%d.candidate'%(self.args.result_path,step)
        gold_path = '%s_step%d.gold' % (self.args.result_path, step)
        with jsonlines.open('Xsum_valid_content.jsonl', mode='a') as content_writer:
            with jsonlines.open('Xsum_valid_idx.jsonl', mode='a') as idx_writer:
                with open(can_path, 'w') as save_pred:
                    with open(gold_path, 'w') as save_gold:
                        with torch.no_grad():
                            for batch in test_iter:
                                if self.args.model == 'clf_albert':
                                    mask = batch.mask
                                    tgt_src = batch.tgt_src
                                    labels = batch.labels
                                    tgt_mask = batch.tgt_mask
                                    tgt_segs = batch.tgt_segs
                                    pred_segs = batch.pred_segs
                                    pred_sent = batch.pred_sent
                                    pred_sent_mask = batch.pred_sent_mask
                                    if self.args.reproduce:
                                        sum_id = batch.sum_src
                                        sum_mask = batch.sum_mask
                                        sum_segs = batch.sum_segs
                                elif self.args.model == 'pointer':
                                    clss = batch.clss
                                    src = batch.src
                                    labels = batch.labels
                                    segs = batch.segs
                                    mask = batch.mask
                                    mask_cls = batch.mask_cls
                                    out_clss = batch.out_clss
                                    mask_out_clss = batch.mask_out_cls
                                    mask_target_clss = batch.mask_target_cls
                                    mask_label = batch.mask_label
                                else:
                                    src = batch.src
                                    labels = batch.labels
                                    segs = batch.segs
                                    clss = batch.clss
                                    mask = batch.mask
                                    mask_cls = batch.mask_cls
                                    if self.args.model=='gcn':
                                        graph = batch.graph
                                    elif self.args.model == 'albert_com':
                                        com_mask = batch.com_mask
                                        com_ids = batch.com_ids
                                        com_segs = batch.com_segs
                                        com_cls = batch.com_clss

                                gold = []
                                pred = []

                                if (cal_lead):
                                    selected_ids = [list(range(batch.clss.size(1)))] * batch.batch_size
                                elif (cal_oracle):
                                    selected_ids = [[j for j in range(batch.clss.size(1)) if labels[i][j] == 1] for i in
                                                    range(batch.batch_size)]
                                else:
                                    if self.args.model=='gcn':
                                        sent_scores, mask = self.model(src, segs, clss, mask, mask_cls, graph)
                                        loss = self.loss(sent_scores, labels.float())
                                        loss = (loss * mask.float()).sum()
                                    elif self.args.model == 'clf_albert':
                                        if self.args.reproduce:
                                            sum_score, sent_scores, mask = self.model(tgt_src, tgt_segs, tgt_mask, pred_sent,
                                                                                      pred_segs,
                                                                                      pred_sent_mask, labels, mask, sum_id,
                                                                                      sum_mask,
                                                                                      sum_segs)  #
                                            loss = self.loss(sum_score, sent_scores)
                                        else:
                                            sent_scores, mask = self.model(tgt_src, tgt_segs, tgt_mask, pred_sent, pred_segs,
                                                                           pred_sent_mask,
                                                                           labels, mask)  # sum_score, sum_id, sum_mask
                                            loss = self.loss(sent_scores)
                                    elif self.args.model=='albert_com':
                                        sent_scores, mask = self.model(src, segs, clss, mask, mask_cls, com_mask, com_ids,
                                                                       com_segs,com_cls)
                                        loss = self.loss(sent_scores, labels.float())
                                        loss = (loss * mask.float()).sum()
                                    elif self.args.model=="pointer":
                                        sent_scores, _ = self.model(src, segs, clss, mask, mask_cls, mask_out_clss, out_clss,
                                                                    mask_target_clss)
                                        loss = self.loss(
                                            torch.log(sent_scores[:, :labels.size(-1), :]).view(-1, sent_scores.size(-1)),
                                            labels.view(-1).long())
                                        loss = (loss.view(labels.size()) * mask_label.float()).sum()
                                    else:
                                        sent_scores, mask = self.model(src, segs, clss, mask,
                                                                       mask_cls)
                                        loss = self.loss(sent_scores, labels.float())
                                        loss = (loss * mask.float()).sum()

                                    batch_stats = Statistics(float(loss.cpu().data.numpy()), len(labels))
                                    stats.update(batch_stats)
                                    if self.args.model=='albert' or self.args.model=='bert':
                                        sent_scores = sent_scores + mask.float()
                                    sent_scores = sent_scores.cpu().data.numpy()
                                    selected_ids = np.argsort(-sent_scores, 1)
                                if self.args.model == 'clf_albert':
                                    for i, idx in enumerate(selected_ids):
                                        cand=idx[0]
                                        _pred=[]
                                        for j in batch.pred_idx[i][cand]:
                                            candidate = batch.src_str[i][j].strip()
                                            _pred.append(candidate)
                                        _pred = '<q>'.join(_pred)
                                        pred.append(_pred)
                                        gold.append(batch.tgt_str[i])

                                    for i in range(len(gold)):
                                        save_gold.write(gold[i].strip()+'\n')
                                    for i in range(len(pred)):
                                        save_pred.write(pred[i].strip()+'\n')
                                elif self.args.model == 'pointer':
                                    for i, idx in enumerate(_):
                                        _pred = []
                                        x=torch.zeros(100,dtype=int)
                                        if (len(batch.src_str[i]) == 0):
                                            continue
                                        for ids, j in enumerate(_[i]):
                                            if x[j]==1:
                                                continue
                                            else:
                                                x[j]=1
                                            candidate = batch.src_str[i][j].strip()
                                            if (self.args.block_trigram):
                                                if (not _block_tri(candidate, _pred)):
                                                    _pred.append(candidate)
                                            else:
                                                _pred.append(candidate)
                                            if ((not cal_oracle) and (not self.args.recall_eval) and len(_pred) == 3):
                                                break
                                        _pred = '<q>'.join(_pred)
                                        pred.append(_pred)
                                        gold.append(batch.tgt_str[i])
                                    for i in range(len(gold)):
                                        save_gold.write(gold[i].strip() + '\n')
                                    for i in range(len(pred)):
                                        save_pred.write(pred[i].strip() + '\n')

                                else:
                                    for i, idx in enumerate(selected_ids):
                                        _pred = []
                                        if(len(batch.src_str[i])==0):
                                            continue
                                        # sent_ids = {}
                                        # text={}
                                        # sent_id = []
                                        for ids,j in enumerate(selected_ids[i][:len(batch.src_str[i])]):
                                            if j>=len( batch.src_str[i]):
                                                continue
                                            # sent_id.append(np.uint32(j).item())
                                            candidate = batch.src_str[i][j].strip()
                                            if(self.args.block_trigram):
                                                if(not _block_tri(candidate,_pred)):
                                                    _pred.append(candidate)
                                            else:
                                                _pred.append(candidate)
                                                # sent_id.append(j)
                                            if ((not cal_oracle) and (not self.args.recall_eval) and len(_pred) == 3):
                                                break
                                        _pred = '<q>'.join(_pred)
                                        # sent_ids['sent_id'] = sent_id
                                        if(self.args.recall_eval):
                                            _pred = ' '.join(_pred.split()[:len(batch.tgt_str[i].split())])
                                        # text['text'] = batch.src_str[i]
                                        # text['summary'] = batch.tgt_str[i].split('<q>')
                                        # content_writer.write(text)
                                        # idx_writer.write(sent_ids)
                                        pred.append(_pred)
                                        gold.append(batch.tgt_str[i])

                                    for i in range(len(gold)):
                                        save_gold.write(gold[i].strip() + '\n')
                                    for i in range(len(pred)):
                                        save_pred.write(pred[i].strip() + '\n')
        if(step!=-1 and self.args.report_rouge):
            rouges = test_rouge(self.args.temp_dir, can_path, gold_path)
            logger.info('Rouges at step %d \n%s' % (step, rouge_results_to_str(rouges)))
        self._report_step(0, step, valid_stats=stats)

        return stats



    def process(self, test_iter, step):
        ref_dir = join(temp_path, 'reference')
        cls_vid = 2#tokenizer.convert_tokens_to_ids('[CLS]')
        sep_vid = 3#tokenizer.convert_tokens_to_ids('[SEP]')
        """ Validate model.
            valid_iter: validate data iterator
        Returns:
            :obj:`nmt.Statistics`: validation loss statistics
        """
        # Set model in validating mode.
        # full_can=open('3_can','w')
        # full_gold=open('3_gold','w')
        self.model.eval()
        stats = Statistics()
        sen_dst={}
        dict_data=[]
        num=0
        total=0
        with torch.no_grad():
            for batch in test_iter:
                src = batch.src
                labels = batch.labels
                segs = batch.segs
                clss = batch.clss
                mask = batch.mask
                mask_cls = batch.mask_cls

                if self.args.model=='gcn':
                    graph = batch.graph

                gold = []
                pred = []

                if self.args.model=='gcn':
                    sent_scores, mask = self.model(src, segs, clss, mask, mask_cls, graph)
                else:
                    sent_scores, mask = self.model(src, segs, clss, mask,
                                                   mask_cls)
                loss = self.loss(sent_scores, labels.float())
                loss = (loss * mask.float()).sum()
                batch_stats = Statistics(float(loss.cpu().data.numpy()), len(labels))
                stats.update(batch_stats)

                sent_scores = sent_scores + mask.float()
                sent_scores = sent_scores.cpu().data.numpy()
                selected_ids = np.argsort(-sent_scores, 1)

                for i, idx in enumerate(selected_ids):
                    _pred = []
                    if(len(batch.src_str[i])==0):
                        continue#5,6
                    for ids,j in enumerate(selected_ids[i][:len(batch.src_str[i])]):
                        if (j>=len( batch.src_str[i])) :#or ids>7
                            continue
                        _pred.append(j)
                        if len(_pred) == 5:
                            break
                    indices = list(combinations(_pred, 2))
                    indices += list(combinations(_pred, 3))
                    # indices += list(combinations(_pred, 1))

                    if len(_pred) < 2:
                        indices = [_pred]
                    score = []
                    cand_labels=[]
                    n=len(indices)
                    pool = multiprocessing.Pool(n)#len(indices)
                    results=[]
                    for id in range (n):#len(indices)
                        result=pool.apply_async(run_pro,(indices,batch.src_str,batch.tgt_str,temp_path,id,i,n,))#[i][i]
                        results.append(result)
                    pool.close()
                    pool.join()
                    for result in results:
                        score.append(result.get())
                    # print('first',score)
                    score.sort(key=lambda x: x[1], reverse=True)
                    # print('second',score)
                    total+=score[0][1]
                    num+=1
                    pred_sent=[]
                    pred_segs=[]
                    pred_idx=[]
                    for id,item in enumerate(score):
                        cur_sent=[cls_vid]
                        for num_id in item[0]:
                            cur_sent += batch.src_tokens_idxs[i][num_id]
                        if id==0:
                            cand_labels.append(1)
                            pred_idx.append(item[0])
                        else:
                                cand_labels.append(0)
                                pred_idx.append(item[0])
                        _segs = [-1] + [k for k, t in enumerate(cur_sent) if t == sep_vid]
                        segs = [_segs[k] - _segs[k - 1] for k in range(1, len(_segs))]
                        segments_ids = []
                        for k, s in enumerate(segs):
                            if (k % 2 == 0):
                                segments_ids += s * [0]
                            else:
                                segments_ids += s * [1]
                        pred_sent.append(cur_sent)
                        pred_segs.append(segments_ids)
                    data = {
                            'tgt_ids':batch.tgt_src[i],'tgt_segs':batch.pre_tgt_segs[i],'pred_sent':pred_sent,'pred_segs':pred_segs,
                            'pred_labels':cand_labels,'pred_idx':pred_idx,'src_txt':batch.src_str[i],'tgt_txt':batch.tgt_str[i]
                            }
                    dict_data.append(data)

            name=self.args.bert_data_path.split('/')[-1]
            torch.save(dict_data,join('/home/gwy/ALL/BertSum/clf_data',name))


    def _gradient_accumulation(self, true_batchs, normalization, total_stats,
                               report_stats,step):
        if self.grad_accum_count > 1:
            self.model.zero_grad()

        for batch in true_batchs:
            if self.grad_accum_count == 1:
                self.model.zero_grad()
            if self.args.model=='clf_albert':
                mask=batch.mask
                tgt_src=batch.tgt_src
                labels=batch.labels
                tgt_mask=batch.tgt_mask
                tgt_segs=batch.tgt_segs
                pred_segs=batch.pred_segs
                pred_sent=batch.pred_sent
                pred_sent_mask=batch.pred_sent_mask
                if self.args.reproduce:
                    sum_id=batch.sum_src
                    sum_mask=batch.sum_mask
                    sum_segs=batch.sum_segs
            else:
                clss = batch.clss
                src = batch.src
                labels = batch.labels
                segs = batch.segs
                mask = batch.mask
                mask_cls = batch.mask_cls
            if self.args.model == 'gcn':
                graph = batch.graph
                sent_scores, mask = self.model(src, segs, clss, mask, mask_cls, graph)
                loss = self.loss(sent_scores, labels.float())
                loss = (loss * mask.float()).sum()
            elif self.args.model == 'pointer':
                out_clss = batch.out_clss
                mask_out_clss = batch.mask_out_cls
                mask_target_clss = batch.mask_target_cls
                mask_label =batch.mask_label
                sent_scores,_ = self.model(src, segs, clss, mask, mask_cls, mask_out_clss, out_clss, mask_target_clss)
                loss = self.loss(torch.log(sent_scores).view(-1,sent_scores.size(-1)), labels.view(-1).long())
                loss = (loss.view(labels.size()) * mask_label.float()).sum()
            elif self.args.model == 'clf_albert':
                if self.args.reproduce:
                    sum_score, sent_scores, mask = self.model(tgt_src, tgt_segs, tgt_mask, pred_sent, pred_segs,
                                                              pred_sent_mask, labels, mask, sum_id, sum_mask,sum_segs)  #
                    loss = self.loss(sum_score, sent_scores)
                else:
                    sent_scores, mask = self.model(tgt_src, tgt_segs, tgt_mask, pred_sent, pred_segs, pred_sent_mask,
                                                   labels, mask)  # sum_score,, sum_id, sum_mask
                    loss = self.loss(sent_scores)
            elif self.args.model=='albert_com':
                com_mask = batch.com_mask
                com_ids = batch.com_ids
                com_segs = batch.com_segs
                com_cls = batch.com_clss
                sent_scores, mask = self.model(src, segs, clss, mask, mask_cls, com_mask, com_ids, com_segs,com_cls)
                loss = self.loss(sent_scores, labels.float())
                loss = (loss * mask.float()).sum()
            else:
                sent_scores, mask = self.model(src, segs, clss, mask, mask_cls)
                loss = self.loss(sent_scores, labels.float())
                loss = (loss * mask.float()).sum()
            (loss/loss.numel()).backward()
            batch_stats = Statistics(float(loss.cpu().data.numpy()), normalization)

            total_stats.update(batch_stats)
            report_stats.update(batch_stats)

            # 4. Update the parameters and statistics.
            if self.grad_accum_count == 1:
                # Multi GPU gradient gather
                if self.n_gpu > 1:
                    grads = [p.grad.data for p in self.model.parameters()
                             if p.requires_grad
                             and p.grad is not None]
                    distributed.all_reduce_and_rescale_tensors(
                        grads, float(1))
                self.optim.step()

        if self.grad_accum_count > 1:
            if self.n_gpu > 1:
                grads = [p.grad.data for p in self.model.parameters()
                         if p.requires_grad
                         and p.grad is not None]
                distributed.all_reduce_and_rescale_tensors(
                    grads, float(1))
            self.optim.step()

    def _save(self, step):
        real_model = self.model
        model_state_dict = real_model.state_dict()
        # generator_state_dict = real_generator.state_dict()
        checkpoint = {
            'model': model_state_dict,
            # 'generator': generator_state_dict,
            'opt': self.args,
            'optim': self.optim,
        }
        checkpoint_path = os.path.join(self.args.model_path, 'model_step_%d.pt' % step)
        logger.info("Saving checkpoint %s" % checkpoint_path)
        # checkpoint_path = '%s_step_%d.pt' % (FLAGS.model_path, step)
        if (not os.path.exists(checkpoint_path)):
            torch.save(checkpoint, checkpoint_path)
            return checkpoint, checkpoint_path

    def _start_report_manager(self, start_time=None):
        """
        Simple function to start report manager (if any)
        """
        if self.report_manager is not None:
            if start_time is None:
                self.report_manager.start()
            else:
                self.report_manager.start_time = start_time

    def _maybe_gather_stats(self, stat):
        """
        Gather statistics in multi-processes cases

        Args:
            stat(:obj:onmt.utils.Statistics): a Statistics object to gather
                or None (it returns None in this case)

        Returns:
            stat: the updated (or unchanged) stat object
        """
        if stat is not None and self.n_gpu > 1:
            return Statistics.all_gather_stats(stat)
        return stat

    def _maybe_report_training(self, step, num_steps, learning_rate,
                               report_stats):
        """
        Simple function to report training stats (if report_manager is set)
        see `onmt.utils.ReportManagerBase.report_training` for doc
        """
        if self.report_manager is not None:
            return self.report_manager.report_training(
                step, num_steps, learning_rate, report_stats,
                multigpu=self.n_gpu > 1)

    def _report_step(self, learning_rate, step, train_stats=None,
                     valid_stats=None):
        """
        Simple function to report stats (if report_manager is set)
        see `onmt.utils.ReportManagerBase.report_step` for doc
        """
        if self.report_manager is not None:
            return self.report_manager.report_step(
                learning_rate, step, train_stats=train_stats,
                valid_stats=valid_stats)

    def _maybe_save(self, step):
        """
        Save the model if a model saver is set
        """
        if self.model_saver is not None:
            self.model_saver.maybe_save(step)
def run_pro(indices,src_str,tgt_str,tmp_dir,id,i,n):#
    tmp_dir=join(tmp_dir,str(id))
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)
    ref_dir=join(tmp_dir,'reference')
    if not os.path.exists(ref_dir):
        os.makedirs(ref_dir)
    # total = []
    for ids,cand_idxs in enumerate(indices):
        if ids%n==id:
            cand_idxs = list(cand_idxs)
            cand_idxs.sort()
            dec = []
            for j in cand_idxs:
                sent = src_str[i][j]
                dec.append(sent)

            with open(join(ref_dir, '0.ref'), 'w') as f:
                for sentence in tgt_str[i].split('<q>'):
                    print(sentence, file=f)
            # print('rouge',(cand_idxs, get_rouge(tmp_dir, dec)))
            # total.append((cand_idxs, get_rouge(tmp_dir, dec)))
            return (cand_idxs, get_rouge(tmp_dir, dec))

def get_rouge(path, dec):
    log.get_global_console_logger().setLevel(logging.WARNING)
    dec_pattern = '(\d+).dec'
    ref_pattern = '#ID#.ref'
    dec_dir = join(path, 'decode')
    if not os.path.exists(dec_dir):
        os.mkdir(dec_dir)
    ref_dir = join(path, 'reference')

    with open(join(dec_dir, '0.dec'), 'w') as f:
        for sentence in dec:
            print(sentence, file=f)

    cmd = '-c 95 -r 1000 -n 2 -m'
    tmp_dir=join(temp_path,'tmp')
    with tempfile.TemporaryDirectory() as tmp_dir:
        Rouge155.convert_summaries_to_rouge_format(
            dec_dir, join(tmp_dir, 'dec'))
        Rouge155.convert_summaries_to_rouge_format(
            ref_dir, join(tmp_dir, 'ref'))
        Rouge155.write_config_static(
            join(tmp_dir, 'dec'), dec_pattern,
            join(tmp_dir, 'ref'), ref_pattern,
            join(tmp_dir, 'settings.xml'), system_id=1
        )
        cmd = (join(_ROUGE_PATH, 'ROUGE-1.5.5.pl')
               + ' -e {} '.format(join(_ROUGE_PATH, 'data'))
               + cmd
               + ' -a {}'.format(join(tmp_dir, 'settings.xml')))
        output = sp.check_output(cmd.split(' '), universal_newlines=True)

        line = output.split('\n')
        rouge1 = float(line[3].split(' ')[3])
        rouge2 = float(line[7].split(' ')[3])
        rougel = float(line[11].split(' ')[3])
    return (rouge1 + rouge2 + rougel) / 3
if __name__ == '__main__':
    # test_rouge('../../temp','/home/gwy/ALL/neusum/code/NeuSum/neusum_pt/candidate','/home/gwy/ALL/neusum/code/NeuSum/neusum_pt/gold')
    results_dict=test_rouge('../../temp','../3_can','../3_gold')
    print(results_dict)# 5{'rouge_1_recall': 0.54861, 'rouge_1_recall_cb': 0.54613, 'rouge_1_recall_ce': 0.55094, 'rouge_1_precision': 0.50461, 'rouge_1_precision_cb': 0.50194, 'rouge_1_precision_ce': 0.50705, 'rouge_1_f_score': 0.51334, 'rouge_1_f_score_cb': 0.51117, 'rouge_1_f_score_ce': 0.51528, 'rouge_2_recall': 0.2886, 'rouge_2_recall_cb': 0.28579, 'rouge_2_recall_ce': 0.29116, 'rouge_2_precision': 0.26925, 'rouge_2_precision_cb': 0.26635, 'rouge_2_precision_ce': 0.27205, 'rouge_2_f_score': 0.27165, 'rouge_2_f_score_cb': 0.26892, 'rouge_2_f_score_ce': 0.27415, 'rouge_l_recall': 0.50609, 'rouge_l_recall_cb': 0.50354, 'rouge_l_recall_ce': 0.50838, 'rouge_l_precision': 0.46684, 'rouge_l_precision_cb': 0.46415, 'rouge_l_precision_ce': 0.4694, 'rouge_l_f_score': 0.47424, 'rouge_l_f_score_cb': 0.4719, 'rouge_l_f_score_ce': 0.47633}
                       # 4{'rouge_1_recall': 0.4684, 'rouge_1_recall_cb': 0.46582, 'rouge_1_recall_ce': 0.47087, 'rouge_1_precision': 0.51158, 'rouge_1_precision_cb': 0.50859, 'rouge_1_precision_ce': 0.5144, 'rouge_1_f_score': 0.47378, 'rouge_1_f_score_cb': 0.4717, 'rouge_1_f_score_ce': 0.47585, 'rouge_2_recall': 0.23566, 'rouge_2_recall_cb': 0.23312, 'rouge_2_recall_ce': 0.23822, 'rouge_2_precision': 0.26628, 'rouge_2_precision_cb': 0.26311, 'rouge_2_precision_ce': 0.26951, 'rouge_2_f_score': 0.24138, 'rouge_2_f_score_cb': 0.23884, 'rouge_2_f_score_ce': 0.24397, 'rouge_l_recall': 0.42835, 'rouge_l_recall_cb': 0.4259, 'rouge_l_recall_ce': 0.43072, 'rouge_l_precision': 0.46986, 'rouge_l_precision_cb': 0.46662, 'rouge_l_precision_ce': 0.47278, 'rouge_l_f_score': 0.43404, 'rouge_l_f_score_cb': 0.43189, 'rouge_l_f_score_ce': 0.43617}
