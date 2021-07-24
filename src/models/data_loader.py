import argparse
import bisect
import gc
import glob
import random
import sys

import torch
import sys
sys.path.append('../')
from others.logging import logger
# from torch_geometric.data import Batch,Data

# class PairData(Data):
#     def __init__(self, edge_index_s, edge_weight_s, edge_index_t, edge_weight_t, x_s=None):
#         super(PairData, self).__init__()
#         self.edge_index_s = edge_index_s
#         self.edge_weight_s = edge_weight_s
#         self.edge_index_t = edge_index_t
#         self.edge_weight_t=edge_weight_t
#
#     def __inc__(self, key, value):
#         if key == 'edge_index_s':
#             return self.x_s.size(0)
#         if key == 'edge_index_t':
#             return self.x_s.size(0)
#         else:
#             return super(PairData, self).__inc__(key, value)
#
class Batch(object):
    def _pad(self, data, pad_id, width=-1):
        if (width == -1):
            width = max(len(d) for d in data)
        rtn_data = [d + [pad_id] * (width - len(d)) for d in data]
        return rtn_data

    def __init__(self, data=None, device=None,  is_test=False):
        """Create a Batch from a list of examples."""
        if data is not None:
            self.batch_size = len(data)
            pre_src = [x[0] for x in data]
            pre_labels = [x[1] for x in data]
            pre_segs = [x[2] for x in data]
            pre_clss = [x[3] for x in data]
            # graph = [x[4] for x in data]
            src = torch.tensor(self._pad(pre_src, 0))

            labels = torch.tensor(self._pad(pre_labels, 0))
            segs = torch.tensor(self._pad(pre_segs, 0))
            mask = ~ (src == 0)

            clss = torch.tensor(self._pad(pre_clss, -1))
            mask_cls = ~ (clss == -1)
            setattr(self, 'clss', clss.to(device))
            setattr(self, 'mask_cls', mask_cls.to(device))
            setattr(self, 'src', src.to(device))
            setattr(self, 'labels', labels.to(device))
            setattr(self, 'segs', segs.to(device))
            setattr(self, 'mask', mask.to(device))
            # setattr(self, 'graph', graph)

            if (is_test):
                src_str = [x[-2] for x in data]
                setattr(self, 'src_str', src_str)
                tgt_str = [x[-1] for x in data]
                setattr(self, 'tgt_str', tgt_str)

    def __len__(self):
        return self.batch_size


def batch(data, batch_size):
    """Yield elements from data in chunks of batch_size."""
    minibatch, size_so_far = [], 0
    for ex in data:
        minibatch.append(ex)
        size_so_far = simple_batch_size_fn(ex, len(minibatch))
        if size_so_far == batch_size:
            yield minibatch
            minibatch, size_so_far = [], 0
        elif size_so_far > batch_size:
            yield minibatch[:-1]
            minibatch, size_so_far = minibatch[-1:], simple_batch_size_fn(ex, 1)
    if minibatch:
        yield minibatch


def load_dataset(args, corpus_type, shuffle):
    """
    Dataset generator. Don't do extra stuff here, like printing,
    because they will be postponed to the first loading time.

    Args:
        corpus_type: 'train' or 'valid'
    Returns:
        A list of dataset, the dataset(s) are lazily loaded.
    """
    assert corpus_type in ["train", "valid", "test"]

    def _lazy_dataset_loader(pt_file, corpus_type):
        dataset = torch.load(pt_file)
        logger.info('Loading %s dataset from %s, number of examples: %d' %
                    (corpus_type, pt_file, len(dataset)))
        return dataset

    # Sort the glob output by file name (by increasing indexes).
    pts = sorted(glob.glob(args.bert_data_path + '.' + corpus_type + '.[0-9]*.pt'))
    if pts:
        if (shuffle):
            random.shuffle(pts)

        for pt in pts:
            yield _lazy_dataset_loader(pt, corpus_type)
    else:
        # Only one inputters.*Dataset, simple!
        pt = args.bert_data_path + '.' + corpus_type + '.pt'
        yield _lazy_dataset_loader(pt, corpus_type)


def simple_batch_size_fn(new, count):
    src, labels = new[0], new[1]
    global max_n_sents, max_n_tokens, max_size
    if count == 1:
        max_size = 0
        max_n_sents=0
        max_n_tokens=0
    max_n_sents = max(max_n_sents, len(src))
    max_size = max(max_size, max_n_sents)
    src_elements = count * max_size
    return src_elements


class Dataloader(object):
    def __init__(self, args, datasets,  batch_size,
                 device, shuffle, is_test):
        self.args = args
        self.datasets = datasets
        self.batch_size = batch_size
        self.device = device
        self.shuffle = shuffle
        self.is_test = is_test
        self.cur_iter = self._next_dataset_iterator(datasets)

        assert self.cur_iter is not None

    def __iter__(self):
        dataset_iter = (d for d in self.datasets)
        while self.cur_iter is not None:
            for batch in self.cur_iter:
                yield batch
            self.cur_iter = self._next_dataset_iterator(dataset_iter)


    def _next_dataset_iterator(self, dataset_iter):
        try:
            # Drop the current dataset for decreasing memory
            if hasattr(self, "cur_dataset"):
                self.cur_dataset = None
                gc.collect()
                del self.cur_dataset
                gc.collect()

            self.cur_dataset = next(dataset_iter)
        except StopIteration:
            return None

        return DataIterator(args = self.args,
            dataset=self.cur_dataset,  batch_size=self.batch_size,
            device=self.device, shuffle=self.shuffle, is_test=self.is_test)


class DataIterator(object):
    def __init__(self, args, dataset,  batch_size,  device=None, is_test=False,
                 shuffle=True):
        self.args = args
        self.batch_size, self.is_test, self.dataset = batch_size, is_test, dataset
        self.iterations = 0
        self.device = device
        self.shuffle = shuffle

        self.sort_key = lambda x: len(x[1])

        self._iterations_this_epoch = 0

    def data(self):
        if self.shuffle:
            random.shuffle(self.dataset)
        xs = self.dataset
        return xs


    def preprocess(self, ex, is_test):
        src = ex['src']
        if('labels' in ex):
            labels = ex['labels']
        else:
            labels = ex['src_sent_labels']

        segs = ex['segs']
        if(not self.args.use_interval):
            segs=[0]*len(segs)
        clss = ex['clss']
        src_txt = ex['src_txt']
        tgt_txt = ex['tgt_txt']

        end_id = [src[-1]]
        src = src[:-1][:512 - 1] + end_id
        segs = segs[:512]
        max_sent_id = bisect.bisect_left(clss, 512)
        labels = labels[:max_sent_id]
        clss = clss[:max_sent_id]

        graph = None
        # if self.args.model=='gcn':
        #     graph=PairData(edge_index_s = torch.tensor(ex['syn']).to(device=self.device), edge_weight_s = torch.tensor(ex['syn_attr']).to(device=self.device), edge_index_t = torch.tensor(ex['seq']), edge_weight_t = torch.tensor(ex['seq_attr']))#.to(device=self.device).to(device=self.device))
        if(is_test):
            return src, labels, segs, clss, graph, src_txt, tgt_txt,
        else:
            return src, labels, segs, clss, graph

    def batch_buffer(self, data, batch_size):
        minibatch, size_so_far = [], 0
        for ex in data:
            if(len(ex['src'])==0):
                continue
            ex = self.preprocess(ex, self.is_test)
            if(ex is None):
                continue
            minibatch.append(ex)
            size_so_far = simple_batch_size_fn(ex, len(minibatch))
            if size_so_far == batch_size:
                yield minibatch
                minibatch, size_so_far = [], 0
            elif size_so_far > batch_size:
                yield minibatch[:-1]
                minibatch, size_so_far = minibatch[-1:], simple_batch_size_fn(ex, 1)
        if minibatch:
            yield minibatch

    def create_batches(self):
        """ Create batches """
        data = self.data()
        for buffer in self.batch_buffer(data, self.batch_size * 50):

            p_batch = sorted(buffer, key=lambda x: len(x[3]))
            p_batch = batch(p_batch, self.batch_size)

            p_batch = list(p_batch)
            if (self.shuffle):
                random.shuffle(p_batch)
            for b in p_batch:
                yield b

    def __iter__(self):
        while True:
            self.batches = self.create_batches()
            for idx, minibatch in enumerate(self.batches):
                # fast-forward if loaded from state
                if self._iterations_this_epoch > idx:
                    continue
                self.iterations += 1
                self._iterations_this_epoch += 1
                batch = Batch(minibatch, self.device, self.is_test)
                yield batch
            return
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("-encoder", default='classifier', type=str)
    parser.add_argument("-bert_data_path", default='/home/gwy/ALL/gcn_data/cnndm')
    parser.add_argument("-hidden_dim", default=126, type=int)
    parser.add_argument("-output_channel", default=126, type=int)
    parser.add_argument("-param_init", default=0, type=int)

    args = parser.parse_args()
    # tg=load_dataset(args,'train',False)
    args.bert_data_path='/home/gwy/ALL/BertSum/bert_data/cnndm'
    t=load_dataset(args,'train',False)
    tgs={}
    tgsg={}
    positive=0
    total=0
    negtive0=0
    negtive1=0
    negtive2=0
    equal=0
    for ids,exs in enumerate(t):
        for ex in exs:
            negtive2+=(sum(ex['labels'])==2)
            negtive1+=(sum(ex['labels'])==1)
            negtive0+=(sum(ex['labels'])==0)
            positive+=(sum(ex['labels'])>3)
            equal+=(sum(ex['labels'])==3)
            total+=1
            print(len(ex['labels']),len(ex['clss']))
            # if sum(ex['labels'])==0:
            #     print('#'*100)
            #     print(ex['tgt_txt'])
            #     print(ex['labels'])

    print('positive',positive,'total',total,'negtive1',negtive1,'negtive2',negtive2,'negtive0',negtive0,'equal',equal)
#     # for ids, exs in enumerate(tg):
#     #     tgsg[ids] = len(exs)
#
#     # for k in tgsg.keys():
#     #     if tgs[k]!=tgsg[k]:
#     #         print('this is an error')
#  positive 0 total 287084 negtive1 56687 negtive2 117908 negtive0 9064 equal 103425  train
# (base) gwy@cherry:~$ positive 0 total 11489 negtive1 1819 negtive2 4591 negtive0 198 equal 4881  test
# (base) gwy@cherry:~$ positive 0 total 13367 negtive1 1976 negtive2 5247 negtive0 243 equal 5901 valid
