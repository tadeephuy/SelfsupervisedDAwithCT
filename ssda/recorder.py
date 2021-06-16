from collections import defaultdict
from copy import deepcopy
import pandas as pd
import os
import tensorflow as tf
import datetime

class Recorder:
    def __init__(self, tags=['train_batch', 'train_epoch', 'valid_epoch'], name='record'):
        self.tags = tags
        self.logs = {k: defaultdict(lambda: defaultdict(lambda: AverageMeter())) for k in self.tags}
        self.name = name
        self.logs_df = 'logs'
        os.makedirs(self.logs_df, exist_ok=True)
        current = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.tbwriter = {tag: tf.summary.create_file_writer('./logs/' + current + '/' + tag) for tag in tags}
        self.index = {}

    def update(self, log, tag):
        
        for k, v in log.items():
            if k == 'losses':
                for name in v.keys():
                    self.logs[tag][k][name].update(log[k][name]) # loss is running average
            elif k == 'metrics':
                for name in v.keys():
                    self.logs[tag][k][name] = log[k][name] # metrics is updated by replacement

    def visualize(self, tag, *args):
        def calc_index(self, graph_name):
            if graph_name not in self.index.keys():
                self.index[graph_name] = 0
            else:
                self.index[graph_name] += 1
            return self.index[graph_name]

        with self.tbwriter[tag].as_default():
            if tag in ['train_batch']:
                for type_ in ['losses', 'metrics']:
                    for name, value in self.logs[tag][type_].items():  # metric name
                        graph_name = '{}/{}/{}'.format(tag, type_, name)
                        tf.summary.scalar(graph_name, value.avg, step=calc_index(self, graph_name))
            elif tag in ['train_epoch', 'valid_epoch']:
                for type_ in ['losses', 'metrics']:
                    graph_name = '{}/{}'.format(tag, type_)
                    try:
                        ### type(metrics) is dictionary
                        for name, value in self.logs[tag][type_].items():
                            graph_name = '{}/{}/{}'.format(tag, type_, name)
                            tf.summary.scalar(graph_name, value.avg, step=calc_index(self, graph_name))
                    except:
                        ### type(losses) is numeric
                        tf.summary.scalar(graph_name, self.logs[tag][type_].avg, step=calc_index(self, graph_name))

    def log(self, idx, tag):
        x = deepcopy(self.logs[tag])
        if 'metrics' in x:
            x['losses'].update(x['metrics'])
        x = pd.DataFrame(x['losses'], index=[idx])
        if self.logs_df is None:
            self.logs_df = x
        else:
            self.logs_df = pd.concat([self.logs_df, x])
        self.logs_df.to_csv(f'{self.name}.csv')
        display(x)
        


class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
    
    def __repr__(self):
        return str(self.val)

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count