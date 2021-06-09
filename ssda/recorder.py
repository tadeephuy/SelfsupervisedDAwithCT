from collections import defaultdict
from copy import deepcopy
import pandas as pd

class Recorder:
    def __init__(self, tags=['train_batch', 'train_epoch', 'valid_epoch'], name='record'):
        self.tags = tags
        self.logs = {k: defaultdict(lambda: defaultdict(lambda: AverageMeter())) for k in self.tags}
        self.name = name
        self.logs_df = None

    def update(self, log, tag):
        
        for k, v in log.items():
            if k == 'losses':
                for name in v.keys():
                    self.logs[tag][k][name].update(log[k][name]) # loss is running average
            elif k == 'metrics':
                for name in v.keys():
                    self.logs[tag][k][name] = log[k][name] # metrics is updated by replacement

    def visualize(self, *args):
        pass        

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