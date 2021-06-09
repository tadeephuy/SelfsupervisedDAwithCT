from collections import defaultdict

class Recorder:
    def __init__(self, tags=['train_batch', 'train_epoch', 'valid_epoch']):
        self.tags = tags
        self.logs = {k: defaultdict(lambda: defaultdict(lambda: AverageMeter())) for k in self.tags}

    def update(self, log, tag):
        for k, v in self.logs[tag].items():
            if k == 'losses':
                for name in v.keys():
                    v[name].update(log[k][name]) # loss is running average
            elif k == 'metrics':
                for name in v.keys():
                    v[name] = log[k][name] # metrics is updated by replacement

    def visualize(self, *args):
        pass        

    def log(self, idx, tag):
        print(idx)
        print(self.logs[tag])


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