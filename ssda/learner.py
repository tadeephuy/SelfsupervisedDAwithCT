import numpy as np
import torch
from torch import nn
from torchvision.transforms import functional as tF
from loss import KLD, entropy_minimization
from recorder import Recorder
from utils import aggregate_logs, split_src_tgt
from fastprogress import progress_bar, master_bar

"""
! TODO: 
3. tensorboard visualizer
"""

def auxs(img):
    aux_labels = [90, 45, 0, -45, -90]
    aux_label = torch.randint(0, len(aux_labels), [1])
    angle = aux_labels[aux_label.item()]
    aux_img = tF.rotate(img, angle=angle)
    return aux_img, aux_label

kld_loss = KLD(reduction='batchmean')
class_loss = nn.BCEWithLogitsLoss()
aux_class_loss = nn.CrossEntropyLoss()
ent_loss = entropy_minimization

W = {
    'w_src_aux': 0.5,
    'w_src_kld': 0.5,
    'w_tgt_aux': 0.5,
    'w_tgt_kld': 0.5,
    'w_tgt_ent': 0.1,
}


class Learner:
    def __init__(self, model, dataloaders, name='model', W=W):
        self.model = model
        self.dataloaders = dataloaders
        self.name = name
        self.W = W

        self.optimizer = None
        self.scheduler = None

        self.recorder = Recorder()
        self.bar = None

    @torch.no_grad()
    def validate(self, dataloader):
        self.model.eval()
        valid_logs = []
        
        bar = progress_bar(dataloader, parent=self.bar)
        for src, tgt in bar:
            batch_logs = self.feed_one_batch(src, tgt, mode='valid')
            valid_logs.append(batch_logs)
        valid_logs = aggregate_logs(valid_logs)
        return valid_logs

    def save(self, name=None, mode=None):
        name = f'{self.name}.pth' if name is None else name
        if mode is None:
            name = name
        elif mode=='finish':
            name = 'finish_' + name
        else:
            name = name
        torch.save(self.model.state_dict(), name)

    def load(self, path, strict=True):
        state_dict = torch.load(path)
        self.model.load_state_dict(state_dict, strict=strict)

    def feed_one_batch(self, src, tgt, mode='train'):
        if mode=='train':
            self.optimizer.zero_grad()

        x_src, y_src, xa_src, a_src = src['x'], src['y'], src['xa'], src['a']
        x_tgt, xa_tgt, a_tgt = tgt['x'], tgt['xa'], tgt['a']

        # classification logits of the original versions
        inputs = torch.cat([x_src, x_tgt]).cuda()
        cls_logits_org = self.model(inputs)
        p_src, p_tgt = split_src_tgt(cls_logits_org)

        # classification and auxiliary logits of the augmented version
        inputs = torch.cat([xa_src, xa_tgt]).cuda()
        cls_logits_aug, aux_logits_aug = self.model(inputs, mode='aux')
        pa_src, pa_tgt = split_src_tgt(cls_logits_aug)
        aux_pred_src, aux_pred_tgt = split_src_tgt(aux_logits_aug)
        
        # label to CUDA
        y_src = y_src.cuda()
        a_src = a_src[:,0].cuda()
        a_tgt = a_tgt[:,0].cuda()

        # SOURCE
        src_class_loss = class_loss(p_src, y_src) # classification loss for main task
        src_aux_loss = aux_class_loss(aux_pred_src, a_src) # classification loss for auxiliary task
        src_kld_loss = kld_loss(p_src, pa_src) # consitency loss

        src_loss = src_class_loss + self.W['w_src_aux']*src_aux_loss + self.W['w_src_kld']*src_kld_loss

        # TARGET
        tgt_aux_loss = aux_class_loss(aux_pred_tgt, a_tgt) # classification loss for auxiliary task
        tgt_kld_loss = kld_loss(p_tgt, pa_tgt) # consitency loss
        tgt_ent_loss = ent_loss(p_tgt) # entropy minimization loss to sharpen prediction of the pseudo-label

        tgt_loss = self.W['w_tgt_aux']*tgt_aux_loss + self.W['w_tgt_kld']*tgt_kld_loss + self.W['w_tgt_ent']*tgt_ent_loss

        loss = src_loss + tgt_loss

        if mode=='train':
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

        batch_logs = {
            'losses': {
                'loss': loss.item(),
                'src_loss': src_loss.item(),
                'src_class_loss': src_class_loss.item(),
                'src_aux_loss': src_aux_loss.item(),
                'src_kld_loss': src_kld_loss.item(),
                'tgt_loss': tgt_loss.item(),
                'tgt_aux_loss': tgt_aux_loss.item(),
                'tgt_kld_loss': tgt_kld_loss.item(),
                'tgt_ent_loss': tgt_ent_loss.item(),
            },
            'preds': {
                'p_src': p_src.detach().cpu().numpy(),
                'aux_pred_src': np.argmax(aux_pred_src.detach().cpu().numpy(), axis=1),
                'aux_pred_tgt': np.argmax(aux_pred_tgt.detach().cpu().numpy(), axis=1),
            },
            'labels': {
                'y_src': y_src.detach().cpu().numpy(),
                'a_src': a_src.detach().cpu().numpy(),
                'a_tgt': a_tgt.detach().cpu().numpy(),
            }
        }

        del loss, src_loss, tgt_loss
        del tgt_ent_loss, tgt_kld_loss, tgt_aux_loss, src_kld_loss, src_aux_loss, src_class_loss
        del p_src, p_tgt, pa_src, pa_tgt, aux_pred_src, aux_pred_tgt

        return batch_logs

    def fit(self, n_epochs, optimizer, scheduler):
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.bar = master_bar(range(n_epochs))
        train_loader, valid_loader = self.dataloaders
        for epoch in self.bar:
            ## train
            self.model.train()
            for src, tgt in progress_bar(train_loader, parent=self.bar):
                batch_logs = self.feed_one_batch(src,tgt,mode='train')
                self.recorder.update(batch_logs, 'train_batch')
            self.save(mode='train_epoch')

            ## valid
            self.model.eval()
            valid_logs = self.validate(valid_loader)
            self.recorder.update(valid_logs, 'valid_epoch')
            self.save(mode='valid_epoch')

            ## visualize
            self.recorder.visualize()
            self.recorder.log(epoch, 'valid_epoch')

        self.save(mode='finish')


