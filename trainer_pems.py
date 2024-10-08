from LibMTL import Trainer
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from LibMTL._record import _PerformanceMeter
from LibMTL.utils import count_parameters


class Trainer_pems(Trainer):
    def __init__(self, task_dict, weighting, architecture, encoder_class, decoders,
                 rep_grad, multi_input, optim_param, scheduler_param, save_path, **kwargs):
        super().__init__(task_dict, weighting, architecture, encoder_class, decoders,
                 rep_grad, multi_input, optim_param, scheduler_param, save_path, **kwargs)


    def _process_data(self, loader):
        try:
            data, label = next(loader[1])    # loader[1].next()
        except:
            loader[1] = iter(loader[0])
            data, label = next(loader[1])   # loader[1].next()

        if not self.multi_input:
            for task in self.task_name:
                data[task] = data[task].to(self.device, non_blocking=True)
                label[task] = label[task].to(self.device, non_blocking=True)
        else:
            label = label.to(self.device, non_blocking=True)
        return data, label

    def train(self, train_dataloaders, test_dataloaders, epochs,
              val_dataloaders=None, return_weight=False):
        r'''The training process of multi-task learning.

        Args:
            train_dataloaders (dict or torch.utils.data.DataLoader): The dataloaders used for training. \
                            If ``multi_input`` is ``True``, it is a dictionary of name-dataloader pairs. \
                            Otherwise, it is a single dataloader which returns data and a dictionary \
                            of name-label pairs in each iteration.

            test_dataloaders (dict or torch.utils.data.DataLoader): The dataloaders used for the validation or testing. \
                            The same structure with ``train_dataloaders``.
            epochs (int): The total training epochs.
            return_weight (bool): if ``True``, the loss weights will be returned.
        '''
        train_loader, train_batch = self._prepare_dataloaders(train_dataloaders)
        train_batch = max(train_batch) if self.multi_input else train_batch

        self.batch_weight = np.zeros([self.task_num, epochs, train_batch])
        self.model.train_loss_buffer = np.zeros([self.task_num, epochs])
        self.model.epochs = epochs
        early_stop = 0
        for epoch in range(epochs):
            self.model.epoch = epoch
            self.model.train()
            self.meter.record_time('begin')
            for batch_index in range(train_batch):
                if not self.multi_input:
                    train_inputs, train_gts = self._process_data(train_loader)
                    train_preds = self.model(train_inputs)
                    train_preds = self.process_preds(train_preds)
                    train_losses = self._compute_loss(train_preds, train_gts)
                    self.meter.update(train_preds, train_gts)
                else:
                    train_losses = torch.zeros(self.task_num).to(self.device)
                    for tn, task in enumerate(self.task_name):
                        train_input, train_gt = self._process_data(train_loader[task])
                        train_pred = self.model(train_input, task)
                        train_pred = train_pred[task]
                        train_pred = self.process_preds(train_pred, task)
                        train_losses[tn] = self._compute_loss(train_pred, train_gt, task)
                        self.meter.update(train_pred, train_gt, task)

                self.optimizer.zero_grad()
                w = self.model.backward(train_losses, **self.kwargs['weight_args'])
                if w is not None:
                    self.batch_weight[:, epoch, batch_index] = w
                self.optimizer.step()

            self.meter.record_time('end')
            self.meter.get_score()
            self.model.train_loss_buffer[:, epoch] = self.meter.loss_item
            self.meter.display(epoch=epoch, mode='train')
            self.meter.reinit()
            early_stop += 1
            if val_dataloaders is not None:
                self.meter.has_val = True
                val_improvement = self.test(val_dataloaders, epoch, mode='val', return_improvement=True)
            self.test(test_dataloaders, epoch, mode='test')
            if self.scheduler is not None:
                if self.scheduler_param['scheduler'] == 'reduce' and val_dataloaders is not None:
                    self.scheduler.step(val_improvement)
                else:
                    self.scheduler.step()
            if self.save_path is not None and self.meter.best_result['epoch'] == epoch:
                torch.save(self.model.state_dict(), os.path.join(self.save_path, 'best.pt'))
                print('Save Model {} to {}'.format(epoch, os.path.join(self.save_path, 'best.pt')))
                early_stop = 0
            # if early_stop >= 5 or epoch == (epochs-1):
            #     np.save(os.path.join(self.save_path, 'loss.npz'), self.model.train_loss_buffer)
            #     print('Save Loss {} to {}'.format(epoch, os.path.join(self.save_path, 'loss.npz')))
            #     break
        self.meter.display_best_result()
        if return_weight:
            return self.batch_weight