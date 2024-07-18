"""Trainer for training and generation with diffusion models."""

import os
import torch
import numpy as np
import pandas as pd
from .utils import XYCTabDataModule
from .ddpm import GaussianMultinomialDiffusion

class XYCTabTrainer:
    def __init__(
        self,
        n_epochs: int = 100,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        max_non_improve: int = 10,
        is_fair: bool = False,
        device: str = 'cpu',
    ) -> None:
        self.n_epochs = n_epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.max_non_improve = max_non_improve
        self.is_fair = is_fair
        self.device = torch.device(device)
    
    def updata_args(
        self, 
        n_epochs: int = None, 
        lr: float = None, 
        weight_decay: float = None, 
        max_non_improve: int = None,
        is_fair: bool = None,
        device: str = None,
    ) -> None:
        if n_epochs is not None:
            self.n_epochs = n_epochs
        if lr is not None:
            self.lr = lr
        if weight_decay is not None:
            self.weight_decay = weight_decay
        if max_non_improve is not None:
            self.max_non_improve = max_non_improve
        if is_fair is not None:
            self.is_fair = is_fair
        if device is not None:
            self.device = torch.device(device)

    def prepare_model(self, model: GaussianMultinomialDiffusion):
        self.model = model
        self.model.to(self.device)

    def prepare_data(self, data: XYCTabDataModule, normalize: bool = True):
        self.data = data
        self.train_loader = data.get_dataloader('train', normalize=normalize)
        self.eval_loader = data.get_dataloader('eval', normalize=normalize)
        self.test_loader = data.get_dataloader('test', normalize=normalize)
        
    def fit(self, model: GaussianMultinomialDiffusion, data: XYCTabDataModule, exp_dir: str = None):
        # prepare
        self.prepare_model(model)
        self.prepare_data(data)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=self.lr, 
            weight_decay=self.weight_decay,
        )
        
        # record
        self.state = {'epoch': 0}
        self.loss_record = {'mloss': np.inf, 'gloss': np.inf, 'tloss': np.inf, 'keeps': 0, 'break': False, 'epoch': 0}
        self.epoch_loss_history = pd.DataFrame(columns=['epoch', 'mloss', 'gloss', 'tloss'])
        
        # train
        t_loss_min = np.inf
        for epoch in range(self.n_epochs):
            self.state['epoch'] = epoch + 1
            mloss, gloss, tloss = self._fit_epoch(self.model, self.train_loader)
            self.loss_record['keeps'] += 1
            if tloss < t_loss_min:
                self.loss_record['mloss'] = mloss
                self.loss_record['gloss'] = gloss
                self.loss_record['tloss'] = tloss
                self.loss_record['keeps'] = 0
                self.loss_record['epoch'] = epoch + 1
                t_loss_min = tloss
                if exp_dir is not None:
                    self.save_model(os.path.join(exp_dir, 'diffusion.pt'))
            if self.loss_record['break']:
                break
            curr_idx = len(self.epoch_loss_history)
            self.epoch_loss_history.loc[curr_idx] = [epoch + 1, mloss, gloss, tloss]
            self.epoch_loss_history.to_csv(os.path.join(exp_dir, 'loss.csv'), index=False)
            self._anneal_lr(epoch)
        
        print()
        print('training complete ^_^')
    
    def save_model(self, path: str) -> None:
        torch.save(self.model.state_dict(), path)
    
    def _fit_epoch(self, model: callable, data_loader: torch.utils.data.DataLoader) -> tuple:
        total_mloss = 0
        total_gloss = 0
        curr_count = 0
        for batch_idx, (x, y) in enumerate(data_loader):
            self.optimizer.zero_grad()
            if self.is_fair:
                x, y = x.to(self.device), y.to(self.device)
                loss_multi, loss_gauss = model.mixed_loss(x, y)
            else:
                y = y[:, 0].unsqueeze(1)  # only use the first column which is the label
                x, y = x.to(self.device), y.to(self.device)
                loss_multi, loss_gauss = model.mixed_loss(x, y)
            loss = loss_multi + loss_gauss
            loss.backward()
            self.optimizer.step()
            
            # record loss
            with torch.no_grad():
                total_mloss += loss_multi.item() * x.shape[0]
                total_gloss += loss_gauss.item() * x.shape[0]
                curr_count += x.shape[0]
                mloss = np.around(total_mloss / curr_count, 4)
                gloss = np.around(total_gloss / curr_count, 4)
                tloss = np.around(mloss + gloss, 4)
            
            keeps = self.loss_record['keeps']
            curr_epoch = self.state['epoch']
            upper_limit = self.max_non_improve
            if keeps > upper_limit:
                msg = f'best results so far -> mloss: {mloss:.4f}, gloss: {gloss:.4f}, tloss: {tloss:.4f} -- record: {self.loss_record["tloss"]:.4f}, '
                msg += f'keeps: {keeps:04}, epoch: {curr_epoch:04} / {self.n_epochs:04}'
                print(msg)
                self.loss_record['break'] = True
                print(f'the model has not improved for {self.max_non_improve} epochs, stopping training')
                break
            
            if batch_idx == len(data_loader) - 1 and curr_epoch == self.n_epochs:
                msg = f'best results so far -> mloss: {mloss:.4f}, gloss: {gloss:.4f}, tloss: {tloss:.4f} -- record: {self.loss_record["tloss"]:.4f}, '
                msg += f'keeps: {keeps:04}, epoch: {curr_epoch:04} / {self.n_epochs:04}'
                print(msg)
            else:
                msg = f'best results so far -> mloss: {mloss:.4f}, gloss: {gloss:.4f}, tloss: {tloss:.4f} -- record: {self.loss_record["tloss"]:.4f}, '
                msg += f'keeps: {keeps:04}, epoch: {curr_epoch:04} / {self.n_epochs:04}'
                print(msg, end='\r')
        return mloss, gloss, tloss
        
    def _anneal_lr(self, epoch):
        frac_done = epoch / self.n_epochs
        lr = self.lr * (1 - frac_done / 2)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
