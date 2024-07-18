"""Trainer for training and generation with diffusion models."""

import os
import torch
import numpy as np
import pandas as pd
from .utils import XYCTabDataModule
from .unet import Unet
from .configs import DenoiseFnCfg, DataCfg, GuidCfg
from .estimator import PosteriorEstimator, DenoiseFn
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
        """Init.
        
        Args:
            n_epochs: number of epochs.
            lr: learning rate.
            weight_decay: weight decay.
            max_non_improve: maximum number of epochs to keep training if no improvement.
            is_fair: whether to use fair diffusion.
            device: device.
        """
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
        """Update args.
        
        Args:
            n_epochs: number of epochs.
            lr: learning rate.
            weight_decay: weight decay.
            max_non_improve: maximum number of epochs to keep training if no improvement.
            is_fair: whether to use fair diffusion.
            device: device.
        """
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
        """Prepare model.
        
        Args:
            model: diffusion model.
        """
        self.model = model
        self.model.to(self.device)

    def prepare_data(self, data: XYCTabDataModule):
        """Prepare data.
        
        Args:
            data: data module that has `get_train_loader`, `get_val_loader`, and `get_test_loader` methods and more.
        """
        self.data = data
        self.train_loader = data.get_train_loader()
        self.val_loader = data.get_val_loader()
        self.test_loader = data.get_test_loader()
        
    def fit(self, model: GaussianMultinomialDiffusion, data: XYCTabDataModule, exp_dir: str = None):
        """Fit.
        
        Args:
            model: diffusion model.
            data: data module that has `get_train_loader`, `get_val_loader`, and `get_test_loader` methods and more.
            exp_dir: directory to save model.
        """
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
        """Save model.
        
        Args:
            path: path to save model.
        """
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


def _test() -> None:
    # configs
    d_oh_x = 24
    n_channels = 1
    d_x_emb = 16
    d_t_emb = 16
    d_cond_emb = 16
    n_base_channels = 32
    n_groups = 1
    data_dir = '/home/tom/github/comp-fair/db-seq/unimodal/depression/'
    batch_size = 32
    device = 'cpu'
    
    # data
    data = XYCTabDataModule(data_dir, batch_size)
    
    # denoising function
    denoise_fn = DenoiseFn(
        denoise_fn_cfg=DenoiseFnCfg(
            d_x_emb=d_x_emb,
            d_t_emb=d_t_emb,
            d_cond_emb=d_cond_emb,
        ),
        data_cfg=DataCfg(
            d_oh_x=d_oh_x,
            n_channels=n_channels,
            n_unq_c_lst=[2, 2, 4],
        ),
        guid_cfg=GuidCfg(
            cond_guid_weight=0.5,
            cond_guid_threshold=1.0,
            cond_momentum_weight=1.0,
            cond_momentum_beta=0.2,
            warmup_steps=10,
            overall_guid_weight=1.0,
        ),
        posterior_est=PosteriorEstimator(
            Unet(
                n_in_channels=n_channels,
                n_out_channels=n_channels,
                n_base_channels=n_base_channels,
                n_channels_factors=[2, 2, 2],
                n_res_blocks=1,
                attention_levels=[0],
                d_t_emb=d_t_emb,
                d_cond_emb=d_cond_emb * 4,
                n_groups=n_groups,
            ),
        ),
    )
    
    # diffusion model
    diffusion = GaussianMultinomialDiffusion(
        num_classes=np.array([2, 4, 2, 2]),
        num_numerical_features=14,
        denoise_fn=denoise_fn,
        device=device,
    )
    diffusion.to(device)
    
    # trainer
    lr = 1e-3
    weight_decay = 1e-4
    trainer = XYCTabTrainer(
        n_epochs=100,
        lr=lr,
        weight_decay=weight_decay,
        is_fair=True,
        device=device,
    )
    trainer.fit(diffusion, data)


if __name__ == '__main__':
    _test()
