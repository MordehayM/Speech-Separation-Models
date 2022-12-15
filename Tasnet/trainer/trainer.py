import numpy as np
import torch
from torchvision.utils import make_grid
from base import BaseTrainer
from utils import inf_loop, MetricTracker
from torch import autograd


class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, criterion, metric_ftns, optimizer, config, device,
                 data_loader, valid_data_loader=None, lr_scheduler=None, len_epoch=None):
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        self.config = config
        cfg_trainer = config['trainer']
        #self.max_clip = cfg_trainer['max_clip']
        self.device = device
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))

        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns["separation"]],
                                            *[m.__name__ for m in self.metric_ftns["separation_mix"]], writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns["separation"]],
                                          *[m.__name__ for m in self.metric_ftns["separation_mix"]], writer=self.writer)
        print(self.max_clip)
    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        #with autograd.detect_anomaly():
        def nan_hook(self, inp, output):
            if not isinstance(inp, tuple):
                inps = [inp]
            else:
                inps = inp
            for i, inp1 in enumerate(inps):
                nan_mask = torch.isnan(inp1)
                if nan_mask.any():
                    print("there is nan")
            if not isinstance(output, tuple):
                outputs = [output]
            else:
                outputs = output

            for i, out in enumerate(outputs):
                nan_mask = torch.isnan(out)
                if nan_mask.any():
                    print("In", self.__class__.__name__)
                    print(out.shape)
                    print(torch.var(out[0], unbiased=False))
                    print(nan_mask.nonzero().shape)
                    print(nan_mask.shape)
                    raise RuntimeError(f"Found NAN in output {i} at indices: ", nan_mask.nonzero(), "where:",
                                       out[nan_mask.nonzero()[:, 0].unique(sorted=True)])

        #for submodule in self.model.modules():
        #    submodule.register_forward_hook(nan_hook)
        for batch_idx, sample in enumerate(self.data_loader):
           
            data = sample['mixed_signals']
            #print(data.shape)
            # print(data.get_device())
            # print(list(data.size()))
            # data=data[:, 1, :]
            # print(list(data.size()))
            target = sample['clean_speeches']
            #doa = sample["doa"]
            #df = sample["df"]
            #df = df.float()
            data, target = data.to(self.device), target.to(self.device)
            # print(data.get_device())
            #print(f"target shape is: {target.shape}")
            self.optimizer.zero_grad()
            output = self.model(data)
            #reduce_kwargs = {'src': target}
            #loss = self.criterion(output, target, reduce_kwargs=reduce_kwargs)
            loss = self.criterion(output, target)
            #print(loss)
            loss.backward() 
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.max_clip, norm_type=2)
            self.optimizer.step()
            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss.item())
            """for met in self.metric_ftns:
                met = met.to(self.device)
                self.train_metrics.update(met.__name__, met(output, target).item())"""
            
            for met in self.metric_ftns["separation"]:
                    met = met.to(self.device)
                    self.train_metrics.update(met.__name__, met(output, target).item())
             
            for met in self.metric_ftns["separation_mix"]:
                    met = met.to(self.device)
                    self.train_metrics.update(met.__name__, met(output, target, data).item())
            #print(self.train_metrics.result())
            
            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item()))
                self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

            if batch_idx == self.len_epoch:
                break
        log = self.train_metrics.result()
        #print(log)
        if self.do_validation:
            val_log, val_loss = self._valid_epoch(epoch)
            log.update(**{'val_'+k : v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step(log["val_loss"])
            self.logger.info(f"The lr is: {self.lr_scheduler._last_lr[0]:.05f}")
        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            for batch_idx, sample in enumerate(self.valid_data_loader):
                data = sample['mixed_signals']
                # data = data[:, 1, :]
                target = sample['clean_speeches']
                #doa = sample["doa"]
                #df = sample["df"]
                #df = df.float()
                data, target = data.to(self.device), target.to(self.device)
                #data, target, doa = data.to(self.device), target.to(self.device), doa.to(self.device)

                output = self.model(data)
                reduce_kwargs = {'src': target}
                # loss = self.criterion(output, target, reduce_kwargs=reduce_kwargs)
                loss = torch.mean(self.criterion(output, target))

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.valid_metrics.update('loss', loss.item())
                """for met in self.metric_ftns:
                    met = met.to(self.device)
                    self.valid_metrics.update(met.__name__, met(output, target))"""
                    
                for met in self.metric_ftns["separation"]:
                    met = met.to(self.device)
                    self.valid_metrics.update(met.__name__, met(output, target).item())
             
                for met in self.metric_ftns["separation_mix"]:
                        met = met.to(self.device)
                        self.valid_metrics.update(met.__name__, met(output, target, data).item())
                        
                self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')
        return self.valid_metrics.result(), loss

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
