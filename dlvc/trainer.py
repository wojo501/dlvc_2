import collections
import torch
from typing import Tuple
from abc import ABCMeta, abstractmethod
from pathlib import Path
from tqdm import tqdm

from dlvc.wandb_logger import WandBLogger
from dlvc.dataset.oxfordpets import OxfordPetsCustom

class BaseTrainer(metaclass=ABCMeta):
    '''
    Base class of all Trainers.
    '''

    @abstractmethod
    def train(self) -> None:
        '''
        Returns the number of samples in the dataset.
        '''

        pass

    @abstractmethod
    def _val_epoch(self) -> Tuple[float, float]:
        '''
        Returns the number of samples in the dataset.
        '''

        pass

    @abstractmethod
    def _train_epoch(self) -> Tuple[float, float]:
        '''
        Returns the number of samples in the dataset.
        '''

        pass

class ImgSemSegTrainer(BaseTrainer):
    """
    Class that stores the logic for training a model for image classification.
    """
    def __init__(self, 
                 model, 
                 optimizer,
                 loss_fn,
                 lr_scheduler,
                 train_metric,
                 val_metric,
                 train_data,
                 val_data,
                 device,
                 num_epochs: int, 
                 training_save_dir: Path,
                 batch_size: int = 4,
                 val_frequency: int = 5):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.lr_scheduler = lr_scheduler

        self.device = device

        self.num_epochs = num_epochs
        self.train_metric = train_metric

        self.val_frequency = val_frequency
        self.val_metric = val_metric

        self.subtract_one = isinstance(train_data, OxfordPetsCustom)
        
        self.train_data_loader = torch.utils.data.DataLoader(train_data,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          num_workers=4)
    
        self.val_data_loader = torch.utils.data.DataLoader(val_data,
                                          batch_size=batch_size,
                                          shuffle=False,
                                          num_workers=4)
        self.num_train_data = len(train_data)
        self.num_val_data = len(val_data)

        self.checkpoint_dir = training_save_dir
        self.wandb_logger = WandBLogger(enabled=True, model=self.model, run_name=model.net._get_name())
        
    def _train_epoch(self, epoch_idx: int) -> Tuple[float, float]:
        self.model.train()
        epoch_loss = 0.
        self.train_metric.reset()
        
        # train epoch
        for i, batch in tqdm(enumerate(self.train_data_loader), desc="train", total=len(self.train_data_loader)):
            self.optimizer.zero_grad()

            inputs, labels = batch
            
            labels = labels.squeeze(1) #error might be here
            if self.subtract_one: 
                labels = labels - 1

            batch_size = inputs.shape[0]

            outputs = self.model(inputs.to(self.device))
            if isinstance(outputs, collections.OrderedDict):
                outputs = outputs['out']

            loss = self.loss_fn(outputs, labels.to(self.device))
            loss.backward()

            self.optimizer.step()

            epoch_loss += (loss.item() * batch_size)
            self.train_metric.update(outputs.detach().cpu(), labels.detach().cpu())

        self.lr_scheduler.step()
        epoch_loss /= self.num_train_data
        epoch_mIoU = self.train_metric.mIoU()

        print(f"______epoch {epoch_idx} \n")
        print(f"Loss: {epoch_loss}")
        print(self.train_metric)

        return epoch_loss, epoch_mIoU

    def _val_epoch(self, epoch_idx: int) -> Tuple[float, float]:
        self.val_metric.reset()
        epoch_loss = 0.
        for batch_idx, batch in tqdm(enumerate(self.val_data_loader), desc="eval", total=len(self.val_data_loader)):
            self.model.eval()
            with torch.no_grad():
                inputs, labels = batch
                labels = labels.squeeze(1) #error might be here
                if self.subtract_one:
                    labels = labels - 1
                batch_size = inputs.shape[0]

                outputs = self.model(inputs.to(self.device))
                if isinstance(outputs, collections.OrderedDict):
                    outputs = outputs['out']

                loss = self.loss_fn(outputs, labels.to(self.device))
                epoch_loss += (loss.item() * batch_size)
                self.val_metric.update(outputs.cpu(), labels.cpu())

        epoch_loss /= self.num_val_data
        epoch_mIoU = self.val_metric.mIoU()
        
        if epoch_mIoU is None:
            epoch_mIoU = 0.0  # Ensure epoch_mIoU is not None
        
        print(f"______epoch {epoch_idx} - validation \n")
        print(f"Loss: {epoch_loss}")
        print(self.val_metric)

        return epoch_loss, epoch_mIoU

    def train(self) -> None:
        best_mIoU = 0.
        for epoch_idx in range(self.num_epochs):
            train_loss, train_mIoU = self._train_epoch(epoch_idx)

            wandb_log = {'epoch': epoch_idx}

            wandb_log.update({"train/loss": train_loss})
            wandb_log.update({"train/mIoU": train_mIoU})

            if epoch_idx % self.val_frequency == 0:
                val_loss, val_mIoU = self._val_epoch(epoch_idx)
                wandb_log.update({"val/loss": val_loss})
                wandb_log.update({"val/mIoU": val_mIoU})

                if val_mIoU is None:
                    val_mIoU = 0.0  # Ensure val_mIoU is not None

                if best_mIoU <= val_mIoU:
                    print(f"####best mIoU: {val_mIoU}")
                    print(f"####saving model to {self.checkpoint_dir}")
                    self.model.save(Path(self.checkpoint_dir), suffix="best")
                    best_mIoU = val_mIoU
                if epoch_idx == self.num_epochs - 1:
                    self.model.save(Path(self.checkpoint_dir), suffix="last")

            self.wandb_logger.log(wandb_log)

    def dispose(self) -> None:
        self.wandb_logger.finish()
