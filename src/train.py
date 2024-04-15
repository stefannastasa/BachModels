from torch.utils.data import DataLoader
from torch.optim import Optimizer
import wandb
from tqdm import tqdm

import math
from torch.utils.data import DataLoader
from src.IAMDataset import IAMDataset
from src.model import FullPageHTR
import gc
import torch
from copy import copy
import wandb


class ModelTrainer:

    def __init__(self, run_name: str,
                 model: FullPageHTR,
                 ds_name: str,
                 train_data: DataLoader,
                 val_data: DataLoader,
                 optimizer: Optimizer,
                 num_epochs: int,
                 device: torch.device,
                 normalization_steps: int):

        self.normalization_steps = normalization_steps

        self.model = model
        self.train_data, self.val_data = train_data, val_data
        self.num_epochs = num_epochs
        self.optimizer = optimizer
        self.ds_name = ds_name
        self.run_name = run_name
        self.device = device

    def _init_wandb(self):

        wandb.init(project="fullpage-htr-base",
                   config={
                       "run_name": self.run_name,
                       "learning_rate": self.optimizer.param_groups[0]["lr"],
                       "epochs": self.num_epochs,
                       "dataset": self.ds_name
                   })
        wandb.define_metric('Train')
        wandb.define_metric('Val')

    def train_epoch_ga(self, epoch_nr, ds_size):
      self.model.train()
      total_loss = 0.0
      total_cer = 0.0
      total_wer = 0.0

      nr_batches = 0
      b_loss = 0.0
      b_cer = 0.0
      b_wer = 0.0
      for idx, batch in enumerate(tqdm(self.train_data)):
        inputs, labels = batch
        inputs = inputs.to(self.device)
        labels = labels.to(self.device)

        outputs, loss = self.model.forward_teacher_forcing(inputs, labels)
        loss = loss / (self.normalization_steps * labels.size(0))
        loss.backward()
        b_loss += loss.item()

        _, preds = outputs.max(-1)
        res = self.model.calculate_metrics(preds, labels)

        b_cer += res["CER"] / (self.normalization_steps * labels.size(0))
        b_wer += res["WER"] / (self.normalization_steps * labels.size(0))

        if idx > 0 and (idx % self.normalization_steps == 0 or idx + 1 == len(self.train_data)):

          self.optimizer.step()
          self.optimizer.zero_grad()
          if self.wanda:
            wandb.log({
              'Train Loss': b_loss ,
              'Train CER' : b_cer ,
              'Train WER' : b_wer ,
              'Train': idx + ds_size * epoch_nr
            })

          total_loss += b_loss
          total_cer  += b_cer
          total_wer  += b_wer
          b_cer = 0.0
          b_wer = 0.0
          b_loss = 0.0

      total_loss /= (ds_size // self.normalization_steps)
      total_cer  /= (ds_size // self.normalization_steps)
      total_wer  /= (ds_size // self.normalization_steps)

      return total_loss, total_cer, total_wer




    def train_epoch(self, epoch_nr, ds_size):
      self.model.train()
      total_loss = 0.0
      total_cer = 0.0
      total_wer = 0.0

      nr_batches = 0
      b_cer = 0.0
      b_wer = 0.0
      for i, mb in enumerate(tqdm(self.train_data)):

        inputs, labels = mb
        inputs = inputs.to(self.device)
        labels = labels.to(self.device)

        output_logits, loss = self.model.forward_teacher_forcing(inputs, labels)

        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        _, preds = output_logits.max(-1)
        res = self.model.calculate_metrics(preds, labels)


        b_cer = res["CER"]
        b_wer = res["WER"]
        if self.wanda:
          wandb.log({
              'Train Loss': loss.item() / labels.size(0),
              'Train CER' : b_cer / labels.size(0),
              'Train WER' : b_wer / labels.size(0),
              'Train': i + ds_size * epoch_nr
            })

        total_loss += loss.item() / labels.size(0)
        total_cer += b_cer / labels.size(0)
        total_wer += b_wer / labels.size(0)



      total_loss /= ds_size
      total_cer /= ds_size
      total_wer /= ds_size

      return total_loss, total_cer, total_wer

    def val_epoch(self, epoch_nr, ds_size):
      self.model.eval()
      total_loss = 0.0
      total_cer = 0.0
      total_wer = 0.0
      nr_batches = 0
      b_loss = 0.0
      b_cer = 0.0
      b_wer = 0.0
      with torch.no_grad():
        for i, mb in enumerate(tqdm(self.val_data)):

          inputs, labels = mb
          inputs = inputs.to(self.device)
          labels = labels.to(self.device)
          self.optimizer.zero_grad()

          output_logits, _, loss = self.model.forward(inputs, labels)


          b_loss = loss.item()
          _, preds = output_logits.max(-1)
          res = self.model.calculate_metrics(preds, labels)
          b_cer = res["CER"]
          b_wer = res["WER"]
          if self.wanda:
            wandb.log({
                'Val Loss': b_loss / labels.size(0),
                'Val CER' : b_cer / labels.size(0),
                'Val WER' : b_wer / labels.size(0),
                'Val' : i  + ds_size * epoch_nr})

          total_loss += b_loss / labels.size(0)
          total_cer += b_cer / labels.size(0)
          total_wer += b_wer / labels.size(0)


          torch.cuda.empty_cache()


        total_loss /= ds_size
        total_cer /= ds_size
        total_wer /= ds_size
      return total_loss, total_cer, total_wer

    def train(self, train_len, val_len, wanda=True):
        self.wanda = wanda
        if wanda:
          self._init_wandb()
        for i in range(self.num_epochs):
            print(f'#.Epoch {i}')
            torch.cuda.empty_cache()
            train_loss, train_cer, train_wer = self.train_epoch_ga(i, train_len)
            val_loss, val_cer, val_wer = self.val_epoch(i, val_len)
            print(f"Train Loss avg: {train_loss}, Train CER avg: {train_cer}, Train WER avg: {train_wer}")
            print(f"Val Loss avg: {val_loss}, Val CER avg: {val_cer}, Val WER avg: {val_wer}")
        if wanda:
          wandb.finish()