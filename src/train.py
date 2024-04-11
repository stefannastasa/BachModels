from torch.utils.data import DataLoader
from torch.optim import Optimizer
import wandb
from tqdm import tqdm

import math
from torch.utils.data import DataLoader
from src.IAMDataset import IAMDataset
from src.model import FullPageHTR

import torch
from copy import copy


class ModelTrainer:

    def __init__(self, run_name: str,
                 model: FullPageHTR,
                 ds_name: str,
                 train_data: DataLoader,
                 val_data: DataLoader,
                 optimizer: Optimizer,
                 num_epochs: int,
                 device: torch.device):
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

    def train_epoch(self):
        self.model.train()
        total_loss = 0.0
        for batch in tqdm(self.train_data):
            inputs, labels = batch
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            self.optimizer.zero_grad()
            output_logits, loss = self.model.forward_teacher_forcing(inputs, labels)

            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            res = self.model.calculate_metrics(output_logits, labels)
        return total_loss / len(self.train_data), res["CER"], res["WER"]

    def val_epoch(self):
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for batch in tqdm(self.val_data):
                inputs, labels = batch
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                output_logits, sampled_ids, loss = self.model(inputs, labels)
                total_loss += loss.item()
                res = self.model.calculate_metrics(output_logits, labels)
        return total_loss / len(self.val_data), res["CER"], res["WER"]

    def train(self):
        self._init_wandb()
        for i in range(self.num_epochs):
            print(f'#.Epoch {i}')
            train_loss, train_cer, train_wer = self.train_epoch()
            val_loss, val_cer, val_wer = self.val_epoch()
            wandb.log({'Train Loss': train_loss, 'Val Loss': val_loss,
                       'Cer Train': train_cer, 'Cer Val': val_cer,
                       'Wer Train': train_wer, 'Wer Val': val_wer})

        wandb.finish()


if __name__ == '__main__':

    ds = IAMDataset(base_dir="/home/tefan/projects/BachModels/data/raw", embedding_loader=None, sample_set="train")
    ds_train, ds_val = torch.utils.data.random_split(ds, [math.ceil(0.8 * len(ds)), math.floor(0.2 * len(ds))])

    ds_val.data = copy(ds)
    ds_val.data.set_transform_pipeline("val")

    from functools import partial

    batch_size = 1
    pad_tkn_idx, eos_tkn_idx = ds.embedding_loader.encode_labels(["<PAD>", "<EOS>"])
    collate_fn = partial(
        IAMDataset.collate_fn, pad_val=pad_tkn_idx, eos_tkn_idx=eos_tkn_idx
    )
    num_workers = 4
    dl_train = DataLoader(
        ds_train,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
    )
    dl_val = DataLoader(
        ds_val,
        batch_size=2 * batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()
    model = FullPageHTR(ds.embedding_loader, max_seq_len=ds.max_seq_length()+1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    trainer = ModelTrainer("Testing_run", model, ds_name="IAM_forms", train_data=dl_train, val_data=dl_val,
                           optimizer=optimizer, num_epochs=100, device=device)

    trainer.train()
