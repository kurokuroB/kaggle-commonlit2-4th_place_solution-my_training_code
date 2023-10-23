# import
import sys
import os
import gc
import math
import warnings

from box import Box

import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint

from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoConfig,
    get_cosine_schedule_with_warmup,
)

# setting warnings
warnings.filterwarnings("ignore")
# For descriptive error messages
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# setting const
OUTPUT_DIR = sys.argv[1]
DATA_PATH = sys.argv[2]

# config
config = {
    "seed": 99,
    "epoch": 2,
    "max_len": 768,
    "gradient_checkpointing_enable": False,
    "model": "microsoft/deberta-v3-large",
    "trainer": {
        "devices": 1,
        "accelerator": "gpu",
        "accumulate_grad_batches": 4,
        "fast_dev_run": False,
        "num_sanity_val_steps": 0,
        "deterministic": False,
        "val_check_interval": 0.2,
        "precision": 16,
    },
    "train_loader": {
        "batch_size": 2,
        "shuffle": True,
        "num_workers": 4,
        "pin_memory": False,
        "drop_last": True,
    },
    "val_loader": {
        "batch_size": 32,
        "shuffle": False,
        "num_workers": 8,
        "pin_memory": False,
        "drop_last": False,
    },
    "optimizer": {
        "name": "optim.AdamW",
        "params": {"lr": 3e-5},
    },
    "params_dir": OUTPUT_DIR,
}

config = Box(config)

# set seed
seed_everything(config.seed, workers=True)

# data
summaries = pd.read_csv(f"{DATA_PATH}/summaries_train.csv")
prompts = pd.read_csv(f"{DATA_PATH}/prompts_train.csv")

summaries_prompts = summaries.merge(prompts, on="prompt_id", how="left")
summaries_prompts["others"] = (
    summaries_prompts["prompt_title"]
    + " "
    + "[SEP]"
    + " "
    + summaries_prompts["prompt_question"]
    + " "
    + "[SEP]"
    + " "
    + summaries_prompts["prompt_text"]
)


# dataset
class CommonLit2Dataset(Dataset):
    def __init__(self, df, tokenizer, max_length):
        self.df = df
        self.max_len = max_length
        self.tokenizer = tokenizer
        self.text = df["text"].values
        self.others = df["others"].values

        target_cols = [
            "content",
            "wording",
        ]
        self.target = df[target_cols].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        text = self.text[index]
        others = self.others[index]
        inputs_text = self.tokenizer.encode_plus(
            text,
            others,
            truncation=True,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
        )

        input_ids = inputs_text["input_ids"]
        attention_mask = inputs_text["attention_mask"]
        token_type_ids = inputs_text["token_type_ids"]

        # target
        target = self.target[index]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "target": torch.tensor(target, dtype=torch.float),
        }


# util
class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, yhat, y):
        return torch.sqrt(self.mse(yhat, y))


def calc_mcrmse(preds, targets):
    """
    preds:二次元のtensor
    targets:二次元のtensor
    """
    mcrmse = 0
    col_num = preds.shape[1]
    for i in range(col_num):
        rmse = RMSELoss()
        mcrmse += rmse(preds[:, i], targets[:, i])
    mcrmse /= col_num

    return mcrmse


def collate(batch):
    """batchをcollateする。"""
    mask_len = int(batch["attention_mask"].sum(axis=1).max())

    batch["input_ids"] = batch["input_ids"][:, :mask_len]
    batch["attention_mask"] = batch["attention_mask"][:, :mask_len]
    batch["token_type_ids"] = batch["token_type_ids"][:, :mask_len]
    batch["target"] = batch["target"]


# model
class CommonLit2Model(pl.LightningModule):
    def __init__(self, config, t_dataloader):
        super().__init__()
        self.cfg = config
        self.save_hyperparameters(self.cfg)

        # dataloader
        self._train_dataloader = t_dataloader

        self.transformer_config = AutoConfig.from_pretrained(self.cfg.model)
        self.transformer_config.output_hidden_states = True
        self.transformer_config.hidden_dropout_prob = 0.007
        self.transformer_config.attention_probs_dropout_prob = 0.007

        self.transformer = AutoModel.from_pretrained(
            self.cfg.model, config=self.transformer_config
        )

        if self.cfg.gradient_checkpointing_enable:
            self.transformer.gradient_checkpointing_enable()

        self.dropout = nn.Dropout(self.transformer_config.hidden_dropout_prob)

        self.output = nn.Linear(self.transformer_config.hidden_size, 2)

    def configure_optimizers(self):
        for layer in [
            self.transformer.embeddings,
            self.transformer.encoder.layer[:10],
        ]:
            for n, p in layer.named_parameters():
                p.requires_grad = False

        param_optimizer = list(self.named_parameters())

        no_decay = ["bias", "LayerNorm.bias"]
        optimizer_parameters = [
            {
                "params": [
                    p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": 1e-6,
            },
            {
                "params": [
                    p for n, p in param_optimizer if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 1e-6,
            },
        ]
        optimizer = eval(self.cfg.optimizer.name)(
            optimizer_parameters, lr=self.cfg.optimizer.params.lr
        )

        num_training_steps = (
            math.ceil(
                len(self._train_dataloader) / self.cfg.trainer.accumulate_grad_batches
            )
            * self.cfg.epoch
        )

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(num_training_steps * 0.2),
            num_training_steps=num_training_steps,
            num_cycles=0.5,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    def forward(self, batch):
        collate(batch)

        x = self.transformer(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            token_type_ids=batch["token_type_ids"],
        )
        x = x[0][:, 0, :]

        x = self.dropout(x)

        x = self.output(x)

        return x

    def _loss(self, out, target):
        return calc_mcrmse(out, target)

    def training_step(self, batch, batch_idx):
        out = self.forward(batch)
        target = batch["target"]

        loss = self._loss(out, target)
        self.log_dict({"train_loss": loss}, on_step=True, prog_bar=True)

        return {"loss": loss}

    def train_dataloader(self):
        return self._train_dataloader


# train
tokenizer = AutoTokenizer.from_pretrained(config.model)

dataset = CommonLit2Dataset(summaries_prompts, tokenizer, config.max_len)

dataloader = DataLoader(dataset, **config.train_loader)

model = CommonLit2Model(config, dataloader)

loss_checkpoint = ModelCheckpoint(
    dirpath=config.params_dir,
    save_top_k=1,
    filename="model1",
    save_weights_only=True,
    every_n_epochs=config.epoch,
)

trainer = pl.Trainer(
    max_epochs=config.epoch,
    callbacks=[loss_checkpoint],
    **config.trainer,
)

trainer.fit(model)

# reflesh memory
del model
gc.collect()
torch.cuda.empty_cache()
