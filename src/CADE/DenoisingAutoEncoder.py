import sys
sys.path.append("src")
from DataModule.dataModule_cade import pytorchDataModuleCADE
from pytorch_lightning  import LightningModule, Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
import torch.nn as nn
from torchmetrics import MeanSquaredError
import numpy as np
import torch

class MODEL(LightningModule):
    def __init__(self, inp_dimension, max_user):
        super().__init__()
        self.inp_dimension = inp_dimension
        self.max_user = max_user
        self.W = nn.Linear(inp_dimension, out_features=512)
        self.V = nn.Embedding(max_user+1, embedding_dim=512)

        self.bias = nn.Parameter(torch.zeros(inp_dimension), requires_grad=True)

        self.decoder = nn.Linear(in_features=512,  out_features=inp_dimension)

        self.total_train_loss = []
        self.total_dev_loss   = []

        self.loss_fn = nn.BCELoss()

    def forward(self, x, user_ids):
        encoder = torch.relu(self.W(x) + self.V(user_ids))
        decode  = self.decoder(encoder) + self.bias
        return torch.sigmoid(decode)


        
    def training_step(self, batch, batch_idx):
        data = batch
        out = self(data["input_ids"], data["userIds"])
        loss = self.loss_fn(out, data["labels"])
        self.total_train_loss.append(round(loss.item(),2))
        return loss

    def validation_step(self, batch, batch_idx):
        out = self(batch["input_ids"], batch["userIds"])
        dev_loss = self.loss_fn(out, batch["input_ids"])
        self.total_dev_loss.append(dev_loss.item())


    def on_train_epoch_end(self):
        self.log("Train_Loss", round(np.mean(np.array(self.total_train_loss)),2), prog_bar=True)
        self.total_train_loss.clear()
    
    def on_validation_epoch_end(self):
        self.log("Dev_Loss", round(np.mean(np.array(self.total_dev_loss)),2), prog_bar=True)
        self.total_dev_loss.clear()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(),  lr=1e-3)
    
dimension = 22187
dataModule = pytorchDataModuleCADE()
wandbLogger = WandbLogger(project="RecommenDationSystem", name="DenoisingAutoEncoder",  log_model="all")
trainer = Trainer(accelerator="cpu", check_val_every_n_epoch=5, max_epochs=15, logger=wandbLogger, callbacks=[EarlyStopping(monitor="Dev_Loss")])
md = MODEL(dimension, dataModule.max_user)
trainer.fit(md, dataModule)