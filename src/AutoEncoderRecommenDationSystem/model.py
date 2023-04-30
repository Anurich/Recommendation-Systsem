import sys
sys.path.append("src")
from DataModule.dataModule_autoEncoder import pytorchDataModuleAutoEncoder
from pytorch_lightning  import LightningModule, Trainer
from pytorch_lightning.loggers import WandbLogger
import torch.nn as nn
from torchmetrics import MeanSquaredError
import numpy as np
import torch

class MODEL(LightningModule):
    def __init__(self, inp_dimension):
        super().__init__()
        self.inp_dimension = inp_dimension
        self.V = nn.Linear(inp_dimension, out_features=512, bias=False)
        self.W = nn.Linear(in_features=512, out_features=inp_dimension, bias=False)

        self.mu = nn.Parameter(torch.zeros(512), requires_grad=True)
        self.b  = nn.Parameter(torch.zeros(inp_dimension), requires_grad=True)
        self.identity = nn.Identity()
        self.total_train_loss = []
        self.total_dev_loss   = []


    def forward(self, x):
        Encoder = torch.sigmoid(self.V(x)+self.mu)
        Decoder = self.identity(self.W(Encoder) + self.b)
        return Decoder


    def loss_fn(self, input, pred, masked):
        W_norm = torch.square(torch.linalg.matrix_norm(self.W.weight, ord="fro"))
        V_norm = torch.square(torch.linalg.matrix_norm(self.V.weight, ord="fro"))
        lambda_val = 0.2
        regularisation = lambda_val/2*(W_norm + V_norm)
        # let's compute the loss for observed value 
        meanSquaredError = torch.multiply((input-pred), masked)
        error = torch.square(torch.linalg.matrix_norm(meanSquaredError, ord=2))
        total_error = error + regularisation
        return total_error
        
    def training_step(self, batch, batch_idx):
        data = batch
        out = self(data["input_ids"])
        loss = self.loss_fn(data["input_ids"], out, data["masked_inpt"])
        self.total_train_loss.append(round(loss.item(),2))
        return loss

    def validation_step(self,batch, batch_idx):
        out = self(batch["input_ids"])
        dev_loss = self.loss_fn(batch["input_ids"], out, batch["masked_inpt"])
        self.total_dev_loss.append(dev_loss.item())


    def on_train_epoch_end(self):
        self.log("Train Loss", round(np.mean(np.array(self.total_train_loss)),2), prog_bar=True)
        self.total_train_loss.clear()
    
    def on_validation_epoch_end(self):
        self.log("Dev Loss", round(np.mean(np.array(self.total_dev_loss)),2), prog_bar=True)
        self.total_dev_loss.clear()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(),  lr=1e-3)
    
dimension = 17694
dataModule = pytorchDataModuleAutoEncoder()
wandbLogger = WandbLogger(project="RecommenDationSystem", name="AutoEncoderRecommendation",  log_model="all")
trainer = Trainer(accelerator="cpu", check_val_every_n_epoch=5, max_epochs=15, logger=wandbLogger)
md = MODEL(dimension)
trainer.fit(md, dataModule)