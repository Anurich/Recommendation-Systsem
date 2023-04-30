import sys 
sys.path.append("src/")
from DataModule.dataModule_seq import  pytorchLightDataModule
from pytorch_lightning import LightningModule, Trainer
from transformers import AutoModelForMaskedLM, AdamW, get_scheduler
import numpy as np
from pytorch_lightning.loggers import WandbLogger
import wandb

class MODEL(LightningModule):
    def __init__(self):
        super().__init__()
        self.model = AutoModelForMaskedLM.from_pretrained("roberta-base")
        self.total_train_loss = []
        self.total_val_loss   = []
    def forward(self,x):
        return self.model(**x)

    def training_step(self, batch, batch_idx):
        output = self(batch)
        loss = output.loss
        self.total_train_loss.append(loss.item())
        return loss


    def on_train_epoch_end(self):
        self.log("Train Loss", np.mean(np.array(self.total_train_loss)))
        self.total_train_loss.clear()

    def validation_step(self, batch, batch_idx):
        output = self(batch)
        self.total_val_loss.append(output.loss.item())

    def on_validation_epoch_end(self):
        self.log("Validation Loss", np.mean(np.array(self.total_val_loss)))
        self.total_val_loss.clear()

    def configure_optimizers(self):
        
        optimizer =  AdamW(self.parameters(), lr=1e-5)
        max_training_step = len(self.train_dataloader()) * self.hparams.epochs
        scheduler = get_scheduler(optimizer=optimizer, name="linear", num_warmup_steps =0, num_training_steps=max_training_step)
        return [optimizer], [scheduler]
       


datModule = pytorchLightDataModule()
wandbLogger = WandbLogger(project="RecommenDationSystem", name="SequentialRecommendation", log_model="all")
trainer = Trainer(accelerator="mps", check_val_every_n_epoch=5, max_epochs=15, logger=wandbLogger)
mymodel = MODEL()
trainer.fit(mymodel, datModule)
