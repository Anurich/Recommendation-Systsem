from pytorch_lightning import LightningModule, Trainer
import torchtext.vocab as vocab
import torch
import torch.nn as nn
import sys
import pandas as pd
sys.path.append("src")
from DataModule.dataModule_collab import MyDateModule
import warnings
warnings.filterwarnings("ignore")
from sklearn.preprocessing import LabelEncoder
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error
from pytorch_lightning.loggers import WandbLogger
from torchmetrics import  MeanSquaredError
import wandb

class MODEL(LightningModule):

    def __init__(self, num_user, num_movies, embedding_dim=100,load_pretrained_embedding=False):
        super().__init__()

        if load_pretrained_embedding:
            embedding = vocab.GloVe(name="6B", dim=100)
            matrix_users = torch.zeros((num_user, embedding_dim))
            for i in range(num_user):
                matrix_users[i] = embedding[str(i)].clone().detach().requires_grad_(True)
            
            
            matrix_movies = torch.zeros((num_movies, embedding_dim))
            for i in range(num_movies):
                matrix_movies[i] = embedding[str(i)].clone().detach().requires_grad_(True)
            
            # user embedding 
            self.user_embedding_matrix = nn.Embedding(num_embeddings=num_user, embedding_dim=embedding_dim)
            self.user_embedding_matrix.weight.data.copy_(matrix_users)
            
            # movie embedding
            self.movie_embedding_matrix = nn.Embedding(num_embeddings=num_movies, embedding_dim=embedding_dim)
            self.movie_embedding_matrix.weight.data.copy_(matrix_movies)
        else:
            # here we simply initialise two embedding matrix 
            self.user_embedding_matrix = nn.Embedding(num_embeddings=num_user, embedding_dim=embedding_dim)
            self.movie_embedding_matrix= nn.Embedding(num_embeddings=num_movies, embedding_dim=embedding_dim)


        # let's create two user and movie bias 
        self.user_bias = nn.Embedding(num_embeddings=num_user, embedding_dim=1)
        self.movie_bias = nn.Embedding(num_embeddings=num_movies, embedding_dim=1)
        
        # let's create few layers 
        
        self.layer1 = nn.Linear(in_features=64, out_features=64)
        self.layer2 = nn.Linear(in_features=64, out_features=1)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.counter = 0
        #self.loss_fn = nn.BCELoss()
        self.total_train_loss = []
        self.total_val_loss   = []

        self.val_mean_squared_error = MeanSquaredError()
        self.val_rmse               = MeanSquaredError(squared=False)
        
        self.train_mean_squared_error = MeanSquaredError()
        self.train_rmse               = MeanSquaredError(squared=False)


    def attention(self, user_vec, movie_vec, u_b, m_b):
        # score. = softmax(Q*k^T + U_b + M_b)
        UM = torch.matmul(movie_vec, user_vec.T)
        concatenated = UM + u_b +m_b                        
        # let's compute the softmax
        attention_weight = torch.softmax(concatenated, dim=-1)
        return concatenated

    def forward(self, movieId, userId):
        user_embed = self.user_embedding_matrix(userId)
        user_bias  = self.user_bias(userId)
        
        # for movie case
        movie_embed = self.movie_embedding_matrix(movieId)
        movie_bias  = self.movie_bias(movieId)
        
        
        # apply attention mechanism
       
        attn_out = self.attention(user_embed, movie_embed, user_bias, movie_bias)

        layer1out = self.relu(self.dropout(self.layer1(attn_out)))
        output = self.layer2(layer1out)
        return torch.sigmoid(output)
        

    def training_step(self, batch, batch_idx):
        movieId, userId, labels = batch["movieId"], batch["userId"], batch["label"]
        y_hat = self(movieId, userId)
        loss = F.binary_cross_entropy(y_hat,labels.unsqueeze(1))
        
        #self.log("train_loss", loss, on_epoch=True, on_step=False, prog_bar=True, logger=True)
        self.total_train_loss.append(loss.item())
        self.train_mean_squared_error.update(labels, y_hat.squeeze())
        self.train_rmse.update(labels, y_hat.squeeze())
        return loss 

    def on_train_epoch_end(self):
        mean_loss = np.mean(np.array(self.total_train_loss))
        # let's call the metrics 
        MSE, RMSE = self.train_mean_squared_error.compute(), self.train_rmse.compute()
        self.log("MSE-RMSE-LOSS", {"MSE":MSE, "RMSE":RMSE, "Train loss": round(mean_loss,2)}, prog_bar=True,logger=True)
        self.total_train_loss.clear()


    def validation_step(self, batch, batch_idx):
        movieId, userId, labels = batch["movieId"], batch["userId"], batch["label"]
        y_hat = self(movieId, userId)
        loss = F.binary_cross_entropy(y_hat,labels.unsqueeze(1))
        self.total_val_loss.append(loss.item())
        self.val_mean_squared_error.update(labels, y_hat.squeeze())
        self.val_rmse.update(labels, y_hat.squeeze())


    
    def on_validation_epoch_end(self):
        mean_loss = np.mean(np.array(self.total_val_loss))
        # let's call the metrics 
        MSE, RMSE = self.val_mean_squared_error.compute(), self.val_rmse.compute()
        self.log("MSE-RMSE-LOSS-VAL", {"MSE":MSE, "RMSE":RMSE, "Validation loss": round(mean_loss,2)}, prog_bar=True,logger=True)
        self.total_val_loss.clear()


    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-5)



datModule = MyDateModule()
user_encoder = LabelEncoder()
movie_encoder = LabelEncoder()
movie_encoder.classes_ = np.load("src/artifacts/label_movie.npy")
user_encoder.classes_  = np.load("src/artifacts/label_user.npy")
wandb.init(project="RecommenDationSystem", sync_tensorboard=True)
wandb_logger = WandbLogger(project="RecommenDationSystem", name="collaborative_filtering", log_model="all")
trainer = Trainer(accelerator="cpu", check_val_every_n_epoch=5, max_epochs=15, logger=wandb_logger)
mymodel = MODEL(len(user_encoder.classes_),len(movie_encoder.classes_), load_pretrained_embedding=True)
trainer.fit(mymodel, datModule)