import sys
from pytorch_lightning import  LightningDataModule
import numpy as np
import pandas as  pd 
sys.path.append("src/")
from utils.util import readData, preprocess
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import torch

class dataset(Dataset):
    def __init__(self, data, corrupt =0.2):
        self.data= data 
        # pivot the table 
        self.data[self.data != 0.0] = 1.0
        mask = np.random.binomial(1, 1-corrupt, size=self.data.shape)
        self.data_X = self.data*mask
        self.userIds = self.data.index
    def __getitem__(self, idx):
        """
            1. So We need to return the masked value along with the df 
        """
        return {
            "labels":  torch.tensor(self.data.values[idx]).float(),
            "input_ids": torch.tensor(self.data_X.values[idx]).float(),
            "userIds": torch.tensor(self.userIds.values[idx])
        }

    def __len__(self):
        return len(self.data)

class pytorchDataModuleCADE(LightningDataModule):
    def __init__(self) -> None:
        super().__init__()
        
        self.movie, self.ratings = readData()
        self.data = preprocess(self.ratings)
        self.max_user = max(set(self.data.userId.values.tolist()))
        self.max_movie = max(self.data["movieId"].unique())

    def setup(self, stage: str) :
        if stage == "fit":
            self.data = pd.pivot_table(self.data, values="rating", index="userId", columns="movieId").fillna(0.0)
            self.train, self.test = train_test_split(self.data, test_size=0.3, random_state=42)
            self.train_dataset =  dataset(self.train)
            self.test_dataset  =  dataset(self.test)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=32, drop_last=True)
    

    def val_dataloader(self) :
        return DataLoader(self.test_dataset, batch_size=32, drop_last=True)