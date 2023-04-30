import sys
sys.path.append("src")
from utils.util import readData, preprocess 
from pytorch_lightning import LightningDataModule
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import torch


class dataset(Dataset):
    def __init__(self, data) -> None:
        super().__init__()
        self.data = data
        self.data.reset_index(inplace=True, drop=True)
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        out = self.data.loc[index]
        ratings, userId, movieId = out["rating"], out["userId"], out["movieId"]
        return {
            "label": torch.tensor(ratings).float(),
            "userId": torch.tensor(userId, dtype=torch.long),
            "movieId": torch.tensor(movieId, dtype=torch.long)
        }


class MyDateModule(LightningDataModule):
    def __init__(self) -> None:
        super().__init__()
        self.movies, self.ratings = readData()
        self.ratings = preprocess(self.ratings)

        self.lbn_user = LabelEncoder()
        self.lbn_movie = LabelEncoder()

        self.ratings.userId = self.lbn_user.fit_transform(self.ratings.userId)
        self.ratings.movieId = self.lbn_movie.fit_transform(self.ratings.movieId)

        self.minMaxScaler = MinMaxScaler()
        self.ratings.rating = self.minMaxScaler.fit_transform(self.ratings.rating.values.reshape(-1,1))
        # let's save the encoder 


        np.save("src/artifacts/min_max_scaler.npy", self.minMaxScaler)
        np.save("src/artifacts/label_user.npy", self.lbn_user.classes_)
        np.save("src/artifacts/label_movie.npy", self.lbn_movie.classes_) 

        # let's split the dataset 

    def setup(self, stage: str) -> None:
        if stage == "fit":
            self.train, self.test = train_test_split(self.ratings, test_size=0.2, random_state=42)
            # i should only consider the userId and Movie ID 
            self.train = self.train[["userId", "movieId","rating"]]
            self.test  = self.test[["userId", "movieId", "rating"]]

            self.trainDataset = dataset(self.train)
            self.testDatset = dataset(self.test)

        
    def train_dataloader(self):
        return DataLoader(self.trainDataset, batch_size=64, shuffle=True, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.testDatset, batch_size=64, drop_last=True)
    
