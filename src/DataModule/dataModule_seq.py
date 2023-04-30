import sys
sys.path.append("src")
from utils.util import readData, preprocess 
from torch.utils.data import Dataset
from transformers import AutoTokenizer,  DataCollatorForLanguageModeling
from sklearn.model_selection import train_test_split
import pandas as pd 
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch

"""
1.   When we tokenize the data the problem is that legth greater than 512 is removed/truncated 
2.   we will not allow the tokenizer to truncate the length 
3.   Any data with length greater than 512 will be considered as another sequence.
4.   I don't care about the rating part I am only consered about the MovieId.
"""


class SeqDatase(Dataset):
    def __init__(self, df: pd.DataFrame) -> None:
        super().__init__()
        self.df = df    
        self.tokenizer = AutoTokenizer.from_pretrained("roberta-base", add_prefix_space=True)
        # roberta the mask token is <mask> 
        self.mask_token_id = self.tokenizer.vocab["<mask>"]

        # let's tokenize single row
        self.max_length = 510
        self.start = 0
        examples = []
        for idx, rows in tqdm(self.df.iterrows()):
            self.tokenized_data = self.tokenizer.encode_plus(rows["movieId"], add_special_tokens=False, is_split_into_words=True)
            input_ids  = self.tokenized_data["input_ids"]
            attention_mask  = self.tokenized_data["attention_mask"]

            input_ids_lists = self.chunks(input_ids)
            attention_mask_lists = self.chunks(attention_mask, "mask")

            # data_inputs.extend(list(input_ids_lists))
            # data_attention.extend(list(attention_mask_lists))
            # user_ids.append(rows["userId"])
            for inp, att in zip(list(input_ids_lists), list(attention_mask_lists)):
                examples.append({
                    "input_ids": torch.tensor(inp),
                    "attention_mask": torch.tensor(att)
                })

        # masking the dataset 
        data_collator = DataCollatorForLanguageModeling(self.tokenizer, mlm=True, mlm_probability=0.15)
        self.preprocessed_data = data_collator(examples)

    def chunks(self,ids,type=""):
        for i in range(self.start,len(ids), self.max_length):
            input_ids = ids[self.start: self.start+self.max_length]
            input_ids = [0]+input_ids + [2] if type =="" else [1] + input_ids + [1]
            # check if the length is 512 otherwise we pad 
            input_ids += [1]*(self.max_length - len(input_ids) + 2) if type == "" else [0] *(self.max_length - len(input_ids) +2)
            yield input_ids
        

    def __getitem__(self, idx):
        # in this section we need to mask the data so that model learns to predict the masked value 
        # we will randomly mask the value from the given dataset and replace it with masked index from vocabulary 
        attention_mask = self.preprocessed_data["attention_mask"][idx]
        input_ids      = self.preprocessed_data["input_ids"][idx]
        labels         = self.preprocessed_data["labels"][idx]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }
        


    def __len__(self):
        return self.preprocessed_data["input_ids"].shape[0]

class pytorchLightDataModule(LightningDataModule):
    def __init__(self):
        super().__init__()
        movies, ratings = readData()
        ratings = preprocess(ratings)

        groups = ratings.groupby("userId")
        self.moveid2title = dict(list(zip(movies["movieId"], movies["title"])))
        self.title2movieid = {value: key  for key, value in self.moveid2title.items()}
        self.df = pd.DataFrame({
            "userId": list(groups.groups.keys()),
            "movieId": groups.movieId.apply(list),
        })

        self.df["movieId"] = self.df["movieId"].map(self.conversion)


    def conversion(self,vals):
        return [self.moveid2title.get(x) for x in vals]
    
    def setup(self, stage: str) -> None:
        if stage=="fit":
            # we split the data into train and validation 
            train, test = train_test_split(self.df, test_size=0.3, random_state=42)
            print("Training Processing....")
            self.train_dataset = SeqDatase(train)
            print("Testing Processing....")
            self.test_dataset  = SeqDatase(test)


    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=4, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=4,shuffle=False, drop_last=True)
