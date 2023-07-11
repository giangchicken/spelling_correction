import warnings
from datasets import load_dataset
import pytorch_lightning as pl
from torch.utils.data.dataloader import DataLoader
from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
)

from tokenizers import Tokenizer
from torch.nn.utils.rnn import pad_sequence
import torch

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

import random

cache_dir = './cache/'
processor = Wav2Vec2Processor.from_pretrained("nguyenvulebinh/wav2vec2-base-vietnamese-250h", cache_dir=cache_dir)
tokenizer = Tokenizer.from_file("MyBPETokenizer.json")

BOS_IDX = 5
EOS_IDX = 6

def insert(index_to_insert, number_to_insert, tensor):
    # Chia tensor thành 2 phần
    part1 = tensor[:index_to_insert]
    part2 = tensor[index_to_insert:]

    # Tạo tensor mới chứa số đã chèn
    new_tensor = torch.cat((part1, torch.tensor([number_to_insert]), part2))
    
    return new_tensor


def generate_error_input(input_):
    len_input = input_.size()[0]
    
    num = random.random()
    
    if (num >0.3) & (num < 0.7):
    
        nb_random = int(0.06*len_input)
        
        rand_idx = [random.randint(1, len_input-1) for i in range(nb_random)]
        rand_number = [random.randint(0, 19000) for i in range(nb_random)]
        for i in range(nb_random):
            input_[rand_idx[i]] = rand_number[i]   
            
        rand_idx_insert = [random.randint(1, len_input-1) for i in range(nb_random)]
        rand_number_insert = [random.randint(6, 19000) for i in range(nb_random)]
        
        for i in range(nb_random):
            input_ = insert(rand_idx_insert[i], rand_number_insert[i], input_)
        
        input_ = torch.cat((torch.tensor([BOS_IDX]), 
                        torch.tensor(input_), 
                        torch.tensor([EOS_IDX])))
        return input_ 
    
    if (num < 0.3):
        nb_random = int(0.06*len_input)
        
        rand_idx_insert = [random.randint(1, len_input-1) for i in range(nb_random)]
        rand_number_insert = [random.randint(6, 19000) for i in range(nb_random)]
        
        for i in range(nb_random):
            input_ = insert(rand_idx_insert[i], rand_number_insert[i], input_)
        
        input_ = torch.cat((torch.tensor([BOS_IDX]), 
                        torch.tensor(input_), 
                        torch.tensor([EOS_IDX])))
        return input_ 
    
    if (num > 0.7):
        nb_random = int(0.06*len_input)
            
        rand_idx_insert = [random.randint(1, len_input-1) for i in range(nb_random)]
        rand_number_insert = [random.randint(6, 19000) for i in range(nb_random)]
        
        for i in range(nb_random):
            input_ = insert(rand_idx_insert[i], rand_number_insert[i], input_)
        
        input_ = torch.cat((torch.tensor([BOS_IDX]), 
                        torch.tensor(input_), 
                        torch.tensor([EOS_IDX])))
        return input_                 

def generate_error_label(input_):
    len_input = input_.size()[0]
    
    input_ = torch.cat((torch.tensor([BOS_IDX]), 
                      torch.tensor(input_), 
                      torch.tensor([EOS_IDX])))
    return input_ 


@dataclass
class DataCollatorCTCWithPadding:

    def __call__(self, 
                features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need
        # different padding methods
        # print(features)
        input_features = [generate_error_input(torch.tensor(feature["input_values"])) for feature in features]
        label_features = [generate_error_label(torch.tensor(feature["labels"])) for feature in features]

        # print(input_features)
        input_ = pad_sequence(input_features, batch_first=True, padding_value=3)
        label = pad_sequence(label_features, batch_first=True, padding_value=3)
        
        # with processor.as_target_processor():
        #     batch = processor.pad(
        #         input_features,
        #         padding=False,
        #         pad_to_multiple_of=3,
        #         return_tensors="pt",
        #     )

        #     labels_batch = processor.pad(
        #         label_features,
        #         padding=False,
        #         pad_to_multiple_of=3,
        #         return_tensors="pt",
        #     )

        # replace padding with -100 to ignore loss correctly
        # labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # print(labels_batch)
        
        batch = {}
        batch["input_ids"] = input_
        batch["labels"] = label

        return batch




class LMDataModule(pl.LightningDataModule):
    def __init__(self, model_name_or_path, train_file, validation_file,
                 preprocessing_num_workers, overwrite_cache,
                 train_batch_size, val_batch_size, dataloader_num_workers):
        super().__init__()
        self.train_file = train_file
        self.validation_file = validation_file
        self.model_name_or_path = model_name_or_path
        self.preprocessing_num_workers = preprocessing_num_workers
        self.overwrite_cache = overwrite_cache
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.dataloader_num_workers = dataloader_num_workers

    def setup(self, stage):
        
        extension = self.train_file.split(".")[-1]
        if extension in ("csv", "raw"):
            extension = "csv"

        data_files = {}
        data_files["train"] = self.train_file
        data_files["validation"] = self.validation_file
        datasets = load_dataset(extension, data_files=data_files)

        column_names = datasets["train"].column_names

        def tokenize_function(examples):
            # Remove empty lines

            examples["input_values"] = [torch.tensor(tokenizer.encode(line).ids)
                                for line in examples["out_wav2vec2"]
                                if len(line) > 0 and not line.isspace()]

            examples["labels"] = [torch.tensor(tokenizer.encode(line).ids)
                                for line in examples["label"]
                                if len(line) > 0 and not line.isspace()]
            
            return examples

        tokenized_datasets = datasets.map(
            tokenize_function,
            batched=True,
            num_proc=self.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not self.overwrite_cache,
        )

        data_collator = DataCollatorCTCWithPadding()

        train_dataset = tokenized_datasets["train"]
        eval_dataset = tokenized_datasets["validation"]
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.data_collator = data_collator

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            shuffle=True,
            collate_fn=self.data_collator,
            num_workers=self.dataloader_num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.eval_dataset,
            batch_size=self.val_batch_size,
            shuffle=True,
            collate_fn=self.data_collator,
            num_workers=self.dataloader_num_workers,
        )