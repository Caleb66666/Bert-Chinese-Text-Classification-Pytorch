# @Author : Caleb
# @Email: VanderLancer@gmail.com

import pandas as pd
import numpy as np
import torch
import time
from datetime import timedelta

PAD, CLS = "[PAD]", "[CLS]"


def process_line(line, config):
    pad_size = config.pad_size
    tokens = [CLS]
    tokens.extend(config.tokenizer.tokenize(line.strip()))
    seq_len = len(tokens)
    token_ids = config.tokenizer.convert_tokens_to_ids(tokens)

    if len(tokens) < pad_size:
        mask = [1] * len(token_ids) + [0] * (pad_size - len(tokens))
        token_ids.extend(([0] * (pad_size - len(tokens))))
    else:
        mask = [1] * pad_size
        token_ids = token_ids[: pad_size]
        seq_len = pad_size

    return token_ids, seq_len, np.array(mask)


def build_dataset(config):
    def _build_ds(file_path):
        df = pd.read_csv(file_path, header=0, sep="\t")
        df["token_ids"], df["seq_len"], df["mask"] = zip(*df["content"].apply(
            lambda line: process_line(line, config)))
        return df

    train_df = _build_ds(config.train_path)
    valid_df = _build_ds(config.valid_path)
    test_df = _build_ds(config.test_path)

    return train_df, valid_df, test_df


class DatasetIterater(object):
    def __init__(self, batches, batch_size, device):
        self.batch_size = batch_size
        self.batches = batches
        self.n_batches = len(batches) // batch_size
        self.residue = False
        if len(batches) / self.n_batches != 0:
            self.residue = True
        self.index = 0
        self.device = device

    def _to_tensor(self, df):
        token_ids = torch.LongTensor(df["token_ids"]).to(self.device)
        label = torch.LongTensor(df["label"]).to(self.device)
        seq_len = torch.LongTensor(df["seq_len"]).to(self.device)
        mask = torch.LongTensor(df["mask"]).to(self.device)
        return (token_ids, seq_len, mask), label

    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batch = self.batches[self.index * self.batch_size: len(self.batches)]
            self.index += 1
            batch = self._to_tensor(batch)
            return batch
        elif self.index > self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batch = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            batch = self._to_tensor(batch)
            return batch

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        return self.n_batches


def build_iterator(dataset, config):
    iter_ = DatasetIterater(dataset, config.batch_size, config.device)
    return iter_


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


if __name__ == '__main__':
    from models.bert import Config

    config_ = Config("THUCNews")
    build_dataset(config_)
