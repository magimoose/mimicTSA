from google.cloud import bigquery
import bigframes.pandas as bpd
import pandas as pd
from torch.utils.data import DataLoader
from reader import ReadmissionReader
from dataset import MIMICDataset, SortedSampler
from reader import ReadmissionReader


train_reader = ReadmissionReader("/home/magnusjg/1TB/database/readmission/test")

train_dataset = MIMICDataset(train_reader)

train_lengths = [len(seq) for seq, _ in train_dataset]
train_sampler = SortedSampler(train_dataset, train_lengths)

print(iter(train_sampler))