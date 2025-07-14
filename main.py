from google.cloud import bigquery
import bigframes.pandas as bpd
import pandas as pd
from torch.utils.data import DataLoader
from reader import ReadmissionReader
from dataset import MIMICDataset, SortedSampler
from reader import ReadmissionReader


test_reader = ReadmissionReader("/home/magnusjg/1TB/database/readmission/test")

test_dataset = MIMICDataset(test_reader)

l = len(test_dataset)

test_lengths = [(test_dataset[i][0].shape[0], test_dataset[i][2]) for i in range(l)]

test_sampler = SortedSampler(test_dataset, test_lengths)

print(test_sampler)

print(iter(test_sampler))