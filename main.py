from google.cloud import bigquery
import bigframes.pandas as bpd
import pandas as pd
from torch.utils.data import DataLoader
from reader import ReadmissionReader
from dataset import MIMICDataset
from reader import ReadmissionReader


test_reader = ReadmissionReader("/home/magnusjg/1TB/database/readmission/test")
# train_reader = ReadmissionReader("/home/magnusjg/1TB/database/readmission/train")

test_dataset = MIMICDataset(test_reader)
# train_dataset = MIMICDataset(train_reader)

test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)
# train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

test_features, test_labels = next(iter(test_loader))