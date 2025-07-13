from reader import Reader, ReadmissionReader
from torch.utils.data import Dataset
from torch import tensor, float32, float64, int
import numpy as np

class MIMICDataset(Dataset):
    def __init__(self, reader: Reader):
        self.reader = reader

    def __len__(self):
        return self.reader.get_number_of_examples()
    
    def __getitem__(self, idx):
        sample = self.reader.read_example(idx)
        X = sample['X']
        y = sample['y']

        X = np.where(X == '', np.nan, X)
        X = np.array(X, dtype=np.float32)

        X = tensor(X, dtype=float32)
        y = tensor(y, dtype=int)
        return X, y
    
    