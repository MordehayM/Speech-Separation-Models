from torchvision import datasets, transforms
from base import BaseDataLoader
from data_loader.Noisy_WSJ_dataset import New_dataset
from data_loader.Noisy_WSJ_dataset import Old_Partial_dataset
from data_loader.Noisy_WSJ_dataset import Whamr_dataset
from data_loader.Noisy_WSJ_dataset import Dataset8k
from data_loader.Noisy_WSJ_dataset import ShortDataset

class Old_Partial_DataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, csv_file, cds_lables, batch_size, type_dataset, shuffle=True, validation_split=0.0, num_workers=1):
        self.csv_file = csv_file
        if type_dataset == "whamr":  
            self.dataset = Whamr_dataset(csv_file, cds_lables)
        elif type_dataset == "old":
            self.dataset = Old_Partial_dataset(csv_file, cds_lables)
        elif type_dataset == "new":
            self.dataset = New_dataset(csv_file, cds_lables)
        elif type_dataset == "8k":
            self.dataset = Dataset8k(csv_file, cds_lables)
        elif type_dataset == "short":
            self.dataset = ShortDataset(csv_file, cds_lables)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)