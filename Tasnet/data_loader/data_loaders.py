from torchvision import datasets, transforms
from base import BaseDataLoader
from data_loader.Noisy_WSJ_dataset import NoisyWsjDataSet

class NoisyWsjDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, csv_file, cds_lables, batch_size, type_dataset, shuffle=True, validation_split=0.0, num_workers=1):
        self.csv_file = csv_file
        self.dataset = NoisyWsjDataSet(csv_file, cds_lables)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
