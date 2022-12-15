import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Dataset
import pandas as pd
import torch ##Asasasasa
import pickle
import numpy as np
from scipy.io.wavfile import write
from scipy.io.wavfile import read
import os 
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data.dataloader import default_collate

class Old_Partial_dataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, cds_lables, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.cds_lables = cds_lables
        self.recording_df = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        #a = 50
        #return a
        #print(self.recording_df.shape[0])
        return self.recording_df.shape[0] - 5



    def __getitem__(self, idx):
        idx = idx + 5
        if torch.is_tensor(idx):
            idx = idx.tolist()
            print('idx:', idx)

        record_path = self.recording_df.loc[idx, "path_file"]
        with open(record_path, "rb") as f:
            #mixed_sig_np, speakers_target, s_thetas_array, dir_feature = pickle.load(f)
            """ for RandMicNoise dataset
            mixed_sig_np, speakers_target, s_thetas_array = pickle.load(f)
            mixed_sig_np = mixed_sig_np[0]"""
            #mixed_sig_np = np.sum(speakers_target, axis=0)
            #max_conv_val = np.max(mixed_sig_np)
            #min_conv_val = np.min(mixed_sig_np)
            #mixed_sig_np = 2 * (mixed_sig_np - min_conv_val) / (max_conv_val - min_conv_val) - 1
            #for wham dataset
            #for wham dataset
            #for wham dataset
            mix_without_noise, noisy_signal, _, speakers_target, s_thetas_array = pickle.load(f) #for delayed speakers, shlomi. also for partial wham
            # for the fixed/new data(withput duplication) 
            #_, noisy_signal, speakers_target, _, s_thetas_array = pickle.load(f) #for reverbed speakers, shlomi
            #mix_without_noise, noisy_signal, speakers_target, _, s_thetas_array = pickle.load(f) #for our sir0 and for the fixed/new data(without duplication) 
            
            #mixed_sig_np = noisy_signal

            # for shlomi dataset
            #_, noisy_signal, speakers_target, reverb_signals_targets, s_thetas_array = pickle.load(f)

            #_, noisy_signal, speakers_target, s_thetas_array = pickle.load(f)
            #mixed_sig_np = noisy_signal

            '''(input_time, speakers_target, doa, dir_feature
            mixed_sig_np, speakers_delay, s_thetas_array = mixed_sig_np,\
                                                           speakers_delay.squeeze(), \
                                                           s_thetas_array
            '''

        path_label_idx = os.path.join(self.cds_lables, "scenario_{0}.npz".format(idx))
        reverb = self.recording_df.loc[idx, "rt60"]
        snr = self.recording_df.loc[idx, "mic_snr"]#it's actually wham snr
        label = np.load(path_label_idx)

        #sample_separation = {'mixed_signals': mixed_sig_np, 'clean_speeches': speakers_target, "doa": s_thetas_array, 'df': dir_feature, "reverb":reverb} ToDO df
        #noisy_signal = noisy_signal - np.mean(noisy_signal)
        #speakers_target = speakers_target - np.mean(speakers_target, axis=1, keepdims=True)
        sample_separation = {'mix_without_noise': mix_without_noise[0], 'mixed_signals': noisy_signal, 'clean_speeches': speakers_target, "doa": s_thetas_array, "reverb":reverb, "snr":snr}
        label_csd = {"vad_frames_sum": label["vad_frames_sum"], "vad_frames_individual": label["vad_frames_individual"]}
        # mixed_signals.shape = [num of channels, num of sampples got this particular audio]
        # clean_speeches.shape = [num of speakers, num of sampples got this particular audio]
        # doa.shape = number of speakers, the doa to the center of the glass
        return sample_separation, label_csd


class Old_Partial_DataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, csv_file, cds_lables, batch_size, type_dataset, shuffle=True, validation_split=0.0, num_workers=1):
        self.csv_file = csv_file
        self.dataset = Old_Partial_dataset(csv_file, cds_lables)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)






class BaseDataLoader(DataLoader):
    """
    Base class for all data loaders
    """
    def __init__(self, dataset, batch_size, shuffle, validation_split, num_workers, collate_fn=default_collate):
        self.validation_split = validation_split
        self.shuffle = shuffle

        self.batch_idx = 0
        self.n_samples = len(dataset)

        self.sampler, self.valid_sampler = self._split_sampler(self.validation_split)

        self.init_kwargs = {
            'dataset': dataset,
            'batch_size': batch_size,
            'shuffle': self.shuffle,
            'collate_fn': collate_fn,
            'num_workers': num_workers
        }
        super().__init__(sampler=self.sampler, **self.init_kwargs)

    def _split_sampler(self, split):
        if split == 0.0:
            return None, None

        idx_full = np.arange(self.n_samples)

        np.random.seed(0)
        np.random.shuffle(idx_full)

        if isinstance(split, int):
            assert split > 0
            assert split < self.n_samples, "validation set size is configured to be larger than entire dataset."
            len_valid = split
        else:
            len_valid = int(self.n_samples * split)

        valid_idx = idx_full[0:len_valid]
        train_idx = np.delete(idx_full, np.arange(0, len_valid))

        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        # turn off shuffle option which is mutually exclusive with sampler
        self.shuffle = False
        self.n_samples = len(train_idx)

        return train_sampler, valid_sampler

    def split_validation(self):
        if self.valid_sampler is None:
            return None
        else:
            return DataLoader(sampler=self.valid_sampler, **self.init_kwargs)