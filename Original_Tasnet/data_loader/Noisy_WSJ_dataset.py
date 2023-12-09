from torch.utils.data import Dataset
import pandas as pd
import torch
import pickle
import numpy as np
from scipy.io.wavfile import write
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data.dataloader import default_collate


class NoisyWsjDataSet(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, cds_lables, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        print(cds_lables)
        print(csv_file)
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
            
            mix_without_noise, noisy_signal, _, speakers_target, s_thetas_array = pickle.load(f) #for delayed speakers, shlomi. also for partial wham
      
        reverb = self.recording_df.loc[idx, "rt60"]
        snr = self.recording_df.loc[idx, "mic_snr"]#it's actually wham snr
  

        #sample_separation = {'mixed_signals': mixed_sig_np, 'clean_speeches': speakers_target, "doa": s_thetas_array, 'df': dir_feature, "reverb":reverb} ToDO df
        #noisy_signal = noisy_signal - np.mean(noisy_signal)
        #speakers_target = speakers_target - np.mean(speakers_target, axis=1, keepdims=True)
        sample_separation = {'mix_without_noise': mix_without_noise[0], 'mixed_signals': noisy_signal, 'clean_speeches': speakers_target, "doa": s_thetas_array, "reverb":reverb, "snr":snr}
        
        # mixed_signals.shape = [num of channels, num of sampples got this particular audio]
        # clean_speeches.shape = [num of speakers, num of sampples got this particular audio]
        # doa.shape = number of speakers, the doa to the center of the glass
        return sample_separation
        # mixed_signals.shape = [num of channels, num of sampples got this particular audio]
        # clean_speeches.shape = [num of speakers, num of sampples got this particular audio]
        # doa.shape = number of speakers, the doa to the center of the glass
        return sample


if __name__ == '__main__':
    data = NoisyWsjDataSet("/mnt/dsi_vol1/shared/sharon_db/mordehay/train/csv_files/with_white_noise_res.csv")
    indx = np.arange(4)
    datloader = DataLoader(data, batch_size=2, shuffle=False, sampler=SubsetRandomSampler(indx), collate_fn=default_collate)
    for sample in data:
        print("bb")
    print("dddd")
