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
import torchaudio.transforms as T


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


class New_dataset(Dataset):
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
        #a = 100
        #return a
        #print(self.recording_df.shape[0])
        return self.recording_df.shape[0] - 5



    def __getitem__(self, idx):
        #idx = 10
        idx = idx + 5
        if torch.is_tensor(idx):
            idx = idx.tolist()
            

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
            """_, noisy_signal, _, speakers_target, s_thetas_array = pickle.load(f)
            mixed_sig_np = noisy_signal"""

            # for shlomi dataset
            #print(idx)
            mix_without_noise, noisy_signal, reverb_signals_targets, delayed_speakers, s_thetas_array = pickle.load(f)
            mixed_sig_np = noisy_signal

            '''(input_time, speakers_target, doa, dir_feature
            mixed_sig_np, speakers_delay, s_thetas_array = mixed_sig_np,\
                                                           speakers_delay.squeeze(), \
                                                           s_thetas_array
            '''


        reverb = self.recording_df.loc[idx, "rt60"]
        snr = self.recording_df.loc[idx, "mic_snr"]#it's actually wham snr


        #sample_separation = {'mixed_signals': mixed_sig_np, 'clean_speeches': speakers_target, "doa": s_thetas_array, 'df': dir_feature, "reverb":reverb} ToDO df
        sample_separation = {'mix_without_noise': mix_without_noise[0], 'mixed_signals': mixed_sig_np, 'clean_speeches': reverb_signals_targets, "delayed_target": delayed_speakers, "doa": s_thetas_array, "reverb":reverb, "snr":snr}
        # mixed_signals.shape = [num of channels, num of sampples got this particular audio]
        # clean_speeches.shape = [num of speakers, num of sampples got this particular audio]
        # doa.shape = number of speakers, the doa to the center of the glass
        return sample_separation




class Whamr_dataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, path_mix, cds_lables="", transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.recording_df = pd.read_csv(path_mix)
        self.transform = transform

    def __len__(self):
        # a = 100
        # return a
        # print(self.recording_df.shape[0])
        return self.recording_df.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        #print(idx)
        reverb_signals_targets1_path = self.recording_df.loc[idx, "path_file_spk1"]
        reverb_signals_targets2_path = self.recording_df.loc[idx, "path_file_spk2"]
        mixed_sig_np_path = self.recording_df.loc[idx, "path_file"]
        _, reverb_signals_targets1 = read(reverb_signals_targets1_path)
        _, reverb_signals_targets2 = read(reverb_signals_targets2_path)
       
        #print(f"s1 shape is: {reverb_signals_targets1.shape}")
        #print(f"s2 shape is: {reverb_signals_targets2.shape}")
        reverb_signals_targets = np.vstack((reverb_signals_targets1, reverb_signals_targets2)).copy()
        
        _, mixed_sig_np = read(mixed_sig_np_path)
        mixed_sig_np = mixed_sig_np.copy()

        reverb = self.recording_df.loc[idx, "rt60"]
        snr = self.recording_df.loc[idx, "mic_snr"]  # it's actually wham snr

        sample_separation = {'mix_without_noise': 0, 'mixed_signals': mixed_sig_np, 'clean_speeches': reverb_signals_targets,  "doa": 0,
                            "reverb": reverb, "snr": snr}


        return sample_separation




class Dataset8k(Dataset):
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
        self.resample = T.Resample(16000, 8000, dtype=torch.float32)

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


        reverb = self.recording_df.loc[idx, "rt60"]
        snr = self.recording_df.loc[idx, "mic_snr"]#it's actually wham snr

        mix_without_noise =self.resample(torch.tensor(mix_without_noise))
        noisy_signal =self.resample(torch.tensor(noisy_signal))
        speakers_target =self.resample(torch.tensor(speakers_target))
        #sample_separation = {'mixed_signals': mixed_sig_np, 'clean_speeches': speakers_target, "doa": s_thetas_array, 'df': dir_feature, "reverb":reverb} ToDO df
        #noisy_signal = noisy_signal - np.mean(noisy_signal)
        #speakers_target = speakers_target - np.mean(speakers_target, axis=1, keepdims=True)
        sample_separation = {'mix_without_noise': mix_without_noise[0], 'mixed_signals': noisy_signal, 'clean_speeches': speakers_target, "doa": s_thetas_array, "reverb":reverb, "snr":snr}
        # mixed_signals.shape = [num of channels, num of sampples got this particular audio]
        # clean_speeches.shape = [num of speakers, num of sampples got this particular audio]
        # doa.shape = number of speakers, the doa to the center of the glass
        return sample_separation



        
        
class ShortDataset(Dataset):
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
        #print(mix_without_noise.shape)
        rand_start_point = 5 #+ np.random.randn(0, 1)
        #start_point = np.ceil(16000 * rand_start_point)
        start_point = rand_start_point*16000 #.astype(int)
        end_point = start_point + 16000*5
        mix_without_noise = mix_without_noise[:, start_point: end_point]
        
        noisy_signal = noisy_signal[start_point: end_point]
        speakers_target = speakers_target[:, start_point: end_point]

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


if __name__ == '__main__':
    cds_lables = "/dsi/gannot-lab/datasets/mordehay/data_wham_libri_all_overlap_shlomi/train/labels_npz/"
    data = New_dataset("/mnt/dsi_vol1/shared/sharon_db/mordehay/train/csv_files/with_white_noise_res.csv", cds_lables)
   
    indx = np.arange(4)
    datloader = DataLoader(data, batch_size=2, shuffle=False, sampler=SubsetRandomSampler(indx), collate_fn=default_collate)
    for sample in data:
        print("bb")
    print("dddd")
