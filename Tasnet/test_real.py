from email.policy import strict
#from pydub import AudioSegment
import numpy as np  
import argparse
import torch
from tqdm import tqdm
import data_loader.data_loaders as module_data
import model.pit_wrapper as module_loss
import model.metric as module_metric
from utility import conv_tasnet as module_arch
from parse_config import ConfigParser
import matplotlib.pyplot as plt
import model.sdr as module_func_loss
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.io.wavfile import write
from scipy.io.wavfile import read
import torchaudio.transforms as T
#from model.combined_loss import reorder_source_mse
import os
from utils import prepare_device

'''"""
class OnlineSaving:
    def __init__(self, model, save_path, criterion_similarity=None) -> None:
        self.indx = 0
        self.fs = 16000
        self.max_len = 5
        self.save_sec = 1
        self.model = model
        self.save_path = save_path
        self.online_sisdr = []
        self.reference_sisdr = []
        self.num_save_samples = 30
        self.similarity = False
        if criterion_similarity is not None:
            self.similarity = True
            self.criterion_similarity = criterion_similarity
        
            
    def reset(self):
        self.indx = 0
    def update_online_signal(self, est_signals):
        """_summary_

        Args:
            signals: the signal are after ordering (PIT)
        """
        if self.indx == 0:
            self.online_signal = est_signals[:, :, est_signals.shape[-1] - int(np.floor(self.fs*self.save_sec)):]
        else:
            self.online_signal = torch.cat((self.online_signal, est_signals[:, :, est_signals.shape[-1] - int(np.floor(self.fs * self.save_sec)):]), dim=-1)
            
    def get_truncated_signal(self, full_signal_mix):
        truncated_signal_mix = full_signal_mix[:, int(np.floor(self.fs * self.indx * self.save_sec)): int(np.floor(self.fs * self.indx * self.save_sec)) + self.max_len * self.fs]
        return truncated_signal_mix
            
    def increase_indx(self):
        self.indx += 1
        
    def get_indx(self):
        return self.indx
    
    def save_audio(self, name_folder, separated_signals, mix):
        
        separated_audio1 = separated_signals[0, 0, :].cpu().detach().numpy() #sample 0 from batch
        separated_audio2 = separated_signals[0, 1, :].cpu().detach().numpy() #sample 0 from batch
        mix_waves = mix[0, :].cpu().detach().numpy()  #sample 0 from batch and 0 from channel
        Path(f"{self.save_path}/{name_folder}/indx_{self.indx}").mkdir(parents=True, exist_ok=True)
        write(f"{self.save_path}/{name_folder}/indx_{self.indx}/mixed.wav", self.fs, mix_waves.astype(np.float32))
        write(f"{self.save_path}/{name_folder}/indx_{self.indx}/output_0.wav", self.fs, separated_audio1.astype(np.float32))
        write(f"{self.save_path}/{name_folder}/indx_{self.indx}/output_1.wav", self.fs, separated_audio2.astype(np.float32))
        
        
    def save_last_online_audio(self, name_folder, online_signal, mixed_signal_t):
        
        online_signal = online_signal[0, :, :].cpu().detach().numpy() #sample 0 from batch
        mixed_signal_t = mixed_signal_t[0, :].cpu().detach().numpy() #sample 0 from batch
        Path(f"{self.save_path}/{name_folder}").mkdir(parents=True, exist_ok=True)
        #save online signal
        write(f"{self.save_path}/{name_folder}/online_signal0.wav", self.fs, online_signal[0].astype(np.float32))
        write(f"{self.save_path}/{name_folder}/online_signal1.wav", self.fs, online_signal[1].astype(np.float32))
        #save true signal
        
        write(f"{self.save_path}/{name_folder}/ref_mix.wav", self.fs, mixed_signal_t.astype(np.float32))
        
    def calc_online(self, full_signal_mix, name_folder, sample_indx):
        if full_signal_mix.shape[-1] < self.fs * self.max_len:
            full_signal_mix = torch.nn.functional.pad(full_signal_mix, (0, self.fs * self.max_len - full_signal_mix.shape[-1]))
            
        #print(full_signal_mix.shape[-1])
        max_indx = np.floor(((full_signal_mix.shape[-1] - self.fs * self.max_len) / (self.fs * self.save_sec)))
        #print(max_indx)
        
        while self.indx <= max_indx:

            truncated_signal_mix = self.get_truncated_signal(full_signal_mix)
            pred_separation, _, _, _, _, _ = self.model(truncated_signal_mix)
            if self.indx == 0:
                self.update_online_signal(pred_separation)
            pred_separation_sim = pred_separation[:, :, - int(np.floor(self.fs*self.save_sec)) - self.online_signal.shape[-1]: - int(np.floor(self.fs*self.save_sec))]
            truncated_onlinet_sim = self.online_signal[:, :, - self.fs * self.max_len + int(np.floor(self.fs*self.save_sec)): ]

            _, batch_indices_separation = self.criterion_similarity(pred_separation_sim, truncated_onlinet_sim,
                                                                            return_incides=True)
            
            pred_separation = reorder_source_mse(pred_separation, batch_indices_separation)
            self.update_online_signal(pred_separation)
            if sample_indx < self.num_save_samples:
                self.save_audio(name_folder, pred_separation, truncated_signal_mix)
            self.increase_indx()

       
        mixed_signal_t = full_signal_mix[:, int(np.floor(self.fs * (self.max_len - self.save_sec))): int(np.floor(self.fs * (self.max_len + (self.indx - 1) * self.save_sec)))]
        #mixed_signal_t = mixed_signal_t[:, 6*self.fs:] ##test
        if sample_indx < self.num_save_samples:
            self.save_last_online_audio(name_folder, self.online_signal, mixed_signal_t)
        
        self.reset() 
"""'''



def plot_spectrogram(masks, title, save_path, batch_indx, ylabel='freq_bin', aspect='auto', xmax=None):
    masks  =masks.cpu()
    for indx_mask in range(masks.shape[1]):
        fig, axs = plt.subplots(1, 1)
        axs.set_title(f"Spectrogram (db) - {title}")
        axs.set_ylabel(ylabel)
        axs.set_xlabel('frame')
        im = axs.imshow(masks[0, indx_mask, :, :].detach().numpy(), origin='lower', aspect=aspect) #sample 0 from batch
        if xmax:
            axs.set_xlim((0, xmax))
        fig.colorbar(im, ax=axs)
        Path(f"{save_path}Batch_{batch_indx}/").mkdir(parents=True, exist_ok=True)
        plt.savefig(f"{save_path}Batch_{batch_indx}/Mask_Speaker_{indx_mask}")
        plt.close('all')

def save_audio(mix_waves, separated_signals, save_path, samplerate):

    separated_audio1 = separated_signals[0, 0, :].cpu().detach().numpy() #sample 0 from batch
    separated_audio2 = separated_signals[0, 1, :].cpu().detach().numpy() #sample 0 from batch
    mix_waves = mix_waves[0, :].cpu().detach().numpy()  #sample 0 from batch and 0 from channel
    Path(save_path).mkdir(parents=True, exist_ok=True)
    
    separated_audio1 = 2*(separated_audio1 - np.min(separated_audio1)) / (np.max(separated_audio1) - np.min(separated_audio1)) - 1
    separated_audio2 = 2*(separated_audio2 - np.min(separated_audio2)) / (np.max(separated_audio2) - np.min(separated_audio2)) - 1
    
    write(f"{save_path}/mixed.wav", samplerate, mix_waves.astype(np.float32))
    write(f"{save_path}/output_0.wav", samplerate, separated_audio1.astype(np.float32))
    write(f"{save_path}/output_1.wav", samplerate, separated_audio2.astype(np.float32))


def save_csd(csd_output, target_csd, batch_indx, save_path):
    Path(f"{save_path}Batch_{batch_indx}/").mkdir(parents=True, exist_ok=True)
    if torch.tensor(target_csd).ndim == 0:
        target_csd = torch.zeros((1, csd_output.shape[1], csd_output.shape[2]))
    csd_output = csd_output.cpu()
    target_csd = target_csd.cpu()
    for spk in range(target_csd.shape[1]):
        plt.plot(target_csd[0, spk])
        plt.savefig(f"{save_path}Batch_{batch_indx}/True_Csd_{spk}.png")
        plt.close()
        estimated_csd = torch.argmax(csd_output[0, spk], dim=0)
        plt.plot(estimated_csd)
        plt.savefig(f"{save_path}Batch_{batch_indx}/Estimated_Csd_{spk}.png")
        plt.close()


def main(config):
    logger = config.get_logger('test_real')

    # build model architecture
    model = config.init_obj('arch', module_arch)
    logger.info(model)

    device, device_ids = prepare_device(config['n_gpu'])
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    checkpoint = torch.load(config.resume, map_location=torch.device('cpu'))
    state_dict = checkpoint['state_dict']
    save_test_path = config["tester"]["save_test_real"]
    
    model.load_state_dict(state_dict, strict=True)

    # prepare model for testing

    model.eval()
    doa = 2 # for filling only, it has no meaning right now
    wav_bool = False
    two_bool = False
    m4a_bool = True
    bit16 = False
    online_bool = False
    fs = 16000
    max_audio_len = 101
    n_fft = 512
    cut_len_temp = int(max_audio_len * fs / n_fft)
    cut_len = cut_len_temp * n_fft
    if wav_bool:
        if two_bool:
            wav_fname1 = "/home/dsi/moradim/Audio-Visual-separation-using-RTF/f2_3deg.wav"
            samplerate, data1 = read(wav_fname1) #shape = [samples, channels]
            wav_fname2 = "/home/dsi/moradim/Audio-Visual-separation-using-RTF/m2_3deg.wav"
            samplerate, data2 = read(wav_fname2)
            cut_new = min(cut_len, len(data1[:, 0]), len(data2[:, 0]))
            data1 = data1[:cut_new, :]
            data2 = data2[:cut_new, :]
            audio = data1 + data2
            audio = audio[:, 0] #choose channel 0
        else:
            #"/dsi/gannot-lab/datasets2/wsj_mix/EN_scenarios/1/SNR15/sent_1_snr_15_50_overlap.wav"
            # "/dsi/gannot-lab/datasets2/wsj_mix/EN_scenarios/1/SNR15/sent_1_snr_15_50_overlap.wav"
            #"/dsi/gannot-lab/datasets2/wsj_mix/mix_15.wav"
            wav_path = "/home/dsi/moradim/Audio-Visual-separation-using-RTF/Looking to Listen - Stand-up-NzZDnRni-8A.wav"
            samplerate, data = read(wav_path) 
            if data.ndim == 2:
                if data.shape[0] > 10:
                    data = data[:, 0]
                elif data.shape[1] > 10:
                    data = data[0, :]
            data = np.float32(data)
            if samplerate != 16000:
                resample = T.Resample(samplerate, 16000, dtype=torch.float32)
                data = resample(torch.tensor(data))
            #print(data.shape)
            #cut_new = min(cut_len, len(data))
            #audio = data[:cut_new + 1]
            audio = data#data[:, 2]
            
    if m4a_bool:    
        path_audio = "/home/dsi/moradim/OurBaselineModels/Tasnet/real_recordings_wav/renana_daniel_english.wav"
        #name_run = os.path.basename(path_audio)[:-4]
        wav_name= path_audio[:-4]
        #os.system("ffmpeg -i {0}  {1}.wav".format(path_audio, wav_name))
        """audio = AudioSegment.from_file(path_audio, "m4a")
        samplerate = audio.frame_rate
        
        audio = np.array(audio.get_array_of_samples(), dtype=np.float32)
        print(audio.dtype)"""
        samplerate, audio = read(wav_name + ".wav")
        audio = np.array(audio, dtype=np.float32)
        #os.remove(wav_name + ".wav")
        if samplerate != 16000:
            resample = T.Resample(samplerate, 16000, dtype=torch.float32)
            audio = resample(torch.tensor(audio))
        #print(len(audio/samplerate))
    audio = np.float32(audio)
    normalized_audio = 1.8*(audio - audio.min()) / (audio.max() - audio.min()) - 0.9
    #write("try.wav", samplerate, normalized_audio.astype(np.float32))
    normalized_audio = torch.from_numpy(normalized_audio)
    normalized_audio = torch.unsqueeze(normalized_audio, dim=0)
    i = 0
    """if online_bool:
        similarity_func = torch.nn.L1Loss()
        criterion_similarity = module_loss.PITLossWrapper(loss_func=similarity_func, pit_from="pw_pt")
        onlinesaving = OnlineSaving(model, save_test_path, criterion_similarity)
        name_folder = f"Batch_{i}/"
        onlinesaving.calc_online(normalized_audio, name_folder, i)"""
    
    out_separation = model(normalized_audio)
    save_audio(normalized_audio, out_separation, save_test_path, 16000)
    #save_csd(csd_output, 0, i, save_test_path)

# ="/dsi/gannot-lab/datasets/mordehay/Result/Separation_data_wham_partial_overlap_MultiplyDialation/models/AV_model/1024_074543/model_best.pth"
# "/dsi/gannot-lab/datasets/mordehay/Result/SeparationVadsum099_GN_withoutDF_partial_wham_16_drop2d005_lr0001_decay095_Wen/models/AV_model/0804_133309/model_best.pth"
# "/dsi/gannot-lab/datasets/mordehay/Result/Separation_data_wham_all_overlap_TrainOn3Sec/models/AV_model/1020_090116/model_best.pth"
# "/dsi/gannot-lab/datasets/mordehay/Result/Separation_GN_withoutDF_whamr3Sec_drop2d005_lr0001_commit/models/AV_model/0918_114233/model_best.pth"
# "/dsi/gannot-lab/datasets/mordehay/Result/Separation_data_wham_partial_overlap_MultiDial_Vad099_decay095_inputActivity_CosineAnnealing/models/AV_model/1115_135213/model_best.pth"
# "/dsi/gannot-lab/datasets/mordehay/Result/Separation_data_wham_partial_overlap_WithVadNormalizeMSE0.8/models/AV_model/1003_170317/model_best.pth"
if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default="/home/dsi/moradim/OurBaselineModels/Tasnet/config.json", type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default="/dsi/gannot-lab/datasets/mordehay/Result/ConvTasnetBaseLine/models/AV_model/1128_120624/model_best.pth",
                                    type=str, help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    config = ConfigParser.from_args(args, trainer_or_tester="tester", save_path="save_test_real")
    main(config)

