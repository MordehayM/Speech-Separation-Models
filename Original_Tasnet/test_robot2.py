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
from model.combined_loss import reorder_source_mse
import os
from model.combined_loss import scale_invariant_signal_distortion_ratio_loss
from model.combined_loss import scale_invariant_signal_distortion_ratio
import glob 
#from pesq import pesq
from scipy import signal
from model.metric import SI_SDRi
import seaborn as sns
from utils import prepare_device




class OnlineSaving:
    def __init__(self, criterion_separation, model, save_path, device, criterion_similarity=None) -> None:
        self.indx = 0
        self.fs = 16000
        self.max_len = 3
        self.save_sec = 1
        self.criterion_separation = criterion_separation
        self.model = model
        self.save_path = save_path
        self.online_sisdr = []
        self.reference_sisdr = []
        self.num_save_samples = 2000000
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
            
    def get_truncated_signal(self, full_signal_mix, target_signal):
        truncated_signal_mix = full_signal_mix[:, int(np.floor(self.fs * self.indx * self.save_sec)): int(np.floor(self.fs * self.indx * self.save_sec)) + self.max_len * self.fs]
        truncated_signal_target = target_signal[:, :, int(np.floor(self.fs * self.indx * self.save_sec)): int(np.floor(self.fs * self.indx * self.save_sec)) + self.max_len * self.fs]        
        return truncated_signal_mix, truncated_signal_target
            
    def increase_indx(self):
        self.indx += 1
        
    def get_indx(self):
        return self.indx
    
    def save_audio(self, name_folder, separated_signals, target, mix, si_sdr):
        target_audio1 = target[0, 0, :].cpu().detach().numpy() #sample 0 from batch
        target_audio2 = target[0, 1, :].cpu().detach().numpy() #sample 0 from batch
        separated_audio1 = separated_signals[0, 0, :].cpu().detach().numpy() #sample 0 from batch
        separated_audio2 = separated_signals[0, 1, :].cpu().detach().numpy() #sample 0 from batch
        mix_waves = mix[0, :].cpu().detach().numpy()  #sample 0 from batch and 0 from channel
        Path(f"{self.save_path}/{name_folder}/indx_{self.indx}_sisdr_{si_sdr:.2f}").mkdir(parents=True, exist_ok=True)
        write(f"{self.save_path}/{name_folder}/indx_{self.indx}_sisdr_{si_sdr:.2f}/mixed.wav", self.fs, mix_waves.astype(np.float32))
        write(f"{self.save_path}/{name_folder}/indx_{self.indx}_sisdr_{si_sdr:.2f}/output_0.wav", self.fs, separated_audio1.astype(np.float32))
        write(f"{self.save_path}/{name_folder}/indx_{self.indx}_sisdr_{si_sdr:.2f}/output_1.wav", self.fs, separated_audio2.astype(np.float32))
        write(f"{self.save_path}/{name_folder}/indx_{self.indx}_sisdr_{si_sdr:.2f}/clean_0.wav", self.fs, target_audio1.astype(np.float32))
        write(f"{self.save_path}/{name_folder}/indx_{self.indx}_sisdr_{si_sdr:.2f}/clean_1.wav", self.fs, target_audio2.astype(np.float32))
        
        
    def save_last_online_audio(self, name_folder, online_signal, true_truncated_signal, online_sisdr, ref_sisdr, mixed_signal_t):
        true_truncated_signal = true_truncated_signal[0, :, :].cpu().detach().numpy() #sample 0 from batch
        #online_signal = online_signal[0, :, :].cpu().detach().numpy() #sample 0 from batch
        mixed_signal_t = mixed_signal_t[0, :].cpu().detach().numpy() #sample 0 from batch
        Path(f"{self.save_path}/{name_folder}").mkdir(parents=True, exist_ok=True)
        #save online signal
        asr_bool = True
        if asr_bool:
            f = open(f"{self.save_path}/{name_folder}/The Si-Sdr is {online_sisdr:.2f}.txt", "w")
            online_signal = 2*(online_signal - torch.min(online_signal, dim=-1, keepdim=True)[0]) / (torch.max(online_signal, dim=-1, keepdim=True)[0] - torch.min(online_signal, dim=-1, keepdim=True)[0]) - 1
            online_signal = online_signal[0, :, :].cpu().detach().numpy() #sample 0 from batch
            write(f"{self.save_path}/{name_folder}/online_signal0.wav", self.fs, online_signal[0].astype(np.float32))
            write(f"{self.save_path}/{name_folder}/online_signal1.wav", self.fs, online_signal[1].astype(np.float32))
            write(f"{self.save_path}/{name_folder}/ref_mix.wav", self.fs, mixed_signal_t.astype(np.float32))
        else:
            write(f"{self.save_path}/{name_folder}/online_signal0_onlineSisdr_{online_sisdr:.2f}_refSisdr_{ref_sisdr:.2f}.wav", self.fs, online_signal[0].astype(np.float32))
            write(f"{self.save_path}/{name_folder}/online_signal1_onlineSisdr_{online_sisdr:.2f}_refSisdr_{ref_sisdr:.2f}.wav", self.fs, online_signal[1].astype(np.float32))
            #save true signal
            write(f"{self.save_path}/{name_folder}/true_signal0_onlineSisdr_{online_sisdr:.2f}_refSisdr_{ref_sisdr:.2f}.wav", self.fs, true_truncated_signal[0].astype(np.float32))
            write(f"{self.save_path}/{name_folder}/true_signal1_onlineSisdr_{online_sisdr:.2f}_refSisdr_{ref_sisdr:.2f}.wav", self.fs, true_truncated_signal[1].astype(np.float32))
            write(f"{self.save_path}/{name_folder}/ref_mix.wav", self.fs, mixed_signal_t.astype(np.float32))
        
    def calc_online(self, full_signal_mix, target_signal, name_folder, sample_indx):
        if full_signal_mix.shape[-1] < self.fs * self.max_len:
            full_signal_mix = torch.nn.functional.pad(full_signal_mix, (0, self.fs * self.max_len - full_signal_mix.shape[-1]))
            target_signal = torch.nn.functional.pad(target_signal, (0, self.fs * self.max_len - target_signal.shape[-1]))
        #print(full_signal_mix.shape[-1])
        #print(full_signal_mix.shape)
        #print(target_signal.shape)
        full_signal_mix = torch.nn.functional.pad(full_signal_mix, (int(self.fs * (self.max_len - self.save_sec)), 0))
        target_signal = torch.nn.functional.pad(target_signal, (int(self.fs * (self.max_len - self.save_sec)), 0))
        pad_zero = (full_signal_mix.shape[-1] - self.fs * self.max_len) % int((self.fs * self.save_sec))
        if pad_zero:  
            full_signal_mix = torch.nn.functional.pad(full_signal_mix, (0, pad_zero))
            target_signal = torch.nn.functional.pad(target_signal, (0, pad_zero))
        max_indx = np.floor(((full_signal_mix.shape[-1] - self.fs * self.max_len) / (self.fs * self.save_sec)))
        #print(max_indx)
        #self.online_signal = target_signal[:, :, self.max_len * self.fs - int(np.floor(self.fs * 2 * self.save_sec)): self.max_len * self.fs - int(np.floor(self.fs * 1 * self.save_sec))] ######maybe i dont need this
        while self.indx <= max_indx:
            #print("INNNNN")
            truncated_signal_mix, truncated_signal_target = self.get_truncated_signal(full_signal_mix, target_signal)
            with torch.no_grad():
                pred_separation = self.model(truncated_signal_mix)
                #pred_separation = pred_separation / pred_separation.max(dim=-1, keepdim=True)[0] ## ADDED
            separation_loss, batch_indices_separation = self.criterion_separation(pred_separation, truncated_signal_target,
                                                                                return_incides=True)
            if self.similarity:
                if self.indx == 0:
                #self.online_signal = pred_separation[:, :, pred_separation.shape[-1] - int(np.floor(self.fs*self.save_sec)):]
                    self.update_online_signal(pred_separation)
                pred_separation_sim = pred_separation[:, :, - int(np.floor(self.fs*self.save_sec)) - self.online_signal.shape[-1]: - int(np.floor(self.fs*self.save_sec))]
                truncated_onlinet_sim = self.online_signal[:, :, - self.fs * self.max_len + int(np.floor(self.fs*self.save_sec)): ]
                #print(f"the pred is: {pred_separation_sim.shape}")
                #print(f"the online is: {truncated_onlinet_sim.shape}")
                _, batch_indices_separation = self.criterion_similarity(pred_separation_sim, truncated_onlinet_sim,
                                                                                return_incides=True)
            pred_separation = reorder_source_mse(pred_separation, batch_indices_separation)
            self.update_online_signal(pred_separation)
            if sample_indx < self.num_save_samples:
                pass ######
                #self.save_audio(name_folder, pred_separation, truncated_signal_target, truncated_signal_mix, -separation_loss)
            self.increase_indx()
        
        ##get refernce score
        pred_separation = self.model(full_signal_mix)
        #pred_separation = pred_separation / pred_separation.max(dim=-1, keepdim=True)[0] ## ADDED
        separation_loss, batch_indices_separation = self.criterion_separation(pred_separation, target_signal,
                                                                                return_incides=True)
        pred_separation = reorder_source_mse(pred_separation, batch_indices_separation)
        pred_separation_t = pred_separation[:, :, int(np.floor(self.fs * (self.max_len - self.save_sec))): int(np.floor(self.fs * (self.max_len + (self.indx - 1) * self.save_sec)))]
        target_signal_t = target_signal[:, :, int(np.floor(self.fs * (self.max_len - self.save_sec))): int(np.floor(self.fs * (self.max_len + (self.indx - 1) * self.save_sec)))]
        #pred_separation_t = pred_separation_t[:, :, 6*self.fs:] ##test
        #target_signal_t = target_signal_t[:, :, 6*self.fs:] ##test
        ref_pred_and_true_signal_sisdr = torch.mean(scale_invariant_signal_distortion_ratio(pred_separation_t, target_signal_t)).item()
        self.reference_sisdr.append(ref_pred_and_true_signal_sisdr)
        
        ##get online score
        true_truncated_signal = target_signal[:, :, int(np.floor(self.fs * (self.max_len - self.save_sec))): int(np.floor(self.fs * (self.max_len + (self.indx - 1) * self.save_sec)))]
        #true_truncated_signal = true_truncated_signal[:, :, 6*self.fs:] ##test
        #self.online_signal = self.online_signal[:, :, 6*self.fs:] ##test
        _, batch_indices_separation = self.criterion_separation(self.online_signal, true_truncated_signal,
                                                                                return_incides=True)
        self.online_signal = reorder_source_mse(self.online_signal, batch_indices_separation)
        online_pred_and_true_signal_sisdr = torch.mean(scale_invariant_signal_distortion_ratio(self.online_signal, true_truncated_signal)).item()
        self.online_sisdr.append(online_pred_and_true_signal_sisdr)
        mixed_signal_t = full_signal_mix[:, int(np.floor(self.fs * (self.max_len - self.save_sec))): int(np.floor(self.fs * (self.max_len + (self.indx - 1) * self.save_sec)))]
        #mixed_signal_t = mixed_signal_t[:, 6*self.fs:] ##test
        if sample_indx < self.num_save_samples:
            #pass ###########
            self.save_last_online_audio(name_folder, self.online_signal, true_truncated_signal, online_pred_and_true_signal_sisdr, ref_pred_and_true_signal_sisdr, mixed_signal_t)
        
        
        
        self.reset()



def plot_spectrogram(masks, title, save_path, ylabel='freq_bin', aspect='auto', xmax=None):
    Path(save_path).mkdir(parents=True, exist_ok=True)
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
        #Path(f"{save_path}").mkdir(parents=True, exist_ok=True)
        plt.savefig(f"{save_path}Mask_Speaker_{indx_mask+1}")
        plt.close('all')
        
def save_audio(mix_waves, separated_signals, target_speeches, save_path, samplerate, bit16, si_sdr):
    Path(save_path).mkdir(parents=True, exist_ok=True)
    f = open(f"{save_path}The Si-Sdr is {si_sdr:.2f}.txt", "w")
    separated_audio1 = separated_signals[0, 0, :].cpu().detach().numpy() #sample 0 from batch
    separated_audio2 = separated_signals[0, 1, :].cpu().detach().numpy() #sample 0 from batch
    target_audio1 = target_speeches[0, 0, :].cpu().detach().numpy() #sample 0 from batch
    target_audio2 = target_speeches[0, 1, :].cpu().detach().numpy() #sample 0 from batch
    mix_waves = mix_waves[0, :].cpu().detach().numpy()  #sample 0 from batch and 0 from channel
    #Path(f"{save_path}Batch_{batch_indx}/").mkdir(parents=True, exist_ok=True)
    
    target_audio1 = 2*(target_audio1 - np.min(target_audio1)) / (np.max(target_audio1) - np.min(target_audio1)) - 1
    target_audio2 = 2*(target_audio2 - np.min(target_audio2)) / (np.max(target_audio2) - np.min(target_audio2)) - 1
    separated_audio1 = 2*(separated_audio1 - np.min(separated_audio1)) / (np.max(separated_audio1) - np.min(separated_audio1)) - 1
    separated_audio2 = 2*(separated_audio2 - np.min(separated_audio2)) / (np.max(separated_audio2) - np.min(separated_audio2)) - 1
    
    if bit16:
        write(f"{save_path}Mixed.wav", samplerate, mix_waves.astype(np.float16))
        write(f"{save_path}Speaker1.wav", samplerate, separated_audio1.astype(np.float16))
        write(f"{save_path}Speaker2.wav", samplerate, separated_audio2.astype(np.float16))
    else:
        write(f"{save_path}Mixed.wav", samplerate, mix_waves.astype(np.float32))
        write(f"{save_path}Speaker1.wav", samplerate, separated_audio1.astype(np.float32))
        write(f"{save_path}Speaker2.wav", samplerate, separated_audio2.astype(np.float32))
        write(f"{save_path}Target1.wav", samplerate, target_audio1.astype(np.float32))
        write(f"{save_path}Target2.wav", samplerate, target_audio2.astype(np.float32))


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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # build model architecture
    model = config.init_obj('arch', module_arch)
    device, device_ids = prepare_device(config['n_gpu'])
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)
    logger.info(model)

    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    checkpoint = torch.load(config.resume)#, map_location=torch.device('cpu'))
    state_dict = checkpoint['state_dict']
    save_test_path = config["tester"]["save_test_robot"]

    model.load_state_dict(state_dict, strict=True)
    func_loss_separation = getattr(module_func_loss, config["loss_separation"]["loss_func"])
    reduce = False
    if config["loss_separation"]["perm_reduce"] is not False:
        reduce = getattr(module_loss, config["loss_separation"]["perm_reduce"])
    else:
        reduce=None

    kw_separation = {"loss_func": func_loss_separation, "perm_reduce":reduce}
    criterion_separation = config.init_obj('loss_separation', module_loss, **kw_separation)

    # prepare model for testing
    model = model.to(device)
    model.eval()
    doa = 2 # for filling only, it has no meaning right now
    wav_bool = False
    two_bool = False
    m4a_bool = True
    bit16 = False
    online_bool = config["tester"]["online"]["online_bool"]
    robot_sample = False
    chirp_bool = True
    abort_offset = False
    online_add_bool = True
    fs = 16000
    max_audio_len = 101
    n_fft = 512
    cut_len_temp = int(max_audio_len * fs / n_fft)
    cut_len = cut_len_temp * n_fft
    sisdr_list = []
    sisdr_init_list = []
    sisdri_list = []
    sisdr_online_list = []
    pit_si_sdr = module_loss.PITLossWrapper(loss_func=scale_invariant_signal_distortion_ratio_loss, pit_from="pw_pt")
    si_sdri_func = SI_SDRi().to(device)
    path2samples = config["tester"]["path2samples_robot"] #### need to modify
    num_folders = len(glob.glob(path2samples + "*"))
    print(f"Th device is: {device}")
    print(f"the number of folder is {num_folders}")
    for i in tqdm(range(num_folders)):
        try:
            samplerate, target_speech1 = read(f"{path2samples}sample{i}/spk1_ros.wav")
        except:
            continue
        print(samplerate)
        target_speech1 = target_speech1[:, 1]
        samplerate, target_speech2 = read(f"{path2samples}sample{i}/spk2_ros.wav")
        target_speech2 = target_speech2[:, 1]
        #print(target_speech1.shape)
        #print(target_speech2.shape)
        target_speeches = np.expand_dims(np.stack((target_speech1, target_speech2), axis=0), axis=0)
        target_speeches = np.array(target_speeches, dtype=np.float32)
        #target_speeches = torch.from_numpy(target_speeches)
        
        samplerate, audio = read(f"{path2samples}sample{i}/mix_ros.wav")
        
        audio = audio[:, 1]
        audio = np.array(audio, dtype=np.float32)
        if chirp_bool:
            
            
            
            if abort_offset:
                target_speeches_chirp = target_speeches[0, :, :int(1.1*samplerate)]
                audio_chirp = audio[:int(1.1*samplerate)]
                correlation_mix_spk0 = signal.correlate(audio_chirp, target_speeches_chirp[0], mode="same")
                #print(audio_chirp.size, target_speeches_chirp[0].size)
                lags = signal.correlation_lags(audio_chirp.size, target_speeches_chirp[0].size, mode="same")
                plt.plot(lags, correlation_mix_spk0)
                Path(f"{save_test_path}sample{i}/").mkdir(parents=True, exist_ok=True)
                plt.savefig(f"{save_test_path}sample{i}/correlation_0.png")
                plt.close()
                lag_mix_spk0 = lags[np.argmax(correlation_mix_spk0)]
                if lag_mix_spk0 > 0:
                    target_speeche0_chirp_pad = np.pad(target_speeches_chirp[0], (lag_mix_spk0, 0), 'constant')
                    target_speeche0_chirp_pad = target_speeche0_chirp_pad[:-lag_mix_spk0]
                else:
                    target_speeche0_chirp_pad = np.pad(target_speeches_chirp[0], (0, -lag_mix_spk0), 'constant')
                    target_speeche0_chirp_pad = target_speeche0_chirp_pad[-lag_mix_spk0:]
                correlation_mix_spk1 = signal.correlate(audio_chirp, target_speeches_chirp[1], mode="same")
                lags = signal.correlation_lags(audio_chirp.size, target_speeches_chirp[0].size, mode="same")
                plt.plot(lags, correlation_mix_spk1)
                plt.savefig(f"{save_test_path}sample{i}/correlation_1.png")
                plt.close()
                lag_mix_spk1 = lags[np.argmax(correlation_mix_spk1)]
                print(lag_mix_spk0, lag_mix_spk1)
                
                target_speeches = target_speeches[:, :, int(1.1*samplerate):]
                if lag_mix_spk0 > 0:
                    target_speeche0_pad = np.pad(target_speeches[0, 0], (lag_mix_spk0, 0), 'constant')
                    target_speeche0_pad = target_speeche0_pad[:-lag_mix_spk0]
                else:
                    target_speeche0_pad = np.pad(target_speeches[0, 0], (0, -lag_mix_spk0), 'constant')
                    target_speeche0_pad = target_speeche0_pad[-lag_mix_spk0:]
                if lag_mix_spk1 > 0:
                    target_speeche1_pad = np.pad(target_speeches[0, 1], (lag_mix_spk1, 0), 'constant')
                    target_speeche1_pad = target_speeche1_pad[:-lag_mix_spk1]
                else:
                    target_speeche1_pad = np.pad(target_speeches[0, 1], (0, -lag_mix_spk1), 'constant')
                    target_speeche1_pad = target_speeche1_pad[-lag_mix_spk1:]
                #target_speeches = np.expand_dims(np.stack((target_speeche0_pad, target_speeche1_pad), axis=0), axis=0)
                audio = audio[int(1.1*samplerate):]
            else:
                target_speeches = target_speeches[:, :, int(1.27*samplerate):] #1.27
                audio = np.sum(target_speeches[0], axis=0)
        else:
            audio = np.sum(target_speeches[0], axis=0)
            
        
        
        
        if samplerate != 16000:
            resample = T.Resample(samplerate, 16000, dtype=torch.float32)
            audio = resample(torch.tensor(audio))
            #print(len(audio/samplerate))
        audio = np.float32(audio)
        if online_add_bool:
            pass
            #audio = np.pad(audio, (2 * 16000, 0), 'constant')
            #target_speeches = np.pad(target_speeches, ((0,0),(0,0),(2 * 16000, 0)), 'constant')
        target_speeches = torch.from_numpy(target_speeches)
        audio_sub = audio - np.mean(audio)
        #normalized_audio = audio_sub / np.abs(audio_sub).max()
        normalized_audio = 2*(audio - audio.min()) / (audio.max() - audio.min()) - 1

        #write("try.wav", samplerate, normalized_audio.astype(np.float32))
        normalized_audio = torch.from_numpy(normalized_audio)
        normalized_audio = torch.unsqueeze(normalized_audio, dim=0)
        normalized_audio = normalized_audio.to(device)
        #print(normalized_audio.device)
        target_speeches = target_speeches.to(device)
        if online_bool:
            if config["tester"]["online"]["l1_loss_similarity"]:
                similarity_func = torch.nn.L1Loss()
            elif config["tester"]["online"]["si_sdr_loss_similarity"]:
                similarity_func = scale_invariant_signal_distortion_ratio_loss
            criterion_similarity = module_loss.PITLossWrapper(loss_func=similarity_func, pit_from="pw_pt")
            onlinesaving = OnlineSaving(criterion_separation, model, f"{save_test_path}sample{i}/", criterion_similarity)
            name_folder = "Online/"
            onlinesaving.calc_online(normalized_audio, target_speeches, name_folder, i)
            sisdr_online = onlinesaving.online_sisdr[0]
            sisdr_online_list.append(sisdr_online)
        if online_add_bool:
            pass
            #target_speeches = target_speeches[:, :, 2*16000:]
            #normalized_audio = normalized_audio[:, 2*16000:]
        with torch.no_grad():
            out_separation = model(normalized_audio)
        minus_si_sdr, indices = pit_si_sdr(out_separation, target_speeches, return_incides=True)
        si_sdr = -minus_si_sdr.item()
        #print(minus_si_sdr)
        out_separation = reorder_source_mse(out_separation, indices)
        #print(scale_invariant_signal_distortion_ratio_loss(out_separation[0, 0], target_speeches[0, 0]))
        #print(pesq(16000, target_speeches[0, 1].cpu().detach().numpy(), out_separation[0, 1].cpu().detach().numpy(), 'wb'))
        if not online_add_bool:
            pass    
        save_audio(normalized_audio, out_separation, target_speeches, f"{save_test_path}sample{i}/", 16000, bit16, si_sdr)
        
        sisdr_list.append(si_sdr)
        sisdri = si_sdri_func(out_separation, target_speeches, normalized_audio)
        sisdri = sisdri.item()
        sisdr_init = sisdri - si_sdr
        sisdr_init_list.append(sisdr_init)
        sisdri_list.append(sisdri)
        #print(sisdri_list)
        
    
    mean_sdr = np.mean(sisdr_list)      
    print(f"the mean si_sdr is: {mean_sdr}")
    fig, axs = plt.subplots(1, 1)
    axs.hist(sisdr_list, bins="sqrt", density=False)
    axs.set_title(f"The mean sdr is {mean_sdr}")
    plt.savefig(f"{save_test_path}hist_si_sdr.png")
    plt.close()
    if online_bool:
        mean_sdr_online = np.mean(sisdr_online_list)      
        print(f"the mean online si_sdr is: {mean_sdr_online}")
        fig, axs = plt.subplots(1, 1)
        axs.hist(sisdr_online_list, bins="sqrt", density=False)
        axs.set_title(f"The mean sdr is {mean_sdr_online}")
        plt.savefig(f"{save_test_path}hist_online_si_sdr.png")
        plt.close()
    #save_csd(csd_output, 0, i, save_test_path)
    scenario_specific = list(np.arange(0, num_folders))
    df_result = pd.DataFrame(list(zip(scenario_specific, sisdr_list, sisdri_list, sisdr_init_list)),
                               columns =['scenario', 'si_sdr', 'sisdri', 'sisdr_init'])
    if online_bool:
        df_result["si_sdr_online"] = sisdr_online_list
    df_result.to_csv(save_test_path + 'results_information.csv')
    
    sns.histplot(data=df_result, x="sisdr_init", kde=True)
    sns.histplot(data=df_result, x="si_sdr", kde=True, color="purple")
    label_input = f'si_sdr_input = {round(df_result["sisdr_init"].mean(), 2)}'
    label_output = f'si_sdr_output = {round(df_result["si_sdr"].mean(), 2)}'
    plt.legend(labels=[label_input, label_output])
    plt.xlabel("si_sdr")
    plt.ylabel("Number of time")
    plt.savefig(f"{save_test_path}sisdrStart_final_kde.png")
    plt.show()
    plt.close()

# ="/dsi/gannot-lab/datasets/mordehay/Result/Separation_data_wham_partial_overlap_MultiplyDialation/models/AV_model/1024_074543/model_best.pth"
# "/dsi/gannot-lab/datasets/mordehay/Result/SeparationVadsum099_GN_withoutDF_partial_wham_16_drop2d005_lr0001_decay095_Wen/models/AV_model/0804_133309/model_best.pth"
# "/dsi/gannot-lab/datasets/mordehay/Result/Separation_data_wham_all_overlap_TrainOn3Sec/models/AV_model/1020_090116/model_best.pth"
# "/dsi/gannot-lab/datasets/mordehay/Result/Separation_GN_withoutDF_whamr3Sec_drop2d005_lr0001_commit/models/AV_model/0918_114233/model_best.pth"
# "/dsi/gannot-lab/datasets/mordehay/Result/Separation_data_wham_partial_overlap_MultiDial_Vad099_decay095_inputActivity_CosineAnnealing/models/AV_model/1115_135213/model_best.pth"
# "/dsi/gannot-lab/datasets/mordehay/Result/Separation_data_wham_partial_overlap_WithVadNormalizeMSE0.8/models/AV_model/1003_170317/model_best.pth"
# "/dsi/gannot-lab/datasets/mordehay/Result/Separation_data_wham_partial_overlap_Multidialiation_CS_07_inputActivity_CosineAnnealing/models/AV_model/1205_231029/model_best.pth"
# "/dsi/gannot-lab/datasets/mordehay/Result/Separation_data_wham_partial_overlap_mod4p1dialiation_Vad099_decay095_inputActivity__fft1024_CosineAnnealing/models/AV_model/1208_131809/model_best.pth"
# "/dsi/scratch/from_netapp/users/mordehay/Results/Separation_data_wham_partial_overlap_ModoluDial_inputActivity_AttentionBeforeSum_ResidualLN_resume/models/AV_model/0115_172306/model_best.pth"
# "/dsi/scratch/from_netapp/users/mordehay/Results/Separation_data_wham_partial_overlap_ModoluDial_inputActivity_AttentionBeforeSum_Hop32_Batch12_ResidualLN_resume/models/AV_model/0227_140315/model_best.pth"
if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default="/home/dsi/moradim/OurBaselineModels/Original_Tasnet/config.json", type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default="/dsi/gannot-lab1/datasets/mordehay/Result/ConvTasnetBaseLine/models/AV_model/1128_120624/model_best.pth",
                                    type=str, help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    config = ConfigParser.from_args(args, trainer_or_tester="tester", save_path="save_test_real")
    main(config)

