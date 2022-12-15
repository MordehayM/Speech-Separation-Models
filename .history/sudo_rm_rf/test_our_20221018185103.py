import numpy as np
# from speechbrain.pretrained import SepformerSeparation as separator
import torchaudio
import glob
from scipy.io.wavfile import read
from scipy.io.wavfile import write
from pathlib import PurePosixPath
import torch
import os
from pathlib import Path
import subprocess
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle
import sys
sys.path.append("/home/dsi/moradim/OurBaselineModels/sudo_rm_rf/")
from sudo_rm_rf.utils.sdr import pairwise_neg_sisdr
from sudo_rm_rf.utils.pit_wrapper import PITLossWrapper
import sudo_rm_rf.dnn.experiments.utils.improved_cmd_args_parser_v2 as parser
import sudo_rm_rf.dnn.models.improved_sudormrf as improved_sudormrf
import sudo_rm_rf.dnn.models.sudormrf as initial_sudormrf
import logging



save_path = '/dsi/gannot-lab/datasets/mordehay/Result/test_for_git/Test_Sudo_data_partial_overlap_wham_sudo1/'
###########Logger
logger = logging.getLogger(__name__)
    # set log level
logger.setLevel(logging.INFO)

# define file handler and set formatter
file_handler = logging.FileHandler(save_path + 'logfile.log', mode='w')  # mode='w'
formatter = logging.Formatter('%(asctime)s : %(levelname)s : %(name)s : %(message)s')
file_handler.setFormatter(formatter)
# add file handler to logger
logger.addHandler(file_handler)
consoleHandler = logging.StreamHandler(sys.stdout)
consoleHandler.setFormatter(formatter)

logger.addHandler(consoleHandler)

"""consoleHandlererr = logging.StreamHandler(sys.stderr)
consoleHandlererr.setFormatter(formatter)

logger.addHandler(consoleHandlererr)"""
class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        #Path(filename).mkdir(parents=True, exist_ok=True)
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        
sys.stdout = Logger(save_path + 'logfile_all.log')
args = parser.get_args()
hparams = vars(args)



def reorder_source_mse(preds, batch_indices):
    r"""Reorder sources according to the best permutation.

    Args:
        preds (torch.Tensor): Tensor of shape :math:`[B, num_spk, num_frames]`
        batch_indices (torch.Tensor): Tensor of shape :math:`(batch, n_src)`.
            Contains optimal permutation indices for each batch.

    Returns:
        :class:`torch.Tensor`: Reordered sources.
    """
    reordered_sources = torch.stack(
        [torch.index_select(s, 0, b) for s, b in zip(preds, batch_indices)]
    )
    return reordered_sources


def save_audio(path, name, separated_signals, target, save_path,y_sudorm1, y_sudorm2, samplerate):
    target_audio1 = target[0, 0, :].cpu().detach().numpy()  # sample 0 from batch
    target_audio2 = target[0, 1, :].cpu().detach().numpy()  # sample 0 from batch
    separated_audio1 = separated_signals[0, 0, :].cpu().detach().numpy()  # sample 0 from batch
    separated_audio2 = separated_signals[0, 1, :].cpu().detach().numpy()  # sample 0 from batch
    # mix_waves = mix_waves[0, :].cpu().detach().numpy()  #sample 0 from batch and 0 from channel
    Path(save_path).mkdir(parents=True, exist_ok=True)
    subprocess.run(["cp", path, save_path + name])
    write(f"{save_path}/output_0.wav", samplerate, separated_audio1.astype(np.float32))
    write(f"{save_path}/output_1.wav", samplerate, separated_audio2.astype(np.float32))
    #torchaudio.save(f"{save_path}/output_0.wav", y_sudorm1.detach().cpu(), 8000)
    #torchaudio.save(f"{save_path}/output_1.wav", y_sudorm2.detach().cpu(), 8000)
    #write(f"{save_path}/output_0.wav", samplerate, y_sudorm1)
    #write(f"{save_path}/output_1.wav", samplerate, y_sudorm2)
    write(f"{save_path}/clean_0.wav", samplerate, target_audio1.astype(np.float32))
    write(f"{save_path}/clean_1.wav", samplerate, target_audio2.astype(np.float32))

CPU = False #True
GPU = True #False


if CPU:
    device = torch.device('cpu')
    mix_path = "/dsi/gannot-lab/datasets/whamr/wav8k/min/tr/mix_both_reverb/011a0101_0.061105_401c020r_-0.061105.wav"
    sudorm_model_path = "/dsi/gannot-lab/Improved_Sudormrf_U36_Bases4096_WHAMRexclmark.pt"
    sudorm_model = torch.load(sudorm_model_path)
    sudorm_model.to(device)
    model_name = os.path.basename(sudorm_model_path)

    criterion_separation = PITLossWrapper(pairwise_neg_sisdr)
    resample = torchaudio.transforms.Resample(orig_freq=16000, new_freq=8000).to("cpu")
    path_data = "/dsi/gannot-lab/datasets/mordehay/data_wham_partial_overlap/test/with_wham_noise_audio/*.p"  # "/dsi/gannot-lab/datasets/whamr/wav8k/max/tt/mix_both_reverb/*.wav"
    #path_data = "/dsi/gannot-lab/datasets/whamr/wav8k/max/tt/mix_both_reverb/*.wav"
    pathes_file = glob.glob(path_data)
    si_sdrs = []
    #save_path = "/dsi/gannot-lab/datasets/mordehay/Result/test_for_git/Sudormrf_Whamr_tt8kmax_anechoic_target"
    save_path = '/dsi/gannot-lab/datasets/mordehay/Result/test_for_git/Test_Sudo_data_partial_overlap_wham_sudo1'
    Path(save_path).mkdir(parents=True, exist_ok=True)
    for i, path in tqdm(enumerate(pathes_file[:3000])):
        with open(path, "rb") as f:
            _, noisy_signal, _, speakers_target, _ = pickle.load(f)
            speakers_target = torch.tensor(speakers_target, dtype=torch.float32)
            path_temp = save_path + "/mix.wav"
            write(path_temp, 16000, noisy_signal.astype(np.float32))

    for i, path in tqdm(enumerate(pathes_file)):
        name = PurePosixPath(path).name
        mix, sample_rate = torchaudio.load(path)
        # mix = mix.to('cpu')
        anechoic_sampled_mixture = mix.unsqueeze(0)
        input_mix_std = anechoic_sampled_mixture.std(-1, keepdim=True)
        input_mix_mean = anechoic_sampled_mixture.mean(-1, keepdim=True)
        input_mix = (anechoic_sampled_mixture - input_mix_mean) / (input_mix_std + 1e-9)
        input_mix = resample(input_mix)
        y_sudorm = sudorm_model(input_mix)
        y_sudorm = (y_sudorm * input_mix_std) + input_mix_mean
        y_sudorm = torch.squeeze(y_sudorm)
        y_sudorm1, y_sudorm2 = y_sudorm[0, :].unsqueeze(0), y_sudorm[1, :].unsqueeze(0)

        torchaudio.save('output_sudorm1.wav', y_sudorm1.detach().cpu(), 8000)
        torchaudio.save('output_sudorm2.wav', y_sudorm2.detach().cpu(), 8000)

        # est_sources = model.separate_file(path=path)  # shape = [1, T, 2]
        # est_sources = torch
        s1_path = "/dsi/gannot-lab/datasets/whamr/wav8k/max/tt/s1_anechoic/" + name
        s2_path = "/dsi/gannot-lab/datasets/whamr/wav8k/max/tt/s2_anechoic/" + name
        # est_sources = torch.permute(y_sudorm.unsqueeze(0), (0, 2, 1))
        est_sources = y_sudorm.unsqueeze(0)

        _, s1 = read(s1_path)
        _, s2 = read(s2_path)
        s1 = torch.tensor(s1, device="cpu")
        s2 = torch.tensor(s2, device="cpu")
        target_separation = torch.vstack((s1, s2)).unsqueeze(dim=0)
        print(target_separation.shape)
        print(est_sources.shape)
        separation_loss, batch_indices_separation = criterion_separation(est_sources, target_separation,
                                                                          return_incides=True)
        print(-separation_loss)
        si_sdrs.append((-separation_loss).item())
        if i < 30:
            #est_sources = reorder_source_mse(est_sources, batch_indices_separation)
            save_path_s = save_path + f"/Sample_{i}_{-separation_loss:.3f}/"
            save_audio(path, name, est_sources, target_separation, save_path_s, samplerate=8000)

if GPU:
    
    if hparams['model_type'] == 'relu':
        sudorm_model = improved_sudormrf.SuDORMRF(out_channels=hparams['out_channels'],
                                        in_channels=hparams['in_channels'],
                                        num_blocks=hparams['num_blocks'],
                                        upsampling_depth=hparams[
                                            'upsampling_depth'],
                                        enc_kernel_size=hparams[
                                            'enc_kernel_size'],
                                        enc_num_basis=hparams['enc_num_basis'],
                                        num_sources=hparams['max_num_sources'])
    elif hparams['model_type'] == 'softmax':
        sudorm_model = initial_sudormrf.SuDORMRF(out_channels=hparams['out_channels'],
                                        in_channels=hparams['in_channels'],
                                        num_blocks=hparams['num_blocks'],
                                        upsampling_depth=hparams[
                                            'upsampling_depth'],
                                        enc_kernel_size=hparams[
                                            'enc_kernel_size'],
                                        enc_num_basis=hparams['enc_num_basis'],
                                        num_sources=hparams['max_num_sources'])
    
    #device = torch.device('cpu')
    #sudorm_model = torch.nn.DataParallel(sudorm_model)
    device = torch.device('cuda:0')
    # mix_path = "/dsi/gannot-lab/datasets/whamr/wav8k/min/tr/mix_both_reverb/011a0101_0.061105_401c020r_-0.061105.wav"
    sudorm_model_path = "/home/dsi/moradim/sudo/checkpoints/improved_sudo_epoch_30"
    sudorm_model_state_dict = torch.load(sudorm_model_path)
    #print(sudorm_model_state_dict)
    sudorm_model.load_state_dict(sudorm_model_state_dict)
    sudorm_model.to(device)
    model_name = os.path.basename(sudorm_model_path)


    criterion_separation = PITLossWrapper(pairwise_neg_sisdr)
    resample = torchaudio.transforms.Resample(orig_freq=16000, new_freq=8000).to("cuda:0")
    path_data = "/dsi/gannot-lab/datasets/mordehay/data_wham_partial_overlap/train/with_wham_noise_audio/*.p"
    # "/dsi/gannot-lab/datasets/whamr/wav8k/max/tt/mix_both_reverb/*.wav"
    #path_data = "/dsi/gannot-lab/datasets/whamr/wav8k/min/tt/mix_both_reverb/*.wav"
    pathes_file = glob.glob(path_data)
    si_sdrs = []
    Path(save_path).mkdir(parents=True, exist_ok=True)

    # for i, path in tqdm(enumerate(pathes_file)):
    for i, path in tqdm(enumerate(pathes_file[:])):
        name = PurePosixPath(path).name
        with open(path, "rb") as f:
            # for our data
            _, mix, _, speakers_target, _ = pickle.load(f)
            speakers_target = torch.tensor(speakers_target, dtype=torch.float32)
            path_temp = save_path + "/mix.wav"
            write(path_temp, 16000, mix.astype(np.float32))

        # for whamr data
        '''
        name = PurePosixPath(path).name
        mix, sample_rate = torchaudio.load(path)
        mix = mix.to('cuda')
        '''

        mix = torch.tensor(mix, dtype=torch.float32)
        mix.to('cuda:0')
        anechoic_sampled_mixture = mix.unsqueeze(0)
        input_mix_std = anechoic_sampled_mixture.std(-1, keepdim=True)
        input_mix_mean = anechoic_sampled_mixture.mean(-1, keepdim=True)
        input_mix = (anechoic_sampled_mixture - input_mix_mean) / (input_mix_std + 1e-9)
        input_mix = input_mix.unsqueeze(0)
        #input_mix = torch.vstack((input_mix, input_mix)).unsqueeze(dim=0)
        #print(input_mix.shape)
        sudorm_model.eval()
        with torch.no_grad():
            y_sudorm = sudorm_model(input_mix.to('cuda:0'))
        #output_mix_std = y_sudorm.std(-1, keepdim=True)
        #output_mix_mean = y_sudorm.mean(-1, keepdim=True)

        y_sudorm = (y_sudorm.to('cuda:0') * input_mix_std.to('cuda:0')) + input_mix_mean.to('cuda:0')
        # y_sudorm = (y_sudorm * output_mix_std) - output_mix_mean
        y_sudorm = torch.squeeze(y_sudorm)
        y_sudorm1, y_sudorm2 = y_sudorm[0, :].unsqueeze(0), y_sudorm[1, :].unsqueeze(0)

        #torchaudio.save('output_sudorm1.wav', y_sudorm1.detach().cpu(), 8000)
        #torchaudio.save('output_sudorm2.wav', y_sudorm2.detach().cpu(), 8000)

        #est_sources = model.separate_file(path=path)  # shape = [1, T, 2]
        # est_sources = torch
        # est_sources = torch.permute(y_sudorm.unsqueeze(0), (0, 2, 1))
        est_sources = y_sudorm.unsqueeze(0)

        #for whamr
        '''
        s1_path = "/dsi/gannot-lab/datasets/whamr/wav8k/min/tt/s1_anechoic/" + name
        s2_path = "/dsi/gannot-lab/datasets/whamr/wav8k/min/tt/s2_anechoic/" + name
        _, s1 = read(s1_path)
        _, s2 = read(s2_path)
        s1 = torch.tensor(s1, device="cuda:0")
        s2 = torch.tensor(s2, device="cuda:0")
        target_separation = torch.vstack((s1, s2)).unsqueeze(dim=0)
        '''
        # our data
        target_separation = torch.tensor(speakers_target, device="cuda:0")
        target_separation = target_separation.unsqueeze(dim=0)
        #print(target_separation.shape)
        #print(est_sources.shape)
        separation_loss, batch_indices_separation = criterion_separation(est_sources, target_separation,
                                                                          return_incides=True)
        #print(-separation_loss)
        si_sdrs.append((-separation_loss).item())
        if i < 5:
            #est_sources = reorder_source_mse(est_sources, batch_indices_separation)
            save_path_s = save_path + f"/Sample_{i}_{-separation_loss:.3f}/"
            save_audio(path, name, est_sources, target_separation, save_path_s,y_sudorm1, y_sudorm2, samplerate=8000)

mean_sdr = np.mean(si_sdrs)
print(f"the mean si_sdr is: {mean_sdr}")
fig, axs = plt.subplots(1, 1)
axs.hist(si_sdrs, bins="sqrt", density=True)
axs.set_title(f"The mean sdr is {mean_sdr}")
plt.savefig(f"{save_path}/hist_si_sdr.png")





#y_sudorm = (y_sudorm * input_mix_std) + input_mix_mean
#y_sudorm = torch.squeeze(y_sudorm)
#y_sudorm1, y_sudorm2 = y_sudorm[0, :].unsqueeze(0), y_sudorm[1, :].unsqueeze(0)

#torchaudio.save('output_sudorm1.wav', y_sudorm1.detach().cpu(), 8000)
#torchaudio.save('output_sudorm2.wav', y_sudorm2.detach().cpu(), 8000)



