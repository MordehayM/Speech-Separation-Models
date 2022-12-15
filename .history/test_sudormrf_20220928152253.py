import numpy as np
# from speechbrain.pretrained import SepformerSeparation as separator
import torchaudio
import glob
from scipy.io.wavfile import read
from scipy.io.wavfile import write
from pathlib import PurePosixPath
from sdr import pairwise_neg_sisdr
from pit_wrapper import PITLossWrapper
import torch
import os
from pathlib import Path
import subprocess
from tqdm import tqdm
#import matplotlib.pyplot as plt
import sys
sys.path.append("/home/dsi/moradim/OurBaselineModels/")

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


def save_audio(path, name, separated_signals, target, save_path, samplerate):
    target_audio1 = target[0, 0, :].cpu().detach().numpy()  # sample 0 from batch
    target_audio2 = target[0, 1, :].cpu().detach().numpy()  # sample 0 from batch
    separated_audio1 = separated_signals[0, 0, :].cpu().detach().numpy()  # sample 0 from batch
    separated_audio2 = separated_signals[0, 1, :].cpu().detach().numpy()  # sample 0 from batch
    # mix_waves = mix_waves[0, :].cpu().detach().numpy()  #sample 0 from batch and 0 from channel
    Path(save_path).mkdir(parents=True, exist_ok=True)
    subprocess.run(["cp", path, save_path + name])
    write(f"{save_path}/output_0.wav", samplerate, separated_audio1.astype(np.float32))
    write(f"{save_path}/output_1.wav", samplerate, separated_audio2.astype(np.float32))
    write(f"{save_path}/clean_0.wav", samplerate, target_audio1.astype(np.float32))
    write(f"{save_path}/clean_1.wav", samplerate, target_audio2.astype(np.float32))

CPU=True
GPU=False


if CPU:
    device = torch.device('cpu')
    mix_path = "/dsi/gannot-lab/datasets/whamr/wav8k/min/tr/mix_both_reverb/011a0101_0.061105_401c020r_-0.061105.wav"
    sudorm_model_path = "/dsi/gannot-lab/Improved_Sudormrf_U36_Bases4096_WHAMRexclmark.pt"
    sudorm_model = torch.load(sudorm_model_path)
    sudorm_model.to(device)
    model_name = os.path.basename(sudorm_model_path)

    criterion_separation = PITLossWrapper(pairwise_neg_sisdr)
    path_data = "/dsi/gannot-lab/datasets/whamr/wav8k/max/tt/mix_both_reverb/*.wav"
    pathes_file = glob.glob(path_data)
    si_sdrs = []
    #save_path = "/dsi/gannot-lab/datasets/mordehay/Result/test_for_git/Sudormrf_Whamr_tt8kmax_anechoic_target"
    save_path = "/home/dsi/moradim/Result_sudo"

    for i, path in tqdm(enumerate(pathes_file)):
        name = PurePosixPath(path).name
        mix, sample_rate = torchaudio.load(path)
        # mix = mix.to('cpu')
        anechoic_sampled_mixture = mix.unsqueeze(0)
        input_mix_std = anechoic_sampled_mixture.std(-1, keepdim=True)
        input_mix_mean = anechoic_sampled_mixture.mean(-1, keepdim=True)
        input_mix = (anechoic_sampled_mixture - input_mix_mean) / (input_mix_std + 1e-9)
        with torch.no_grad():
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
    #device = torch.device('cpu')
    device = torch.device('cuda')
    mix_path = "/dsi/gannot-lab/datasets/whamr/wav8k/min/tr/mix_both_reverb/011a0101_0.061105_401c020r_-0.061105.wav"
    sudorm_model_path = "/dsi/gannot-lab/Improved_Sudormrf_U36_Bases4096_WHAMRexclmark.pt"
    sudorm_model = torch.load(sudorm_model_path)
    sudorm_model.to(device)
    model_name = os.path.basename(sudorm_model_path)


    criterion_separation = PITLossWrapper(pairwise_neg_sisdr)
    path_data = "/dsi/gannot-lab/datasets/whamr/wav8k/max/tt/mix_both_reverb/*.wav"
    pathes_file = glob.glob(path_data)
    si_sdrs = []
    save_path = "/dsi/gannot-lab/datasets/mordehay/Result/test_for_git/Sudormrf_Whamr_tt8kmax_anechoic_target"
    #'/home/lab/renana/PycharmProjects/sudo_rm_rf/results/'


    for i, path in tqdm(enumerate(pathes_file)):
        name = PurePosixPath(path).name
        mix, sample_rate = torchaudio.load(path)
        mix = mix.to('cuda')
        anechoic_sampled_mixture = mix.unsqueeze(0)
        input_mix_std = anechoic_sampled_mixture.std(-1, keepdim=True)
        input_mix_mean = anechoic_sampled_mixture.mean(-1, keepdim=True)
        input_mix = (anechoic_sampled_mixture - input_mix_mean) / (input_mix_std + 1e-9)

        with torch.no_grad():
            y_sudorm = sudorm_model(input_mix.to('cuda').unsqueeze(1))
        #output_mix_std = y_sudorm.std(-1, keepdim=True)
        #output_mix_mean = y_sudorm.mean(-1, keepdim=True)
        y_sudorm = (y_sudorm * input_mix_std) + input_mix_mean
        # y_sudorm = (y_sudorm * output_mix_std) - output_mix_mean
        y_sudorm = torch.squeeze(y_sudorm)
        y_sudorm1, y_sudorm2 = y_sudorm[0, :].unsqueeze(0), y_sudorm[1, :].unsqueeze(0)

        torchaudio.save('output_sudorm1.wav', y_sudorm1.detach().cpu(), 8000)
        torchaudio.save('output_sudorm2.wav', y_sudorm2.detach().cpu(), 8000)

        #est_sources = model.separate_file(path=path)  # shape = [1, T, 2]
        # est_sources = torch
        s1_path = "/dsi/gannot-lab/datasets/whamr/wav8k/max/tt/s1_anechoic/" + name
        s2_path = "/dsi/gannot-lab/datasets/whamr/wav8k/max/tt/s2_anechoic/" + name
        #est_sources = torch.permute(y_sudorm.unsqueeze(0), (0, 2, 1))
        est_sources = y_sudorm.unsqueeze(0)

        # _, s1 = read(s1_path)
        # _, s2 = read(s2_path)
        # s1 = torch.tensor(s1, device="cuda")
        # s2 = torch.tensor(s2, device="cuda")
        # target_separation = torch.vstack((s1, s2)).unsqueeze(dim=0)
        # print(target_separation.shape)
        # print(est_sources.shape)
        # separation_loss, batch_indices_separation = criterion_separation(est_sources, target_separation,
        #                                                                  return_incides=True)
        # print(-separation_loss)
        # si_sdrs.append((-separation_loss).item())
        # if i < 30:
        #     #est_sources = reorder_source_mse(est_sources, batch_indices_separation)
        #     save_path_s = save_path + f"/Sample_{i}_{-separation_loss:.3f}/"
        #     save_audio(path, name, est_sources, target_separation, save_path_s, samplerate=8000)

mean_sdr = np.mean(si_sdrs)
print(f"the mean si_sdr is: {mean_sdr}")
"""fig, axs = plt.subplots(1, 1)
axs.hist(si_sdrs, bins="sqrt", density=True)
axs.set_title(f"The mean sdr is {mean_sdr}")
plt.savefig(f"{save_path}/hist_si_sdr.png")"""





#y_sudorm = (y_sudorm * input_mix_std) + input_mix_mean
#y_sudorm = torch.squeeze(y_sudorm)
#y_sudorm1, y_sudorm2 = y_sudorm[0, :].unsqueeze(0), y_sudorm[1, :].unsqueeze(0)

#torchaudio.save('output_sudorm1.wav', y_sudorm1.detach().cpu(), 8000)
#torchaudio.save('output_sudorm2.wav', y_sudorm2.detach().cpu(), 8000)



