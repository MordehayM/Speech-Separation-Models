import numpy as np
from speechbrain.pretrained import SepformerSeparation as separator
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
import matplotlib.pyplot as plt
import pickle 
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

def save_audio(noisy_signal, separated_signals, target, save_path):
    target_audio1 = target[0, 0, :].cpu().detach().numpy() #sample 0 from batch
    target_audio2 = target[0, 1, :].cpu().detach().numpy() #sample 0 from batch
    separated_audio1 = separated_signals[0, 0, :].cpu().detach().numpy() #sample 0 from batch
    separated_audio2 = separated_signals[0, 1, :].cpu().detach().numpy() #sample 0 from batch
    #mix_waves = mix_waves[0, :].cpu().detach().numpy()  #sample 0 from batch and 0 from channel
    Path(save_path).mkdir(parents=True, exist_ok=True)
    #subprocess.run(["cp", path, save_path+name])
    write(f"{save_path}/output_0.wav", 8000, separated_audio1.astype(np.float32))
    write(f"{save_path}/output_1.wav", 8000, separated_audio2.astype(np.float32))
    write(f"{save_path}/clean_0.wav", 16000, target_audio1.astype(np.float32))
    write(f"{save_path}/clean_1.wav", 16000, target_audio2.astype(np.float32))
    write(f"{save_path}/mix.wav", 16000, noisy_signal.astype(np.float32))

model = separator.from_hparams(source="speechbrain/sepformer-whamr", savedir='pretrained_models/sepformer-whamr', run_opts={"device":"cuda"})
criterion_separation = PITLossWrapper(pairwise_neg_sisdr)
path_data = "/dsi/gannot-lab/datasets/mordehay/data_wham_partial_overlap/test/with_wham_noise_audio/*.p" #"/dsi/gannot-lab/datasets/whamr/wav8k/max/tt/mix_both_reverb/*.wav"
pathes_file = glob.glob(path_data)
si_sdrs = []
save_path = "/dsi/gannot-lab/datasets/mordehay/Result/test_for_git/Sepformer_our_data_reverb_target"
Path(save_path).mkdir(parents=True, exist_ok=True)
for i, path in tqdm(enumerate(pathes_file)):
    with open(path, "rb") as f:
            _, noisy_signal, _, speakers_target, _ = pickle.load(f)
            path_temp = save_path + "/mix.wav"
            write(path_temp, 16000, noisy_signal.astype(np.float32))
    
    est_sources = model.separate_file(path=f"{save_path}/mix.wav") #shape = [1, T, 2]
    est_sources = torch.permute(est_sources, (0, 2, 1))
    

    
    
    speakers_target = torch.tensor(speakers_target, device="cuda").unsqueeze(dim=0)
    print(speakers_target.shape)
    print(est_sources.shape)
    separation_loss, batch_indices_separation = criterion_separation(est_sources, speakers_target,
                                                                              return_incides=True)
    si_sdrs.append((-separation_loss).item())
    if i < 30:
        est_sources = reorder_source_mse(est_sources, batch_indices_separation)
        save_path_s = save_path + f"/Sample_{i}_{-separation_loss:.3f}/"
        save_audio(noisy_signal ,est_sources, speakers_target, save_path_s)
        
os.remove("/home/dsi/moradim/OurBaselineModels/mix.wav")
os.remove(path_temp)         
mean_sdr = np.mean(si_sdrs)      
print(f"the mean si_sdr is: {mean_sdr}")
fig, axs = plt.subplots(1, 1)
axs.hist(si_sdrs, bins="sqrt", density=True)
axs.set_title(f"The mean sdr is {mean_sdr}")
plt.savefig(f"{save_path}/hist_si_sdr.png")
