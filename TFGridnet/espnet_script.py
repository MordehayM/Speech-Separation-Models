import os
import librosa
import soundfile
import numpy as np
import torch
import torchaudio
from espnet_model_zoo.downloader import ModelDownloader
from espnet2.bin.enh_inference import SeparateSpeech
from pathlib import Path

#tag = "https://zenodo.org/records/8433041"
tag = "https://zenodo.org/records/8433041/files/enh_train_enh_tfgridnetv2_tf_lr-patience3_patience5_I_1_J_1_D_128_raw_valid.loss.ave.zip?download=1"
#tag = "espnet/yoshiki_wsj0_2mix_spatialized_enh_tfgridnet_waspaa2023_raw"
#tag = "lichenda/wsj0_2mix_skim_noncausal"
print(Path(__file__).parent)
d = ModelDownloader()
cfg = d.download_and_unpack(tag)

separate_speech = SeparateSpeech(
    train_config=cfg["train_config"],
    model_file=cfg["model_file"],
    normalize_segment_scale=False,
    normalize_output_wav=True,
    device="cuda:0",
)

def espnet_main(input_filepath: str, output_dir: str):

    mixwav, sr = soundfile.read(input_filepath)
    #mixwav /= np.std(mixwav)
    name_file = Path(input_filepath).stem

    # While the algorithm can handle any sample rate, performance is best at 8kHz.
    # From limited testing, it is best to resample.
    if sr != 8000:
        mixwav = librosa.resample(mixwav, orig_sr=sr, target_sr=8000)
        sr = 8000

    waves_wsj = separate_speech(mixwav[None, ...], fs=sr)

    spk1 = waves_wsj[0] #.squeeze()
    spk2 = waves_wsj[1]

    spk1_torch = torch.from_numpy(spk1)
    spk2_torch = torch.from_numpy(spk2)

    os.makedirs(output_dir, exist_ok=True)
    torchaudio.save(os.path.join(output_dir, f"{name_file}_output_spk1.wav"), spk1_torch, sr)
    torchaudio.save(os.path.join(output_dir, f"{name_file}_output_spk2.wav"), spk2_torch, sr)

    return


if __name__ == "__main__":

    input_file = "/home/dsi/moradim/OurBaselineModels/050a0502_1.9707_440c020w_-1.9707.mix.wav"
    dest_dir = "/home/dsi/moradim/OurBaselineModels/"
    espnet_main(input_file, dest_dir)