from speechbrain.pretrained import SepformerSeparation as separator
import torchaudio

model = separator.from_hparams(source="speechbrain/sepformer-whamr", savedir='pretrained_models/sepformer-whamr')

# for custom file, change path
est_sources = model.separate_file(path="/dsi/gannot-lab/datasets/whamr/wav8k/max/tr/mix_both_reverb/011a0101_0.061105_401c020r_-0.061105.wav")

torchaudio.save("source1hat.wav", est_sources[:, :, 0].detach().cpu(), 8000)
torchaudio.save("source2hat.wav", est_sources[:, :, 1].detach().cpu(), 8000)