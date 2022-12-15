from speechbrain.pretrained import SepformerSeparation as separator
import torchaudio
import glob
from scipy.io.wavfile import read
from pathlib import PurePosixPath
from sdr import pairwise_neg_sisdr
from pit_wrapper import PITLossWrapper
import torch

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

criterion_separation = PITLossWrapper(pairwise_neg_sisdr)

separation_loss, batch_indices_separation = criterion_separation(pred_separation, target_separation,
                                                                              return_incides=True)
path_data = "/dsi/gannot-lab/datasets/whamr/wav16k/max/tt/mix_both_reverb/"
pathes_file = glob.glob(path_data)

for i, path in enumerate(pathes_file):
    
    _, mix = read(path)
    
    name = PurePosixPath(path).name
    s1_path = "/dsi/gannot-lab/datasets/whamr/wav16k/max/tt/s1_reverb/" + name
    s2_path = "/dsi/gannot-lab/datasets/whamr/wav16k/max/tt/s2_reverb/" + name
    _, s1 = read(s1_path)
    _, s2 = read(s2_path)

    model = separator.from_hparams(source="speechbrain/sepformer-whamr", savedir='pretrained_models/sepformer-whamr')

    # for custom file, change path
    est_sources = model.separate_file(path="/dsi/gannot-lab/datasets/whamr/wav8k/max/tr/mix_both_reverb/20no010n_0.67178_407a0107_-0.67178.wav")

    torchaudio.save("source1hat.wav", est_sources[:, :, 0].detach().cpu(), 8000)
    torchaudio.save("source2hat.wav", est_sources[:, :, 1].detach().cpu(), 8000)