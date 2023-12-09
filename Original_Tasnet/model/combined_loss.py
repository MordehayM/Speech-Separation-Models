from turtle import forward
from torch import nn
import torch

from torchaudio import transforms
from torch import Tensor

from torch.nn import MSELoss
import numpy as np
import torch.nn.functional as F
from functools import partial

#from speechbrain.pretrained import EncoderClassifier


def cosine_similarity(x, y):
    cs =  torch.sum(x * y, dim=-1) / (torch.sqrt(torch.sum(torch.pow(x, 2), dim=-1)) * torch.sqrt(torch.sum(torch.pow(y, 2), dim=-1)))
    return torch.mean(torch.exp(2 * cs)) #mean over the batch

def cosine_similarity_target_limit(x, y, x_target, y_target):
    cs =  torch.sum(x * y, dim=-1) / (torch.sqrt(torch.sum(torch.pow(x, 2), dim=-1)) * torch.sqrt(torch.sum(torch.pow(y, 2), dim=-1)))
    cs_target =  torch.sum(x_target * y_target, dim=-1) / (torch.sqrt(torch.sum(torch.pow(x_target, 2), dim=-1)) * torch.sqrt(torch.sum(torch.pow(y_target, 2), dim=-1)))
    cs_limit = torch.maximum(cs, cs_target) #limit the cs by the target cs
    return torch.mean(torch.exp(2 * cs_limit)) #mean over the batch

def _check_same_shape(preds: Tensor, target: Tensor) -> None:
    """Check that predictions and target have the same shape, else raise error."""
    if preds.shape != target.shape:
        raise RuntimeError(f"Predictions and targets are expected to have the same shape, pred has shape of {preds.shape} and target has shape of {target.shape}")

def scale_invariant_signal_distortion_ratio(preds: Tensor, target: Tensor, zero_mean: bool = True) -> Tensor:
    """Calculates Scale-invariant signal-to-distortion ratio (SI-SDR) metric. The SI-SDR value is in general
    considered an overall measure of how good a source sound.
    Args:
        preds:
            shape ``[...,time]``
        target:
            shape ``[...,time]``
        zero_mean:
            If to zero mean target and preds or not
    Returns:
        si-sdr value of shape [...]
    Example:
        #>>> from torchmetrics.functional.audio import scale_invariant_signal_distortion_ratio
        #>>> target = torch.tensor([3.0, -0.5, 2.0, 7.0])
        #>>> preds = torch.tensor([2.5, 0.0, 2.0, 8.0])
        #>>> scale_invariant_signal_distortion_ratio(preds, target)
        tensor(18.4030)
    References:
        [1] Le Roux, Jonathan, et al. "SDR half-baked or well done." IEEE International Conference on Acoustics, Speech
        and Signal Processing (ICASSP) 2019.
    """
    #print(f"shape preds: {preds.shape} \nshape target: {target.shape}")
    _check_same_shape(preds, target)
    EPS = torch.finfo(preds.dtype).eps

    if zero_mean:
        target = target - torch.mean(target, dim=-1, keepdim=True)
        preds = preds - torch.mean(preds, dim=-1, keepdim=True)

    alpha = (torch.sum(preds * target, dim=-1, keepdim=True) + EPS) / (
        torch.sum(target ** 2, dim=-1, keepdim=True) + EPS
    )
    target_scaled = alpha * target

    noise = target_scaled - preds

    val = (torch.sum(target_scaled ** 2, dim=-1) + EPS) / (torch.sum(noise ** 2, dim=-1) + EPS)
    val = 10 * torch.log10(val)

    return val

def scale_invariant_signal_distortion_ratio_loss(preds: Tensor, target: Tensor, zero_mean: bool = True) -> Tensor:
    """Calculates Scale-invariant signal-to-distortion ratio (SI-SDR) metric. The SI-SDR value is in general
    considered an overall measure of how good a source sound.
    Args:
        preds:
            shape ``[...,time]``
        target:
            shape ``[...,time]``
        zero_mean:
            If to zero mean target and preds or not
    Returns:
        si-sdr value of shape [...]
    Example:
        #>>> from torchmetrics.functional.audio import scale_invariant_signal_distortion_ratio
        #>>> target = torch.tensor([3.0, -0.5, 2.0, 7.0])
        #>>> preds = torch.tensor([2.5, 0.0, 2.0, 8.0])
        #>>> scale_invariant_signal_distortion_ratio(preds, target)
        tensor(18.4030)
    References:
        [1] Le Roux, Jonathan, et al. "SDR half-baked or well done." IEEE International Conference on Acoustics, Speech
        and Signal Processing (ICASSP) 2019.
    """
    #print(f"shape preds: {preds.shape} \nshape target: {target.shape}")
    _check_same_shape(preds, target)
    EPS = torch.finfo(preds.dtype).eps

    if zero_mean:
        target = target - torch.mean(target, dim=-1, keepdim=True)
        preds = preds - torch.mean(preds, dim=-1, keepdim=True)

    alpha = (torch.sum(preds * target, dim=-1, keepdim=True) + EPS) / (
        torch.sum(target ** 2, dim=-1, keepdim=True) + EPS
    )
    target_scaled = alpha * target

    noise = target_scaled - preds

    val = (torch.sum(target_scaled ** 2, dim=-1) + EPS) / (torch.sum(noise ** 2, dim=-1) + EPS)
    val = 10 * torch.log10(val)
    
    return -val

def scale_invariant_signal_distortion_ratio_vad(preds: Tensor, target: Tensor, est_vad: Tensor, zero_mean: bool = True) -> Tensor:
    """Calculates Scale-invariant signal-to-distortion ratio (SI-SDR) metric. The SI-SDR value is in general
    considered an overall measure of how good a source sound.
    Args:
        preds:
            shape ``[...,time]``
        target:
            shape ``[...,time]``
        zero_mean:
            If to zero mean target and preds or not
    Returns:
        si-sdr value of shape [...]
    Example:
        #>>> from torchmetrics.functional.audio import scale_invariant_signal_distortion_ratio
        #>>> target = torch.tensor([3.0, -0.5, 2.0, 7.0])
        #>>> preds = torch.tensor([2.5, 0.0, 2.0, 8.0])
        #>>> scale_invariant_signal_distortion_ratio(preds, target)
        tensor(18.4030)
    References:
        [1] Le Roux, Jonathan, et al. "SDR half-baked or well done." IEEE International Conference on Acoustics, Speech
        and Signal Processing (ICASSP) 2019.
    """
    #print(f"shape preds: {preds.shape} \nshape target: {target.shape}")
    _check_same_shape(preds, target)
    EPS = torch.finfo(preds.dtype).eps

    if zero_mean:
        target = target - torch.mean(target, dim=-1, keepdim=True)
        preds = preds - torch.mean(preds, dim=-1, keepdim=True)

    alpha = (torch.sum(preds * target, dim=-1, keepdim=True) + EPS) / (
        torch.sum(target ** 2, dim=-1, keepdim=True) + EPS
    )
    target_scaled = alpha * target

    noise = target_scaled - preds

    val = (torch.sum(target_scaled ** 2, dim=-1) + EPS) / (torch.sum(noise ** 2, dim=-1) + EPS)
    val = 10 * torch.log10(val)

    return val


def reorder_source(preds, batch_indices):
        r"""Reorder sources according to the best permutation.

        Args:
            preds (torch.Tensor): Tensor of shape :math:`[B, num_class, num_spk, num_frames]`
            batch_indices (torch.Tensor): Tensor of shape :math:`(batch, n_src)`.
                Contains optimal permutation indices for each batch.

        Returns:
            :class:`torch.Tensor`: Reordered sources.
        """
        reordered_sources = torch.stack(
            [torch.index_select(s, 1, b) for s, b in zip(preds, batch_indices)]
        )
        return reordered_sources


def reorder_source_mse(preds, batch_indices):
    r"""Reorder sources according to the best permutation.

    Args:
        preds (torch.Tensor): Tensor of shape :math:`[B, num_spk, num_frames]`
        batch_indices (torch.Tensor): Tensor of shape :math:`(batch, n_src)`.
            Contains optimal permutation indices for each batch.

    Returns:
        :class:`torch.Tensor`: Reordered sources.
    """
    #print(preds.shape)
    #print(batch_indices.shape)
    reordered_sources = torch.stack(
        [torch.index_select(s, 0, b) for s, b in zip(preds, batch_indices)]
    )
    return reordered_sources

def reorder_source_mse_fast(preds, batch_indices):
    r"""Reorder sources according to the best permutation.

    Args:
        preds (torch.Tensor): Tensor of shape :math:`[B, num_spk, num_frames]`
        batch_indices (torch.Tensor): Tensor of shape :math:`(batch, n_src)`.
            Contains optimal permutation indices for each batch.

    Returns:
        :class:`torch.Tensor`: Reordered sources.
    """
    batch_indices_r = batch_indices.repeat((preds.shape[-1],1,1))
    batch_indices_r_p = batch_indices_r.permute((1,2,0))
    reordered_sources = torch.gather(preds, dim=1, index=batch_indices_r_p)
    return reordered_sources

class CombinedLoss(nn.Module):
    def __init__(self, criterion_separation, criterion_vad, criterion_csd, criterion_sum_csd, criterion_mse,
                 criterion_sobel, criterion_stoi, criterion_vad_sum,
                 criterion_mse_vad, criterion_mse_vad_only, **weights):
        super().__init__()
        self.criterion_separation = criterion_separation
        self.criterion_vad = criterion_vad
        self.criterion_csd = criterion_csd
        self.criterion_sum_csd = criterion_sum_csd
        self.criterion_mse = criterion_mse
        self.criterion_sobel = criterion_sobel
        self.criterion_stoi = criterion_stoi
        self.criterion_vad_sum = criterion_vad_sum
        self.criterion_mse_vad = criterion_mse_vad
        self.criterion_mse_vad_only = criterion_mse_vad_only
        self.bce = BinaryCrossEntropyLoss_Mean()
        self.weights = weights
        if self.weights["weight_cosine_similarity"]:
            self.classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", run_opts={"device":"cuda:0"})
        
    def forward(self, pred_separation, target_separation, pred_vad, target_vad, pred_csd, target_csd, pred_csd_sum,
                target_csd_sum, estimated_stfts, mix_without_noise, pred_vad_sum):
        #print(torch.isnan(pred_vad).any())
        #print(pred_separation.shape)
        #print(target_separation.shape)
        if self.weights["weight_separation_loss"]:
            separation_loss, batch_indices_separation = self.criterion_separation(pred_separation, target_separation,
                                                                              return_incides=True)
            pred_separation = reorder_source_mse(pred_separation, batch_indices_separation)
        else:
            separation_loss, batch_indices_separation = torch.tensor(0), torch.tensor(0)
        
        
        
        if self.weights["min_over_spk"]:
            #print("Inside Min Loss")
            #pred_separation = reorder_source_mse(pred_separation, batch_indices_separation)
            si_sdr_loss = scale_invariant_signal_distortion_ratio(pred_separation, target_separation) #shape=[B, num_spk]
            #emphasize the min si-sdr
            #si_sdr_min_spk = torch.min(si_sdr_loss, dim=-1)[0]
            #si_sdr_loss_min = torch.mean(50 * (torch.exp(-si_sdr_min_spk / 14) - 1)) #torch.mean(26.6 * (torch.exp(-si_sdr_min_spk / 4.8) - 1))
            #si_sdr_loss_min = -torch.mean(torch.min(si_sdr_loss, dim=-1)[0])
            si_sdr_loss_min = -torch.mean(torch.sort(torch.min(si_sdr_loss, dim=-1)[0], dim=0)[0][:8])
        else:
            si_sdr_loss_min = torch.tensor(0)

        if self.weights["weight_vad_sum_loss"]:
            vad_sum_loss = torch.mean(self.criterion_vad_sum(pred_vad_sum, target_csd_sum))
        else:
            vad_sum_loss = torch.tensor(0)

        if self.weights["weight_vad_loss"]:
            #print(pred_vad.shape)
            #print(target_vad.shape)
            pred_vad = reorder_source_mse(pred_vad, batch_indices_separation)
            vad_loss = self.bce(pred_vad, target_vad)
            #vad_loss, batch_indices_vad = self.criterion_vad(pred_vad, target_vad, return_incides=True)
            batch_indices_vad = batch_indices_separation
        else:
            vad_loss, batch_indices_vad = torch.tensor(0), torch.tensor(0)

        if self.weights["weight_csd_loss"]:
            csd_loss = self.criterion_csd(pred_csd, target_csd, batch_indices_separation)
        else:
            csd_loss = torch.tensor(0)

        if self.weights["weight_sum_csd_loss"]:
            sum_csd_loss = self.criterion_sum_csd(pred_csd_sum, target_csd_sum)
        else:
            sum_csd_loss = torch.tensor(0)

        if self.weights["weight_mse_loss"]:
            mse_loss = self.criterion_mse(pred_separation, target_separation, batch_indices_separation, target_separation)
        else:
            mse_loss = torch.tensor(0)
            
        if self.weights["weight_sobel_loss"]:
            sobel_loss = self.criterion_sobel(pred_separation, target_separation, batch_indices_separation)
            #print(f"The sobel is {sobel_loss}")
            #print("I am In")
        else:
            sobel_loss = torch.tensor(0)
            
        if self.weights["weight_complement_loss"]:
            #pred_separation = reorder_source_mse(pred_separation, batch_indices_separation)
            #print(mix_without_noise.shape)
            #complement_loss1 = scale_invariant_signal_distortion_ratio(mix_without_noise - pred_separation[:, 0], target_separation[:, 1]) #shape=[B]
            #complement_loss0 = scale_invariant_signal_distortion_ratio(mix_without_noise - pred_separation[:, 1], target_separation[:, 0]) #shape=[B]
             
            #emphasize the min si-sdr
            """if self.weights["min_over_spk_complement_loss"]:
                complement_loss = torch.minimum(complement_loss0, complement_loss1)
            else: 
                complement_loss = 0.5*(complement_loss1 + complement_loss0)"""
            
            complement_loss = scale_invariant_signal_distortion_ratio(torch.sum(pred_separation, dim = 1), mix_without_noise) #shape=[B]
            
            complement_loss = -torch.mean(complement_loss)
            #print(f"The sobel is {sobel_loss}")
            #print("I am In")
        else:
            complement_loss = torch.tensor(0)
            
        if self.weights["weight_stoi_loss"]:
            #if not self.weights["min_over_spk"]:
                #pred_separation = reorder_source_mse(pred_separation, batch_indices_separation) #be carefull
            stoi_loss = self.criterion_stoi(pred_separation, target_separation)
            #stoi_loss = -torch.exp(1/(1 + stoi_loss)) #for exp stoi, delete for regular stoi
            stoi_loss = torch.mean(stoi_loss)
            #separation_loss = torch.tensor(0)
            #print(stoi_loss.requires_grad)
             
            
            #print(f"The sobel is {sobel_loss}")
            #print("I am In")
        else:
            stoi_loss = torch.tensor(0) 
            
        if self.weights["weight_mse_vad_loss"]:
            if not self.weights["min_over_spk"]:
                pred_separation = reorder_source_mse(pred_separation, batch_indices_separation)
            mse_vad_loss = self.criterion_mse_vad(pred_separation, target_separation, target_vad)
            separation_loss = torch.tensor(0) 
        else:
            #print("here")
            mse_vad_loss = torch.tensor(0) 
        
        if self.weights["mse_vad_only_loss"]:
            mse_vad_only_loss, batch_indices_separation = self.criterion_mse_vad_only(pred_separation, target_separation, target_vad, return_incides=True)
        else:
            #print("here")
            mse_vad_only_loss = torch.tensor(0) 

        if self.weights["weight_cosine_similarity"]:
            #print("begin1")
            B, _, _ = pred_separation.shape
            pred_cat = torch.cat((pred_separation[:, 0], pred_separation[:, 1]), dim=0)
            target_cat = torch.cat((target_separation[:, 0], target_separation[:, 1]), dim=0)
            """embedding0 = self.classifier.encode_batch(pred_separation[:, 0])
            print("finish1")
            embedding1 = self.classifier.encode_batch(pred_separation[:, 1])
            print("finish2")"""
            pred_cat_embedding = self.classifier.encode_batch(pred_cat)
            target_cat_embedding = self.classifier.encode_batch(target_cat)
            #print("finish1")
            embedding0 = pred_cat_embedding[:B]
            embedding1 = pred_cat_embedding[B:]
            
            embedding0_target = target_cat_embedding[:B]
            embedding1_target = target_cat_embedding[B:]
            cs_score = cosine_similarity_target_limit(embedding0, embedding1, embedding0_target, embedding1_target)
            #print("finish calc")
        else:
            #print("here")
            cs_score = torch.tensor(0) 
        
        #total_loss = separation_loss * self.weights["weight_separation_loss"] + csd_indiv_loss *
        # self.weights["weight_vad_loss"] + csd_sum_loss * self.weights["weight_csd_loss"]
        #return total_loss
        #return -3 * torch.exp(-0.5 * separation_loss) + 1, -3 * torch.exp(-0.5 * vad_loss) + 1, csd_loss, sum_csd_loss, mse_loss, sobel_loss, si_sdr_loss_min, complement_loss, stoi_loss, batch_indices_vad, batch_indices_separation
        return separation_loss, vad_loss, csd_loss, sum_csd_loss,\
            mse_loss, sobel_loss, si_sdr_loss_min, complement_loss, stoi_loss, \
            batch_indices_vad, batch_indices_separation, vad_sum_loss, mse_vad_loss, mse_vad_only_loss, cs_score
        
class CrossEntropyLoss_Mean(nn.Module):
    def __init__(self):
        """
        this clss is must be wrapped in pit since the return shape is batch
        """
        super().__init__()
        self.CE_none = nn.CrossEntropyLoss(reduction='none')
    def forward(self, pred_vad, target_vad):
        target_vad = target_vad.to(torch.long)
        out = self.CE_none(pred_vad, target_vad) #shape = [B, num_frames]
        output = torch.sum(out, dim=1).to(torch.float32)
        return output

class BinaryCrossEntropyLoss_Mean(nn.Module):
    def __init__(self):
        super().__init__()
        self.BCE_none = nn.BCELoss(reduction='none')
    def forward(self, pred_vad, target_vad):
        """
        pred_csd.shape = [B, num_frames]
        target_csd.shape = [B, num_frames]
        """
        
        target_vad = target_vad.to(torch.float32) 
        target_vad[target_vad == 2] = 1
        
        #print(target_vad.dtype)
        #print(pred_vad.dtype)
        out = self.BCE_none(pred_vad, target_vad) #shape = [B, num_frames]
        
        ### For weighting
        weight = target_vad * 0.36 + 0.64 * (1 - target_vad)
        out = out * weight
        ###
        
        output = torch.mean(torch.sum(out, dim=(-1, -2)).to(torch.float32)) #torch.sum(out, dim=-1).to(torch.float32)
        return output        

class CrossEntropyLoss_Csd(nn.Module):
    def __init__(self):
        super().__init__()
        self.CE_none = nn.CrossEntropyLoss(reduction='none')
    def forward(self, pred_csd, target_csd, batch_indices):
        """
        target shape = [B, num_spk, num_frames]
        pred_csd shape = [B, num_class, num_spk, num_frames]
        """
        target_csd = target_csd.to(torch.long)
        #print(target_csd.shape)
        reorder_pred_csd = reorder_source(pred_csd, batch_indices)
        #print(reorder_pred_csd.shape)
        out = self.CE_none(reorder_pred_csd, target_csd) #shape=[B, num_spk, num_frames]
        output = torch.sum(out).to(torch.float32) / (target_csd.shape[0] * target_csd.shape[1])
        return output


class CSD_sum_loss(nn.Module):
    def __init__(self):
        """
        there is no neccesart for pit in this loss class
        """
        super().__init__()
        self.CE_none = nn.CrossEntropyLoss(reduction='none')
    def forward(self, pred_csd, target_csd):
        """
        target shape = [B, num_frames]
        pred_csd shape = [B, num_class, num_frames]
        """
        target_csd = target_csd.to(torch.long)
        #print(target_vad.dtype)
        #print(pred_vad.dtype)
        out = self.CE_none(pred_csd, target_csd) #shape = [B, num_frames]
        output = torch.sum(out).to(torch.float32) / target_csd.shape[0]
        return output


class BaseLoss(nn.Module):
    def __init__(self, samplerate=16000, nfft=512, f_low=80, f_high=250, w_p=21, alpha_up=0.95, alpha_down=0.5,
                 lambda_voice=2, lambda_unvoice=1, **kwargs):
        super().__init__()
        self.lambda_voice = lambda_voice
        self.lambda_unvoice = lambda_unvoice
        freqs = torch.arange(f_low, f_high + 10, 10)
        times = 1. / freqs * samplerate
        bins = torch.arange(0, nfft / 2+1, 1)  # nfft / 2 + 1 -> remove 1 due to bias removal

        f_t = torch.exp(1j * torch.outer(times, bins) * 2 * np.pi / nfft)
        f_t[:, :2] = 0.
        f_t_p = F.pad(f_t, (w_p // 2, w_p // 2, 0, 0))
        #print('f_t_p.shape', f_t_p.shape)
        self.f_t_u = f_t_p.unfold(1, w_p, 1).unsqueeze(0)
        #print('self.f_t_u.shape', self.f_t_u.shape)
        self.calc_weights = partial(pitch_detect, alpha_up=alpha_up, alpha_down=alpha_down)

    def forward(self, outputs, target, clean=None):
        raise NotImplementedError

    def weight_loss(self, loss, clean):
        #loss and clean have shape [B,num_spk,F,T]
        """ apply harmonic weighted loss as in Ceva's paper by Nir Raviv and Ofer Schwartz """
        all_loss = []
        if self.f_t_u.device != loss.device:
            self.f_t_u = self.f_t_u.to(device=clean.device)
            #print('i range:', loss.shape[1])
        for i in range(loss.shape[1]):  # loop on #speaker
            clean_i = clean[:, i].transpose(-1, -2)
            loss_i = loss[:, i].transpose(-1, -2)
            # weights = self.calc_weights(clean_i[..., 0] ** 2 + 1e-8, f_t_u=self.f_t_u)
            weights = self.calc_weights(clean_i.abs() ** 2 + 1e-8, f_t_u=self.f_t_u) #pitch_detect input must be with this shape [B, T, F]. weights have the same shape
            # our input is abs, clean dim B*T*F
            # weights = self.calc_weights(clean_i[..., 0].abs() ** 2 + 1e-8, f_t_u=self.f_t_u)
            # #original clean dim B*T*F
            # clean_frames = (target.sum(dim=2, keepdim=True) > 20).expand(loss.shape)  # get speech frames
            clean_frames = (clean_i.abs().sum(dim=2, keepdim=True) > 5).expand(loss_i.shape)  # get speech frames(sum over the frequncies)
            if weights.ndim < loss_i.ndim:
                print("Something is wrong")
                weights = weights.unsqueeze(-1).expand(loss_i.shape)
            background_loss = loss_i * (~clean_frames)
            voice_loss = loss_i * ((weights > 0.4) & clean_frames)
            unvoice_loss = loss_i * ((weights <= 0.4) & clean_frames)
            #print('i', i)
            #print('loss_i', loss_i.shape)
            all_loss.append(background_loss + self.lambda_voice * voice_loss + self.lambda_unvoice * unvoice_loss)
        all_loss_spk = torch.stack(all_loss, dim=1) #[B, num_spk, T, F]
        mean_spk_loss = torch.sum(all_loss_spk, dim=(2,3)) #average over the frames and frequncy dims. shape=[B, num_spk]
        #all_loss = (all_loss[0] + all_loss[1])/2
        return all_loss_spk


class Weighted_MSELoss(BaseLoss):
    def __init__(self, samplerate=16000, nfft=512, f_low=80, f_high=250, w_p=21, alpha_up=0.95, alpha_down=0.5,
                 reduction='mean', pitch_weights=False, lambda_voice=2, lambda_unvoice=1, realAndimag=False, **kwargs):
        super().__init__(samplerate, nfft, f_low, f_high, w_p, alpha_up, alpha_down, lambda_voice, lambda_unvoice)
        self.pitch_weights = pitch_weights
        self.realAndimag = realAndimag
        self.mse = MSELoss(reduction='none')
        self.reduction = reduction
        self.spec = transforms.Spectrogram(n_fft=512, hop_length=256, win_length=512,
                                           window_fn=torch.hann_window, power=None)  # for all channels

    def forward(self, outputs, target, batch_indices, clean=None):
        """
        outputs and targets shape = [B, num_spk, samples]
        """
        #real-imag mse
        #reorder_pred_mse = reorder_source_mse(outputs, batch_indices)
        outputs = self.spec(reorder_pred_mse) #[B,num_spk,F,T]
        #print('outputs', outputs)
        # print(target.device)
        target = self.spec(target) #[B,num_spk,F,T]
        #print('target', target)

        #loss = torch.square(torch.abs(target - outputs))

        #print(type(loss))
        #loss = self.mse(outputs, target)  #[B,num_spk,F,T]

        if self.pitch_weights:
            #loss = self.weight_loss(loss, clean)
            loss = self.weight_loss(loss, target)
        elif self.realAndimag:
            loss  =  torch.square(torch.norm(target.real - outputs.real, p='fro', dim=(1,2)))\
                +  torch.square(torch.norm(target.imag - outputs.imag, p='fro', dim=(1,2)))
        else:
            loss = torch.square(torch.abs(target - outputs))    
            
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss
    
    
    
    
class MSE_with_VA(nn.Module):
    def __init__(self):
        """
        there is no neccesart for pit in this loss class
        """
        super().__init__()
        #self.mse_loss = nn.MSELoss()
        self.spec = transforms.Spectrogram(n_fft=512, hop_length=256, win_length=512,
                                      window_fn=torch.hann_window, power=None) 
        self.am_to_db = transforms.AmplitudeToDB(stype="power")
    def forward(self, pred, target, target_vad):
        """
        target shape = [B, num_spk]
        pred shape = [B, num_spk, frames]
        target_vad = [B, num_spk, frames]
        """
        #pred = 2*(pred - torch.min(pred, dim=-1, keepdim=True)[0]) / (torch.max(pred, dim=-1, keepdim=True)[0] - torch.min(pred, dim=-1, keepdim=True)[0]) - 1
        #target_csd = target_csd.to(torch.long)
        #print(target_vad.dtype)
        #print(pred_vad.dtype)
        #print(torch.sum(target_vad, dim=-1))
        """stft_preds = self.spec(pred)
        #power_preds = torch.pow(torch.abs(stft_preds), 2)
        #spectrum_preds = self.am_to_db(power_preds) #shape = [B, num_spk, F, T]
        stft_preds = stft_preds * torch.unsqueeze(target_vad, dim=2)
        
        stft_target = self.spec(target)
        stft_target = stft_target * torch.unsqueeze(target_vad, dim=2)
        #For NMSE
        eng_target = torch.sum(torch.pow(torch.abs(stft_target), 2), dim=(2,3))
        #output = torch.mean((torch.sum(torch.pow(torch.abs(stft_preds - stft_target), 2), dim=(2,3)) / torch.sum(target_vad, dim=-1)) / eng_target) #option 1
        output = torch.mean(torch.sum(torch.pow(torch.abs(stft_preds - stft_target), 2), dim=(2,3)) / eng_target) #option 2"""
        
        #For MSE
        #output = torch.mean(torch.sum(torch.pow(torch.abs(stft_preds - stft_target), 2), dim=(2,3)) / torch.sum(target_vad, dim=-1))
        #power_target = torch.pow(torch.abs(stft_target), 2) 
        #spectrum_target = self.am_to_db(power_target)
        #print(output)
        #output = self.mse_loss(spectrum_preds, spectrum_target) #shape = [B, num_frames]
        
        #for MSE of spectrograms
        stft_preds = self.spec(pred)
        power_preds = torch.pow(torch.abs(stft_preds), 2)
        spectrum_preds = self.am_to_db(power_preds) #shape = [B, num_spk, F, T]
        spectrum_preds = spectrum_preds * torch.unsqueeze(target_vad, dim=2)
        
        stft_target = self.spec(target)
        power_target = torch.pow(torch.abs(stft_target), 2)
        spectrum_target = self.am_to_db(power_target) #shape = [B, num_spk, F, T]
        spectrum_target = spectrum_target * torch.unsqueeze(target_vad, dim=2)
        
        alpha = torch.sum(spectrum_preds * spectrum_target, dim=(2,3), keepdim=True) / torch.sum(spectrum_preds * spectrum_preds, dim=(2,3), keepdim=True)
        spectrum_preds = spectrum_preds * alpha
        eng_target = torch.sqrt(torch.sum(torch.pow(torch.abs(spectrum_target), 2), dim=(2,3)))
        output = torch.mean(torch.log(torch.sqrt(torch.sum(torch.pow(spectrum_target - spectrum_preds, 2), dim=(2,3))) / eng_target))
        #output = torch.mean(torch.sum(torch.abs(spectrum_target - spectrum_preds), dim=(2,3)) /  (257*torch.sum(target_vad, dim=-1)))
        #output = torch.mean(torch.sum(torch.abs(spectrum_target - spectrum_preds), dim=(2,3)) /  (257*torch.sum(target_vad, dim=-1)))
        """eng_target = torch.sum(torch.pow(torch.abs(stft_target), 2), dim=(2,3))
        output = torch.mean(torch.sum(torch.pow(spectrum_target - spectrum_preds, 2), dim=(2,3)) / eng_target)"""
        #output = -torch.mean(scale_invariant_signal_distortion_ratio(spectrum_preds.flatten(start_dim=2), spectrum_target.flatten(start_dim=2)))

        return output
    
    
    
class MSE_with_VA_only(nn.Module):
    def __init__(self):
        """
        there is no neccesart for pit in this loss class
        """
        super().__init__()
        #self.mse_loss = nn.MSELoss()
        self.spec = transforms.Spectrogram(n_fft=512, hop_length=256, win_length=512,
                                    window_fn=torch.hann_window, power=None) 
        self.am_to_db = transforms.AmplitudeToDB(stype="power")
    def forward(self, pred, target, target_vad):
        """
        target shape = [B]
        pred shape = [B, frames]
        target_vad = [B, frames]
        """
        
        
        #for MSE of spectrograms
        stft_preds = self.spec(pred)
        power_preds = torch.pow(torch.abs(stft_preds), 2)
        spectrum_preds = self.am_to_db(power_preds) #shape = [B, F, T]
        spectrum_preds = spectrum_preds * torch.unsqueeze(target_vad, dim=1)
        
        stft_target = self.spec(target)
        power_target = torch.pow(torch.abs(stft_target), 2)
        spectrum_target = self.am_to_db(power_target) #shape = [B, F, T]
        spectrum_target = spectrum_target * torch.unsqueeze(target_vad, dim=1)
        
        alpha = torch.sum(spectrum_preds * spectrum_target, dim=(1,2), keepdim=True) / torch.sum(spectrum_preds * spectrum_preds, dim=(1,2), keepdim=True)
        spectrum_preds = spectrum_preds * alpha
        output = torch.sum(torch.pow(spectrum_target - spectrum_preds, 2), dim=(1,2))

        return output


