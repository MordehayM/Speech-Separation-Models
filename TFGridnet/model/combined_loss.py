
from torch import nn
import torch
from torch import Tensor
from itertools import permutations

EPS = torch.finfo(torch.float32).eps

def _check_same_shape(preds: Tensor, target: Tensor) -> None:
    """Check that predictions and target have the same shape, else raise error."""
    if preds.shape != target.shape:
        raise RuntimeError(f"Predictions and targets are expected to have the same shape, pred has shape of {preds.shape} and target has shape of {target.shape}")

def calc_sisdr(preds: Tensor, target: Tensor, zero_mean: bool = True) -> Tensor:
    """Calculates Scale-invariant signal-to-distortion ratio (SI-SDR) metric. The SI-SDR value is in general
    considered an overall measure of how good a target sound.
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

    _check_same_shape(preds, target)
    

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

def calc_sisdr_loss(preds: Tensor, target: Tensor, zero_mean: bool = True) -> Tensor:
    val = calc_sisdr(preds, target, zero_mean)
    return -val


def reorder_source_mse(preds, batch_indices):
    r"""Reorder targets according to the best permutation.

    Args:
        preds (torch.Tensor): Tensor of shape :math:`[B, num_spk, num_frames]`
        batch_indices (torch.Tensor): Tensor of shape :math:`(batch, n_src)`.
            Contains optimal permutation indices for each batch.

    Returns:
        :class:`torch.Tensor`: Reordered targets.
    """

    reordered_targets = torch.stack(
        [torch.index_select(s, 0, b) for s, b in zip(preds, batch_indices)]
    )
    return reordered_targets

class CombinedLoss(nn.Module):
    def __init__(self, criterion_separation, criterion_vad, weights):
        super().__init__()
        self.criterion_separation = criterion_separation
        self.criterion_vad = criterion_vad
        self.bce = BinaryCrossEntropyLoss_Mean()
        self.weights = weights
        self.learn_weight_bool = weights["learn_weight_bool"]
        if self.learn_weight_bool: #for learning the weights between vad loss to separation loss
            self.learn_weight_vadLoss = nn.parameter.Parameter(torch.tensor(0.5))
            self.learn_weight_separationLoss = nn.parameter.Parameter(torch.tensor(0.5))
        if self.weights["sa_sisdr_loss"]:
            self.sa_sisdr = SA_SISDR(num_speaker=2)
        if self.weights["sa_sdr_loss"]:
            self.sa_sdr = SA_SDR(num_speaker=2)    
        if self.weights["sisdr_loss"]:
            self.sisdr = SI_SDR(num_speaker=2)
        if self.weights["tsisdr_loss"]:
            self.tsisdr = tSI_SDR(num_speaker=2, sisdr_max=10)
            
            

    def forward(self, pred_separation, target_separation, pred_vad, target_vad):
        if self.weights["sisdr_loss"]:
            # separation_sisdr, batch_indices_separation = self.criterion_separation(pred_separation, target_separation,
            #                                                                     return_incides=True) #batch_indices_separation is the \
                                                                                    #permutation matrix with shape [B, num_spk].\
                                                                                        # This matrix can be applied on the predicted signals in order to\
                                                                                            # be aligned with the target signals. 
            separation_sisdr, batch_indices_separation = self.sisdr(pred_separation, target_separation)
            separation_loss = - separation_sisdr
            pred_separation = reorder_source_mse(pred_separation, batch_indices_separation)
        if self.weights["sa_sisdr_loss"]:
            separation_sa_sisdr, batch_indices_separation = self.sa_sisdr(pred_separation, target_separation)
            separation_loss = - separation_sa_sisdr
            
        if self.weights["sa_sdr_loss"]:
            separation_sa_sdr, batch_indices_separation = self.sa_sdr(pred_separation, target_separation)
            separation_loss = - separation_sa_sdr
            
        if self.weights["tsisdr_loss"]:
            separation_tsisdr, batch_indices_separation = self.tsisdr(pred_separation, target_separation)
            separation_loss = - separation_tsisdr
        
        if self.learn_weight_bool:
            separation_loss = separation_loss * (1 / (2 * torch.square(self.learn_weight_separationLoss))) + \
                torch.log(1 + torch.square(self.learn_weight_separationLoss))

        if self.weights["combined_loss"]:
            separation_loss = - 0.5 * separation_sa_sisdr - 0.5 * separation_sisdr
            
        if self.weights["weight_vad_loss"]:
            #print(pred_vad.shape)
            #print(target_vad.shape)
            pred_vad = reorder_source_mse(pred_vad, batch_indices_separation)
            vad_loss = self.bce(pred_vad, target_vad)
            if self.learn_weight_bool:
                vad_loss = vad_loss * (1 / (2 * torch.square(self.learn_weight_vadLoss))) + torch.log(1 + torch.square(self.learn_weight_vadLoss))
            #vad_loss, batch_indices_vad = self.criterion_vad(pred_vad, target_vad, return_incides=True)
            batch_indices_vad = batch_indices_separation
        else:
            vad_loss, batch_indices_vad = torch.tensor(0), torch.tensor(0)

        return separation_loss, vad_loss, batch_indices_vad, batch_indices_separation
        
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
        out = self.BCE_none(pred_vad, target_vad) #shape = [B, num_frames]
        ### For weighting
        weight = target_vad * 0.36 + 0.64 * (1 - target_vad) #Here we give more weight to the 0's since they are less frequent 
        out = out * weight
        ###
        output = torch.mean(torch.sum(out, dim=(-1, -2)).to(torch.float32)) #mean over the batch, sum over the frames and speakers
        return output
    
class SA_SISDR():
    def __init__(self, num_speaker=2):
        self.C = num_speaker
        
    def __call__(self, preds, target):
        mean_target = torch.mean(target, dim=2, keepdim=True)
        mean_estimate = torch.mean(preds, dim=2, keepdim=True)
        zero_mean_target = target - mean_target
        zero_mean_estimate = preds - mean_estimate

        # Step 2. SI-SNR with PIT
        # reshape to use broadcast
        s_target = torch.unsqueeze(zero_mean_target, dim=2)  # [B, C, 1, T]
        s_estimate = torch.unsqueeze(zero_mean_estimate, dim=1)  # [B, 1, C, T]
        # s_target = <s', s>s / ||s||^2
        pair_wise_dot = torch.sum(s_estimate * s_target, dim=-1, keepdim=True)  # [B, C, C, 1], the first C is the target dim
        s_target_energy = torch.sum(s_target ** 2, dim=-1, keepdim=True) + EPS  # [B, C, 1, 1]
        pair_wise_proj = pair_wise_dot * s_target / s_target_energy  # [B, C, C, T]
        # e_noise = s' - s_target
        e_noise = s_estimate - pair_wise_proj  # [B, C, C, T]

        # permutation and one-hot matrices, [C!, C, C]
        perms = target.new_tensor(list(permutations(range(self.C))), dtype=torch.long) #[C!, C]
        index = torch.unsqueeze(perms, 2) #[C!, C, 1]
        perms_one_hot = target.new_zeros((*perms.size(), self.C)).scatter_(2, index, 1)  #[C!, C, C]

        pair_wise_numerator = torch.sum(pair_wise_proj ** 2, dim=-1) # [B, C, C]
        pair_wise_denominator = torch.sum(e_noise ** 2, dim=-1) # [B, C, C]
        numerator_perm = torch.einsum('bij, pij->bp', [pair_wise_numerator, perms_one_hot]) # [B, C!]
        denominator_perm = torch.einsum('bij, pij->bp', [pair_wise_denominator, perms_one_hot]) # [B, C!]
        sa_sisdr_perm = 10 * torch.log10(numerator_perm / denominator_perm + EPS) # [B, C!]
        max_sa_sisdr, max_sa_sisdr_idx = torch.max(sa_sisdr_perm, dim=-1, keepdim=False) # [B], [B]
        batch_indices_separation = torch.index_select(perms, 0, max_sa_sisdr_idx) #[B, C]
        max_sa_sisdr_mean = torch.mean(max_sa_sisdr)
        return max_sa_sisdr_mean, batch_indices_separation
    
class SA_SDR():
    def __init__(self, num_speaker=2):
        self.C = num_speaker
        
    def __call__(self, preds, target):
        mean_target = torch.mean(target, dim=2, keepdim=True)
        mean_estimate = torch.mean(preds, dim=2, keepdim=True)
        zero_mean_target = target - mean_target
        zero_mean_estimate = preds - mean_estimate

        # Step 2. SI-SNR with PIT
        # reshape to use broadcast
        s_target = torch.unsqueeze(zero_mean_target, dim=2)  # [B, C, 1, T]
        s_estimate = torch.unsqueeze(zero_mean_estimate, dim=1)  # [B, 1, C, T]

        # e_noise = s' - s_target
        e_noise = s_target - s_estimate # [B, C, C, T]

        # permutation and one-hot matrices, [C!, C, C]
        perms = target.new_tensor(list(permutations(range(self.C))), dtype=torch.long) #[C!, C]
        index = torch.unsqueeze(perms, 2) #[C!, C, 1]
        perms_one_hot = target.new_zeros((*perms.size(), self.C)).scatter_(2, index, 1)  #[C!, C, C]

        pair_wise_numerator = torch.sum(torch.square(torch.norm(s_target, dim=-1, p=2)), dim=1) # [B, 1]
        pair_wise_denominator = torch.square(torch.norm(e_noise, dim=-1, p=2)) # [B, C, C]
        numerator_perm = pair_wise_numerator # [B, 1]
        denominator_perm = torch.einsum('bij, pij->bp', [pair_wise_denominator, perms_one_hot]) # [B, C!]
        sa_sdr_perm = 10 * torch.log10(numerator_perm / denominator_perm) # [B, C!]
        max_sa_sdr, max_sa_sisdr_idx = torch.max(sa_sdr_perm, dim=-1, keepdim=False) # [B], [B]
        batch_indices_separation = torch.index_select(perms, 0, max_sa_sisdr_idx) #[B, C]
        max_sa_sdr_mean = torch.mean(max_sa_sdr)
        return max_sa_sdr_mean, batch_indices_separation
    
class SI_SDR():
    def __init__(self, num_speaker=2):
        self.C = num_speaker
        
    def __call__(self, preds, target):
        mean_target = torch.mean(target, dim=2, keepdim=True)
        mean_estimate = torch.mean(preds, dim=2, keepdim=True)
        zero_mean_target = target - mean_target
        zero_mean_estimate = preds - mean_estimate

        # Step 2. SI-SNR with PIT
        # reshape to use broadcast
        s_target = torch.unsqueeze(zero_mean_target, dim=2)  # [B, C, 1, T]
        s_estimate = torch.unsqueeze(zero_mean_estimate, dim=1)  # [B, 1, C, T]
        # s_target = <s', s>s / ||s||^2
        pair_wise_dot = torch.sum(s_estimate * s_target, dim=-1, keepdim=True)  # [B, C, C, 1], the first C is the target dim
        s_target_energy = torch.sum(s_target ** 2, dim=-1, keepdim=True) + EPS  # [B, C, 1, 1]
        pair_wise_proj = pair_wise_dot * s_target / s_target_energy  # [B, C, C, T]
        # e_noise = s' - s_target
        e_noise = s_estimate - pair_wise_proj  # [B, C, C, T]
        # SI-SNR = 10 * log_10(||s_target||^2 / ||e_noise||^2)
        pair_wise_si_snr = torch.sum(pair_wise_proj ** 2, dim=3) / (torch.sum(e_noise ** 2, dim=3) + EPS)
        pair_wise_si_snr = 10 * torch.log10(pair_wise_si_snr + EPS)  # [B, C, C]

        # Get max_snr of each utterance
        # permutations, [C!, C]
        perms = target.new_tensor(list(permutations(range(self.C))), dtype=torch.long)
        # one-hot, [C!, C, C]
        index = torch.unsqueeze(perms, 2)
        perms_one_hot = target.new_zeros((*perms.size(), self.C)).scatter_(2, index, 1)
        # [B, C!] <- [B, C, C] einsum [C!, C, C], SI-SNR sum of each permutation
        sisdr_set = torch.einsum('bij,pij->bp', [pair_wise_si_snr, perms_one_hot])

        max_sisdr, max_sisdr_idx = torch.max(sisdr_set, dim=1, keepdim=False)
        max_sisdr /= self.C
        batch_indices_separation = torch.index_select(perms, 0, max_sisdr_idx) #[B, C]
        max_sisdr_mean = torch.mean(max_sisdr)
        return max_sisdr_mean, batch_indices_separation
    
    
class tSI_SDR():
    def __init__(self, num_speaker=2, sisdr_max=10):
        self.C = num_speaker
        self.tao = 10 ** (-sisdr_max / 10)
        
    def __call__(self, preds, target):
        mean_target = torch.mean(target, dim=2, keepdim=True)
        mean_estimate = torch.mean(preds, dim=2, keepdim=True)
        zero_mean_target = target - mean_target
        zero_mean_estimate = preds - mean_estimate

        # Step 2. SI-SNR with PIT
        # reshape to use broadcast
        s_target = torch.unsqueeze(zero_mean_target, dim=2)  # [B, C, 1, T]
        s_estimate = torch.unsqueeze(zero_mean_estimate, dim=1)  # [B, 1, C, T]
        # s_target = <s', s>s / ||s||^2
        pair_wise_dot = torch.sum(s_estimate * s_target, dim=-1, keepdim=True)  # [B, C, C, 1], the first C is the target dim
        s_target_energy = torch.sum(s_target ** 2, dim=-1, keepdim=True) + EPS  # [B, C, 1, 1]
        pair_wise_proj = pair_wise_dot * s_target / s_target_energy  # [B, C, C, T]
        # e_noise = s' - s_target
        e_noise = s_estimate - pair_wise_proj  # [B, C, C, T]
        # SI-SNR = 10 * log_10(||s_target||^2 / ||e_noise||^2)
        pair_wise_si_snr = torch.sum(pair_wise_proj ** 2, dim=3) / (torch.sum(e_noise ** 2, dim=3) + self.tao * torch.sum(pair_wise_proj, dim=3) + EPS)
        pair_wise_si_snr = 10 * torch.log10(pair_wise_si_snr + EPS)  # [B, C, C]

        # Get max_snr of each utterance
        # permutations, [C!, C]
        perms = target.new_tensor(list(permutations(range(self.C))), dtype=torch.long)
        # one-hot, [C!, C, C]
        index = torch.unsqueeze(perms, 2)
        perms_one_hot = target.new_zeros((*perms.size(), self.C)).scatter_(2, index, 1)
        # [B, C!] <- [B, C, C] einsum [C!, C, C], SI-SNR sum of each permutation
        sisdr_set = torch.einsum('bij,pij->bp', [pair_wise_si_snr, perms_one_hot])

        max_sisdr, max_sisdr_idx = torch.max(sisdr_set, dim=1, keepdim=False)
        max_sisdr /= self.C
        batch_indices_separation = torch.index_select(perms, 0, max_sisdr_idx) #[B, C]
        max_sisdr_mean = torch.mean(max_sisdr)
        return max_sisdr_mean, batch_indices_separation