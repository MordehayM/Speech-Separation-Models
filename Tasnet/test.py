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
from model.stoi_metric import Stoi
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.io.wavfile import write
import subprocess
from utils import prepare_device
from torch import Tensor
from pypesq import pesq

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

def plot_spectrogram(masks, title, save_path, batch_indx, ylabel='freq_bin', aspect='auto', xmax=None):
    masks  =masks.cpu()
    for indx_mask in range(masks.shape[1]):
        fig, axs = plt.subplots(1, 1)
        axs.set_title(f"Spectrogram (db) - {title}")
        axs.set_ylabel(ylabel)
        axs.set_xlabel('frame')
        im = axs.imshow(masks[0, indx_mask, :, :], origin='lower', aspect=aspect) #sample 0 from batch
        if xmax:
            axs.set_xlim((0, xmax))
        fig.colorbar(im, ax=axs)
        Path(f"{save_path}{batch_indx}/").mkdir(parents=True, exist_ok=True)
        plt.savefig(f"{save_path}{batch_indx}/Mask_Speaker_{indx_mask}")
        plt.close('all')
        
def save_audio(mix_waves, separated_signals, target, save_path, samplerate):
    target_audio1 = target[0, 0, :].cpu().detach().numpy() #sample 0 from batch
    target_audio2 = target[0, 1, :].cpu().detach().numpy() #sample 0 from batch
    separated_audio1 = separated_signals[0, 0, :].cpu().detach().numpy() #sample 0 from batch
    separated_audio2 = separated_signals[0, 1, :].cpu().detach().numpy() #sample 0 from batch
    mix_waves = mix_waves[0, :].cpu().detach().numpy()  #sample 0 from batch and 0 from channel
    Path(save_path).mkdir(parents=True, exist_ok=True)
    
    separated_audio1 = 2*(separated_audio1 - np.min(separated_audio1)) / (np.max(separated_audio1) - np.min(separated_audio1)) - 1
    separated_audio2 = 2*(separated_audio2 - np.min(separated_audio2)) / (np.max(separated_audio2) - np.min(separated_audio2)) - 1
    
    write(f"{save_path}/mixed.wav", samplerate, mix_waves.astype(np.float32))
    write(f"{save_path}/output_0.wav", samplerate, separated_audio1.astype(np.float32))
    write(f"{save_path}/output_1.wav", samplerate, separated_audio2.astype(np.float32))
    write(f"{save_path}/clean_0.wav", samplerate, target_audio1.astype(np.float32))
    write(f"{save_path}/clean_1.wav", samplerate, target_audio2.astype(np.float32))

def main(config):
    logger = config.get_logger('test')

    # setup data_loader instances
    data_loader = getattr(module_data, config['data_loader']['type'])(
        config['tester']['csv_file_test'],
        config['tester']['cds_lables'],
        batch_size=1,
        type_dataset=config['tester']['type_dataset'],
        shuffle=False,
        validation_split=0,
        num_workers=0
    )

    # build model architecture
    model = config.init_obj('arch', module_arch)
    logger.info(model)
    
    # get function handles of loss and metrics
    func_loss = getattr(module_func_loss, config["loss"]["loss_func"])

    reduce = False
    if config["loss"]["perm_reduce"] is not False:
        reduce = getattr(module_loss, config["loss"]["perm_reduce"])
    else:
        reduce=None

    kw = {"loss_func": func_loss, "perm_reduce":reduce}
    criterion = config.init_obj('loss', module_loss, **kw)

    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    checkpoint = torch.load(config.resume)
    state_dict = checkpoint['state_dict']
    save_test_path = config["tester"]["save_test"]
    device, device_ids = prepare_device(config['n_gpu'])
    print(save_test_path)
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)
    model.load_state_dict(state_dict)

    # prepare model for testing
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #device = torch.device('cpu')
    #model = model.to(device)
    model.eval()
    
    '''activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook
    model.mask_per_speaker.register_forward_hook(get_activation('masks'))'''
    
    metrics_separation = [getattr(module_metric, met) for met in config['metrics']["separation"]]
    metrics_separation_mix = [getattr(module_metric, met) for met in config['metrics']["separation_mix"]]
    metrics = {"separation": metrics_separation, "separation_mix": metrics_separation_mix}
    
    total_loss = 0.0
    total_metrics_separation = torch.zeros(len(metrics_separation), device=device)
    total_metrics_separation_mix = torch.zeros(len(metrics_separation_mix), device=device)


    



    samplerate = 16000
    reverbs = []
    snrs = []
    si_sdris = []
    si_sdrs = []
    si_sdrs_start = []
    value_separation = {}
    value_separation_start = {}
    si_sdr_per_spk= []
    si_sdr_per_spk_start = []
    stoi_calc = Stoi()
    stoi_list = []
    initial_stoi_list = []
    pesq_spk0_list = []
    pesq_spk1_list = []
    pesq_initial_list = []
    
    with torch.no_grad():
        for batch_idx, sample_separation in enumerate(tqdm(data_loader)):
            reverb = sample_separation["reverb"]
            snr = sample_separation["snr"]
            data = sample_separation['mixed_signals']
            #print(data.shape)
            mix_without_noise = sample_separation['mix_without_noise']
            batch_size = data.shape[0]
            target_separation = sample_separation['clean_speeches']
            doa = sample_separation["doa"]

            """df = sample_separation["df"]
            data, target_separation, doa, label_csd["vad_frames_individual"], df = data.to(device), target_separation.to(device), doa.to(device), label_csd["vad_frames_individual"].to(device), df.to(device)
            out_separation, csd_output = model(data, doa, df)"""
            data, target_separation, = data.to(device), target_separation.to(device)
           
            
            
            out_separation = model(data)
            
            
            ##Loss
            separation_loss, batch_indices_separation =  criterion(out_separation, target_separation, return_incides=True)
            
            #print(batch_idx)
            
            
            out_separation = reorder_source_mse(out_separation, batch_indices_separation)
            #reduce_kwargs = {'src': target} #I dont do reduce with csd
            stoi_value = stoi_calc(out_separation, target_separation)
            print(stoi_value)
            stoi_list.append(stoi_value.item())
            
            si_sdr_spk = scale_invariant_signal_distortion_ratio(out_separation, target_separation) #shape=[B, num_spk]
            #print(si_sdr_spk.shape)
            si_sdr_per_spk.append(np.array(si_sdr_spk.detach().cpu())[0])
            
            ##the start si-sdr for each speaker
            mix = data.unsqueeze(dim=1)
            mix = mix.repeat(1, 2, 1)
            
            si_sdr_spk_start = scale_invariant_signal_distortion_ratio(mix, target_separation)
            si_sdr_per_spk_start.append(np.array(si_sdr_spk_start.detach().cpu())[0])
            
            stoi_initial_value = stoi_calc(mix, target_separation)
            print(stoi_initial_value)
            initial_stoi_list.append(stoi_initial_value.item())
            
            pesq_initial = pesq(target_separation[0, 0].cpu().numpy(), data[0].cpu().numpy(), 16000)
            pesq_initial_list.append(pesq_initial)
            #print(mean_sisdr)
            #print(pesq_initial)
            #pesq_spk0 = pesq(16000, target_separation[0, 0].cpu().numpy(), out_separation[0, 0].cpu().numpy(), 'wb')
            pesq_spk0 = pesq(target_separation[0, 0].cpu().numpy(), out_separation[0, 0].cpu().numpy(), 16000)
            pesq_spk0_list.append(pesq_spk0)
            #pesq_spk1 = pesq(16000, target_separation[0, 1].cpu().numpy(), out_separation[0, 1].cpu().numpy(), 'wb')
            pesq_spk1 = pesq(target_separation[0, 1].cpu().numpy(), out_separation[0, 1].cpu().numpy(), 16000)
            pesq_spk1_list.append(pesq_spk1)
            mean_pesq = (pesq_spk0 + pesq_spk1) / 2
            
            #print(com1)
            #print(com0)
            
            
            
            

            
            #
            # save sample images, or do something with output here
            #

            # computing loss, metrics on test set

        
            
            for i, metric in enumerate(metrics["separation"]):
                metric = metric.to(device)
                value_separation[metric.__name__] = metric(out_separation, target_separation)
                value_separation_start[metric.__name__] = metric(mix, target_separation).detach()
                total_metrics_separation[i] += value_separation[metric.__name__].detach() * batch_size
            """for i, metric in enumerate(metrics["csd_acc"]):
                metric = metric.to(device)
                total_metrics_csd[i] += metric(output_csd, label_csd["vad_frames_sum"]) * batch_size"""
            
            for i, metric in enumerate(metrics["separation_mix"]):
                metric = metric.to(device)
                si_sdri = metric(out_separation, target_separation, data)
                total_metrics_separation_mix[i] += si_sdri.detach() * batch_size
            #print(si_sdri)
            save_test_path_full = f"{save_test_path}Batch_{batch_idx}_SiSDRI_{si_sdri.item():.2f}_SiSDR_{torch.mean(si_sdr_spk):.2f}_Reverb_{reverb.item():.2f}_Snr_{snr.item():.2f}_Stoi_{stoi_value.item():.2f}_Pesq_{mean_pesq:.2f}/"
            """########
            #print(output_vad.shape)
            Path(f"{save_test_path_full}").mkdir(parents=True, exist_ok=True)
            outvad = output_vad.cpu()
            est_vad = torch.where(outvad[0, 0] >= 0.5, 1, 0)
            plt.plot(est_vad)
            plt.savefig(f"{save_test_path_full}/Estimated_Vad_0.png")
            plt.close()
            est_vad = torch.where(outvad[0, 1] >= 0.5, 1, 0)
            plt.plot(est_vad)
            plt.savefig(f"{save_test_path_full}/Estimated_Vad_1.png")
            plt.close()
            m_vad = torch.where(torch.mean(outvad[0], dim=0) >= 0.5, 1, 0)
            plt.plot(m_vad)
            plt.savefig(f"{save_test_path_full}/Mean_Estimated_Vad.png")
            plt.close()
            #########"""
            
            #reduce_kwargs = {'src': target} #I dont do reduce with csd

            
            if batch_idx < 30: 
                save_audio(data, out_separation, target_separation, save_test_path_full, samplerate)

            total_loss += separation_loss.item() * batch_size   
            reverbs.append(reverb.item())
            
            snrs.append(snr.item())
            si_sdris.append(si_sdri.item())
            si_sdrs.append(value_separation["pit_si_sdr"].item())
            si_sdrs_start.append(value_separation_start["pit_si_sdr"].item())
        ###########################################################        
    si_sdr_per_spk = np.array(si_sdr_per_spk)
    si_sdr_per_spk_start = np.array(si_sdr_per_spk_start)
    
    
    #print(si_sdr_per_spk.shape)
    n_samples = len(data_loader.sampler)
    scenario = np.arange(0, n_samples + 1)

    df = pd.DataFrame(list(zip(scenario, reverbs, snrs, si_sdris, si_sdrs, si_sdrs_start, si_sdr_per_spk_start[:, 0], si_sdr_per_spk_start[:, 1],
                              si_sdr_per_spk[:, 0], si_sdr_per_spk[:, 1], stoi_list, initial_stoi_list,
                                   pesq_spk0_list, pesq_spk1_list, pesq_initial_list)),
            columns =['scenario', 'reverb', 'snr', 'si_sdri', 'si_sdr', 'si_sdr_start', 'si_sdr_start_speaker0', 'si_sdr_start_speaker1',
                    'si_sdr_speaker0', 'si_sdr_speaker1', 'stoi', 'initial_stoi', 'pesq_spk0', 'pesq_spk1', 'pesq_initial'])


        

    df.to_csv(save_test_path + 'results_information.csv')
    fig, ax = plt.subplots()
    g = ax.scatter(x = reverbs, 
               y = np.array(snrs),
               c = si_sdrs,
               cmap = "magma")
    fig.colorbar(g)
    ax.set_xlabel("reverbs[ms]")
    ax.set_ylabel("snr[dB]")
    #l1 = ax1.scatter(np.array(reverbs), np.array(si_sdrs), color='red')
    #ax1.set_xlabel("reverbs[ms]")
    #ax2 = ax1.twiny()
    #l2 = ax2.scatter(np.array(snrs), np.array(si_sdrs), color='orange')
    #ax2.set_xlabel("snr[dB]")

    #plt.legend([l1, l2], ["reverbs", "snrs"])
    plt.savefig(f"{save_test_path}Plot_reverb_snr_sisdr.png")
    
    
    mean_sdr = np.mean(si_sdrs)      
    print(f"the mean si_sdr is: {mean_sdr}")
    fig, axs = plt.subplots(1, 1)
    axs.hist(si_sdrs, bins="sqrt", density=True)
    axs.set_title(f"The mean sdr is {mean_sdr}")
    plt.savefig(f"{save_test_path}hist_si_sdr.png")
    
    mean_stoi_list = np.mean(stoi_list) 
    fig, axs = plt.subplots(1, 1)
    axs.hist(stoi_list, bins="sqrt", density=True)
    axs.set_title(f"The mean stoi is {mean_stoi_list}")
    plt.savefig(f"{save_test_path}hist_stoi.png")

        
    log = {'loss': total_loss / n_samples}
    log.update({
        met.__name__: total_metrics_separation[i].item() / n_samples for i, met in enumerate(metrics["separation"])
    })

    log.update({
        met.__name__: total_metrics_separation_mix[i].item() / n_samples for i, met in enumerate(metrics["separation_mix"])
    })
    log.update({
        "stoi": np.mean(stoi_list)
    })

    logger.info(log)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default="/home/dsi/moradim/OurBaselineModels/Tasnet/config.json", type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default="/dsi/gannot-lab/datasets/mordehay/Result/ConvTasnetBaseLine/models/AV_model/1128_120624/model_best.pth",
                                    type=str, help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    config = ConfigParser.from_args(args, trainer_or_tester="tester", save_path="save_test")
    main(config)
