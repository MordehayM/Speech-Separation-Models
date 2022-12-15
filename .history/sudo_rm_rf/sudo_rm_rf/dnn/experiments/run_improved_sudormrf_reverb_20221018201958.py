"""!
@brief Running an experiment with the improved version of SuDoRmRf
and reverberant data.

@author Efthymios Tzinis {etzinis2@illinois.edu}
@copyright University of Illinois at Urbana-Champaign
"""


import os
import sys
sys.path.append("/home/dsi/moradim/OurBaselineModels/sudo_rm_rf")
current_dir = os.path.dirname(os.path.abspath('__file__'))
root_dir = os.path.abspath(os.path.join(current_dir, '../../../'))
sys.path.append(root_dir)

#from __config__ import API_KEY
import math 
from torch import Tensor
import torch
import logging
from torch.nn import functional as F
from tqdm import tqdm
from pprint import pprint
import sudo_rm_rf.dnn.experiments.utils.improved_cmd_args_parser_v2 as parser

import sudo_rm_rf.dnn.losses.sisdr as sisdr_lib
import sudo_rm_rf.dnn.models.improved_sudormrf as improved_sudormrf
import sudo_rm_rf.dnn.models.sudormrf as initial_sudormrf
import sudo_rm_rf.dnn.utils.cometml_loss_report as cometml_report
import sudo_rm_rf.dnn.utils.cometml_log_audio as cometml_audio_logger
#import wandb
from sudo_rm_rf.dnn.dataset_loader.our_dataloader import Old_Partial_DataLoader
from pathlib import Path



###########Logger
logger = logging.getLogger(__name__)
    # set log level
logger.setLevel(logging.INFO)

# define file handler and set formatter
Path('/home/dsi/moradim/sudo/').mkdir(parents=True, exist_ok=True)
file_handler = logging.FileHandler('/home/dsi/moradim/sudo_paper_lr/logfile.log', mode='w')  # mode='w'
formatter = logging.Formatter('%(asctime)s : %(levelname)s : %(name)s : %(message)s')
file_handler.setFormatter(formatter)
# add file handler to logger
logger.addHandler(file_handler)
consoleHandler = logging.StreamHandler(sys.stdout)
consoleHandler.setFormatter(formatter)
logger.addHandler(consoleHandler)

consoleHandlererr = logging.StreamHandler(sys.stderr)
consoleHandler.setFormatter(formatter)
logger.addHandler(consoleHandler)




csv_file = "/dsi/gannot-lab/datasets/mordehay/data_wham_partial_overlap/train/csv_files/with_wham_noise_res.csv"
cds_lables = "/dsi/gannot-lab/datasets/mordehay/data_wham_partial_overlap/train/labels_npz/"
batch_size = 4
shuffle = True
validation_split = 0.2
num_workers = 8
data_loader = Old_Partial_DataLoader(csv_file, cds_lables, batch_size, shuffle=True, validation_split=0.2, num_workers=1)
valid_data_loader = data_loader.split_validation()




def _check_same_shape(preds: Tensor, target: Tensor) -> None:
    """Check that predictions and target have the same shape, else raise error."""
    if preds.shape != target.shape:
        raise RuntimeError("Predictions and targets are expected to have the same shape")

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

def si_sdriAndsisdr(preds, target, mix):
    mix = mix.unsqueeze(dim=1)
    mix = mix.repeat(1, 2, 1)
    si_sdr_spks_start = scale_invariant_signal_distortion_ratio(mix, target, zero_mean=True)
    si_sdr = scale_invariant_signal_distortion_ratio(preds, target)
    sisdri = torch.mean(si_sdr - si_sdr_spks_start)
    return sisdri, torch.mean(si_sdr)

def _progress(batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        current = batch_idx
        total = len(data_loader)
        return base.format(current, total, 100.0 * current / total)

#wandb.init(project="my-AV-project", entity="ren_mor")
#wandb.run.name = os.path.basename(config["trainer"]["save_dir"])
args = parser.get_args()
hparams = vars(args)


if hparams["checkpoints_path"] is not None:
    if hparams["save_checkpoint_every"] <= 0:
        raise ValueError("Expected a value greater than 0 for checkpoint "
                         "storing.")
    if not os.path.exists(hparams["checkpoints_path"]):
        os.makedirs(hparams["checkpoints_path"])

# if hparams["log_audio"]:



os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(
    [cad for cad in hparams['cuda_available_devices']])

back_loss_tr_loss_name, back_loss_tr_loss = (
    'tr_back_loss_SISDRi',
    sisdr_lib.PITLossWrapper(sisdr_lib.PairwiseNegSDR("sisdr"),
                             pit_from='pw_mtx')
)

back_loss_val_loss = sisdr_lib.PermInvariantSISDR(
        batch_size=hparams['batch_size'], n_sources=hparams['max_num_sources'],
        zero_mean=True, backward_loss=False, improvement=True,
        return_individual_results=False)
"""
val_losses = {}
all_losses = []
for val_set in [x for x in generators if not x == 'train']:
    if generators[val_set] is None:
        continue
    val_losses[val_set] = {}
    all_losses.append(val_set + '_SISDRi')
    val_losses[val_set][val_set + '_SISDRi'] = sisdr_lib.PermInvariantSISDR(
        batch_size=hparams['batch_size'], n_sources=hparams['max_num_sources'],
        zero_mean=True, backward_loss=False, improvement=True,
        return_individual_results=True)
all_losses.append(back_loss_tr_loss_name)"""

if hparams['model_type'] == 'relu':
    model = improved_sudormrf.SuDORMRF(out_channels=hparams['out_channels'],
                                       in_channels=hparams['in_channels'],
                                       num_blocks=hparams['num_blocks'],
                                       upsampling_depth=hparams[
                                           'upsampling_depth'],
                                       enc_kernel_size=hparams[
                                           'enc_kernel_size'],
                                       enc_num_basis=hparams['enc_num_basis'],
                                       num_sources=hparams['max_num_sources'])
elif hparams['model_type'] == 'softmax':
    model = initial_sudormrf.SuDORMRF(out_channels=hparams['out_channels'],
                                      in_channels=hparams['in_channels'],
                                      num_blocks=hparams['num_blocks'],
                                      upsampling_depth=hparams[
                                          'upsampling_depth'],
                                      enc_kernel_size=hparams[
                                          'enc_kernel_size'],
                                      enc_num_basis=hparams['enc_num_basis'],
                                      num_sources=hparams['max_num_sources'])
else:
    raise ValueError('Invalid model: {}.'.format(hparams['model_type']))

numparams = 0
for f in model.parameters():
    if f.requires_grad:
        numparams += f.numel()

print('Trainable Parameters: {}'.format(numparams))

model = torch.nn.DataParallel(model).cuda()
opt = torch.optim.Adam(model.parameters(), lr=hparams['learning_rate'])


# lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#     optimizer=opt, mode='max', factor=1. / hparams['divide_lr_by'],
#     patience=hparams['patience'], verbose=True)


def normalize_tensor_wav(wav_tensor, eps=1e-8, std=None):
    mean = wav_tensor.mean(-1, keepdim=True)
    if std is None:
        std = wav_tensor.std(-1, keepdim=True)
    return (wav_tensor - mean) / (std + eps)


tr_step = 0
val_step = 0
len_data_loader = len(data_loader)
len_valid_data_loader = len(valid_data_loader)
print(len_data_loader)
display_num = int(math.sqrt(len_data_loader) / 2)
print(display_num)
display_num_val = int(math.sqrt(len_valid_data_loader) / 2)
print(display_num_val)
for i in range(hparams['n_epochs']):
    logger.info("Improved Rev. Sudo-RM-RF:|| Epoch: {}/{}".format(i + 1,
        hparams['n_epochs']))
    model.train()
    batch_step = 0
    sum_loss = 0.
    sum_loss_val = 0
    sisdri = 0
    sisdri_val = 0
    sisdr = 0
    sisdr_val = 0
    #training_gen_tqdm = tqdm(generators['train'], desc='Training')

    #for data in training_gen_tqdm:
    for batch_idx, (sample_separation, label_csd) in enumerate(data_loader):
        opt.zero_grad()
        #sources_wavs = data[0]
        #targets_wavs = data[-1]

        # Online mixing over samples of the batch. (This might cause to get
        # utterances from the same speaker but it's highly improbable).
        # Keep the exact same SNR distribution with the initial mixtures.
        """s_energies = torch.sum(sources_wavs ** 2, dim=-1, keepdim=True)
        t_energies = torch.sum(targets_wavs ** 2, dim=-1, keepdim=True)
        b_size, n_sources, _ = sources_wavs.shape
        new_sources = []
        new_targets = []
        for k in range(n_sources):
            this_rand_perm = torch.randperm(b_size)
            new_src = sources_wavs[this_rand_perm, k]
            new_trgt = targets_wavs[this_rand_perm, k]
            new_src = new_src * torch.sqrt(
                s_energies[:, k] / (new_src ** 2).sum(-1, keepdims=True))
            new_trgt = new_trgt * torch.sqrt(
                t_energies[:, k] / (new_trgt ** 2).sum(-1, keepdims=True))
            new_sources.append(new_src)
            new_targets.append(new_trgt)"""

        data = sample_separation['mixed_signals']
        target_separation = sample_separation['clean_speeches']
        doa = sample_separation["doa"]
        mix_without_noise = sample_separation['mix_without_noise']

        #new_sources = torch.stack(new_sources, dim=1)
        data = data.cuda()
        new_targets = target_separation.cuda()
        m1wavs = normalize_tensor_wav(data).cuda()

        rec_sources_wavs = model(m1wavs.unsqueeze(1))
        #print(rec_sources_wavs.shape)
        #print(new_targets.shape)
        loss_temp, reorder_est = back_loss_tr_loss(rec_sources_wavs,
                              new_targets[:, :hparams['max_num_sources']], return_est=True)

        l = torch.clamp(
            loss_temp,
            min=-50., max=+50.)
        l.backward()
        if hparams['clip_grad_norm'] > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                           hparams['clip_grad_norm'])
        opt.step()
        np_loss_value = l.detach().item()
        sum_loss += np_loss_value
        #training_gen_tqdm.set_description(
        #    "Training, Running Avg Loss:"
        #    f"{round(sum_loss / (batch_step + 1), 3)}"
        #)
        
        ######Metric
        si_sdri_metric, sisdr_metric = si_sdriAndsisdr(reorder_est, new_targets, data)
        sisdri += si_sdri_metric.item()
        sisdr += sisdr_metric.item()
        batch_step += 1
        if batch_idx % display_num == 0:
            logger.info('Train Epoch: {} {}: The loss is {:.2f}, si-sdri = {:.2f}, si-sdr = {:.2f}'.format(
                    i,
                    _progress(batch_idx),
                    np_loss_value, 
                    si_sdri_metric.item(),
                    sisdr_metric.item()
                    ))
    # lr_scheduler.step(res_dic['val_SISDRi']['mean'])
    
    ###we added
    if hparams['patience'] > 0 and tr_step % hparams['patience'] == 0:
        new_lr = hparams['learning_rate'] / 5
        logger.info('Reducing Learning rate to: {}'.format(new_lr))
        for param_group in opt.param_groups:
            param_group['lr'] = new_lr
    """if hparams['patience'] > 0:
        if tr_step % hparams['patience'] == 0:
            new_lr = (hparams['learning_rate']
                      / (hparams['divide_lr_by'] ** (
                                tr_step // hparams['patience'])))
            logger.info('Reducing Learning rate to: {}'.format(new_lr))
            for param_group in opt.param_groups:
                param_group['lr'] = new_lr"""
    tr_step += 1
    logger.info('Total Train Epoch {} Loss is {:.2f}, sisdri = {:.2f}, sisdr = {:.2f}'.format(tr_step, sum_loss / (batch_idx + 1),
                                                                              sisdri / (batch_idx + 1),
                                                                              sisdr / (batch_idx + 1)))

    #for val_set in [x for x in generators if not x == 'train']:
        #if generators[val_set] is not None:
    for batch_idx, (sample_separation, label_csd) in enumerate(valid_data_loader):
        model.eval()
        with torch.no_grad():
            #for data in tqdm(generators[val_set],
                                #desc='Validation on {}'.format(val_set)):
            data = sample_separation['mixed_signals']
            target_separation = sample_separation['clean_speeches']
            doa = sample_separation["doa"]
            mix_without_noise = sample_separation['mix_without_noise']

            data = data.cuda()
            new_targets = target_separation.cuda()
            m1wavs = normalize_tensor_wav(data).cuda()

            #targets_wavs = data[-1].cuda()
            targets_wavs = new_targets
            #m1wavs = normalize_tensor_wav(sources_wavs.sum(1)).cuda()

            rec_sources_wavs = model(m1wavs.unsqueeze(1))

            """l = back_loss_val_loss(rec_sources_wavs,
                            targets_wavs[:, :hparams['max_num_sources']],
                            initial_mixtures=m1wavs.unsqueeze(1))"""
            l, reorder_est = back_loss_tr_loss(rec_sources_wavs,
                              new_targets[:, :hparams['max_num_sources']], return_est=True)                
            sum_loss_val += l.detach().item()
            si_sdri_metric, sisdr_metric = si_sdriAndsisdr(reorder_est, new_targets, data)
            sisdri_val += si_sdri_metric.item()
            sisdr_val += sisdr_metric.item()
            
            if batch_idx % display_num_val == 0:
                logger.info('Val Epoch: {} {}: The loss is {:.2f}, si-sdri = {:.2f}, si-sdr = {:.2f}'.format(
                    i,
                    _progress(batch_idx),
                    l.detach().item(), 
                    si_sdri_metric.item(),
                    sisdr_metric.item()
                    ))
    val_step += 1
    logger.info('Total Val Epoch {} Loss is {:.2f}, sisdri = {:.2f}, sisdr = {:.2f}'.format(val_step, sum_loss_val / (batch_idx + 1),
                                                                            sisdri_val / (batch_idx + 1),
                                                                            sisdr_val / (batch_idx + 1)))




    if hparams["save_checkpoint_every"] > 0:
        if tr_step % hparams["save_checkpoint_every"] == 0:
            torch.save(
                model.state_dict(),
                os.path.join(hparams["checkpoints_path"],
                             f"improved_sudo_epoch_{tr_step}"),
            )
