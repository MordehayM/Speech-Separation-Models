#from speechbrain.pretrained import SepformerSeparation as separator
import torchaudio
import torch
import os
import torchaudio.transforms as T
from scipy.io.wavfile import write
import numpy as np
import sudo_rm_rf.dnn.experiments.utils.improved_cmd_args_parser_v2 as parser
import sudo_rm_rf.dnn.models.improved_sudormrf as improved_sudormrf

sepformer = False
SUDORMRF = True
our_parameters = True
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cuda:4' if torch.cuda.is_available() else 'cpu')

mix_path = "/home/dsi/moradim/Audio-Visual-separation-using-RTF/our_mix.wav"# "/home/dsi/moradim/Audio-Visual-separation-using-RTF/Looking to Listen - Stand-up-NzZDnRni-8A.wav"

## sepformer ##
if sepformer:
    model = separator.from_hparams(source="speechbrain/sepformer-whamr", savedir='pretrained_models/sepformer-whamr')

    # for custom file, change path
    est_sources = model.separate_file(mix_path)

    torchaudio.save("source1hat_whamr.wav", est_sources[:, :, 0].detach().cpu(), 8000)
    torchaudio.save("source2hat_whamr.wav", est_sources[:, :, 1].detach().cpu(), 8000)

## Sudo rm -rf ##
if SUDORMRF:
    args = parser.get_args()
    hparams = vars(args)
    if our_parameters:
    
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
            
        sudorm_model = torch.nn.DataParallel(sudorm_model) 
        sudorm_model_path = "/home/dsi/moradim/sudo2/checkpoints/improved_sudo_epoch_50" ####the path to the parameters
        sudorm_model_state_dict = torch.load(sudorm_model_path, map_location=torch.device('cpu'))
        #sudorm_model_state_dict = torch.load(sudorm_model_path)
        #print(sudorm_model_state_dict)
        sudorm_model.load_state_dict(sudorm_model_state_dict)
        sudorm_model.to(device)
       
    else:
        sudorm_model_path = "/dsi/gannot-lab/Improved_Sudormrf_U36_Bases4096_WHAMRexclmark.pt"
        sudorm_model = torch.load(sudorm_model_path)
        for i, (name, module) in enumerate(sudorm_model.sm.named_children()):
            #print(f"For module {name}************************")
            for layer_name, mod_name in module.named_children():
                if layer_name == 'upsampler':
                    sudorm_model.sm[i].upsampler = torch.nn.Upsample(scale_factor=2)
                    #print(sudorm_model.sm[i].upsampler)
                    #print(layer_name)
        sudorm_model.to(device)
    
    
    model_name = os.path.basename(sudorm_model_path)

    mix, sample_rate = torchaudio.load(mix_path)
    if mix.ndim == 2:
        if mix.shape[0] > 10:
            mix = mix[:, 0]
        elif mix.shape[1] > 10:
            mix = mix[0, :]
    if sample_rate != 8000:
        resample = T.Resample(sample_rate, 8000, dtype=torch.float32)
        mix = resample(mix)
    mix = mix.to(device)
    anechoic_sampled_mixture = mix.unsqueeze(0)
    input_mix_std = anechoic_sampled_mixture.std(-1, keepdim=True)
    input_mix_mean = anechoic_sampled_mixture.mean(-1, keepdim=True)
    input_mix = (anechoic_sampled_mixture - input_mix_mean) / (input_mix_std + 1e-9)
    print(input_mix.shape)
    #y_sudorm = sudorm_model(input_mix)
    sudorm_model.eval()
    with torch.no_grad():
        y_sudorm = sudorm_model(input_mix.unsqueeze(1))

    y_sudorm = 2*(y_sudorm - torch.min(y_sudorm, dim=-1, keepdim=True)[0]) / (torch.max(y_sudorm, dim=-1, keepdim=True)[0] - torch.min(y_sudorm, dim=-1, keepdim=True)[0]) - 1
    #y_sudorm = (y_sudorm * input_mix_std) + input_mix_mean
    y_sudorm = torch.squeeze(y_sudorm)
    y_sudorm1, y_sudorm2 = y_sudorm[0, :].unsqueeze(0), y_sudorm[1, :].unsqueeze(0)
    
    write('mix.wav', 8000, mix.detach().cpu().squeeze().numpy().astype(np.float32))
    write('output_sudorm1.wav', 8000, y_sudorm1.detach().cpu().squeeze().numpy().astype(np.float32))
    write('output_sudorm2.wav', 8000, y_sudorm2.detach().cpu().squeeze().numpy().astype(np.float32))
    #torchaudio.save('output_sudorm1.wav', y_sudorm1.detach().cpu(), 8000)
    #torchaudio.save('output_sudorm2.wav', y_sudorm2.detach().cpu(), 8000)
