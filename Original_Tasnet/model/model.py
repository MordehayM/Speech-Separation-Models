import torch.nn as nn
import torch.nn.functional as F
# from STFT_Using_Conv1.STFT_Conv import STFT
# from base import BaseModel
import torch
import numpy as np
from torch.autograd import Variable
from torchaudio import transforms


class cLN(nn.Module):
    def __init__(self, dimension, eps=1e-8, trainable=True):
        super(cLN, self).__init__()

        self.eps = eps
        if trainable:
            self.gain = nn.Parameter(torch.ones(1, dimension, 1))
            self.bias = nn.Parameter(torch.zeros(1, dimension, 1))
        else:
            self.gain = Variable(torch.ones(1, dimension, 1), requires_grad=False)
            self.bias = Variable(torch.zeros(1, dimension, 1), requires_grad=False)

    def forward(self, input):
        # input size: (Batch, Freq, Time)
        # cumulative mean for each time step

        batch_size = input.size(0)
        channel = input.size(1)
        time_step = input.size(2)

        step_sum = input.sum(1)  # B, T
        step_pow_sum = input.pow(2).sum(1)  # B, T
        cum_sum = torch.cumsum(step_sum, dim=1)  # B, T
        cum_pow_sum = torch.cumsum(step_pow_sum, dim=1)  # B, T

        entry_cnt = np.arange(channel, channel * (time_step + 1), channel)
        entry_cnt = torch.from_numpy(entry_cnt).type(input.type())
        entry_cnt = entry_cnt.view(1, -1).expand_as(cum_sum)

        cum_mean = cum_sum / entry_cnt  # B, T
        cum_var = (cum_pow_sum - 2 * cum_mean * cum_sum) / entry_cnt + cum_mean.pow(2)  # B, T
        cum_std = (cum_var + self.eps).sqrt()  # B, T

        cum_mean = cum_mean.unsqueeze(1)
        cum_std = cum_std.unsqueeze(1)

        x = (input - cum_mean.expand_as(input)) / cum_std.expand_as(input)
        return x * self.gain.expand_as(x).type(x.type()) + self.bias.expand_as(x).type(x.type())


class DepthConv1d(nn.Module):

    def __init__(self, input_channel, hidden_channel, kernel, padding, dilation=1, skip=False, causal=False):
        super(DepthConv1d, self).__init__()
        #hidden_channel = 512
        #input_channel = 256
        self.causal = causal
        self.skip = skip

        self.conv1d = nn.Conv1d(input_channel, input_channel, 1)
        if self.causal:
            self.padding = (kernel - 1) * dilation
        else:
            self.padding = padding
        groups = int(hidden_channel/2)
        self.dconv1d = nn.Conv1d(input_channel, hidden_channel, kernel, dilation=dilation,
                                 groups=groups,
                                 padding=self.padding) #Depth-wise convolution
        self.res_out = nn.Conv1d(hidden_channel, input_channel, 1)
        self.nonlinearity1 = nn.PReLU()
        self.nonlinearity2 = nn.PReLU()
        if self.causal:
            self.reg1 = cLN(hidden_channel, eps=1e-08)
            self.reg2 = cLN(hidden_channel, eps=1e-08)
        else:
            self.reg1 = nn.GroupNorm(1, input_channel, eps=1e-08)
            self.reg2 = nn.GroupNorm(1, hidden_channel, eps=1e-08)

        if self.skip:
            self.skip_out = nn.Conv1d(hidden_channel, input_channel, 1)

    def forward(self, input):
        output = self.reg1(self.nonlinearity1(self.conv1d(input)))
        if self.causal:
            output = self.reg2(self.nonlinearity2(self.dconv1d(output)[:, :, :-self.padding]))
        else:
            #print(output.shape)
            output = self.reg2(self.nonlinearity2(self.dconv1d(output)))
            #print(output.shape)
        residual = self.res_out(output)
        if self.skip:
            skip = self.skip_out(output)
            return residual, skip
        else:
            return residual


class TCN(nn.Module): #this is the audio blocks
    def __init__(self, input_dim, output_dim, BN_dim, H_dim,
                 layer, R, kernel=3, skip=False,
                 causal=False, dilated=True):
        super(TCN, self).__init__()

        # input is a sequence of features of shape (B, N, L)
        #input_dim = 257
        # BN_dim = 256

        # normalization
        if not causal:
            self.LN = nn.GroupNorm(1, input_dim, eps=1e-8) #this is like layer normalization because the number of groups is equal to one
        else:
            self.LN = cLN(input_dim, eps=1e-8)

        self.BN = nn.Conv1d(input_dim, BN_dim, 1)

        # TCN for feature extraction
        self.receptive_field = 0
        self.dilated = dilated

        self.TCN = nn.ModuleList([])
        for r in range(R):
            for i in range(layer):
                if self.dilated:
                    self.TCN.append(DepthConv1d(BN_dim, H_dim, kernel, dilation=2 ** i, padding=2 ** i, skip=skip,
                                                causal=causal))
                else:
                    self.TCN.append(
                        DepthConv1d(BN_dim, H_dim, kernel, dilation=1, padding=1, skip=skip, causal=causal))
                if i == 0 and r == 0:
                    self.receptive_field += kernel
                else:
                    if self.dilated:
                        self.receptive_field += (kernel - 1) * 2 ** i
                    else:
                        self.receptive_field += (kernel - 1)

        # print("Receptive field: {:3d} frames.".format(self.receptive_field))

        # output layer

        self.output = nn.Sequential(nn.PReLU(),
                                    nn.Conv1d(BN_dim, output_dim, 1)
                                    )

        self.skip = skip

    def forward(self, input):

        # input shape: (B, N, L)

        # normalization
        #print(type(input))
        #print(input.shape)
        x = self.LN(input)
        output = self.BN(x)

        # pass to TCN
        if self.skip:
            skip_connection = 0.
            for i in range(len(self.TCN)):
                residual, skip = self.TCN[i](output)
                output = output + residual
                skip_connection = skip_connection + skip
        else:
            for i in range(len(self.TCN)):
                residual = self.TCN[i](output)
                output = output + residual


        # output layer
        if self.skip:
            output = self.output(skip_connection)
        else:
            output = self.output(output)

        return output #asumming there is no skipping


class SeparationModel(nn.Module):
    def __init__(self, n_fftBins=512, BN_dim=256, H_dim=512, layer=8, stack=3, kernel=3, num_spk=2, skip=False, dilated=True, casual=False):
        super(SeparationModel, self).__init__()
        self.n_fftBins = n_fftBins
        self.n_fftBins_h = n_fftBins//2 + 1
        self.df_freq = (n_fftBins//2 + 1)*(num_spk*2+1)
        self.BN_dim = BN_dim
        self.layer = layer
        self.stack = stack
        self.kernel = kernel
        self.num_spk = num_spk
        self.H_dim = H_dim
        self.skip = skip
        self.dilated = dilated
        self.casual = casual

        print("begin of model")
        # self.STFT = STFT(filter_length=512, hop_length=256, win_length=None, window='hann')
        self.am_to_db = transforms.AmplitudeToDB(stype="power")
        self.TCN = TCN(self.df_freq, self.n_fftBins_h*self.num_spk, self.BN_dim, self.H_dim,
                       self.layer, self.stack, kernel=3, skip=self.skip,
                       causal=self.casual, dilated=self.dilated)
        self.to_spec = nn.Conv1d(self.BN_dim, self.num_spk*self.n_fftBins_h, 1)
        self.spec = transforms.Spectrogram(n_fft=self.n_fftBins, hop_length=256, win_length=self.n_fftBins,
                                      window_fn=torch.hann_window, power=None)  # for all channels
        self.inv_spec = transforms.InverseSpectrogram(n_fft=self.n_fftBins, hop_length=256, win_length=self.n_fftBins,
                                      window_fn=torch.hann_window)
        #self.df_layer = DirectionalFeature()

        self.m = nn.Sigmoid()
        self.softmax = torch.nn.Softmax(dim=1)
    # 512,257


    def forward(self, x, doa, df):

        #the input is samples of signals
        df = df.float()
        num_of_samples = x.shape[-1]
        stft = self.spec(x)
        stft[:, :, 0, :] = 0
        #print(f"the stft shape is: {stft.shape}")
        power = torch.pow(torch.abs(stft[:,0,:,:]), 2)
        spectrum = self.am_to_db(power) # take only the channel zero
        self.masks = self.TCN(df)
        self.masks = self.m(self.masks) ############THIS
        batch_size, freq_mul_speakers_size, frames_number = self.masks.shape
        self.mask_per_speaker = self.masks.reshape(batch_size, self.num_spk, self.n_fftBins_h, frames_number)
        self.estimated_stfts = torch.mul(torch.unsqueeze(stft[:,0,:,:], 1), self.mask_per_speaker)#apply each mask on stft,
                                                                            # the unsqueeze action is intended to

        #  convert the stft's shape to [B, 1, self.n_fftBins_h, frames_number]
        # for broadcasting

        # print(self.estimated_stfts.shape)
        out = self.inv_spec(self.estimated_stfts, length=num_of_samples) # shape = [B, self.num_spk, time]
        #print(f"the out shape is: {out.shape}")
        
        return out



if __name__ == '__main__':
    s = SeparationModel()
    # y= np.random.random(size=(10, 257, 512)).astype(np.double) # + j*np.random.random(size=(1, 257, 512))
    x = torch.rand(size=(10, 6, 48000))
    doa = 360 * torch.rand(size=(10, 1, 2))  # in deg
    # y = torch.from_numpy(y)
    out = s(x, doa)
    print("done")
