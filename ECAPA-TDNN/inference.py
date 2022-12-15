import torchaudio
from speechbrain.pretrained import EncoderClassifier
import torch 
import pickle
from scipy.io.wavfile import write
classifier = EncoderClassifier.from_hparams(source="LanceaKing/spkrec-ecapa-cnceleb", run_opts={"device":"cuda:6"})
"""#signal, fs =torchaudio.load("/dsi/gannot-lab/datasets/LibriSpeech/LibriSpeech/Test/5639/40744/5639-40744-0012.wav")
signal, fs =torchaudio.load("/dsi/gannot-lab/datasets/mordehay/Result/test_for_git/Separation_data_wham_partial_overlap_MultiDial_Vad099_decay095_inputActivity_CosineAnnealing_Test/Batch_23_SiSDRI_11.61_SiSDR_10.39_Reverb_0.24_Snr_7.86/clean_1.wav")

embeddings1 = classifier.encode_batch(signal)

#signal, fs =torchaudio.load("/dsi/gannot-lab/datasets/LibriSpeech/LibriSpeech/Test/5639/40744/5639-40744-0039.wav")
signal, fs =torchaudio.load("/dsi/gannot-lab/datasets/mordehay/Result/test_for_git/Separation_data_wham_partial_overlap_MultiDial_Vad099_decay095_inputActivity_CosineAnnealing_Test/Batch_23_SiSDRI_11.61_SiSDR_10.39_Reverb_0.24_Snr_7.86/output_0.wav")
embeddings2 = classifier.encode_batch(signal)

#signal, fs =torchaudio.load("/dsi/gannot-lab/datasets/LibriSpeech/LibriSpeech/Test/6829/68771/6829-68771-0012.wav")
signal, fs =torchaudio.load("/dsi/gannot-lab/datasets/mordehay/Result/test_for_git/Separation_data_wham_partial_overlap_MultiDial_Vad099_decay095_inputActivity_CosineAnnealing_Test/Batch_23_SiSDRI_11.61_SiSDR_10.39_Reverb_0.24_Snr_7.86/output_1.wav")
embeddings3 = classifier.encode_batch(signal)"""
with open("/dsi/gannot-lab/datasets/mordehay/data_wham_partial_overlap/test/with_wham_noise_audio/scenario_10.p", "rb") as f:
    # for our data
    mix , _, _, speakers_target, _ = pickle.load(f)
    #print(mix.shape)
    mix = torch.from_numpy(mix[0])
    speakers_target = torch.from_numpy(speakers_target)

print("begin")
embeddings1 = classifier.encode_batch(speakers_target[1].repeat(32, 1))
print("finish1")
embeddings2 = classifier.encode_batch(speakers_target[0].repeat(32, 1))
print("finish2")
embeddings3 = classifier.encode_batch(mix.repeat(32, 1))
print("finish3")




write("/home/dsi/moradim/OurBaselineModels/ECAPA-TDNN/mix.wav", 16000, mix.numpy().astype(mix.numpy().dtype))
def cosine_similarity(x, y):
    a =  torch.sum(x * y, dim=-1) / (torch.sqrt(torch.sum(torch.pow(x, 2), dim=-1)) * torch.sqrt(torch.sum(torch.pow(y, 2), dim=-1)))
    return a#torch.log(a+1.0001)

"""def cosine_similarity(x, y):
    a =  torch.mean(torch.pow(x -y, 2))
    return a#torch.log(a+1.0001)"""

"""print(f"the same speaker: {cosine_similarity(embeddings1, embeddings2)}")
print(f"differnet speaker: {cosine_similarity(embeddings1, embeddings3)}")
print(f"differnet speaker: {cosine_similarity(embeddings2, embeddings3)}")"""


