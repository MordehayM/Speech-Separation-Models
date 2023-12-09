import pandas as pd

path_two_speakers = "/dsi/gannot-lab1/datasets/mordehay/data_wham_partial_overlap/train/csv_files/with_wham_noise_res_more_informative_correct.csv"
path_one_speaker = "/dsi/gannot-lab1/datasets/mordehay/data_wham_partial_overlap/train/csv_files/with_wham_noise_res_more_informative_correct_one_speaker.csv"
df_twp_speakers = pd.read_csv(path_two_speakers)
df_one_speaker = pd.read_csv(path_one_speaker)
pd.concat([df_twp_speakers, df_one_speaker], ignore_index=True).to_csv("/dsi/gannot-lab1/datasets/mordehay/data_wham_partial_overlap/train/csv_files/with_wham_noise_res_more_informative_correct_one_speaker_and_two_speakers.csv", index=False)