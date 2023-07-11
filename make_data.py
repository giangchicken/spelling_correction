from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import pandas as pd
import librosa 
import torch
import soundfile as sf
import os
import torchaudio


cache_dir = './cache/'
processor = Wav2Vec2Processor.from_pretrained("nguyenvulebinh/wav2vec2-base-vietnamese-250h", cache_dir=cache_dir)
model = Wav2Vec2ForCTC.from_pretrained("nguyenvulebinh/wav2vec2-base-vietnamese-250h", cache_dir=cache_dir)

path_folder_wav = ""
path_label = "/media/tamtran/D/huyennt/3T/spelling_correction/label_sv.csv"

df_label = pd.read_csv(path_label)

print(df_label["path"][0])

output_wav2vec2 = []

N = df_label.shape[0]
for i in range(N):
    path = str(df_label.iloc[i, :]["path"])
    print(path)
    print(os.path.isfile(path))
    try:
        speech, _ = librosa.load(path, sr=16000)
        input_values = processor(speech, return_tensors="pt").input_values

        logits = model(input_values).logits[0]
        pred_ids = torch.argmax(logits, dim=-1)
        greedy_search_output = processor.decode(pred_ids)
    
    except:
        greedy_search_output = "Null"

    output_wav2vec2.append(greedy_search_output)

df_label["out_wav2vec2"] = output_wav2vec2
df_label = df_label.loc[df_label.out_wav2vec2 != "Null"].reset_index(drop=True)
df_label.shape

df_label.to_csv("results.csv")