import pandas as pd 
from torchmetrics.functional import word_error_rate
from model_bert import LMModel
import librosa
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, AutoTokenizer
import os
import torch
import numpy as np
import pdb
from tokenizers import Tokenizer


cache_dir = './cache/'
processor = Wav2Vec2Processor.from_pretrained("nguyenvulebinh/wav2vec2-base-vietnamese-250h", cache_dir=cache_dir)
model = Wav2Vec2ForCTC.from_pretrained("nguyenvulebinh/wav2vec2-base-vietnamese-250h", cache_dir=cache_dir)
tokenizer = Tokenizer.from_file("MyBPETokenizer.json")

df = pd.read_csv("/media/tamtran/D/huyennt/speech_data/FILE_GHI_AM_CUT_AGENT_10/PSEUDO_LABELS.CSV")

df["output_dir"] = [str(b)[2:] for b in df["output_dir"]]
df["path"] = "/media/tamtran/D/huyennt" + df["output_dir"] + df["audio_part_name"]

df = df[["path", "beam_search_output"]].reset_index(drop=True)

path_model = "/media/tamtran/D/huyennt/3T/spelling_correction/lightning_logs/version_17/checkpoints/epoch=499-step=125000.ckpt"
LM = LMModel.load_from_checkpoint(path_model)

results = pd.read_csv("/media/tamtran/D/huyennt/3T/spelling_correction/results.csv").iloc[15000:, :]

results = results[["path", "out_wav2vec2", "label"]].reset_index(drop=True)

print(results.iloc[0, :]["path"])
print(df.iloc[4, :]["path"])
result = pd.merge(df, results, on="path", how="inner").reset_index(drop=True)
print(result.shape)
print(result.iloc[0, :])
result = result.loc[result.out_wav2vec2.notnull()].reset_index(drop=True)[["path", "out_wav2vec2",
                                                                        "beam_search_output","label"]]

N = result.shape[0]

wer_wav2vec2 = []
wer_lm = []
wer_sc = []
# function to generate output sequence using greedy algorithm 
def greedy_decode(model, src, max_len, start_symbol):
    # pdb.set_trace()
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long)
    for i in range(max_len-1):
        # print("src: ", src.size())
        out = model(x=src, tgt=ys)
        # print("out: ", out.size())

        next_word = torch.argmax(out, axis=-1)[-1]
        # print("ys_size: ", ys.size())
        # print("next_size: ", torch.tensor([next_word]).size())
        # print(next_word)
        # print(next_word)
        ys = torch.cat([ys, torch.ones(1, 1).fill_(int(next_word)).type(torch.long)], dim=1)
        # print("output line: ", ys)
        if next_word == torch.tensor([6]):
            break
    return ys

for i in range(N):

    # pred_ids = torch.tensor(tokenizer(result.iloc[i, :]["out_wav2vec2"])['input_ids'])
    pred_ids = torch.tensor(tokenizer.encode(result.iloc[i, :]["out_wav2vec2"]).ids)

    # pred_ids_ = torch.full(size=(1, pred_ids.size()[0] + 3), fill_value=109)
    # pred_ids_[:, :pred_ids.size()[0]] = pred_ids
    # pred_ids = torch.squeeze(pred_ids_, 0)
    # print(pred_ids.size())
    # print(pred_ids.size())
    # print(LM)
    out = greedy_decode(LM, src=pred_ids, max_len = 1024, start_symbol=5)
    # out = torch.argmax(out, axis=-1)
    # print(out.size())

    # pred_str = processor.decode(out, group_tokens=False)
    out = out.squeeze(0)[1:-1]
    pred_str = tokenizer.decode(list(out))

    wer_sc_ = word_error_rate(pred_str, result.iloc[i, :]['label'])
    wer_or = word_error_rate(result.iloc[i, :]['out_wav2vec2'], result.iloc[i, :]['label'])
    wer_lm_ = word_error_rate(result.iloc[i, :]['beam_search_output'], result.iloc[i, :]['label'])

    wer_wav2vec2.append(wer_or)
    wer_lm.append(wer_lm_)
    wer_sc.append(wer_sc_)

    print("pred lm: ",pred_str)
    print("label: ",result.iloc[i, :]['label'])
    print("pred wav2vec2: ", result.iloc[i, :]['out_wav2vec2'])
    print("pred beam search: ", result.iloc[i, :]['beam_search_output'])

result["wer_wav2vec2"] = wer_wav2vec2
result["wer_lm"] = wer_lm
result["wer_sc"] = wer_sc

print("wer_wav2vec2 mean: ", np.mean(result["wer_wav2vec2"]))
print("wer_lm mean: ", np.mean(result["wer_lm"]))
print("wer_sc mean: ", np.mean(result["wer_sc"]))








