import pandas as pd
from torchmetrics.functional import word_error_rate

results = pd.read_csv("/media/tamtran/D/huyennt/3T/spelling_correction/results.csv")

results = results[["path", "out_wav2vec2", "label"]]

results = results.loc[results.out_wav2vec2.notnull()].reset_index(drop=True)[["path", "out_wav2vec2", "label"]]

print(results.iloc[0, :])

wer_list = []

N = results.shape[0]
for i in range(N):
    wer = word_error_rate(results.iloc[i, :]["out_wav2vec2"], results.iloc[i, :]["label"])
    wer_list.append(float(wer))

results["wer"] = wer_list
print(results.groupby(["wer"])["wer"].count())
print(results.shape)
# train = results.iloc[:12000, :]
# valid = results.iloc[12000:15000, :]
# test = results.iloc[15000:, :]

# train.to_csv("./data/train.csv")
# valid.to_csv("./data/valid.csv")
# test.to_csv("./data/test.csv")