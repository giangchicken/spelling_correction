import torch
from transformers import AutoModel, AutoTokenizer, Wav2Vec2CTCTokenizer
import pandas as pd

results = pd.read_csv("/media/tamtran/D/huyennt/3T/spelling_correction/results.csv")
VIC_data = pd.read_csv("D:\spelling-correction\Vin\MT-Vi-Mono-VLSP2020\corpus.2M.shuf.csv")

results = results[["path","label", "out_wav2vec2"]]
print(VIC_data)
phobert = AutoModel.from_pretrained("vinai/phobert-base-v2")
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")



# INPUT TEXT MUST BE ALREADY WORD-SEGMENTED!
sentence = 'Chúng tôi là những nghiên cứu_viên .'  

input_ids = torch.tensor([tokenizer.encode(sentence)])

with torch.no_grad():
    features = phobert(input_ids)  # Models outputs are now tuples

df = []    

for i in range (results.shape[0]):
    sentence = str(results.iloc[i, :]["label"])
    df += [sentence]

# df_error = []
# for i in range (results.shape[0]):
#     sentence = str(results.iloc[i, :]["out_wav2vec2"])
#     df_error +=tokenizer.encode(sentence)

def unique(list1):
 
    # initialize a null list
    unique_list = []
 
    # traverse for all elements
    for x in list1:
        # check if exists in unique_list or not
        if x not in unique_list:
            unique_list.append(x)
    return unique_list
#64000

# print(len(unique(df)))
# print(len(unique(df_error)))

# unique_df = unique(df)



# print(dic)
# for i in unique_df:
#     print(tokenizer.decode(torch.tensor(i)))



import json
# with open('vocab.json', 'w', encoding='utf8') as vocab_file:
#     json.dump(dic, vocab_file, ensure_ascii=False)

#Build BPE    

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
trainer = BpeTrainer(vocab_size=4048, special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]", "<s>", "</s>"])
tokenizer.pre_tokenizer = Whitespace()

tokenizer.train_from_iterator(df, trainer)

print(tokenizer)

df_ = []
for i in range (results.shape[0]):
    sentence = str(results.iloc[i, :]["label"]) 
    # print(sentence)
    # print(tokenizer.encode(sentence).tokens)
    df_ += tokenizer.encode(sentence).word_ids
    
print(len(unique(df_)))

tokenizer.save("MyBPETokenizer.json")

tokenizer = Tokenizer.from_file("MyBPETokenizer.json")

# for i in range(210):
#     print("i " + str(i) + tokenizer.decode([i]))
    
df_ = []
for i in range (results.shape[0]):
    sentence = str(results.iloc[i, :]["label"]) 
    print(sentence)
    enc = tokenizer.encode(sentence).ids
    batch_label = torch.cat((torch.tensor([5]), 
                      torch.tensor(enc), 
                      torch.tensor([6])))
    print(tokenizer.decode(list(enc)))
    # df_ += tokenizer.encode(sentence).ids
