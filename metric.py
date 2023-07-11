from torchmetrics.functional import word_error_rate
import numpy as np
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, AutoTokenizer
import torch
from tokenizers import Tokenizer


cache_dir = './cache/'
tokenizer = Tokenizer.from_file("MyBPETokenizer.json")


def wer(pred, label):
    pred_ids = torch.argmax(pred, axis=1)
    pred_str = tokenizer.decode(list(pred_ids[0]))
    label_str = tokenizer.decode(list(label[0]))

    print(pred_str)
    print(label_str)
    wer = word_error_rate(pred_str, label_str)
    wer = wer.float()
    return wer


    