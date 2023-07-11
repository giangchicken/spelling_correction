from argparse import ArgumentParser
import pytorch_lightning as pl
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    AutoConfig,
)
from metric_bert import wer
from transformers.optimization import AdamW
from data_bert import LMDataModule
import torch.nn as nn
import torch
import torch.nn.functional as F
from position import PositionalEncoding
import pdb
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

BOS_IDX = 5
EOS_IDX = 6

checkpoints = "/media/tamtran/D/huyennt/3T/spelling_correction/lightning_logs/version_20/checkpoints/epoch=499-step=250000.ckpt"

def generate_square_subsequent_mask(sz: int, num_heads: int):
    r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
        Unmasked positions are filled with float(0.0).
    """
    tgt_mask= torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1)
    tgt_mask = tgt_mask.unsqueeze(0).repeat(1 * num_heads, 1, 1)
    # print("tgt_mask: ", tgt_mask)
    return tgt_mask

def create_mask(src, tgt, num_heads):
    # print("Src size", src.size())
    src_seq_len = src.size()[1]
    tgt_seq_len = tgt.size()[1]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len, 2* num_heads)
    src_mask = torch.zeros((src_seq_len, src_seq_len)).type(torch.bool)
    src_mask = src_mask.unsqueeze(0).repeat(2 * num_heads, 1, 1)

    # print("Src size:", src_mask.size())
    # print("dst size", tgt_mask.size())
    return src_mask, tgt_mask


def create_mask_test(src, tgt, num_heads):
    # print("Src size", src.size())
    src_seq_len = src.size()[1]
    tgt_seq_len = tgt.size()[1]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len, num_heads)
    src_mask = torch.zeros((src_seq_len, src_seq_len)).type(torch.bool)
    src_mask = src_mask.unsqueeze(0).repeat(num_heads, 1, 1)

    # print("Src size:", src_mask.size())
    # print("dst size", tgt_mask.size())
    return src_mask, tgt_mask

class LMModel(pl.LightningModule):
    def __init__(self, input_dim, output_dim, learning_rate, adam_beta1, adam_beta2, adam_epsilon):
        super().__init__()

        self.save_hyperparameters()

        # self.lstm = nn.LSTM(input_size=input_dim, hidden_size=256, num_layers=3, dropout=0.3,
        #                  batch_first=True)

        # self.encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=4, dim_feedforward=512)
        self.position_enc = PositionalEncoding(d_model=512)
        # self.trans_enc = nn.TransformerEncoder(self.encoder_layer, num_layers=2, norm=None, 
        #                     enable_nested_tensor=True, mask_check=True)

        self.linear1 = nn.Linear(20000, 512)
        self.linear3 = nn.Linear(20000, 512)
        self.trans = nn.Transformer(d_model = 512, nhead = 8, num_encoder_layers = 3,
                 num_decoder_layers= 3, dim_feedforward = 1024, dropout= 0.1,
                 activation = "gelu",
                 batch_first= True)
        self.linear2 = nn.Linear(512, 20000)
        
        self.loss = nn.CrossEntropyLoss(ignore_index= -100)
        
    def forward(self, x, tgt):
        x = torch.cat((torch.tensor([BOS_IDX]), 
                      torch.tensor(x), 
                      torch.tensor([EOS_IDX])))
        x = x.unsqueeze(0)

        # print("x: ", x.size())
        # print("tgt: ", tgt.size())

        src_mask, tgt_mask = create_mask_test(x, tgt, 8)
        
        src_key_padding_mask = torch.eq(x, 3)
        tgt_key_padding_mask = torch.eq(tgt, 3)

        x = F.one_hot(x, num_classes=20000).float()
        tgt = F.one_hot(tgt, num_classes=20000).float()

        x = self.linear1(x)
        tgt = self.linear3(tgt)
        
        x = self.position_enc(x)
        tgt = self.position_enc(tgt)        

        logits = self.trans(x, tgt, src_key_padding_mask=src_key_padding_mask,
                        tgt_key_padding_mask=tgt_key_padding_mask,
                        src_mask=src_mask, tgt_mask=tgt_mask)
        logits = self.linear2(logits)
        logits = torch.squeeze(logits, 0)
        return logits

    
    
    def training_step(self, batch, batch_idx):        
        batch_input = batch["input_ids"]
        batch_label = batch["labels"]
        
        
        
        # len_input = batch_input.size()[0]

        # nb_random = int(0.1*len_input)
        
        # rand_idx = [random.randint(1, len_input-1) for i in range(nb_random)]
        # rand_number = [random.randint(0, 5000) for i in range(nb_random)]
        # for i in range(nb_random):
        #     batch_input[rand_idx[i]] = rand_number[i]
        
        # batch_input = batch_input[batch_input != 3]
        # batch_label = batch_label[batch_label != 3]

        # print("train input: ", batch_input.size())
        # print("train label: ", batch_label.size())
        
        # batch_input = torch.cat((torch.tensor([BOS_IDX]).cuda(), 
        #               torch.tensor(batch_input).cuda(), 
        #               torch.tensor([EOS_IDX]).cuda()))

        # batch_label = torch.cat((torch.tensor([BOS_IDX]).cuda(), 
        #               torch.tensor(batch_label).cuda(), 
        #               torch.tensor([EOS_IDX]).cuda()))

        # if batch_input.size()[0] >= batch_label.size()[0]:
        #     batch_label_ = torch.full(size=(1, batch_input.size()[0]), fill_value=109)
        #     batch_label_[:, :batch_label.size()[0]] = batch_label
        #     # batch_label_[:, batch_label.size()[0]: batch_label.size()[0] + 3] = 109
        #     batch_label = batch_label_

        #     batch_input = batch_input.unsqueeze(0)

        # else:
        #     batch_input_ = torch.full(size=(1, batch_input.size()[0]), fill_value=109)
        #     batch_input_[:, :batch_input.size()[0]] = batch_input
        #     batch_input = batch_input_

        #     batch_label = batch_label.unsqueeze(0)

            # if batch_input.size()[1] < batch_label.size()[0]:

            #     batch_label_ = torch.full(size=(1, batch_input.size()[1]), fill_value=-100)
            #     batch_label_[:, :] = batch_label[:batch_input.size()[1]]
            #     batch_label = batch_label_
            # else:

            #     batch_label_ = torch.full(size=(1, batch_input.size()[1]), fill_value=-100)
            #     batch_label_[:, :batch_label.size()[0]] = batch_label
            #     batch_label = batch_label_                
                
            # batch_input = batch_input.unsqueeze(0)

        # batch_input = batch_input.unsqueeze(0)
        
        # batch_label = batch_label.unsqueeze(0)
        # batch_input = batch_input.unsqueeze(0)
        
        print("train input: ", batch_input.size())
        print("train label: ", batch_label.size())

        src_mask, tgt_mask = create_mask(batch_input, batch_label[:, :-1], 8)

        src_key_padding_mask = torch.eq(batch_input, 3)  # Mask các vị trí đệm trong dữ liệu nguồn
        tgt_key_padding_mask = torch.eq(batch_label[:, :-1], 3)  # Mask các vị trí đệm trong dữ liệu đích
        
        # print("src_key_padding_mask: ", src_key_padding_mask.size())
        # print("tgt_key_padding_mask: ",  tgt_key_padding_mask.size())
        

        batch_input = F.one_hot(batch_input, num_classes=20000).float()
        batch_label_ = F.one_hot(batch_label[:, :-1], num_classes=20000).float()

        # def inference(batch_input, max_len=1024):
        #     # pdb.set_trace()
        #     ys = torch.ones(1, 1).fill_(BOS_IDX).type(torch.long)
        #     end = torch.tensor([EOS_IDX])
        #     for i in range(max_len-1):
        #         # print("src: ", src.size())
                
        #         src_mask, tgt_mask = create_mask(batch_input, ys, 8)
        #         tgt_key_padding_mask = torch.eq(ys, 109)
        #         ys_ = F.one_hot(ys, num_classes=112).float()
        #         x_ = self.linear1(batch_input.cuda())
        #         y_ = self.linear3(ys_.cuda())
        #         # print("out: ", out.size())
                            
        #         x_ = self.trans(src = x_, tgt = y_, src_key_padding_mask=src_key_padding_mask.cuda(),
        #                     tgt_key_padding_mask=tgt_key_padding_mask.cuda(),
        #                     src_mask=src_mask.cuda(), tgt_mask=tgt_mask.cuda())
        #         out = self.linear2(x_)
        #         out_ = torch.squeeze(out, 0)
        #         next_word = torch.argmax(out_, axis=-1)[-1]
        #         # print("ys_size: ", ys.size())
        #         # print("next_size: ", torch.tensor([next_word]).size())
        #         # print("ys size: ", ys.size())
        #         # print(next_word)
        #         ys = torch.cat([ys, torch.ones(1, 1).fill_(int(next_word)).type(torch.long)], dim=1)
        #         # print("output line: ", ys)
        #         if next_word == end.cuda():
        #             break
        #     return out
        
        # x, _ = self.lstm(batch_input.cuda())
        x = self.linear1(batch_input.cuda())
        y_label = self.linear3(batch_label_.cuda())
        # x = self.trans_enc(x)
        
        x = self.position_enc(x)
        y_label = self.position_enc(y_label)        
        
        x = self.trans(src = x, tgt = y_label, src_key_padding_mask=src_key_padding_mask.cuda(),
                        tgt_key_padding_mask=tgt_key_padding_mask.cuda(),
                        src_mask=src_mask.cuda(), tgt_mask=tgt_mask.cuda())
        y = self.linear2(x)
        
        
        # y = inference(batch_input= batch_input)   
        y = y.permute(0, 2, 1)
        # y = y.permute(0, 2, 1)
        # print(y.size())
        # print(batch_label.size())
        # if y.size()[-1] > batch_label.size()[-1]:
        #     batch_label_ = torch.full(size=(1, y.size()[-1]), fill_value=-100)
        #     batch_label_[:, :batch_label.size()[-1]] = batch_label
        
        loss = self.loss(y, batch_label[:, 1:].cuda())
        print("loss: ", loss)

        wer_ = wer(y.detach(), batch_label.detach())
        print("train wer: ", wer_)

        self.log('train_loss', loss, on_epoch=True)
        self.log('train_wer', wer_, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        batch_input = batch["input_ids"]
        batch_label = batch["labels"]
        
        # print("batch_input size: ", batch_input.size())


        # len_input = batch_input.size()[0]

        # nb_random = int(0.1*len_input)
        
        # rand_idx = [random.randint(1, len_input-1) for i in range(nb_random)]
        # rand_number = [random.randint(0, 5000) for i in range(nb_random)]
        # for i in range(nb_random):
        #     batch_input[rand_idx[i]] = rand_number[i]
        
        # batch_input = batch_input[batch_input != 3]
        # batch_label = batch_label[batch_label != 3]

        # batch_input = torch.cat((torch.tensor([BOS_IDX]).cuda(), 
        #               torch.tensor(batch_input).cuda(), 
        #               torch.tensor([EOS_IDX]).cuda()))

        # batch_label = torch.cat((torch.tensor([BOS_IDX]).cuda(), 
        #               torch.tensor(batch_label).cuda(), 
        #               torch.tensor([EOS_IDX]).cuda()))


        # batch_label = batch_label.unsqueeze(0)
        # batch_input = batch_input.unsqueeze(0)

        src_mask, tgt_mask = create_mask(batch_input, batch_label[:, :-1], 8)
        
        # print("src_mask: ", src_mask)
        # print("tgt_mask: ", tgt_mask)
        # print("stc_mask size: ", src_mask.size())
        # print("tgt_mask size: ", tgt_mask.size())

        src_key_padding_mask = torch.eq(batch_input, 3)  # Mask các vị trí đệm trong dữ liệu nguồn
        tgt_key_padding_mask = torch.eq(batch_label[:, :-1], 3)  # Mask các vị trí đệm trong dữ liệu đích

        batch_input = F.one_hot(batch_input, num_classes=20000).float()
        batch_label_ = F.one_hot(batch_label[:, :-1], num_classes=20000).float()
        
        x = self.linear1(batch_input.cuda())
        y_label = self.linear3(batch_label_.cuda())
        # x = self.trans_enc(x)
        
        x = self.position_enc(x)
        y_label = self.position_enc(y_label)    
        
        x = self.trans(src = x, tgt = y_label, src_key_padding_mask=src_key_padding_mask.cuda(),
                        tgt_key_padding_mask=tgt_key_padding_mask.cuda(),
                        src_mask=src_mask.cuda(), tgt_mask=tgt_mask.cuda())
        y = self.linear2(x)
        
        # y = inference(batch_input = batch_input)
        y = y.permute(0, 2, 1)

        # print(y.size())
        # print(batch_label.size())
        # if y.size()[-1] > batch_label.size()[-1]:
        #     batch_label_ = torch.full(size=(1, y.size()[-1]), fill_value=-100)
        #     batch_label_[:, :batch_label.size()[-1]] = batch_label
        
        loss = self.loss(y, batch_label[:, 1:].cuda())

        wer_ = wer(y.detach(), batch_label.detach())
        
        
        print("valid wer: ", wer_)
        

        self.log('valid_loss', loss, on_step=True)
        self.log('valid_wer', wer_, on_epoch=True)
        
        return loss

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(),
                          self.hparams.learning_rate,
                          betas=(self.hparams.adam_beta1,
                                 self.hparams.adam_beta2),
                          eps=self.hparams.adam_epsilon,)
        return optimizer

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--learning_rate', type=float, default=0.00005)
        parser.add_argument('--adam_beta1', type=float, default=0.9)
        parser.add_argument('--adam_beta2', type=float, default=0.999)
        parser.add_argument('--adam_epsilon', type=float, default=1e-8)
        return parser


def cli_main():
    pl.seed_everything(1234)

    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str,
                        default="nguyenvulebinh/wav2vec2-base-vietnamese-250h")
    parser.add_argument('--train_file', type=str,
                        default="/media/tamtran/D/huyennt/3T/spelling_correction/data/train.csv")
    parser.add_argument('--validation_file', type=str,
                        default="/media/tamtran/D/huyennt/3T/spelling_correction/data/valid.csv")
    parser.add_argument('--preprocessing_num_workers', type=int, default=1)
    parser.add_argument('--overwrite_cache', action='store_true', default=False)
    parser.add_argument('--input_dim', type=int, default=110)
    parser.add_argument('--output_dim', type=int, default=110)
    parser.add_argument('--train_batch_size', type=int, default=2)
    parser.add_argument('--val_batch_size', type=int, default=2)
    parser.add_argument('--dataloader_num_workers', type=int, default=4)
    parser = pl.Trainer.add_argparse_args(parser)
    parser = LMModel.add_model_specific_args(parser)
    args = parser.parse_args()

    # ------------
    # data
    # ------------
    data_module = LMDataModule(
        model_name_or_path=args.model_name_or_path,
        train_file=args.train_file,
        validation_file=args.validation_file,
        preprocessing_num_workers=args.preprocessing_num_workers,
        overwrite_cache=args.overwrite_cache,
        train_batch_size=args.train_batch_size,
        val_batch_size=args.val_batch_size,
        dataloader_num_workers=args.dataloader_num_workers,
    )

    # ------------
    # model
    # ------------
    lmmodel = LMModel(
        input_dim=args.input_dim,
        output_dim=args.output_dim,
        learning_rate=args.learning_rate,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        adam_epsilon=args.adam_epsilon,
    )

    # ------------
    # training
    # ------------
    
    early_stopping = EarlyStopping(monitor="valid_wer", mode="min")

    trainer = pl.Trainer(callbacks=[EarlyStopping(monitor="val_loss", mode="min", patience=5)])
    trainer = trainer.from_argparse_args(args)
    trainer.fit(lmmodel, data_module)


if __name__ == '__main__':
    cli_main()