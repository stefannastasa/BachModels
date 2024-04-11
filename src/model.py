import math
from typing import Callable, Optional

from torch import nn
import torch
from torchvision import models

from src.IAMDataset import IAMDataset
from src.LabelParser import LabelParser
from src.metrics import CharacterErrorRate, WordErrorRate
from torch.utils.data import Dataset


class PosEmbedding1D(nn.Module):
    """
    Implements 1D sinusoidal embeddings.

    Adapted from 'The Annotated Transformer', http://nlp.seas.harvard.edu/2018/04/03/attention.html
    """

    def __init__(self, d_model, max_len=1000):
        super().__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros((max_len, d_model), requires_grad=False)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Add a 1D positional embedding to an input tensor.

        Args:
            x (Tensor): tensor of shape (B, T, d_model) to add positional
                embedding to
        """
        _, T, _ = x.shape
        # assert T <= self.pe.size(0) \
        assert T <= self.pe.size(1), (
            f"Stored 1D positional embedding does not have enough dimensions for the current feature map. "
            f"Currently max_len={self.pe.size(1)}, T={T}. Consider increasing max_len such that max_len >= T."
        )
        return x + self.pe[:, :T]



class PosEmbedding2D(nn.Module):
    def __init__(self, d_model, max_len=100):
        super().__init__()
        self.d_model = d_model
        pe_x = torch.zeros((max_len, d_model // 2), requires_grad=False)
        pe_y = torch.zeros((max_len, d_model // 2), requires_grad=False)

        pos = torch.arange(0, max_len).unsqueeze(1)

        div_term = torch.exp(
            -math.log(10000.0) * torch.arange(0, d_model // 2, 2) / d_model
        )

        pe_y[:, 0::2] = torch.sin(pos * div_term)
        pe_y[:, 1::2] = torch.cos(pos * div_term)
        pe_x[:, 0::2] = torch.sin(pos * div_term)
        pe_x[:, 1::2] = torch.cos(pos * div_term)

        self.register_buffer("pe_x", pe_x)
        self.register_buffer("pe_y", pe_y)

    def forward(self, x):
        _, w, h, _ = x.shape

        pe_x_ = self.pe_x[:w, :].unsqueeze(1).expand(-1, h, -1)
        pe_y_ = self.pe_y[:h, :].unsqueeze(0).expand(w, -1, -1)

        pe = torch.cat([pe_y_, pe_x_], -1)
        pe = pe.unsqueeze(0)

        return x + pe


class encoderHTR(nn.Module):
    def __init__(self, d_model: int, encoder_type: str, dropout=0.1, bias=True):
        super().__init__()
        assert encoder_type in ["resnet18", "resnet34", "resnet50"], "Model not found"

        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        self.pos_embd = PosEmbedding2D(d_model)

        resnet = getattr(models, encoder_type)(pretrained=False)

        modules = list(resnet.children())
        cnv_1 = modules[0]
        cnv_1 = nn.Conv2d(
            1,
            cnv_1.out_channels,
            cnv_1.kernel_size,
            cnv_1.stride,
            cnv_1.padding,
            bias=cnv_1.bias
        )
        self.encoder = nn.Sequential(cnv_1, *modules[1:-2])
        self.linear = nn.Conv2d(resnet.fc.in_features, d_model, kernel_size=1)

    def forward(self, imgs):
        x = self.encoder(imgs.unsqueeze(1))
        x = self.linear(x).transpose(1, 2).transpose(2, 3)
        x = self.pos_embd(x)
        x = self.dropout(x)
        x = x.flatten(1, 2)

        return x


class decoderHTR(nn.Module):
    def __init__(self,
                 vocab_length,
                 max_seq_len,
                 eos_tkn_idx,
                 sos_tkn_idx,
                 pad_tkn_idx,
                 d_model,
                 num_layers,
                 nhead,
                 dim_ffn,
                 dropout,
                 activation="relu"):
        super().__init__()
        self.vocab_length = vocab_length
        self.max_seq_len = max_seq_len
        self.eos_idx = eos_tkn_idx
        self.sos_idx = sos_tkn_idx
        self.pad_idx = pad_tkn_idx
        self.d_model = d_model
        self.num_layers = num_layers
        self.nhead = nhead
        self.dim_ffn = dim_ffn
        self.pos_emb = PosEmbedding1D(d_model)
        self.emb = nn.Embedding(vocab_length, d_model)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model,
            nhead,
            dim_ffn,
            dropout,
            activation,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.clf = nn.Linear(d_model, vocab_length)
        self.drop = nn.Dropout(dropout)

    def forward(self, memory: torch.Tensor):
        B, _, _ = memory.shape
        all_logits = []
        sampled_ids = [torch.full([B], self.sos_idx).to(memory.device)]
        tgt = self.pos_emb(
            self.emb(sampled_ids[0]).unsqueeze(1) * math.sqrt(self.d_model)
        )
        tgt = self.drop(tgt)
        eos_sampled = torch.zeros(B).bool()
        for t in range(self.max_seq_len):
            tgt_mask = self.subsequent_mask(len(sampled_ids)).to(memory.device)
            out = self.decoder(tgt, memory, tgt_mask=tgt_mask)
            logits = self.clf(out[:, -1, :])
            _, pred = torch.max(logits, -1)
            all_logits.append(logits)
            sampled_ids.append(pred)
            for i, pr in enumerate(pred):
                if pr == self.eos_idx:
                    eos_sampled[i] = True
            if eos_sampled.all():
                break

            tgt_ext = self.drop(
                self.pos_emb.pe[:, len(sampled_ids)]
                + self.emb(pred) * math.sqrt(self.d_model)
            ).unsqueeze(1)
            tgt = torch.cat([tgt, tgt_ext], 1)
        sampled_ids = torch.stack(sampled_ids, 1)
        all_logits = torch.stack(all_logits, 1)

        eos_idxs = (sampled_ids == self.eos_idx).float().argmax(1)
        for i in range(B):
            if eos_idxs[i] != 0:
                sampled_ids[i, eos_idxs[i] + 1:] = self.pad_idx

        return all_logits, sampled_ids

    def forward_teacher_forcing(self, memory: torch.Tensor, tgt: torch.Tensor):
        B, T = tgt.shape
        tgt = torch.cat(
            [
                torch.full([B], self.sos_idx).unsqueeze(1).to(memory.device),
                tgt[:, :-1]
            ],
            1
        )

        tgt_key_masking = tgt == self.pad_idx
        tgt_mask = self.subsequent_mask(T).to(tgt.device)

        tgt = self.pos_emb(self.emb(tgt) * math.sqrt(self.d_model))
        tgt = self.drop(tgt)
        out = self.decoder(
            tgt, memory, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_masking
        )
        logits = self.clf(out)
        return logits

    @staticmethod
    def subsequent_mask(size: int):
        mask = torch.triu(torch.ones(size, size), diagonal=1)
        return mask == 1


class FullPageHTR(nn.Module):
    encoder: encoderHTR
    decoder: decoderHTR
    cer_metric: CharacterErrorRate
    wer_metric: WordErrorRate
    loss_fn: Callable
    label_encoder: LabelParser

    def __init__(self, label_encoder: LabelParser,
                 max_seq_len=500,
                 d_model=260,
                 num_layers=6,
                 nhead=4,
                 dim_feedforward=1024,
                 encoder_name="resnet18",
                 drop_enc=0.1,
                 drop_dec=0.1,
                 activ_dec="gelu",
                 label_smoothing=0.0,
                 vocab_len: Optional[int] = None):
        super().__init__()
        self.eos_token_idx, self.sos_token_idx, self.pad_token_idx = label_encoder.encode_labels(
            ["<EOS>", "<SOS>", "<PAD>"]
        )

        self.encoder = encoderHTR(d_model, encoder_type=encoder_name, dropout=drop_enc)
        self.decoder = decoderHTR(vocab_length=(vocab_len or len(label_encoder.classes)),
                                  max_seq_len=max_seq_len,
                                  eos_tkn_idx=self.eos_token_idx,
                                  sos_tkn_idx=self.sos_token_idx,
                                  pad_tkn_idx=self.pad_token_idx,
                                  d_model=d_model,
                                  num_layers=num_layers,
                                  nhead=nhead,
                                  dim_ffn=dim_feedforward,
                                  dropout=drop_dec,
                                  activation=activ_dec)
        self.label_encoder = label_encoder
        self.cer_metric = CharacterErrorRate(label_encoder)
        self.wer_metric = WordErrorRate(label_encoder)
        self.loss_fn = nn.CrossEntropyLoss(
            ignore_index=self.pad_token_idx,
            label_smoothing=label_smoothing
        )

    def forward(self, imgs: torch.Tensor, targets: Optional[torch.Tensor] = None):
        logits, sampled_ids = self.decoder(self.encoder(imgs))
        loss = None
        if targets is not None:
            loss = self.loss_fn(
                logits[:, : targets.size(1), :].transpose(1, 2),
                targets[:, : logits.size(1)],
            )
        return logits, sampled_ids, loss

    def forward_teacher_forcing(self, imgs: torch.Tensor, targets: torch.Tensor):
        memory = self.encoder(imgs)
        logits = self.decoder.forward_teacher_forcing(memory, targets)
        loss = self.loss_fn(logits.transpose(1, 2), targets)

        return logits, loss

    def calculate_metrics(self, preds: torch.Tensor, targets: torch.Tensor):
        self.cer_metric.reset()
        self.wer_metric.reset()

        cer = self.cer_metric(preds, targets)
        wer = self.wer_metric(preds, targets)
        return {"CER": cer, "WER": wer}

    def set_num_output_classes(self, n_classes: int):
        old_vocab_len = self.decoder.vocab_length
        self.decoder.vocab_length = n_classes
        self.decoder.clf = nn.Linear(self.decoder.d_model, n_classes)

        new_embs = nn.Embedding(n_classes, self.decoder.d_model)
        with torch.no_grad():
            new_embs.weight[:old_vocab_len] = self.decoder.emb.weight
            self.decoder.emb = new_embs


if __name__ == "__main__":
    dataset = IAMDataset("/home/tefan/projects/BachModels/data/raw", "train", None,
                         True)
    torch.cuda.empty_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = FullPageHTR(dataset.embedding_loader, d_model=120, num_layers=6, nhead=8, dim_feedforward=1024).to(device)
    image = torch.Tensor(dataset[1][0]).unsqueeze(0).to(device)
    target = torch.Tensor(dataset[1][1]).unsqueeze(0).to(device)
    print(image.shape)
    logits, _, _ = model(image)
    print(logits.shape)
    print(target.shape)
