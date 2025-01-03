import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer
from .layers.SelfAttention_Family import FullAttention, ProbAttention, AttentionLayer
from .layers.Embed import DataEmbedding
import numpy as np


class Model(nn.Module):
    """
    Informer with Propspare attention in O(LlogL) complexity
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention

        # Embedding
        if configs.embed_type == 0:
            self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                            configs.dropout)
            self.dec_embedding = DataEmbedding(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        elif configs.embed_type == 1:
            self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
            self.dec_embedding = DataEmbedding(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
        # elif configs.embed_type == 2:
        #     self.enc_embedding = DataEmbedding_wo_pos(configs.enc_in, configs.d_model, configs.embed, configs.freq,
        #                                             configs.dropout)
        #     self.dec_embedding = DataEmbedding_wo_pos(configs.dec_in, configs.d_model, configs.embed, configs.freq,
        #                                             configs.dropout)

        # elif configs.embed_type == 3:
        #     self.enc_embedding = DataEmbedding_wo_temp(configs.enc_in, configs.d_model, configs.embed, configs.freq,
        #                                             configs.dropout)
        #     self.dec_embedding = DataEmbedding_wo_temp(configs.dec_in, configs.d_model, configs.embed, configs.freq,
        #                                             configs.dropout)
        # elif configs.embed_type == 4:
        #     self.enc_embedding = DataEmbedding_wo_pos_temp(configs.enc_in, configs.d_model, configs.embed, configs.freq,
        #                                             configs.dropout)
        #     self.dec_embedding = DataEmbedding_wo_pos_temp(configs.dec_in, configs.d_model, configs.embed, configs.freq,
        #                                             configs.dropout)
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        ProbAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            [
                ConvLayer(
                    configs.d_model
                ) for l in range(configs.e_layers - 1)
            ] if configs.distil else None,
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        ProbAttention(True, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model, configs.n_heads),
                    AttentionLayer(
                        ProbAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
            projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
        )

        self.act = F.gelu
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(configs.d_model * configs.seq_len, configs.num_classes)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x_enc, label, x_mark_enc=None, x_dec=None, x_mark_dec=None,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        x_enc = x_enc.permute(0, 2, 1)
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)
        output = self.act(enc_out)
        output = self.dropout(output)
        output = output.reshape(output.shape[0], -1)
        output = self.classifier(output)
        loss = self.loss(output, label)
        return output, loss

        # dec_out = self.dec_embedding(x_dec, x_mark_dec)
        # dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)

        # if self.output_attention:
        #     return dec_out[:, -self.pred_len:, :], attns
        # else:
        #     return dec_out[:, -self.pred_len:, :]  # [B, L, D]
