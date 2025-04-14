import math
import torch
import torch.nn as nn
from ..build import NETWORK_REGISTRY
from src.engine.build import device

from src.modeling.network.transformer.Embed import DataEncoding
from src.modeling.network.transformer.Attention import FullAttention, AttentionLayer
from src.modeling.network.transformer.Encoder_Decoder import EncoderLayer, DecoderLayer, Encoder, Decoder


class Transformer(nn.Module):
    """
    Text classifier based on a pytorch TransformerEncoder.
    """

    def __init__(
        self,
        encoder_input, decoder_input, channel_output,
        d_model=200, n_head=5, encoder_layers=2, decoder_layers=1, ff_dim=256,
        dropout=0.1, activation="gelu", output_attention=False,
        num_classes=3 # number of classes
    ):
        super().__init__()
        # 输入维度必须能被self-attention计算的次数整除
        assert d_model % n_head == 0, "n_heads must divide evenly into d_model, now d_model is {}, n_head is {}".format(d_model, n_head)

        # Encoding
        self.enc_embedding = DataEncoding(encoder_input, d_model, dropout)
        self.dec_embedding = DataEncoding(decoder_input, d_model, dropout)

        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False,
                                      attention_dropout=dropout,
                                      output_attention=output_attention),
                        d_model=d_model, n_heads=n_head
                    ),
                    d_model=d_model,
                    ff_dim=ff_dim,
                    n_heads=n_head,
                    dropout=dropout,
                    activation=activation
                ) for _ in range(encoder_layers)
            ],
            norm_layer=nn.LayerNorm(d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        FullAttention(True, attention_dropout=dropout, output_attention=False),
                        d_model=d_model, n_heads=n_head),
                    AttentionLayer(
                        FullAttention(False, attention_dropout=dropout, output_attention=False),
                        d_model=d_model, n_heads=n_head),
                    d_model=d_model,
                    ff_dim=ff_dim,
                    n_heads=n_head,
                    dropout=dropout,
                    activation=activation
                )
                for _ in range(decoder_layers)
            ],
            norm_layer=nn.LayerNorm(d_model)
        )

        self.projection_decoder = nn.Linear(d_model, channel_output, bias=True)
        
        #self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, data, enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        x_enc = data
        x_dec = data
        enc_out = self.enc_embedding(x_enc)
        dec_out = self.dec_embedding(x_dec)

        enc_out, _ = self.encoder(enc_out, attn_mask=enc_self_mask)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)

        output = self.projection_decoder(dec_out)
        return output



@NETWORK_REGISTRY.register()
def transformer(cfg=None):
    model = Transformer(encoder_input=cfg.MODEL.TRANSFORMER.CHANNELS,
                        decoder_input=cfg.MODEL.TRANSFORMER.CHANNELS,
                        channel_output=3,
                        d_model=cfg.MODEL.TRANSFORMER.D_MODEL,
                        n_head=cfg.MODEL.TRANSFORMER.N_HEAD,
                        encoder_layers=cfg.MODEL.TRANSFORMER.ENC_LAYERS,
                        decoder_layers=cfg.MODEL.TRANSFORMER.DEC_LAYERS,
                        ff_dim=cfg.MODEL.TRANSFORMER.DIM_FEEDFORWARD,
                        activation=cfg.MODEL.TRANSFORMER.ACTIVATION,
                        dropout=cfg.MODEL.DROPOUT,
                        output_attention=cfg.MODEL.TRANSFORMER.OUTPUT_ATTENTION)
    return model.to(device)
