from torch import nn
from collections import OrderedDict
import torch
from torch.nn import functional as F
from espnet.nets.pytorch_backend.transformer.repeat import repeat
from espnet.nets.pytorch_backend.backbones.conv3d_extractor import Conv3dResNet
from espnet.nets.pytorch_backend.transformer.encoder_layer import EncoderLayer
from espnet.nets.pytorch_backend.transformer.embedding import (
    RelPositionalEncoding,  # noqa: H301
)
from espnet.nets.pytorch_backend.transformer.positionwise_feed_forward import (
    PositionwiseFeedForward,  # noqa: H301
)
from src.models.attention import RelPositionMultiHeadedAttention
from espnet.nets.pytorch_backend.transformer.convolution import ConvolutionModule
from src.utils import lens_to_mask

class Video_Encoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        encoder_attn_layer = RelPositionMultiHeadedAttention
        positionwise_layer = PositionwiseFeedForward
        convolution_layer = ConvolutionModule
        pos_enc_class = RelPositionalEncoding
        encoder_attn_layer_args = (cfg.attention_heads, cfg.attention_dim, cfg.attention_dropout_rate)
        positionwise_layer_args = (cfg.attention_dim, cfg.ffn_dim, cfg.dropout_rate)
        convolution_layer_args = (cfg.attention_dim, cfg.cnn_module_kernel)

        self.spk_embedding = nn.Embedding(cfg.num_speakers, cfg.spk_dim)
        self.fusion = nn.Linear(cfg.spk_dim + cfg.attention_dim, cfg.attention_dim)
        self.resnet = Conv3dResNet(relu_type=cfg.relu_type)
        self.embed = torch.nn.Sequential(
            torch.nn.Linear(512, cfg.attention_dim),
            pos_enc_class(cfg.attention_dim, cfg.positional_dropout_rate),
        )
        self.encoder = repeat(
            cfg.n_layers,
            lambda: EncoderLayer(
                cfg.attention_dim,
                encoder_attn_layer(*encoder_attn_layer_args),
                positionwise_layer(*positionwise_layer_args),
                convolution_layer(*convolution_layer_args),
                cfg.dropout_rate,
                normalize_before=True,
                concat_after=True,
                macaron_style=True,                
            ),
        )

    def forward(self, x, x_len, spk_id):
        T = x.size(2)
        x = self.resnet(x)
        # spk_emb = self.spk_embedding(spk_id)
        spk_emb = F.normalize(self.spk_embedding(spk_id))
        spk_emb_re = spk_emb.unsqueeze(1).repeat(1, x.size(1), 1)
        x = self.fusion(torch.cat([x, spk_emb_re], dim=-1))
        x_mask = lens_to_mask(x_len, T).unsqueeze(1)
        x = self.embed(x)
        x, x_mask = self.encoder(x, x_mask)
        return x[0], x_mask.squeeze(1), spk_emb

# from FastSpeech2
class VariancePredictor(nn.Module):
    """Pitch and Energy Predictor"""
    def __init__(self, cfg):
        super(VariancePredictor, self).__init__()

        self.input_size = cfg.encoder_hidden
        self.filter_size = cfg.filter_size
        self.kernel =cfg.kernel_size
        self.conv_output_size = cfg.filter_size
        self.dropout = cfg.dropout
        self.output_channels = cfg.output_channels
        self.n_layer = cfg.n_layer
        conv_layers = []
        for layer_idx in range(self.n_layer):
            in_channels = self.input_size if layer_idx == 0 else self.filter_size
            conv_layers.extend([
                (
                    f"conv1d_{layer_idx+1}",
                    Conv(
                        in_channels,
                        self.filter_size,
                        kernel_size=self.kernel,
                        padding=(self.kernel - 1) // 2,
                    ),
                ),
                (f"relu_{layer_idx+1}", nn.ReLU()),
                (f"layer_norm_{layer_idx+1}", nn.LayerNorm(self.filter_size)),
                (f"dropout_{layer_idx+1}", nn.Dropout(self.dropout)),                
            ])
        self.conv_layer = nn.Sequential(OrderedDict(conv_layers))

        self.linear_layer = nn.Linear(self.conv_output_size, self.output_channels)

    def forward(self, encoder_output, mask):
        out = self.conv_layer(encoder_output)
        out = self.linear_layer(out)
        out = out.masked_fill_(mask.unsqueeze(-1).repeat(1, 1, out.size(-1)), 0.0)

        return out
    
class Conv(nn.Module):
    """
    Convolution Module
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=0,
        dilation=1,
        bias=True,
        w_init="linear",
    ):
        """
        :param in_channels: dimension of input
        :param out_channels: dimension of output
        :param kernel_size: size of kernel
        :param stride: size of stride
        :param padding: size of padding
        :param dilation: dilation rate
        :param bias: boolean. if True, bias is included.
        :param w_init: str. weight inits with xavier initialization.
        """
        super(Conv, self).__init__()

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )

    def forward(self, x):
        x = x.contiguous().transpose(1, 2)
        x = self.conv(x)
        x = x.contiguous().transpose(1, 2)

        return x
    
class Mel_Decoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        pos_enc_class = RelPositionalEncoding
        self.embed = torch.nn.Sequential(
            torch.nn.Linear(512, cfg.attention_dim),
            pos_enc_class(cfg.attention_dim, cfg.positional_dropout_rate),
        )
        decoder_attn_layer = RelPositionMultiHeadedAttention
        positionwise_layer = PositionwiseFeedForward
        convolution_layer = ConvolutionModule
        decoder_attn_layer_args = (cfg.attention_heads, cfg.attention_dim, cfg.attention_dropout_rate)
        positionwise_layer_args = (cfg.attention_dim, cfg.ffn_dim, cfg.dropout_rate)
        convolution_layer_args = (cfg.attention_dim, cfg.cnn_module_kernel)
        self.decoder = repeat(
            cfg.n_layers,
            lambda: EncoderLayer(
                cfg.attention_dim,
                decoder_attn_layer(*decoder_attn_layer_args),
                positionwise_layer(*positionwise_layer_args),
                convolution_layer(*convolution_layer_args),
                cfg.dropout_rate,
                normalize_before=True,
                concat_after=True,
                macaron_style=True,                
            ),
        )
        self.fusion = nn.Linear(cfg.spk_dim + cfg.attention_dim, cfg.attention_dim)
        self.proj_mel = nn.Linear(cfg.attention_dim, 80)

    def forward(self, x, mask, spk_emb):
        x_mask = torch.repeat_interleave(mask, repeats=4, dim=1).unsqueeze(1)
        spk_emb_re = spk_emb.unsqueeze(1).repeat(1, x.size(1), 1)
        x = self.fusion(torch.cat([x, spk_emb_re], dim=-1))
        x = self.embed(x)
        x, x_mask = self.decoder(x, x_mask)
        x = self.proj_mel(x[0])
        return x*x_mask.permute(0, 2, 1), x_mask.squeeze(1), spk_emb