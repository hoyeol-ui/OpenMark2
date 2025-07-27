# watermark/models/encoder.py
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from watermark.models.common import Conv2d, DecBlock, Watermark2Image
import logging

logger = logging.getLogger(__name__)

class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.watermark2image = Watermark2Image(
            config.num_encoded_bits,
            resolution=config.image_shape[0],
            hidden_dim=config.watermark_hidden_dim,
            num_repeats=config.num_noises,
            channel=3
        )
        self.pre = Conv2d(6, config.num_initial_channels, 3, 1, 1)
        self.enc = nn.ModuleList()
        input_channel = config.num_initial_channels
        for _ in range(config.num_down_levels):
            self.enc.append(Conv2d(input_channel, input_channel * 2, 3, 2, 1))
            input_channel *= 2

        self.dec = nn.ModuleList()
        for i in range(config.num_down_levels):
            if i < config.num_down_levels - 1:
                skip_width = input_channel // 2
            else:
                skip_width = input_channel // 2 + 6  # 마지막 skip은 입력까지 포함

            out_channels = input_channel // 2  # 반드시 지정해줘야 함
            self.dec.append(DecBlock(input_channel, skip_width, out_channels, activ='relu', norm='none'))
            input_channel = out_channels  # 다음 단계의 in_channel 업데이트

        self.post = nn.Sequential(
            Conv2d(input_channel, input_channel, 3, 1, 1, activ='None'),
            Conv2d(input_channel, input_channel // 2, 1, 1, 0, activ='silu'),
            Conv2d(input_channel // 2, 3, 1, 1, 0, activ='tanh')
        )

    def forward(self, image: torch.Tensor, watermark: torch.Tensor = None):
        if watermark is None:
            logger.info("Watermark is not provided. Using zeros.")
            watermark = torch.zeros(image.shape[0], self.config.num_encoded_bits, device=image.device)

        watermark = self.watermark2image(watermark)
        inputs = torch.cat((image, watermark), dim=1)

        enc_feats = []
        x = self.pre(inputs)
        for layer in self.enc:
            enc_feats.append(x)
            x = layer(x)

        enc_feats = enc_feats[::-1]
        for i, (layer, skip) in enumerate(zip(self.dec, enc_feats)):
            if i < self.config.num_down_levels - 1:
                x = layer(x, skip)
            else:
                x = layer(x, torch.cat([skip, inputs], dim=1))

        return self.post(x)