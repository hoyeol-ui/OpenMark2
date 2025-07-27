# watermark/models/decoder.py
import torch
import torch.nn as nn
import torchvision
from watermark.configs import ModelConfig

class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.extractor = torchvision.models.convnext_base(
            weights="IMAGENET1K_V1"
        )
        n_inputs = self.extractor.classifier[2].in_features

        # ⚡ Only swap out the old Linear head; leave idx 0 alone
        self.extractor.classifier[2] = nn.Linear(
            n_inputs, config.num_encoded_bits
        )

        self.main = nn.Sequential(self.extractor, nn.Sigmoid())

    def forward(self, image: torch.Tensor):
        if image.shape[-1] != self.config.image_shape[0] or image.shape[-2] != self.config.image_shape[1]:
            image = nn.functional.interpolate(image, size=self.config.image_shape, mode='bilinear')
        return self.main(image)