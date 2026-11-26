"""Custom CNN for CIFAKE real-vs-AI-generated image classification.

Architecture (all weights randomly initialised - no pretrained weights):

    Input   : [B, 3, H, W]  (32 or 64, RGB)
    Stages  : 5 convolutional stages with increasing channel widths:
                3 -> 32 -> 64 -> 96 -> 128 -> 160
              Each stage:
                Conv2d(in, w, k=3, s=2, p=1) -> BatchNorm2d(w) -> ReLU
                Conv2d(w, w, k=3, s=1, p=1)  -> BatchNorm2d(w) -> ReLU
              The stride-2 conv halves the spatial size per stage, so the
              map stays large in early stages and shrinks as width grows -
              no immediate collapse of the resolution.
    Head    : AdaptiveAvgPool2d(1) -> Flatten -> Linear(160, 64) -> ReLU
              -> Linear(64, 1)
    Output  : [B, 1] raw logits (no sigmoid inside the model; training uses
              BCEWithLogitsLoss, verification can apply sigmoid).

The model accepts a configurable input size (32 or 64) via ``input_size``.
"""

import torch
import torch.nn as nn

from .config import IMAGE_SIZE_MAIN, NUM_CLASSES

# Stage widths: increasing channel widths as the image shrinks.
STAGE_WIDTHS = (32, 64, 96, 128, 160)
# Width of the small fully-connected head after global average pooling.
HEAD_WIDTH = 64


class CifakeCNN(nn.Module):
    """Small binary CNN; positive class (FAKE) = 1."""

    def __init__(self, input_size=IMAGE_SIZE_MAIN, widths=STAGE_WIDTHS,
                 head_width=HEAD_WIDTH):
        super().__init__()
        self.input_size = input_size
        layers = []
        in_channels = 3
        for width in widths:
            layers += [
                nn.Conv2d(in_channels, width, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(width),
                nn.ReLU(inplace=True),
                nn.Conv2d(width, width, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(width),
                nn.ReLU(inplace=True),
            ]
            in_channels = width
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, head_width),
            nn.ReLU(inplace=True),
            nn.Linear(head_width, NUM_CLASSES),
        )

    def forward(self, x):
        """Return raw logits of shape [batch_size, NUM_CLASSES]."""
        return self.head(self.features(x))


def count_parameters(model):
    """Return the number of trainable parameters of a model."""
    return sum(p.numel() for p in model.parameters())


def run_shape_test():
    """Create a model, run a dummy batch through it, and print results.

    1. Creates a model (CifakeCNN).
    2. Passes a dummy batch (4 images of 3xHxW).
    3. Verifies the output shape is [batch_size, 1].
    4. Prints the actual trainable parameter count.
    """
    for size in (32, IMAGE_SIZE_MAIN):
        model = CifakeCNN(input_size=size)
        dummy = torch.zeros(4, 3, size, size)  # batch of 4 images
        with torch.no_grad():
            logits = model(dummy)
        assert logits.shape == (4, NUM_CLASSES), f"unexpected shape {tuple(logits.shape)}"
        print(f"input {size}x{size}: logits shape {tuple(logits.shape)}; "
              f"parameters: {count_parameters(model):,}")


if __name__ == "__main__":
    run_shape_test()
