# Copyright 2020 The OATomobile Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Perception (e.g., LIDAR) feature extractors."""

from typing import Callable
from typing import Optional
from typing import Sequence

import torch
import torch.nn as nn

# ==============================================================================
# NFNet
from nfnets import replace_conv, AGC, WSConv2d, ScaledStdConv2d
# ==============================================================================
# ==============================================================================
# Bottleneck Attention
from self_attention_cv.bottleneck_transformer import BottleneckBlock

# ==============================================================================

class NFNets_Attention(nn.Module):
  """A `PyTorch Hub` NFNets model wrapper."""

  def __init__(
      self,
      num_classes: int,
      in_channels: int = 3,
      pretrained=False,
  ) -> None:
    """Constructs a Rsenet50 model."""
    super(NFNets_Attention, self).__init__()

    self._model = torch.hub.load(
      'pytorch/vision:v0.10.0', 'resnet18', \
      pretrained=False)

#     self._model = mobilenet_v2(num_classes=num_classes, pretrained=False)
    # HACK(filangel): enables non-RGB visual features.
#     _tmp = self._model.features._modules['0']._modules['0']
#     self._model.features._modules['0']._modules['0'] = nn.Conv2d(
#         in_channels=in_channels,
#         out_channels=_tmp.out_channels,
#         kernel_size=_tmp.kernel_size,
#         stride=_tmp.stride,
#         padding=_tmp.padding,
#         bias=_tmp.bias,
#     )
#     _tmp = [nn.Conv2d(in_channels, 3, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True)]
#     _tmp.extend(list(self._model.features))  
#     self._model.features = nn.Sequential(*_tmp)
    


    # Input channels alteration
    weight = self._model.conv1.weight.clone()
    self._model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
    for c in range(in_channels):
        self._model.conv1.weight[:, c, :, : ].data[...]  = torch.nn.Parameter(weight[:, 0, :, :])
    self._model.conv1.weight[:, :3, :, :].data[...]  = torch.nn.Parameter(weight)
    
    # Output channels alteration
    self._model.fc = nn.Linear(512, num_classes)

    # NFNets
    replace_conv(self._model, WSConv2d)
    
    # Features
    self.features = nn.Sequential(*list(self._model.children())[:-3])
    # Attention
    self.attention_bottleneck = BottleneckBlock(in_channels=256, \
                                                fmap_size=(8, 8), \
                                                heads=4, \
                                                out_channels=num_classes, \
                                                pooling=True)
    # Classifier
    self.classifier = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), \
                                    nn.Flatten(), \
                                    nn.Linear(512, num_classes),
                                   )
    
  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Forward pass from the MobileNetV2."""
    f = self.features(x)
#     print("feature map shape:", list(f.size()))
    y = self.attention_bottleneck(f)
#     print("attention bottleneck output shape:", list(y.size()))
    y = self.classifier(y)
#     print("output shape:", list(y.size()))

    return y