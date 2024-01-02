"""
Date: December 2023
Author: Jędrzej Smok 
Email: jsmok@man.poznan.pl
Github: ai4eosc-psnc
"""
from typing import List, Callable, Union, Any, TypeVar, Tuple
from abc import abstractmethod

import torch
from torch import nn
from torch import Tensor
from torch.nn import functional as F


class BaseClassifier(nn.Module):
    
    def __init__(self) -> None:
        super(BaseClassifier, self).__init__()

    def encode(self, input: Tensor) -> List[Tensor]:
        raise NotImplementedError

    def decode(self, input: Tensor) -> Any:
        raise NotImplementedError

    def sample(self, batch_size:int, current_device: int, **kwargs) -> Tensor:
        raise NotImplementedError

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def forward(self, *inputs: Tensor) -> Tensor:
        pass

    @abstractmethod
    def loss_function(self, *inputs: Any, **kwargs) -> Tensor:
        pass

class SmallCNNModel(BaseClassifier):
    def __init__(self,
                channels: list,):
        super(SmallCNNModel, self).__init__()
        in_channels = channels[0]
        modules= []
        modules.append(nn.Conv2d(3, channels[0], kernel_size=7, padding="same"))
        modules.append(nn.BatchNorm2d(channels[0]))
        modules.append(nn.ReLU())
        modules.append(nn.MaxPool2d(2))
        for channel in channels:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, channel, kernel_size=5, padding="same"),
                    nn.BatchNorm2d(channel),
                    nn.ReLU(),
                    nn.Conv2d(channel, channel, kernel_size=5, padding="same"),
                    nn.BatchNorm2d(channel),
                    nn.ReLU(),
                    nn.MaxPool2d(2)
                )
            )
            in_channels = channel
        modules.append(nn.Conv2d(channels[-1], channels[-1]*2, kernel_size=3, padding="same"))
        modules.append(nn.BatchNorm2d(channels[-1]*2))
        modules.append(nn.ReLU())
        modules.append(nn.MaxPool2d(2))
        modules.append(nn.Conv2d(channels[-1]*2, channels[-1]*2, kernel_size=3, padding="same"))
        modules.append(nn.BatchNorm2d(channels[-1]*2))
        modules.append(nn.ReLU())
        modules.append(nn.MaxPool2d(2))
        
        self.model = nn.Sequential(*modules)
        self.flatten = nn.Flatten()
        self.linear_1 = nn.Linear(8192, 256)
        self.linear_2 = nn.Linear(256, 1)
        
    def forward(self, input: Tensor) -> List[Tensor]:
        result = self.model(input)
        flatten = torch.flatten(result, start_dim=1)
        linear_1 = self.linear_1(flatten)
        output = self.linear_2(linear_1)

        return output
    def loss_function(self, *args, **kwargs):
        results = args[0]
        labels = args[1]
        labels = labels[:,None]
        loss = F.binary_cross_entropy_with_logits(results, labels.float())
        return loss



class conv_block(nn.Module):
  def __init__(self, in_c, out_c):
    super().__init__()
    self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding='same')
    self.bn1 = nn.BatchNorm2d(out_c)
    self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding='same')
    self.bn2 = nn.BatchNorm2d(out_c)

  def forward(self, inputs):
    x = self.conv1(inputs)
    x = self.bn1(x)
    x = F.leaky_relu(x)
    x = self.conv2(x)
    x = self.bn2(x)
    x = F.leaky_relu(x)
    return x

class encoder_block(nn.Module):
  def __init__(self, in_c, out_c):
    super().__init__()
    self.conv_block = conv_block(in_c, out_c)

  def forward(self, inputs):
    x = self.conv_block(inputs)
    p = F.max_pool2d(x, 2)
    #print(f"Encoder x shape:{x.shape}, p shape: {p.shape}")
    return x, p

class decoder_block(nn.Module):
  def __init__(self, in_c, out_c):
    super().__init__()
    self.up_conv = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2, padding=0, output_padding=0)
    self.conv_block = conv_block(out_c+out_c, out_c)

  def forward(self, inputs, skip):
    x = self.up_conv(inputs)
    d2 = skip.shape[2] - x.shape[2]
    d3 = skip.shape[3] - x.shape[3]
    #print(f"Decoder x shape:{x.shape}, skip shape: {skip.shape}")
    x = torch.cat([x, skip[:,:,d2//2:d2//2+x.shape[2],d3//2:d3//2+x.shape[3]]], axis=1)
    x = self.conv_block(x)
    return x


class Unet(nn.Module):
  def __init__(self):
    super().__init__()
    self.e1 = encoder_block(3, 64)
    self.e2 = encoder_block(64, 128)
    self.e3 = encoder_block(128, 256)
    self.e4 = encoder_block(256, 512)

    self.b = conv_block(512, 1024)

    self.d1 = decoder_block(1024, 512)
    self.d2 = decoder_block(512, 256)
    self.d3 = decoder_block(256, 128)
    self.d4 = decoder_block(128, 64)

    self.outputs = nn.Conv2d(64, 1, kernel_size=1, padding='same')

  def forward(self, inputs):

    s1, p1 = self.e1(inputs)
    s2, p2 = self.e2(p1)
    s3, p3 = self.e3(p2)
    s4, p4 = self.e4(p3)


    b = self.b(p4)
    #print(f"P4: {p4.shape}, b: {b.shape}")

    d1 = self.d1(b, s4)
    d2 = self.d2(d1, s3)
    d3 = self.d3(d2, s2)
    d4 = self.d4(d3, s1)

    outputs = self.outputs(d4)
    outputs = F.sigmoid(outputs)
    return outputs