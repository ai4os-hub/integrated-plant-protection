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