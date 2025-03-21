#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

### YOUR CODE HERE for part 1i

import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, input_c, output_c, kernel_size=5):
        """init CNN.

        Args:
            input_c (int): # of input channel. In our application, argument will be e_{char}.
            output_c (int): # of output channel (== # of kernels). In our application, argument will be e_{word}.
            kernel_size (int): size of the kernel.
        """
        super(CNN, self).__init__()
        self.conv = nn.Conv1d(input_c, output_c, kernel_size)


    def forward(self, X) -> torch.Tensor:
        """forward propagation of CNN.

        Args:
            X (torch.Tensor): word vector embedded on character level. batched version of x_{reshaped}
                            in the pdf with the shape (batch, e_{char}, m_{word}) where e_{char} is the #
                            of character level embedding features and m_{word} is predefined maximum
                            length of word.(In our application, m_{word} is set to 21)

        Returns:
            torch.Tensor: batched version of x_{conv_out} in the pdf with the shape (batch, e_{word})
                            where e_{word} is the # of word embedding features.

        """
        out = self.conv(X)  # (batch, e_{word}, m_{word} - k + 1)
        out = F.relu(out)   # (batch, e_{word}, m_{word} - k + 1)
        assert out.shape == (X.shape[0], self.conv.out_channels, X.shape[2] - self.conv.kernel_size[0] + 1)

        out, _ = torch.max(out, dim=2)  # (batch, e_word)
        assert out.shape == (X.shape[0], self.conv.out_channels)

        return out

### END YOUR CODE

