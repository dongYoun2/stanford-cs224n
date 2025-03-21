#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

### YOUR CODE HERE for part 1h
import torch
import torch.nn as nn
import torch.nn.functional as F

class Highway(nn.Module):
    def __init__(self, num_features):
        """init Highway Network.

        Args:
            num_features (int): # of input features. Same as e_{word} in the pdf.
        """
        super(Highway, self).__init__()

        self.projection = nn.Linear(num_features, num_features)
        self.gate = nn.Linear(num_features, num_features)

    def forward(self, X) -> torch.Tensor:
        """forward propagation of Highway Network.

        Args:
            X (torch.Tensor): x_{conv_out} in the pdf (batched) where shape is (batch, e_{word}).

        Returns:
            torch.Tensor: x_{highway} in the pdf (batched) where shape is (batch, e_{word}).
        """
        X_proj = self.projection(X)
        X_proj = F.relu(X_proj)
        assert X_proj.shape == X.shape

        X_gate = self.gate(X)
        X_gate = F.sigmoid(X_gate)
        assert X_gate.shape == X.shape

        out = X_gate * X_proj + (1 - X_gate) * X
        assert out.shape == X.shape

        return out

### END YOUR CODE

