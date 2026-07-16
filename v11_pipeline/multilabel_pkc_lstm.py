from __future__ import annotations

import torch
from torch import nn


class MultiLabelPKCLSTM(nn.Module):
    """Predict the knowledge concepts covered by the next exercise.

    Each time step represents one observed exercise.  The input contains two
    K-dimensional channels: concepts in an incorrect response and concepts in
    a correct response.  The output has one independent logit per concept so
    that a next exercise may have multiple associated concepts.
    """

    def __init__(self, num_concepts: int, hidden_size: int = 200, dropout: float = 0.2) -> None:
        super().__init__()
        self.num_concepts = int(num_concepts)
        self.hidden_size = int(hidden_size)
        self.input_size = 2 * self.num_concepts
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, batch_first=True)
        self.dropout = nn.Dropout(float(dropout))
        self.output = nn.Linear(self.hidden_size, self.num_concepts)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return unnormalized next-concept logits with shape [B, T, K]."""
        states, _ = self.lstm(inputs)
        return self.output(self.dropout(states))
