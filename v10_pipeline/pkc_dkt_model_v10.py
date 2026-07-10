from __future__ import annotations

import torch
import torch.nn as nn


class PKCDKT(nn.Module):
    """DKT variant for next-knowledge-concept occurrence prediction.

    The input at each time step is the current concept-response pair. The
    output is a multi-label probability vector over all knowledge concepts,
    interpreted as the concepts likely to appear in the next learning step.
    """

    def __init__(self, concept_count: int, embed_size: int = 128, hidden_size: int = 200, dropout: float = 0.2) -> None:
        super().__init__()
        self.concept_count = int(concept_count)
        self.input_embedding = nn.Embedding(self.concept_count * 2 + 1, embed_size, padding_idx=0)
        self.rnn = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_size, self.concept_count)

    def encode_input(self, concepts: torch.Tensor, responses: torch.Tensor) -> torch.Tensor:
        # Padding uses 0. Valid concepts are shifted by +1; incorrect/correct
        # responses occupy two disjoint ranges.
        concepts = concepts.clamp(min=0, max=self.concept_count - 1)
        responses = responses.clamp(min=0, max=1)
        return concepts + 1 + responses * self.concept_count

    def forward(self, concepts: torch.Tensor, responses: torch.Tensor) -> torch.Tensor:
        x = self.input_embedding(self.encode_input(concepts, responses))
        h, _ = self.rnn(x)
        return torch.sigmoid(self.output(self.dropout(h)))
