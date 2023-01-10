import torch
from typing import TypedDict

class Model(TypedDict):
    library: str
    model: str
    hidden_layers: list
    activation: torch.nn.Module
    bias: bool
    dtype: torch.dtype

