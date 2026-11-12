import torch

from neuralforecast.common._modules import PositionalEmbedding
from neuralforecast.models.softssharp import PositionalEmbedding as SharpEmbedding


def test_positional_embedding_is_not_persistent_and_loads_legacy_state():
    embedding = PositionalEmbedding(hidden_size=8, max_len=16)
    legacy_pe = embedding.pe.clone()
    module = torch.nn.Sequential(embedding)

    assert "0.pe" not in module.state_dict()
    module.load_state_dict({"0.pe": legacy_pe}, strict=True)


def test_softssharp_embedding_is_not_persistent_and_loads_legacy_state():
    embedding = SharpEmbedding(d_series=8, max_len=16)
    legacy_pe = embedding.position_embedding.clone()
    module = torch.nn.Sequential(embedding)

    assert "0.position_embedding" not in module.state_dict()
    module.load_state_dict({"0.position_embedding": legacy_pe}, strict=True)
