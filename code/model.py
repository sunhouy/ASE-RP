import math

import torch
from torch import nn
from torch_geometric.nn import GatedGraphConv


class POIGraph(nn.Module):
    def __init__(self, num_nodes: int, embedding_dim: int):
        super().__init__()
        self.embedding_dim = int(embedding_dim)
        self.embedding = nn.Embedding(int(num_nodes), self.embedding_dim)
        self.ggnn = GatedGraphConv(self.embedding_dim, num_layers=2)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        bound = 1.0 / math.sqrt(self.embedding_dim)
        for parameter in self.parameters():
            parameter.data.uniform_(-bound, bound)

    def forward(self, node_indices: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.ggnn(self.embedding(node_indices), edge_index)


class UserEmbeddings(nn.Module):
    def __init__(self, num_users: int, embedding_dim: int):
        super().__init__()
        self.embedding_dim = int(embedding_dim)
        self.embedding = nn.Embedding(int(num_users), self.embedding_dim)
        self.projection = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.activation = nn.LeakyReLU(0.2)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        bound = 1.0 / math.sqrt(self.embedding_dim)
        for parameter in self.parameters():
            parameter.data.uniform_(-bound, bound)

    def forward(
        self,
        user_indices: torch.Tensor,
        preference_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        embeddings = self.embedding(user_indices) + preference_embeddings
        return self.activation(self.projection(embeddings))


class FuseEmbeddings(nn.Module):
    def __init__(self, first_dim: int, second_dim: int):
        super().__init__()
        fused_dim = int(first_dim) + int(second_dim)
        self.projection = nn.Linear(fused_dim, fused_dim)
        self.activation = nn.LeakyReLU(0.2)

    def forward(
        self,
        first: torch.Tensor,
        second: torch.Tensor,
    ) -> torch.Tensor:
        return self.activation(self.projection(torch.cat((first, second), dim=-1)))


class Time2Vec(nn.Module):
    def __init__(self, output_dim: int):
        super().__init__()
        output_dim = int(output_dim)
        if output_dim < 2:
            raise ValueError("Time2Vec output_dim must be at least 2")
        self.periodic_weight = nn.Parameter(torch.randn(1, output_dim - 1))
        self.periodic_bias = nn.Parameter(torch.randn(1, output_dim - 1))
        self.linear_weight = nn.Parameter(torch.randn(1, 1))
        self.linear_bias = nn.Parameter(torch.randn(1, 1))

    def forward(self, time_values: torch.Tensor) -> torch.Tensor:
        periodic = torch.sin(
            time_values @ self.periodic_weight + self.periodic_bias
        )
        linear = time_values @ self.linear_weight + self.linear_bias
        return torch.cat((periodic, linear), dim=-1)


class PositionalEncoding(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embedding_dim = int(embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        length = inputs.size(1)
        position = torch.arange(
            length, device=inputs.device, dtype=inputs.dtype
        ).unsqueeze(1)
        divisor = torch.exp(
            torch.arange(
                0,
                self.embedding_dim,
                2,
                device=inputs.device,
                dtype=inputs.dtype,
            )
            * (-math.log(10000.0) / self.embedding_dim)
        )
        encoding = torch.zeros(
            length,
            self.embedding_dim,
            device=inputs.device,
            dtype=inputs.dtype,
        )
        encoding[:, 0::2] = torch.sin(position * divisor)
        encoding[:, 1::2] = torch.cos(position * divisor)
        return self.dropout(inputs + encoding.unsqueeze(0))


class TransformerModel(nn.Module):
    def __init__(
        self,
        num_pois: int,
        num_regions: int,
        num_times: int,
        embedding_dim: int,
        num_heads: int,
        feedforward_dim: int,
        num_layers: int,
        num_categories: int,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.embedding_dim = int(embedding_dim)
        self.position = PositionalEncoding(self.embedding_dim, dropout)
        layer = nn.TransformerEncoderLayer(
            self.embedding_dim,
            int(num_heads),
            int(feedforward_dim),
            dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, int(num_layers))
        self.decoder_poi = nn.Linear(self.embedding_dim, int(num_pois))
        self.decoder_region = nn.Linear(self.embedding_dim, int(num_regions))
        self.decoder_time = nn.Linear(self.embedding_dim, int(num_times))
        self.decoder_cat = nn.Linear(self.embedding_dim, int(num_categories))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for head in (
            self.decoder_poi,
            self.decoder_region,
            self.decoder_time,
            self.decoder_cat,
        ):
            head.bias.data.zero_()
            head.weight.data.uniform_(-0.1, 0.1)

    def forward(
        self,
        inputs: torch.Tensor,
        causal_mask: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        causal_mask = causal_mask.to(device=inputs.device)
        padding_mask = padding_mask.to(device=inputs.device)
        encoded = self.position(inputs * math.sqrt(self.embedding_dim))
        hidden = self.encoder(encoded, causal_mask, padding_mask)
        return (
            self.decoder_poi(hidden),
            self.decoder_time(hidden),
            self.decoder_region(hidden),
            self.decoder_cat(hidden),
        )
