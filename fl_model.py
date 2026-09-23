"""
Definición del modelo compartida por el algoritmo C2D y el orquestador.

El orquestador y la imagen `fl-local-train` DEBEN usar exactamente la misma
arquitectura: los deltas se agregan tensor a tensor por nombre y forma.
Cualquier cambio aquí implica una nueva versión de la imagen (nuevo digest)
y, por tanto, una nueva aprobación del data owner.
"""
from __future__ import annotations

import torch
from torch import nn

MODEL_VERSION = "mlp-20-64-64-10/v1"


def build_model(n_features: int = 20, n_classes: int = 10, hidden: tuple[int, ...] = (64, 64)) -> nn.Module:
    layers: list[nn.Module] = []
    d = n_features
    for h in hidden:
        layers += [nn.Linear(d, h), nn.ReLU()]
        d = h
    layers.append(nn.Linear(d, n_classes))
    return nn.Sequential(*layers)


def state_to_float32(sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {k: v.detach().to(torch.float32).contiguous().clone() for k, v in sd.items()}
