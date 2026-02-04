from __future__ import annotations
import torch


def tile_design(design: torch.Tensor) -> torch.Tensor:
    """Tile a microstructure design into a 3x3 mosaic with dimmed neighbors.

    Parameters
    ----------
    design : (32, 32) torch.Tensor
        Microstructure design.

    Returns
    -------
    tiled : (96, 96) torch.Tensor
        Tiled microstructure design (with dimmed neighbors).

    """
    light = 0.5 * design

    row0 = torch.cat([light, light, light], dim=1)
    row1 = torch.cat([light, design, light], dim=1)
    row2 = torch.cat([light, light, light], dim=1)

    return torch.cat([row0, row1, row2], dim=0)
