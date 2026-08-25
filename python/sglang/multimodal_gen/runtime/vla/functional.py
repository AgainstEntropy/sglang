# SPDX-License-Identifier: Apache-2.0
"""Model-agnostic mask helpers shared by VLA policy implementations."""

from __future__ import annotations

import torch


def make_att_2d_masks(
    pad_masks: torch.Tensor,
    att_masks: torch.Tensor,
) -> torch.Tensor:
    """Expand big_vision-style 1D block masks into a 2D attention mask.

    ``att_masks`` is 1 where a token starts a new attention block and 0 where it
    shares the previous token's block; a token attends to every valid token
    whose cumulative block id is <= its own.
    """
    if att_masks.ndim != 2 or pad_masks.ndim != 2:
        raise ValueError("pad_masks and att_masks must be [batch, seq]")
    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    return att_2d_masks & pad_2d_masks
