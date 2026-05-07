# =============================================================================
# patch_embed.py — Rozdělení feature mapy na patche pro Vision Transformer
# =============================================================================

import torch
import torch.nn as nn


class PatchEmbed(nn.Module):
    """
    Převede 2D feature mapu [B, C, H, W] na sekvenci tokenů [B, N, C]
    kde N = (H/patch_size) × (W/patch_size) je počet patchů.
    """

    def __init__(self, embed_dim=128, patch_size=4):
        """
        Parametry:
            embed_dim:  počet kanálů vstupní feature mapy (= dimenze embeddingů)
            patch_size: velikost jednoho patche v pixelech (8 × 8 dle configu)
        """
        super().__init__()
        self.patch_size = patch_size

        self.proj = nn.Conv2d(
            embed_dim,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        """
        Parametry:
            x: feature mapa [B, C, H, W]

        Vrátí:
            tokeny [B, N, C] kde N = (H/patch_size) × (W/patch_size)

        Příklad s configem (patch_size=8, feature mapa 256×256):
            vstup:  [B, 256, 256, 256]
            po proj: [B, 256, 32, 32]   → 32×32 = 1024 patchů
            výstup: [B, 1024, 256]       → 1024 tokenů dimenze 256
        """
        x = self.proj(x)       # [B, C, H, W] → [B, C, H/ps, W/ps]  (ps = patch_size)
        B, C, H, W = x.shape

        x = x.flatten(2)       # [B, C, H/ps, W/ps] → [B, C, H/ps × W/ps]  (sloučí prostorové dimenze)
        x = x.transpose(1, 2)  # [B, C, N] → [B, N, C]  (transformer očekává tokeny jako druhý rozměr)

        return x