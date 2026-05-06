# =============================================================================
# payload_decoder.py — MLP hlava pro dekódování payloadu z CLS tokenu
#
# Tento modul je součástí ViT větve SCIFDecoderu.
# Přijme CLS token (vektor dimenze embed_dim) z výstupu Transformer bloků
# a převede ho na 512 logitů — jeden logit pro každý bit payloadu.
#
# =============================================================================

import torch
import torch.nn as nn


class PayloadDecoder(nn.Module):
    """
    Jednoduchá MLP síť: CLS token [B, embed_dim] → 512 logitů [B, 512]

    Architektura:
        Linear(256→512) → LayerNorm → GELU → Dropout(0.1) → Linear(512→512)
    """

    def __init__(self, embed_dim=256, output_dim=512):
        super().__init__()

        # MLP s dvěma lineárními vrstvami
        # LayerNorm po první vrstvě: normalizuje aktivace → stabilnější trénink
        # GELU: nelinearita mezi vrstvami
        # Dropout(0.1): regularizace — zabraňuje overfittingu na tréninková data
        self.net = nn.Sequential(
            nn.Linear(embed_dim, 512),  # rozšíření z embed_dim (256) na 512
            nn.LayerNorm(512),           # normalizace — stabilizuje trénink
            nn.GELU(),                   # nelinearita (hladší než ReLU)
            nn.Dropout(0.1),             # 10% dropout pro regularizaci
            nn.Linear(512, output_dim)   # výstupní projekce: 512 → 512 logitů
        )

    def forward(self, x):
        """
        Parametry:
            x: CLS token [B, embed_dim]

        Vrátí:
            logity [B, output_dim]
        """
        return self.net(x)