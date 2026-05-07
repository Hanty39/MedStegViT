# =============================================================================
# positional_encoding.py — Poziční embeddingý pro Vision Transformer
#
# =============================================================================

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """
    Přidá naučitelné poziční embeddingý ke každému tokenu v sekvenci.

    Používá se v SCIFEncoder i SCIFDecoder před Transformer bloky.
    """

    def __init__(self, max_len, embed_dim):
        """
        Parametry:
            max_len:    maximální délka sekvence tokenů
                        (= počet patchů + 1 pro CLS/payload token)
                        Při patch_size=8 a feature mapě 256×256: (256/8)² + 1 = 1025
            embed_dim:  dimenze embeddingů (musí odpovídat rozměru tokenů = 256)
        """
        super().__init__()

        # Trénovatelný parametr tvaru [1, max_len, embed_dim]
        # Dimenze 1 na začátku = broadcast přes batch
        # Inicializace: randn() × 0.02 = malé náhodné hodnoty
        #
        # PROČ 0.02?
        # Původní randn() - poziční embeddings dominovaly nad vstupními tokeny
        self.pos_embedding = nn.Parameter(
            torch.randn(1, max_len, embed_dim) * 0.02
        )

    def forward(self, x):
        """
        Přičte poziční embeddingý ke vstupním tokenům.
        """
        return x + self.pos_embedding[:, :x.size(1), :]