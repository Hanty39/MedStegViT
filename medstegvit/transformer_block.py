# =============================================================================
# transformer_block.py — Jeden blok Vision Transformeru
#
# =============================================================================

import torch.nn as nn


class TransformerBlock(nn.Module):

    def __init__(self, embed_dim=256, num_heads=4, mlp_dim=768, dropout=0.2):
        """
        Parametry:
            embed_dim:  dimenze tokenů
            num_heads:  počet attention hlaviček

            mlp_dim:    dimenze skryté vrstvy MLP
            dropout:    pravděpodobnost dropout regularizace
        """
        super().__init__()

        # LayerNorm normalizuje každý token zvlášť (přes dimenzi embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)

        # Multi-Head Self-Attention
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

        # Dropout po attention — regularizace, zabraňuje overfittingu
        self.dropout1 = nn.Dropout(dropout)

        # Druhý LayerNorm před MLP
        self.norm2 = nn.LayerNorm(embed_dim)

        # MLP (Feed-Forward Network): dvě lineární vrstvy s GELU aktivací
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        """
        Parametry:
            x: tokeny [B, seq_len, embed_dim]

        Vrátí:
            tokeny po attention a MLP, stejný tvar [B, seq_len, embed_dim]
        """

        # Sub-blok 1: Multi-Head Self-Attention
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)

        x = x + self.dropout1(attn_out)

        # Sub-blok 2: MLP
        x_norm = self.norm2(x)
        x = x + self.mlp(x_norm)

        return x