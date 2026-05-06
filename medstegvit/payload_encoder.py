# =============================================================================
# payload_encoder.py — Zakódování 512 bitů payloadu pro použití v encoderu
#
# Tento modul převede 512 bitů payloadu do dvou reprezentací které encoder
# použije k vložení payloadu do snímku:
#
#   1. ViT token: komprimovaný vektor [B, embed_dim] vložený do transformer sekvence
#      jako první token
#
#   2. FiLM parametry (gamma, beta): vektory [B, embed_dim] pro Feature-wise
#      Linear Modulation — payload přímo moduluje feature mapu encoderu
#      kanál po kanálu: feature_mapa = gamma * feature_mapa + beta
#
# =============================================================================

import torch
import torch.nn as nn


class PayloadEncoder(nn.Module):
    """
    Převede 512 bitů payloadu na ViT token + FiLM parametry (gamma, beta).
    """

    def __init__(self, input_dim=512, embed_dim=256):
        super().__init__()

        # ── Větev 1: ViT token ────────────────────────────────────────
        # Komprimuje 512 bitů → embed_dim vektor (256)
        # Výsledný vektor se vloží jako první token do transformer sekvence.
        # Transformer bloky pak mohou pomocí attention propojit payload token
        # s image tokeny → payload ovlivní celou feature mapu přes attention.
        self.token_mlp = nn.Sequential(
            nn.Linear(input_dim, 256),  # redukce: 512 → 256
            nn.LayerNorm(256),           # normalizace
            nn.GELU(),                   # nelinearita
            nn.Dropout(0.1),             # regularizace
            nn.Linear(256, embed_dim)    # výstup: 256 → embed_dim (256)
        )

        # ── Větev 2: FiLM podmínění ───────────────────────────────────
        # FiLM (Feature-wise Linear Modulation) moduluje feature mapu:
        #   výstup = gamma * feature_mapa + beta
        # kde gamma a beta závisí na payloadu.
        #
        # Efekt: každý kanál feature mapy se škáluje (gamma) a posouvá (beta)
        # podle konkrétního payloadu → delta nese přímý otisk payloadu.
        #
        # Výstup má dimenzi embed_dim * 2 — první polovina = gamma, druhá = beta
        self.film_mlp = nn.Sequential(
            nn.Linear(input_dim, 256),       # redukce: 512 → 256
            nn.LayerNorm(256),                # normalizace
            nn.GELU(),                        # nelinearita
            nn.Linear(256, embed_dim * 2)    # výstup: 256 → 512 (gamma + beta dohromady)
        )

        # Inicializace FiLM vrstvy: gamma=1, beta=0
        # Při gamma=1, beta=0 platí: feature_mapa = 1 * feature_mapa + 0 = identita
        # → na začátku tréninku FiLM neovlivňuje feature mapu vůbec
        # → síť se nejdřív naučí základní pipeline, pak postupně FiLM přebírá
        nn.init.zeros_(self.film_mlp[-1].weight)             # váhy na nulu
        nn.init.zeros_(self.film_mlp[-1].bias)               # bias na nulu
        self.film_mlp[-1].bias.data[:embed_dim] = 1.0        # gamma bias = 1 → gamma = 1 na startu

    def forward(self, x):
        """
        Parametry:
            x: payload bity [B, 512], hodnoty 0.0 nebo 1.0

        Vrátí:
            token: ViT token [B, embed_dim]
            gamma: FiLM škálovací faktor [B, embed_dim]
            beta:  FiLM posun [B, embed_dim]
        """

        # ViT token — jde jako první token do transformer sekvence
        token = self.token_mlp(x)           # [B, 512] → [B, embed_dim]

        # FiLM parametry — gamma a beta jsou zakódovány v jednom vektoru
        film      = self.film_mlp(x)        # [B, 512] → [B, embed_dim * 2]
        embed_dim = token.shape[-1]
        gamma = film[:, :embed_dim]         # první polovina: škálovací faktory [B, embed_dim]
        beta  = film[:, embed_dim:]         # druhá polovina: posuvy [B, embed_dim]

        return token, gamma, beta