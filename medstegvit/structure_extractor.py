# =============================================================================
# structure_extractor.py — Adaptivní extrakce anatomické masky ze snímku
#
# Tento modul generuje anatomickou masku která říká encoderu KDE v snímku
# má vkládat perturbaci (payload). Maska má hodnoty [0, 1]:
#   1.0 = tady je hrana nebo anatomická struktura → encoder smí psát
#   0.0 = tady je homogenní oblast nebo pozadí → encoder by neměl psát
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F


class StructureExtractor(nn.Module):

    def __init__(self, channels=32):
        super().__init__()

        # ----------------------------------------------------------------
        # UNet ENCODER — extrakce příznaků na 3 škálách
        # Každá úroveň zdvojnásobí počet kanálů a zachytí jemnější struktury
        # ----------------------------------------------------------------

        # Úroveň 1: plné rozlišení 512×512, 32 kanálů
        # Zachytí jemné hrany a textury (vysoké frekvence)
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )

        # Úroveň 2: 256×256 (po MaxPool), 64 kanálů
        # Zachytí středně velké struktury (žebra, okraje orgánů)
        self.enc2 = nn.Sequential(
            nn.Conv2d(channels, channels * 2, 3, padding=1),
            nn.BatchNorm2d(channels * 2),
            nn.ReLU(),
            nn.Conv2d(channels * 2, channels * 2, 3, padding=1),
            nn.BatchNorm2d(channels * 2),
            nn.ReLU()
        )

        # Úroveň 3: 128×128 (po MaxPool), 128 kanálů
        # Zachytí globální anatomické struktury (tvar plic, srdce)
        self.enc3 = nn.Sequential(
            nn.Conv2d(channels * 2, channels * 4, 3, padding=1),
            nn.BatchNorm2d(channels * 4),
            nn.ReLU(),
            nn.Conv2d(channels * 4, channels * 4, 3, padding=1),
            nn.BatchNorm2d(channels * 4),
            nn.ReLU()
        )

        # ----------------------------------------------------------------
        # SE BLOKY — Channel Attention pro každou škálu
        # SE blok se naučí které kanály jsou pro detekci struktur důležité
        # a zesílí je, zatímco potlačí méně důležité kanály.
        # ----------------------------------------------------------------
        self.se1 = self._se_block(channels)        # pro enc1 výstup (32 kanálů)
        self.se2 = self._se_block(channels * 2)    # pro enc2 výstup (64 kanálů)
        self.se3 = self._se_block(channels * 4)    # pro enc3 výstup (128 kanálů)

        # ----------------------------------------------------------------
        # PROSTOROVÁ ATTENTION (CBAM styl) — "kde hledat"
        # Aplikuje se na nejhlubší reprezentaci (enc3)
        # Kombinuje průměr a maximum přes kanály → dvoukanálová mapa → konvoluce → sigmoid
        # Výsledek: mapa [0, 1] která říká kde v obraze jsou důležité struktury
        # ----------------------------------------------------------------
        self.spatial_att = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3),
            nn.Sigmoid()
        )

        # ----------------------------------------------------------------
        # UNet DECODER — rekonstrukce masky v plném rozlišení
        # Skip connections: přilepí encoder příznaky k upsamplovaným příznakům
        # → decoder má přístup jak k jemným (enc1) tak globálním (enc3) příznaků
        # ----------------------------------------------------------------

        # Dekóder úroveň 2: spojí enc3 (upsamplované) + enc2 → výstup 64 kanálů
        self.dec2 = nn.Sequential(
            nn.Conv2d(channels * 4 + channels * 2, channels * 2, 3, padding=1),
            nn.BatchNorm2d(channels * 2),
            nn.ReLU()
        )

        # Dekóder úroveň 1: spojí dec2 (upsamplované) + enc1 → výstup 32 kanálů
        self.dec1 = nn.Sequential(
            nn.Conv2d(channels * 2 + channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )

        # ----------------------------------------------------------------
        # VÝSTUPNÍ HLAVA — převede 32 kanálů na masku [0, 1]
        # ----------------------------------------------------------------
        self.out_conv = nn.Conv2d(channels, 1, 1)

        self.mask_temperature = 2.0

        self.pool = nn.MaxPool2d(2)
        self.up   = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def _se_block(self, channels, reduction=4):
        """
        Vytvoří SE (Squeeze-and-Excitation) blok pro zadaný počet kanálů.
        """
        return nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, max(channels // reduction, 4)),
            nn.ReLU(),
            nn.Linear(max(channels // reduction, 4), channels),
            nn.Sigmoid()
        )

    def _apply_se(self, x, se_block):
        """
        Aplikuje SE blok na feature mapu — přenásobí každý kanál jeho váhou.
        """
        # SE blok vrátí [B, C] — potřebujeme přidat prostorové dimenze pro broadcast
        scale = se_block(x).unsqueeze(-1).unsqueeze(-1)
        return x * scale  # broadcast: každý kanál se přenásobí svou vahou

    def _apply_spatial(self, x):
        """
        Aplikuje prostorovou attention na feature mapu — zdůrazní důležité pozice.
        """
        avg = torch.mean(x, dim=1, keepdim=True)
        mx  = torch.max(x,  dim=1, keepdim=True).values
        att = self.spatial_att(torch.cat([avg, mx], dim=1))
        return x * att  # přenásob feature mapu prostorovou attention mapou

    def forward(self, x):
        """
        Parametry:
            x: DICOM snímek [B, 1, 512, 512], hodnoty [0, 1]

        Vrátí:
            mask: anatomická maska [B, 1, 512, 512], hodnoty [0, 1]
                  1.0 = hrana/struktura kde encoder smí psát
                  0.0 = homogenní oblast kde encoder psát nemá
        """

        # ── UNet Encoder ──────────────────────────────────────────────
        e1 = self.enc1(x)              # [B, 1, 512, 512] - [B, 32, 512, 512]
        e1 = self._apply_se(e1, self.se1)  # channel attention na škále 1

        e2 = self.enc2(self.pool(e1))  # MaxPool 512→256, pak conv: [B, 64, 256, 256]
        e2 = self._apply_se(e2, self.se2)  # channel attention na škále 2

        e3 = self.enc3(self.pool(e2))  # MaxPool 256→128, pak conv: [B, 128, 128, 128]
        e3 = self._apply_se(e3, self.se3)  # channel attention na škále 3

        e3 = self._apply_spatial(e3)   # zdůrazní kde jsou globální struktury

        # ── UNet Decoder se skip connections ──────────────────────────
        # up(e3): upsample 128×128 → 256×256
        d2 = self.dec2(torch.cat([self.up(e3), e2], dim=1))  # [B, 192, 256, 256] - [B, 64, 256, 256]

        # up(d2): upsample 256×256 → 512×512
        d1 = self.dec1(torch.cat([self.up(d2), e1], dim=1))  # [B, 96, 512, 512] - [B, 32, 512, 512]

        # Výstupní hlava: 32 kanálů → 1 kanál, sigmoid s teplotou
        mask = torch.sigmoid(self.out_conv(d1) * self.mask_temperature)  # [B, 1, 512, 512]

        return mask