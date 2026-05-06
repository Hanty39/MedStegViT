# =============================================================================
# model_scif_net.py — Hlavní architektura encoderu a decoderu (SCIF-Net)
#
# Obsahuje dvě třídy:
#
#   SCIFEncoder: vloží 512 bitů payloadu do DICOM snímku jako neviditelnou perturbaci
#                Vstup:  snímek [B,1,512,512] + bity [B,512]
#                Výstup: stego snímek [B,1,512,512] + maska [B,1,512,512]
#
#   SCIFDecoder: extrahuje 512 bitů payloadu ze stego snímku
#                Vstup:  stego snímek [B,1,512,512]
#                Výstup: logity [B,512]  (bez sigmoidu)
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F

from medstegvit.structure_extractor import StructureExtractor
from medstegvit.patch_embed import PatchEmbed
from medstegvit.transformer_block import TransformerBlock
from medstegvit.positional_encoding import PositionalEncoding
from medstegvit.payload_encoder import PayloadEncoder
from medstegvit.payload_decoder import PayloadDecoder


# =============================================================================
# SCIF ENCODER
#
# Pipeline:
#   snímek → CNN backbone → feature mapa
#   snímek → StructureExtractor → anatomická maska
#   bity   → PayloadEncoder → ViT token + FiLM (gamma, beta)
#
#   feature mapa → PatchEmbed → tokeny
#   [payload_token, image_tokeny] → poziční encoding → Transformer bloky
#   → image_tokeny → reshape na feature mapu → FiLM modulace
#   → delta (perturbace) → maskování → stego snímek
# =============================================================================

class SCIFEncoder(nn.Module):

    def __init__(self, cfg):
        super().__init__()

        # Rozbalení hyperparametrů z configu
        embed_dim    = cfg["payload"]["embedding_dim"]   # 256
        payload_bits = cfg["payload"]["payload_bits"]    # 512
        patch_size   = cfg["model"]["patch_size"]        # 8
        vit_layers   = cfg["model"]["vit_layers"]        # 6
        vit_heads    = cfg["model"]["vit_heads"]         # 4
        vit_mlp      = cfg["model"]["vit_mlp_dim"]       # 768
        dropout      = cfg["model"]["dropout"]           # 0.2

        # CNN backbone: extrahuje bohaté příznaky ze snímku
        # 3 konvoluční vrstvy postupně zvyšují počet kanálů: 1 → 64 → 128 → 256
        # Výstup má stejné rozlišení jako vstup (padding=1 zachová rozměry)
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.Conv2d(128, embed_dim, 3, padding=1), nn.ReLU()
        )

        # StructureExtractor: generuje anatomickou masku [0, 1]
        # Trénuje se společně s encoderem — naučí se kde v RTG snímku jsou hrany
        self.structure_extractor = StructureExtractor(channels=32)

        # PatchEmbed: rozdělí feature mapu na tokeny pro ViT
        self.patch_embed = PatchEmbed(embed_dim=embed_dim, patch_size=patch_size)

        # PayloadEncoder: převede 512 bitů → ViT token + FiLM parametry
        self.payload_encoder = PayloadEncoder(input_dim=payload_bits, embed_dim=embed_dim)

        # 6 Transformer bloků — zpracují sekvenci [payload_token + image_tokeny]
        # ModuleList registruje všechny bloky jako parametry modelu
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim=embed_dim, num_heads=vit_heads,
                             mlp_dim=vit_mlp, dropout=dropout)
            for _ in range(vit_layers)
        ])

        # Počet image tokenů: snímek se interpoluje na 0.5× → 256×256,
        # pak PatchEmbed s patch_size=8 → (256/8)² = 1024 tokenů
        # +1 pro payload token který je přidán jako první
        _n_patches = (512 // 2 // patch_size) ** 2
        self.positional_encoding = PositionalEncoding(max_len=_n_patches + 1, embed_dim=embed_dim)

        # Residual head: převede transformer výstup (feature mapu) na deltu
        # Delta = perturbace která se přičte ke snímku
        self.residual_head = nn.Sequential(
            nn.Conv2d(embed_dim, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 1, 3, padding=1)   # výstup: 1 kanál = jedna hodnota perturbace na pixel
        )

        # mask_strength: říká jak silně maska filtruje deltu
        # 0.0 = delta prochází bez filtrace (na začátku tréninku)
        # 1.0 = plné maskování (na konci warm-up fáze)
        # Nastavuje se zvenku v main.py každou epochu (warm-up schedule)
        self.mask_strength = 0.0

        # delta_scale: škálovací faktor perturbace
        # stego = img + delta_masked * delta_scale
        # Nižší hodnota = menší perturbace = neviditelnější ale těžší dekódování
        self.delta_scale   = cfg["model"]["delta_scale"]

    def forward(self, img, bits):
        """
        Vloží 512 bitů payloadu do snímku jako neviditelnou perturbaci.
        """

        # CNN příznaky ze snímku: [B,1,512,512] → [B,256,512,512]
        features = self.backbone(img)

        # Anatomická maska: [B,1,512,512] → hodnoty [0,1]
        mask = self.structure_extractor(img)

        # Interpolace feature mapy na 0.5× pro ViT (redukce výpočtu)
        # [B,256,512,512] → [B,256,256,256]
        features_small = F.interpolate(features, scale_factor=0.5,
                                        mode="bilinear", align_corners=False)

        # PatchEmbed: feature mapa → tokeny sekvence
        # [B,256,256,256] → [B,1024,256]  (1024 tokenů, každý 256-dimenzionální)
        tokens = self.patch_embed(features_small)

        # PayloadEncoder: 512 bitů → ViT token + FiLM parametry
        payload_token, gamma, beta = self.payload_encoder(bits)

        # Přidá payload token jako první token v sekvenci
        # [B,1024,256] → [B,1025,256]
        tokens = torch.cat([payload_token.unsqueeze(1), tokens], dim=1)

        # Přičte poziční embeddingý — transformer se dozví kde je který token
        tokens = self.positional_encoding(tokens)

        # Zpracování 6 Transformer bloky — payload token interaguje s image tokeny
        for block in self.transformer_blocks:
            tokens = block(tokens)

        # Extrakce image tokenů (bez payload tokenu na pozici 0)
        image_tokens = tokens[:, 1:, :]   # [B, 1024, 256]

        # Reshape tokenů zpět na 2D feature mapu
        B, N, C = image_tokens.shape
        grid = int(N ** 0.5)             # 1024 tokenů → 32×32 grid
        feature_map = image_tokens.transpose(1, 2).reshape(B, C, grid, grid)
        # [B, 256, 32, 32]

        # Upsample zpět na původní rozlišení feature mapy
        feature_map = F.interpolate(feature_map, size=features.shape[-2:],
                                     mode="bilinear", align_corners=False)
        # [B, 256, 512, 512]

        # FiLM modulace: feature_mapa = gamma * feature_mapa + beta
        # gamma a beta závisí na payloadu → feature mapa nese payload informaci
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)  # [B, 256] → [B, 256, 1, 1] pro broadcast
        beta  = beta.unsqueeze(-1).unsqueeze(-1)
        feature_map = gamma * feature_map + beta

        # Residual head: feature mapa → raw delta, omezena tanh na [-1, 1]
        delta = torch.tanh(self.residual_head(feature_map))  # [B, 1, 512, 512]

        # Přímá payload modulace: každý bit škáluje příslušnou prostorovou oblast delty
        # 512 bitů → prostorová mřížka 16×32 → upsample na rozlišení delty
        # Efekt: oblasti snímku odpovídající bitu=1 mají o 7.5% větší deltu
        #        oblasti odpovídající bitu=0 mají o 7.5% menší deltu
        # mode='nearest': zachová ostré hranice (bilinear by bity rozmazal)
        B, C, H, W = feature_map.shape
        payload_map = bits.view(B, 1, 16, 32)                         # [B, 1, 16, 32]
        payload_map = F.interpolate(payload_map, size=(H, W), mode='nearest')  # [B, 1, H, W]
        delta = delta * (1.0 + 0.15 * (payload_map - 0.5))
        # payload=1 → faktor 1.075, payload=0 → faktor 0.925

        # Vyhlazení masky průměrovacím filtrem 15×15 (zabrání ostrým artefaktům)
        mask_smooth = F.avg_pool2d(mask, kernel_size=15, stride=1, padding=7)

        # Aplikace masky na deltu podle aktuální mask_strength
        # mask_strength=0: delta_masked = delta * 1.0 (maska nemá vliv)
        # mask_strength=1: delta_masked = delta * mask_smooth (plné maskování)
        delta_masked = delta * (mask_smooth * self.mask_strength + (1.0 - self.mask_strength))

        # Přičtení perturbace ke snímku a clamp na [0, 1]
        # clamp je lineární → nezkresluje perturbaci (sigmoid zesiloval 2-4×
        # v jasných/tmavých oblastech RTG, což ničilo PSNR)
        stego = torch.clamp(img + delta_masked * self.delta_scale, 0.0, 1.0)

        return stego, mask_smooth


# =============================================================================
# SCIF DECODER
#
# Dekóduje 512 bitů payloadu ze stego snímku pomocí dvou paralelních větví:
#
#   ViT větev: stego → CNN → PatchEmbed → Transformer → CLS token → logity_vit
#
#
#   CNN větev: stego → CNN → AdaptiveAvgPool(16,32) → Conv1×1 → logity_cnn
#
#
#   Výsledek: logity = branch_weight × logity_vit + (1 - branch_weight) × logity_cnn
#   branch_weight je trénovatelný parametr — naučí se optimální mix obou větví
# =============================================================================

class SCIFDecoder(nn.Module):

    def __init__(self, cfg):
        super().__init__()

        embed_dim    = cfg["payload"]["embedding_dim"]   # 256
        payload_bits = cfg["payload"]["payload_bits"]    # 512
        patch_size   = cfg["model"]["patch_size"]        # 8
        vit_layers   = cfg["model"]["vit_layers"]        # 6
        vit_heads    = cfg["model"]["vit_heads"]         # 4
        vit_mlp      = cfg["model"]["vit_mlp_dim"]       # 768
        dropout      = cfg["model"]["dropout"]           # 0.2

        # ── ViT větev ─────────────────────────────────────────────────
        # Jednodušší backbone než encoder (2 vrstvy místo 3)
        self.vit_backbone = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, embed_dim, 3, padding=1), nn.ReLU()
        )

        self.patch_embed = PatchEmbed(embed_dim=embed_dim, patch_size=patch_size)

        # CLS token: naučitelný vektor přidaný na začátek sekvence
        # Po průchodu transformerem obsahuje agregovanou informaci o celém snímku
        # → PayloadDecoder z něj dekóduje logity
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim=embed_dim, num_heads=vit_heads,
                             mlp_dim=vit_mlp, dropout=dropout)
            for _ in range(vit_layers)
        ])

        _n_patches = (512 // 2 // patch_size) ** 2
        self.positional_encoding = PositionalEncoding(max_len=_n_patches + 1, embed_dim=embed_dim)

        # PayloadDecoder: CLS token [B, embed_dim] → logity [B, 512]
        self.payload_decoder_vit = PayloadDecoder(embed_dim=embed_dim, output_dim=payload_bits)

        # ── CNN větev ─────────────────────────────────────────────────
        # 4 konvoluční vrstvy se stride=2 postupně zmenší rozlišení:
        # 512 → 256 → 128 → 64 → 32, kanály: 1 → 32 → 64 → 128 → 256
        self.cnn_branch = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),    # 512→256, 1→32 kanálů
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),   # 256→128, 32→64 kanálů
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),  # 128→64, 64→128 kanálů
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1), # 64→32, 128→256 kanálů
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((16, 32)),
        )

        self.cnn_head = nn.Sequential(
            nn.Flatten(),
        )

        # 1×1 konvoluce: redukuje 256 kanálů na 1 kanál = jeden logit na prostorovou buňku
        # [B, 256, 16, 32] → [B, 64, 16, 32] → [B, 1, 16, 32] → flatten → [B, 512]
        self.cnn_spatial_conv = nn.Sequential(
            nn.Conv2d(256, 64, 1),  # 256 → 64 kanálů (1×1 konvoluce = lineární projekce)
            nn.ReLU(),
            nn.Conv2d(64, 1, 1),    # 64 → 1 kanál = jeden logit na pixel
        )

        # Trénovatelný váhový koeficient pro mix ViT a CNN větve
        # branch_weight prochází sigmoid → w ∈ (0, 1)
        # logity = w * logity_vit + (1-w) * logity_cnn
        # Inicializace 0.5 → sigmoid(0.5) ≈ 0.62 → mírně preferuje ViT větev
        self.branch_weight = nn.Parameter(torch.tensor(0.5))

    def forward(self, img):
        """
        Extrahuje 512 logitů payloadu ze stego snímku.
        """

        # ── ViT větev ─────────────────────────────────────────────────
        features = self.vit_backbone(img)              # [B,1,512,512] → [B,256,512,512]
        features_small = F.interpolate(features, scale_factor=0.5,
                                        mode="bilinear", align_corners=False)  # → [B,256,256,256]
        tokens = self.patch_embed(features_small)      # → [B, 1024, 256]
        B = tokens.shape[0]

        # Přidá CLS token na začátek sekvence — po transformeru z něj dekódujeme payload
        cls_tokens = self.cls_token.expand(B, -1, -1)  # [1,1,256] → [B,1,256]
        tokens = torch.cat([cls_tokens, tokens], dim=1)  # [B,1025,256]
        tokens = self.positional_encoding(tokens)

        for block in self.transformer_blocks:
            tokens = block(tokens)

        # CLS token (pozice 0) po transformeru obsahuje info o celém snímku
        logits_vit = self.payload_decoder_vit(tokens[:, 0])  # [B,256] → [B,512]

        # ── CNN větev ─────────────────────────────────────────────────
        cnn_features = self.cnn_branch(img)               # [B,1,512,512] → [B,256,16,32]
        logits_cnn   = self.cnn_spatial_conv(cnn_features) # [B,256,16,32] → [B,1,16,32]
        logits_cnn   = logits_cnn.view(logits_cnn.shape[0], -1)  # [B,1,16,32] → [B,512]

        # ── Kombinace obou větví ───────────────────────────────────────
        w = torch.sigmoid(self.branch_weight)  # naučitelná váha ∈ (0, 1)
        logits = w * logits_vit + (1.0 - w) * logits_cnn

        return logits