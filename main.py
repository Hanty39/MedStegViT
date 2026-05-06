# =============================================================================
# main.py — Hlavní tréninkový skript pro MedStegViT
#
# Celková architektura systému:
#   1. Načteme DICOM snímky (RTG hrudník) z disku
#   2. Pro každý snímek vytvoříme payload = SHA-256 hash PatientID (256 bitů),
#      zakódovaný LDPC kódem → 512 bitů (rate 0.50, soft-decision dekódování)
#   3. Encoder (SCIFEncoder) vloží 512 bitů do snímku jako neviditelné změny
#      → výstup je "stego snímek" vizuálně nerozeznatelný od originálu
#   4. Decoder (SCIFDecoder) ze stego snímku extrahuje 512 logitů zpět
#      → LDPC soft-decision dekódér opraví bitové chyby a rekonstruuje SHA-256 hash
# =============================================================================

import os
import sys
import io
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.amp as amp
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from datetime import datetime

# Vlastní moduly projektu
from medstegvit.config import load_config           # načte default.yaml
from medstegvit.dicom_loader import load_dicom_512  # načte .dcm soubor jako 512×512 float tensor
from medstegvit.payload import hash_patient_id, bytes_to_bits  # SHA-256 hash + převod na bity
from medstegvit.rs_codec import rs_encode           # Reed-Solomon
from medstegvit.ldpc_codec import ldpc_encode, ldpc_decode_soft, get_codec as get_ldpc_codec  # LDPC soft-decision
from medstegvit.model_scif_net import SCIFEncoder, SCIFDecoder  # enkodér a dekodér neuronové sítě
from medstegvit.losses import (
    ssim_loss,
    gradient_loss,
    energy_loss,
    mask_aware_loss,  # penalizuje změny MIMO anatomickou masku
    sparsity_loss  # nutí masku být řídkou — penalizuje odchylku průměrné hodnoty masky
                    # od cílových 15 % aktivních pixelů
                    # Bez této penalizace by SE síť zvolila nejjednodušší řešení
)

# =============================================================================
# TIMESTAMPED OUTPUT FILES
# Každý run dostane unikátní ID (timestamp) → logy a checkpointy se nepřepisují
# =============================================================================

# Unikátní identifikátor runu ve formátu YYYYMMDD_HHMMSS
RUN_ID   = datetime.now().strftime("%Y%m%d_%H%M%S")
# Název souboru pro log tohoto runu
LOG_FILE = f"log_{RUN_ID}.txt"

# -----------------------------------------------------------------------------
# Třída Tee: přesměruje stdout zároveň na terminál i do log souboru
# -----------------------------------------------------------------------------
class Tee:
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log      = open(filepath, "w", encoding="utf-8")  # otevře log soubor pro zápis
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
    def flush(self):
        self.terminal.flush()
        self.log.flush()
    def close(self):
        self.log.close()

sys.stdout = Tee(LOG_FILE)

print("=== MedStegViT · Single-Stage Training ===")
print(f"Run ID : {RUN_ID}")
print(f"Log    : {LOG_FILE}")
print("-" * 60)

# Načte konfiguraci z cfg/default.yaml
cfg    = load_config()
# Použije GPU pokud je dostupná, jinak CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("CUDA available:", torch.cuda.is_available())
print("Device:", device)

# =============================================================================
# CONFIG — rozbalení hyperparametrů z yaml do lokálních proměnných
# =============================================================================

total_epochs  = cfg["training"]["total_epochs"]       # celkový počet epoch tréninku
mask_start    = cfg["training"]["mask_start_epoch"]   # epocha kdy maska dosáhne plné síly (warm-up target)
psnr_start    = cfg["training"]["psnr_start_epoch"]   # epocha od které se aktivuje PSNR penalizace
target_psnr   = cfg["training"]["target_psnr"]        # cílová hodnota PSNR v dB
psnr_weight   = cfg["training"]["psnr_weight"]        # váha PSNR penalizace v loss funkci
lr            = cfg["training"]["learning_rate"]      # základní learning rate

pw  = cfg["training"]["payload_weight"]    # váha payload loss (jak moc záleží na správném dekódování)
mw  = cfg["training"]["mask_weight"]       # maximální váha mask_aware_loss (dosažena na epoch mask_start)
spw = cfg["training"]["sparsity_weight"]   # váha sparsity loss (nutí masku být řídkou)

write_loss_threshold = cfg["training"]["write_loss_threshold"]  # minimální požadovaný průměrný |delta|
write_loss_weight    = cfg["training"]["write_loss_weight"]     # penalizace pokud encoder píše příliš slabě
# lambda_outside: kolikrát více penalizovat změny MIMO masku oproti uvnitř
# fallback na 3.0 pokud klíč v yaml chybí (zpětná kompatibilita)
lambda_outside       = cfg["training"].get("lambda_outside", 3.0)

grad_clip    = cfg["training"]["grad_clip"]    # maximální norma gradientu před clippingem
use_amp      = cfg["training"]["use_amp"]      # automatická smíšená přesnost (False = vypnuto)
log_interval = cfg["training"]["log_interval"] # každých N epoch vypíše statistiky
batch_size   = cfg["training"].get("batch_size", 1)  # počet snímků per gradient step (výchozí 1 = online)

n_train   = cfg["data"]["n_train"]               # počet tréninkových snímků (před rozdělením)
n_test    = cfg["data"]["n_test"]                # počet testovacích snímků
val_split = cfg["data"].get("val_split", 0.15)  # podíl val z train (výchozí 15 %)

# Nastavení checkpointů
SAVE_CKPT      = cfg["checkpoints"]["save_checkpoint"]   # True = ukládat checkpointy
LOAD_CKPT      = cfg["checkpoints"]["load_checkpoint"]   # "" = trénink od nuly, jinak cesta k .pt souboru
CKPT_INTERVAL  = cfg["checkpoints"]["checkpoint_interval"]  # každých N epoch uloží periodický checkpoint
CKPT_DIR       = "checkpoints"
os.makedirs(CKPT_DIR, exist_ok=True)  # vytvoří adresář pokud neexistuje
# Cesty k checkpointovým souborům pro tento run
CKPT_FILE      = os.path.join(CKPT_DIR, f"ckpt_{RUN_ID}.pt")       # finální checkpoint po tréninku
CKPT_BEST_FILE = os.path.join(CKPT_DIR, f"ckpt_{RUN_ID}_best.pt")  # nejlepší checkpoint podle BitAcc

# Složky pro výstupní soubory
GRAPHS_DIR      = "graphs"        # grafy průběhu metrik během tréninku
FINAL_IMGS_DIR  = "final_images"  # jednotlivé vizualizační obrázky z evaluace
os.makedirs(GRAPHS_DIR,     exist_ok=True)
os.makedirs(FINAL_IMGS_DIR, exist_ok=True)

# Globální historie metrik pro checkpointy — aktualizuje se v tréninkovém cyklu
_checkpoint_history = {
    "epochs": [], "steps": [], "loss": [], "bacc": [], "psnr": [], "ssim": [],
    "val_loss": [], "val_bacc": [], "val_psnr": [], "val_ssim": []
}

# =============================================================================
# NAČTENÍ DICOM SNÍMKŮ
# =============================================================================

def load_dicom_folder(folder, max_count):
    files = []
    # Rekurzivně prochází podsložky
    for root, _, fnames in os.walk(folder):
        for f in sorted(fnames):
            if f.endswith(".dcm"):
                files.append(os.path.join(root, f))

    if not files:
        raise FileNotFoundError(f"Žádný .dcm soubor nenalezen v {folder}")

    files = sorted(files)  # deterministické pořadí před shuffle

    # Seed pro reprodukovatelný výběr snímků (seed=-1 = náhodný)
    data_seed = cfg["data"].get("seed", -1)
    if data_seed >= 0:
        random.seed(data_seed)
        print(f"  Seed: {data_seed} (reprodukovatelný výběr)")
    else:
        random.seed()

    random.shuffle(files)  # náhodně promíchá před výběrem

    if max_count > 0:
        files = files[:max_count]  # vezme prvních max_count souborů

    print(f"  Načítám {len(files)} snímků z {folder}")

    dataset = []
    for fpath in files:
        # Načte snímek jako 512×512 float32 normalizovaný na [0,1] + PatientID string
        img, patient_id = load_dicom_512(fpath)
        img_tensor = (torch.tensor(img, dtype=torch.float32)
                      .unsqueeze(0).unsqueeze(0))

        # Debug módy: fixní payload nebo N různých payloadů (pro testování sítě)
        _debug_mode = cfg.get("debug", {}).get("fixed_payload", False)
        _debug_n_payloads = cfg.get("debug", {}).get("n_payloads", 0)
        if _debug_mode:
            # Jeden fixní payload pro všechny snímky — síť se naučí konstantní bity
            original_hash = hash_patient_id("DEBUG_FIXED_PATIENT")
        elif _debug_n_payloads > 0:
            # N různých payloadů — snímky se cyklicky střídají
            idx = len(dataset) % _debug_n_payloads
            original_hash = hash_patient_id(f"DEBUG_PATIENT_{idx}")
        else:
            # Produkční mód: hash skutečného PatientID
            original_hash = hash_patient_id(patient_id)

        # LDPC kódování: 32 datových bajtů (SHA-256) → 512 bitů (rate 0.5)
        bits = ldpc_encode(original_hash)
        bit_tensor = (torch.tensor(bits, dtype=torch.float32)
                      .unsqueeze(0))

        dataset.append({
            "path":       fpath,
            "patient_id": patient_id,
            "img":        img_tensor,
            "bits":       bit_tensor,
        })
        print(f"    ✓ {os.path.basename(fpath)} | PatientID: {patient_id}")

    return dataset


print("\n--- Tréninkové snímky ---")
BASE_DIR   = cfg["data"]["base_dir"]
# Načte všechna tréninková data ze složky data/raw_dicom/train
all_train_data = load_dicom_folder(os.path.join(BASE_DIR, "train"), n_train)

# ── Rozdělení na train / val ──────────────────────────────────────────────────
data_seed = cfg["data"].get("seed", -1)
rng = random.Random(data_seed if data_seed >= 0 else None)
rng.shuffle(all_train_data)

n_val      = max(1, int(len(all_train_data) * val_split))  # počet validačních snímků
val_data   = all_train_data[:n_val]                         # první n_val snímků → validace
train_data = all_train_data[n_val:]                         # zbytek → trénink

print(f"\n  Celkem načteno : {len(all_train_data)} snímků")
print(f"  Trénink        : {len(train_data)} snímků  ({100*(1-val_split):.0f} %)")
print(f"  Validace       : {len(val_data)} snímků  ({100*val_split:.0f} %)")
print(f"  (val_split={val_split}, seed={data_seed})")

print("\n--- Testovací snímky ---")
# Načte testovací data ze složky data/raw_dicom/test
test_data  = load_dicom_folder(os.path.join(BASE_DIR, "test"), n_test)

print(f"\n  Trénink: {len(train_data)} snímků | Validace: {len(val_data)} snímků | Test: {len(test_data)} snímků")

# =============================================================================
# MODELS
# Inicializace encoderu a decoderu
# =============================================================================

# Globální seed pro reprodukovatelnost inicializace vah modelu.
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
print(f"  Torch seed: 42 (reprodukovatelná inicializace vah)")

# SCIFEncoder: vezme snímek + 512 bitů → vrátí stego snímek + anatomická maska
encoder = SCIFEncoder(cfg).to(device)
# SCIFDecoder: vezme stego snímek → vrátí 512 logitů (neaplikovaný sigmoid)
decoder = SCIFDecoder(cfg).to(device)

# -----------------------------------------------------------------------------
# Inicializace vah sítě
# Xavier uniform: inicializuje váhy tak aby rozptyl aktivací byl konzistentní
# přes celou hloubku sítě → zabraňuje vanishing/exploding gradients na startu
# -----------------------------------------------------------------------------
def init_weights(m):
    if isinstance(m, nn.Conv2d):
        # Konvoluční vrstvy
        nn.init.xavier_uniform_(m.weight, gain=1.0)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Linear):
        # Lineární vrstvy
        nn.init.xavier_uniform_(m.weight, gain=1.0)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)

# Aplikuje init_weights rekurzivně na všechny vrstvy encoderu a decoderu
encoder.apply(init_weights)
decoder.apply(init_weights)

# FiLM (Feature-wise Linear Modulation): gamma * feature_mapa + beta
if hasattr(encoder.payload_encoder, 'film_mlp'):
    nn.init.zeros_(encoder.payload_encoder.film_mlp[-1].weight)
    nn.init.zeros_(encoder.payload_encoder.film_mlp[-1].bias)
    encoder.payload_encoder.film_mlp[-1].bias.data[:256] = 1.0  # gamma offset → gamma=1 na startu

# Výpis počtu parametrů pro orientaci (měřítko složitosti modelu)
n_enc = sum(p.numel() for p in encoder.parameters())
n_dec = sum(p.numel() for p in decoder.parameters())
print(f"\nParametry — Encoder: {n_enc:,}  Decoder: {n_dec:,}")

# BCEWithLogitsLoss: Binary Cross-Entropy loss se zabudovaným sigmoidem
# Porovnává decoded_logits vs. cílové bity (0.0/1.0)
payload_criterion = nn.BCEWithLogitsLoss()

# -----------------------------------------------------------------------------
# Rozdělení parametrů do skupin pro různé learning rates
# Důvod: StructureExtractor (SE) trénujeme 10× pomaleji než zbytek encoderu,
# protože potřebuje čas stabilizovat se před tím, než začne silně ovlivňovat masku.
# Decoder trénujeme 20× rychleji — musí rychle sledovat měnící se enkodér.
# -----------------------------------------------------------------------------
se_params    = list(encoder.structure_extractor.parameters())
se_param_ids = set(id(p) for p in se_params)
other_params = [p for p in encoder.parameters()
                if id(p) not in se_param_ids]
dec_params   = list(decoder.parameters())

# Jeden společný AdamW optimizer se třemi skupinami parametrů
optimizer = optim.AdamW([
    {"params": other_params, "lr": lr,         "name": "encoder"},          # lr = 0.0005
    {"params": se_params,    "lr": lr * 0.1,   "name": "se_extractor"},     # lr = 0.00005
    {"params": dec_params,   "lr": lr * 3.0,   "weight_decay": 0.0,         # lr = 0.0015
                              "name": "decoder"},                            # 10× způsobovalo gradient collapse s 4310 snímky/epochu
], weight_decay=cfg["training"]["weight_decay"])  # weight_decay=0.0002 pro encoder skupiny

# ReduceLROnPlateau: snižuje LR pokud se loss nezlepšuje
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=10, min_lr=lr * 0.01
)

# Aliasy pro kompatibilitu se starším kódem
encoder_optimizer = optimizer
decoder_optimizer = optimizer
encoder_scheduler = scheduler
decoder_scheduler = scheduler

# =============================================================================
# CHECKPOINT FUNKCE
# Ukládají a načítají stav sítě, optimizeru a scheduleru
# =============================================================================

def save_checkpoint(filepath, epoch):
    torch.save({
        "encoder":   encoder.state_dict(),    # váhy a bias všech vrstev encoderu
        "decoder":   decoder.state_dict(),    # váhy a bias všech vrstev decoderu
        "optimizer": optimizer.state_dict(),  # momentová statistika AdamW (m, v vektory)
        "scheduler": scheduler.state_dict(),  # aktuální LR a počítadlo trpělivosti
        "epoch":     epoch,                   # číslo epochy pro navazující trénink
        "run_id":    RUN_ID,                  # ID runu pro dohledatelnost
        "history":   _checkpoint_history,      # historie metrik pro spojité grafy
    }, filepath)
    print(f"\n  Checkpoint uložen: {filepath} (epocha {epoch})")

def load_checkpoint(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Checkpoint nenalezen: {filepath}\n"
            f"  → Nastav load_checkpoint: \"\" pro trénink od nuly"
        )
    ckpt = torch.load(filepath, map_location=device)  # načte na správné zařízení (CPU/GPU)

    enc_res = encoder.load_state_dict(ckpt["encoder"], strict=False)
    dec_res = decoder.load_state_dict(ckpt["decoder"], strict=False)

    # Výpis varování pro chybějící nebo přebývající klíče
    if enc_res.missing_keys:
        print(f"  Encoder — chybějící klíče (nová init): {enc_res.missing_keys}")
    if enc_res.unexpected_keys:
        print(f"  Encoder — ignorované staré klíče: {enc_res.unexpected_keys}")
    if dec_res.missing_keys:
        print(f"  Decoder — chybějící klíče (nová init): {dec_res.missing_keys}")
    if dec_res.unexpected_keys:
        print(f"  Decoder — ignorované staré klíče: {dec_res.unexpected_keys}")

    try:
        # Zkusí obnovit optimizer a scheduler — může selhat pokud yaml změnil LR
        optimizer.load_state_dict(ckpt.get("optimizer", ckpt.get("encoder_optimizer")))
        scheduler.load_state_dict(ckpt.get("scheduler", ckpt.get("encoder_scheduler")))
    except Exception as e:
        # Při nekompatibilitě začne optimizer od nuly (ne fatální chyba)
        print(f"  Optimizer/scheduler nelze načíst ({e}) — začínám od nuly")

    # Načtení historie metrik pro spojité grafy při pokračování tréninku
    global _checkpoint_history
    saved_history = ckpt.get("history", None)
    if saved_history and saved_history.get("epochs"):
        _checkpoint_history = saved_history
        print(f"     Historie metrik: {len(saved_history['epochs'])} epoch načteno → grafy budou spojité")
    else:
        print(f"     Historie metrik: checkpoint neobsahuje historii → grafy začnou od ep {ckpt.get('epoch', 0)}")

    start_epoch = ckpt.get("epoch", 0)
    print(f"\n  Checkpoint načten: {filepath}")
    print(f"     Původní Run ID: {ckpt.get('run_id', 'neznámý')} | Epocha: {start_epoch}")
    return start_epoch

# =============================================================================
# METRIKY
# =============================================================================

def psnr(x, y):
    """
    PSNR (Peak Signal-to-Noise Ratio) v decibelech.
    """
    mse = torch.mean((x - y) ** 2)
    mse = torch.clamp(mse, min=1e-10)
    return 10 * torch.log10(1.0 / mse)

def bit_accuracy(logits, targets):
    """
    Podíl správně dekódovaných bitů (0.0 – 1.0).
    BitAcc = 1.0 znamená perfektní dekódování všech 512 bitů.
    BitAcc = 0.5 = náhodné hádání.
    """
    return ((torch.sigmoid(logits) > 0.5).float() == targets).float().mean()

# =============================================================================
# LDPC CONFIG — inicializace kodeku pro evaluaci
# LDPC (rate 0.50, n=512, k=256): soft-decision dekódování opraví ~35 bitů (BER ~7 %)
# rs_k_cfg / rs_n_cfg jsou legacy parametry z yaml (původně Reed-Solomon) —
# max_bit_errors=128 se aktivně nepoužívá, práh v tréninku je fixní 35 bitů
# =============================================================================

rs_k_cfg       = cfg["payload"]["rs_k"]           # 32 datových bajtů (legacy RS parametr)
rs_n_cfg       = cfg["payload"]["rs_n"]           # 64 celkových bajtů
max_errors     = (rs_n_cfg - rs_k_cfg) // 2       # 16 bajtů
max_bit_errors = max_errors * 8                   # 128 bitů (legacy RS metrika)
# LDPC soft-decision dekódování se používá ve finální evaluaci (pomalé pro per-snímek)
# Během tréninku se používá hard BitAcc + max_bit_errors jako rychlá aproximace
ldpc_codec = get_ldpc_codec()                     # inicializuje LDPC kodek (rate 0.5, n=512, k=256)

# =============================================================================
# ATTACK AUGMENTATION — robustnost vůči degradaci stego snímku
#
# Funkce apply_attack() aplikuje jeden ze čtyř útoků na stego snímek
# před tím než ho dostane decoder.
#
# Curriculum learning: intenzita útoku roste lineárně od attack_start_epoch
# do attack_max_epoch. Na začátku tréninku je útoků méně a jsou slabší —
# síť se nejdřív naučí základní pipeline, pak postupně přidáváme těžší úkoly.
# =============================================================================

# Načtení konfigurace augmentace z yaml
_aug = cfg.get("augmentation", {})
ATTACK_PROB          = _aug.get("attack_prob",          0.3)
ATTACK_START_EPOCH   = _aug.get("attack_start_epoch",   12)
ATTACK_MAX_EPOCH     = _aug.get("attack_max_epoch",     40)
JPEG_ENABLED         = _aug.get("jpeg_enabled",         True)
JPEG_QUALITY_MIN     = _aug.get("jpeg_quality_min",     85)
JPEG_QUALITY_MAX     = _aug.get("jpeg_quality_max",     95)
NOISE_ENABLED        = _aug.get("noise_enabled",        False)
NOISE_SIGMA_MAX      = _aug.get("noise_sigma_max",      0.015)
BLUR_ENABLED         = _aug.get("blur_enabled",         True)
BLUR_STRENGTH_MAX    = _aug.get("blur_strength_max",    0.25)
BRIGHTNESS_ENABLED   = _aug.get("brightness_enabled",   True)
BRIGHTNESS_DELTA_MAX = _aug.get("brightness_delta_max", 0.05)


def jpeg_compress(stego_tensor, quality):
    """
    Aplikuje JPEG kompresi na stego tensor přes PIL.
    """
    # Převod na numpy uint8 pro PIL
    np_img = (stego_tensor.squeeze().detach().cpu().numpy() * 255).astype(np.uint8)
    pil_img = Image.fromarray(np_img, mode='L')   # 'L' = grayscale (1 kanál)

    # JPEG encode do paměti (ne na disk) a zpět decode
    buf = io.BytesIO()
    pil_img.save(buf, format='JPEG', quality=quality)
    buf.seek(0)
    pil_decoded = Image.open(buf).convert('L')

    # Zpět na tensor na správném zařízení
    result = torch.tensor(
        np.array(pil_decoded, dtype=np.float32) / 255.0,
        device=stego_tensor.device
    )
    return result.unsqueeze(0).unsqueeze(0)   # zpět na [1, 1, H, W]


def apply_attack(stego, epoch):
    """
    Náhodně vybere a aplikuje jeden útok na stego snímek.

    Útok se aplikuje pouze pokud:
      1. epoch >= ATTACK_START_EPOCH (curriculum: nejdřív základní trénink)
      2. random() < ATTACK_PROB (ne každý snímek je napaden)

    Intenzita útoku roste lineárně s epochou:
      epoch = ATTACK_START_EPOCH → intenzita 0 %
      epoch = ATTACK_MAX_EPOCH   → intenzita 100 %

    Vrátí:
        poškozený tensor stejného tvaru — nebo původní stego pokud útok nevybrán
    """
    # Před attack_start_epoch žádný útok
    if epoch < ATTACK_START_EPOCH:
        return stego

    # Náhodně přeskočí útok (50 % šance)
    if random.random() >= ATTACK_PROB:
        return stego

    # Intenzita roste lineárně od 0 do 1 v rozsahu [attack_start, attack_max]
    progress = min(1.0, (epoch - ATTACK_START_EPOCH) /
                        max(1, ATTACK_MAX_EPOCH - ATTACK_START_EPOCH))

    # Vyber náhodný útok ze seznamu povolených
    available = []
    if JPEG_ENABLED:       available.append("jpeg")
    if NOISE_ENABLED:      available.append("noise")
    if BLUR_ENABLED:       available.append("blur")
    if BRIGHTNESS_ENABLED: available.append("brightness")

    if not available:
        return stego

    attack = random.choice(available)

    # ── JPEG komprese ─────────────────────────────────────────────────
    # Quality klesá s intenzitou: při progress=0 quality=95, při progress=1 quality=70
    # Náhodnost: quality se volí náhodně z [quality_min, quality_max * (1-progress)]
    if attack == "jpeg":
        quality_range_max = int(JPEG_QUALITY_MAX - (JPEG_QUALITY_MAX - JPEG_QUALITY_MIN) * progress)
        quality_range_max = max(quality_range_max, JPEG_QUALITY_MIN)
        quality = random.randint(JPEG_QUALITY_MIN, quality_range_max)
        return jpeg_compress(stego, quality)

    # ── Gaussovský šum ────────────────────────────────────────────────
    # sigma roste s intenzitou; šum je nezávislý pro každý pixel
    elif attack == "noise":
        sigma = NOISE_SIGMA_MAX * progress
        if sigma < 1e-6:
            return stego
        noise = torch.randn_like(stego) * sigma
        return torch.clamp(stego + noise, 0.0, 1.0)

    # ── Gaussovské rozostření ─────────────────────────────────────────
    # Váhovaný průměr původního a rozostřeného snímku
    # Kernel 3×3 průměrování jako aproximace Gaussovského rozostření
    elif attack == "blur":
        strength = BLUR_STRENGTH_MAX * progress
        if strength < 1e-6:
            return stego
        kernel = torch.ones(1, 1, 3, 3, device=stego.device) / 9.0
        blurred = F.conv2d(stego, kernel, padding=1)
        # Lineárně mixuje originál a rozostřený snímek
        return stego * (1.0 - strength) + blurred * strength

    # ── Změna jasu ────────────────────────────────────────────────────
    # Simuluje windowing (posun okna jasu) v DICOM prohlížečích
    # Náhodný posun v rozmezí [-delta_max, +delta_max]
    elif attack == "brightness":
        delta_max = BRIGHTNESS_DELTA_MAX * progress
        if delta_max < 1e-6:
            return stego
        shift = (random.random() * 2.0 - 1.0) * delta_max  # náhodný posun v [-delta_max, +delta_max]
        return torch.clamp(stego + shift, 0.0, 1.0)

    return stego


# =============================================================================
# TRAINING LOOP
# =============================================================================

def train():
    """
     Hlavní tréninková smyčka.

    Každá epocha:
      1. Promíchá tréninková data
      2. Pro každý batch provede forward pass → spočítá loss → backward → optimizer step
      3. Každých log_interval epoch vypíše statistiky a varování
      4. Ukládá checkpointy při zlepšení nebo periodicky

    Celková loss funkce:
      L = p_w·L_payload + p_w·L_robust + μ_mask·L_mask + μ_sparse·L_sparse + L_write + L_aux

      payload_loss    — BCEWithLogits: správnost dekódování bitů (hlavní cíl)
      robustness_loss — payload_loss na attacked stegu (robustnost vůči degradaci)
      mask_l          — mask_aware_loss: změny mimo anatomickou masku jsou dražší
      sparse_l        — sparsity_loss: nutí masku být řídkou (~15 % aktivních pixelů)
      write_loss      — penalizace pokud encoder píše příliš slabě (delta < threshold)
      auxiliary_loss  — penalizace pokud energie delty je příliš nízká
    """

    print(f"\n{'='*60}")
    print(f"  TRÉNINK  ({len(train_data)} snímků / epocha)")
    print(f"  PSNR target: {target_psnr} dB  (penalizace od epochy {psnr_start})")
    print(f"  Maska: zapíná se od epochy {mask_start}")
    print(f"  write_loss: threshold={write_loss_threshold}  weight={write_loss_weight}")
    print(f"{'='*60}\n")

    start_epoch = 0
    if LOAD_CKPT:
        # Pokud je nastaven checkpoint, načte ho a pokračuje od uložené epochy
        start_epoch = load_checkpoint(LOAD_CKPT)

    best_bit_acc       = 0.0   # nejlepší dosažená BitAcc (pro ukládání best checkpointu)
    best_psnr_val      = 0.0   # nejlepší dosažené PSNR (jen pro výpis)
    stagnation_counter = 0     # počet epoch bez zlepšení BitAcc
    epoch_times        = []    # klouzavý průměr dob trvání epoch (pro ETA výpočet)
    train_start_time   = datetime.now()  # čas zahájení tréninku

    # Historie metrik pro vykreslení grafů průběhu tréninku
    # Pokud pokračujeme z checkpointu, načteme uloženou historii → spojité grafy
    global _checkpoint_history
    history_epochs    = list(_checkpoint_history.get("epochs",   []))
    history_steps     = list(_checkpoint_history.get("steps",    []))
    history_loss      = list(_checkpoint_history.get("loss",     []))
    history_bacc      = list(_checkpoint_history.get("bacc",     []))
    history_psnr      = list(_checkpoint_history.get("psnr",     []))
    history_ssim      = list(_checkpoint_history.get("ssim",     []))
    history_val_loss  = list(_checkpoint_history.get("val_loss", []))
    history_val_bacc  = list(_checkpoint_history.get("val_bacc", []))
    history_val_psnr  = list(_checkpoint_history.get("val_psnr", []))
    history_val_ssim  = list(_checkpoint_history.get("val_ssim", []))
    if history_epochs:
        print(f"  Historie metrik: načteno {len(history_epochs)} epoch z předchozího runu")

    for epoch in range(start_epoch, total_epochs):

        encoder.train()  # zapne dropout a BatchNorm v tréninkovém módu
        decoder.train()

        # -----------------------------------------------------------------
        # MASK WARM-UP
        # -----------------------------------------------------------------
        mask_active     = True
        warmup_progress = min(1.0, epoch / max(1, mask_start))  # lineární warm-up [0, 1]
        effective_mw    = mw  * warmup_progress   # skutečná váha mask_aware_loss pro tuto epochu
        effective_spw   = spw * warmup_progress   # skutečná váha sparsity_loss pro tuto epochu

        # Nastaví mask_strength v encoderu — určuje jak silně maska filtruje deltu
        # 0.0 = delta prochází bez filtrace, 1.0 = plné maskování
        encoder.mask_strength = warmup_progress

        epoch_start = datetime.now()
        random.shuffle(train_data)  # náhodné pořadí snímků v každé epoše

        # Akumulátory pro průměrné statistiky epochy
        epoch_loss     = 0.0
        epoch_bacc     = 0.0
        epoch_psnr     = 0.0
        epoch_ssim     = 0.0
        epoch_pl       = 0.0   # raw payload_loss průměr přes epochu
        last_mask      = None  # maska posledního batche (pro výpis mask density)
        last_delta     = None  # delta posledního batche (pro výpis δ_in/out ratio)
        n_steps        = 0     # počet gradient stepů (= počet batchů) v epoše

        # -----------------------------------------------------------------
        # VNITŘNÍ SMYČKA: iterace přes tréninková data po batchích
        # batch_size snímků → jeden forward pass → jeden optimizer.step()
        # -----------------------------------------------------------------
        for batch_start in range(0, len(train_data), batch_size):
            batch = train_data[batch_start: batch_start + batch_size]

            # Skládáme snímky a bity do batchů [B, 1, 512, 512] a [B, 512]
            img_t = torch.cat([s["img"] for s in batch], dim=0).to(device)
            bit_t = torch.cat([s["bits"] for s in batch], dim=0).to(device)

            optimizer.zero_grad()

            # Forward pass encoderu:
            #   img_t + bit_t → stego snímek + anatomická maska
            #   Encoder vloží 512 bitů do snímku jako neviditelnou perturbaci (delta)
            stego, emb_mask = encoder(img_t, bit_t)

            # Postupné zapojení encoderu do payload gradientu
            # Ep 0–1:  stego.detach() → payload_loss teče pouze do decoderu
            #          → decoder si vybuduje základní dekódovací schopnost
            # Ep 2+:   bez detach → payload_loss teče i do encoderu
            #          → encoder se učí jak zapsat payload aby byl lépe dekódovatelný
            ENCODER_GRAD_START = 2  # epocha od které payload_loss ovlivňuje encoder

            # Cesta 1: dekódování čistého stega
            if epoch < ENCODER_GRAD_START:
                decoded_logits = decoder(stego.detach())
            else:
                decoded_logits = decoder(stego)
            payload_loss = payload_criterion(decoded_logits, bit_t)

            # Cesta 2: pokud je aktivní útok, decoder navíc dekóduje attacked stego
            # apply_attack pracuje per-snímek [1,1,H,W] (JPEG přes PIL není batchovatelné)
            # → aplikujeme na každý snímek zvlášť a skládáme zpět do batche
            attacked_list = [apply_attack(stego[i:i + 1].detach(), epoch)
                             for i in range(len(batch))]
            stego_attacked = torch.cat(attacked_list, dim=0)
            _attacked = not torch.equal(stego_attacked, stego.detach())
            if _attacked:
                decoded_logits_atk = decoder(stego_attacked)
                robustness_loss = payload_criterion(decoded_logits_atk, bit_t)
            else:
                robustness_loss = torch.tensor(0.0, device=device)

            # Delta = rozdíl stego − originál (perturbace vložená encoderem)
            delta = stego - img_t
            mean_delta = torch.mean(torch.abs(delta))  # průměrná absolutní změna
            delta_energy = torch.mean(delta ** 2)  # energie (rozptyl) perturbace

            # PSNR penalizace: vypnuto (psnr_start=99999 → podmínka nikdy nenastane)
            # relu(target - current) = penalizuje pouze pokud PSNR < target
            # PSNR roste organicky díky masce bez přímé penalizace
            current_psnr = psnr(stego, img_t)
            psnr_penalty = (
                psnr_weight * torch.relu(target_psnr - current_psnr)
                if epoch >= psnr_start
                else torch.tensor(0.0, device=device)
            )

            # Write loss: penalizuje encoder pokud píše příliš slabě
            write_loss = write_loss_weight * torch.relu(write_loss_threshold - mean_delta)

            # Auxiliary loss: penalizuje nízkou energii delty
            auxiliary_loss = 1000.0 * torch.relu(1e-4 - delta_energy)

            # Mask-aware loss: změny MIMO anatomickou masku penalizuje lambda_outside× více
            # Nutí encoder psát payload preferenčně do oblastí hran a struktur
            mask_l   = mask_aware_loss(img_t, stego, emb_mask, lambda_outside=lambda_outside)

            # Sparsity loss: nutí SE masku být řídkou
            # Cíl: ~15% aktivních pixelů = realistické anatomické hrany
            # Bez sparsity_loss by se maska naučila být 1.0 všude (triviální řešení)
            sparse_l = sparsity_loss(emb_mask, target_density=0.15)

            # mask_loss_weight = 0 pokud jsme před warm-up fází (epoch < mask_start v původním schématu)
            # Nyní warm-up probíhá od ep 0 → effective_mw roste lineárně od začátku
            mask_loss_weight = effective_mw if epoch >= mask_start else 0.0

            # Celková loss funkce — vážená suma všech složek
            loss = (
                pw               * payload_loss      +  # správné dekódování bitů (gradient → encoder + decoder)
                pw               * robustness_loss   +  # robustnost vůči útokům (gradient → jen decoder)
                psnr_penalty                         +  # PSNR penalizace (vypnuto)
                write_loss                           +  # encoder musí psát dostatečně silně
                auxiliary_loss                       +  # delta musí mít energii
                mask_loss_weight * mask_l            +  # změny preferenčně do masky
                effective_spw    * sparse_l             # maska musí být řídká
            )

            # NaN/Inf detekce — při numerické nestabilitě přeskočí snímek
            if torch.isnan(loss) or torch.isinf(loss):
                checks = {
                    "payload":    payload_loss,
                    "robustness": robustness_loss,
                    "psnr_pen":   psnr_penalty,
                    "write":      write_loss,
                    "auxiliary":  auxiliary_loss,
                    "mask":       mask_l,
                }
                # Identifikuje která složka způsobila NaN/Inf pro debugging
                bad = [k for k, v in checks.items() if torch.isnan(v) or torch.isinf(v)]
                print(f"  NaN/Inf ep {epoch+1} — {bad if bad else 'neznámá složka'}")
                optimizer.zero_grad()
                continue  # přeskočí backward a optimizer.step()

            # Zpětná propagace: spočítá gradienty pro všechny parametry
            loss.backward()

            # DEBUG: zobrazí maximální gradient před clippingem (jen první 3 epochy)
            _enc_before = 0.0
            _dec_before = 0.0
            for p in encoder.parameters():
                if p.grad is not None:
                    _enc_before = max(_enc_before, p.grad.abs().max().item())
            for p in decoder.parameters():
                if p.grad is not None:
                    _dec_before = max(_dec_before, p.grad.abs().max().item())
            if epoch < 3 and batch_start == 0:
                print(f"    [DEBUG] enc_grad_max={_enc_before:.2e}  dec_grad_max={_dec_before:.2e}")

            # Gradient clipping: omezí normu gradientů na grad_clip=1.0
            # Zabraňuje exploding gradients (náhlé velké skoky ve vahách)
            torch.nn.utils.clip_grad_norm_(
                list(encoder.parameters()) + list(decoder.parameters()), grad_clip)

            # Aktualizace vah (jeden krok AdamW optimizeru)
            optimizer.step()

            # Akumulace statistik — torch.no_grad() zabraňuje zbytečnému výpočtu gradientů
            with torch.no_grad():
                epoch_loss  += loss.item()
                epoch_pl    += payload_loss.item()
                epoch_bacc  += bit_accuracy(decoded_logits, bit_t).item()
                epoch_psnr  += psnr(stego, img_t).item()
                epoch_ssim  += (1 - ssim_loss(stego, img_t)).item()
                last_mask   = emb_mask       # uloží masku pro výpis mask density
                last_delta  = delta          # uloží deltu pro výpis δ_in/out ratio
                last_logits = decoded_logits
                last_bits_t = bit_t

            # Gradient diagnostika — jednou za epochu na prvním batchi
            # Sleduje průměrnou absolutní velikost gradientu pro encoder a decoder
            # enc_grad ≈ 0 = encoder se přestal učit (PSNR stagnuje)
            # dec_grad ≈ 0 = decoder se přestal učit
            n_steps += 1
            if batch_start == 0:
                dec_grad = None
                enc_grad = None
                for name, p in decoder.named_parameters():
                    if p.grad is not None:
                        dec_grad = p.grad.abs().mean().item()
                        break
                for name, p in encoder.named_parameters():
                    if p.grad is not None:
                        enc_grad = p.grad.abs().mean().item()
                        break
                _grad_dec = dec_grad if dec_grad else 0.0
                _grad_enc = enc_grad if enc_grad else 0.0

        # -----------------------------------------------------------------
        # KONEC EPOCHY — průměrování statistik
        # Dělíme n_steps (počet batchů), ne len(train_data):
        # každý batch již vrací průměr přes své snímky → průměr průměrů = epochový průměr
        # -----------------------------------------------------------------
        n        = max(1, n_steps)
        avg_loss = epoch_loss / n     # průměrná celková loss
        avg_pl   = epoch_pl   / n     # průměrná payload loss
        avg_bacc = epoch_bacc / n     # průměrná BitAcc přes epochu
        avg_psnr = epoch_psnr / n     # průměrné PSNR přes epochu
        avg_ssim = epoch_ssim / n     # průměrné SSIM přes epochu

        # -----------------------------------------------------------------
        # VALIDAČNÍ LOOP
        # Vyhodnotí model na validační množině bez gradientů a bez změny vah.
        # Validační loss slouží k:
        #   1. sledování generalizace (val_loss >> train_loss = overfitting)
        #   2. řízení ReduceLROnPlateau scheduleru (místo train_loss)
        #   3. výběru best checkpointu podle val_bacc místo train_bacc
        # -----------------------------------------------------------------
        encoder.eval()
        decoder.eval()
        val_loss = 0.0
        val_bacc = 0.0
        val_psnr = 0.0
        val_ssim = 0.0

        with torch.no_grad():
            for vsample in val_data:
                v_img = vsample["img"].to(device)
                v_bit = vsample["bits"].to(device)

                v_stego, v_mask   = encoder(v_img, v_bit)
                v_logits          = decoder(v_stego)
                v_payload_loss    = payload_criterion(v_logits, v_bit)
                v_delta           = v_stego - v_img
                v_mean_delta      = torch.mean(torch.abs(v_delta))
                v_delta_energy    = torch.mean(v_delta ** 2)
                v_write_loss      = write_loss_weight * torch.relu(write_loss_threshold - v_mean_delta)
                v_auxiliary_loss  = 1000.0 * torch.relu(1e-4 - v_delta_energy)
                v_mask_l          = mask_aware_loss(v_img, v_stego, v_mask, lambda_outside=lambda_outside)
                v_sparse_l        = sparsity_loss(v_mask, target_density=0.15)
                v_effective_mw    = mw  * warmup_progress
                v_effective_spw   = spw * warmup_progress
                v_mask_lw         = v_effective_mw if epoch >= mask_start else 0.0

                v_loss = (
                    pw             * v_payload_loss  +
                    v_write_loss                     +
                    v_auxiliary_loss                 +
                    v_mask_lw      * v_mask_l        +
                    v_effective_spw * v_sparse_l
                )

                val_loss += v_loss.item()
                val_bacc += bit_accuracy(v_logits, v_bit).item()
                val_psnr += psnr(v_stego, v_img).item()
                val_ssim += (1 - ssim_loss(v_stego, v_img)).item()

        n_val_actual = len(val_data)
        avg_val_loss = val_loss / n_val_actual
        avg_val_bacc = val_bacc / n_val_actual
        avg_val_psnr = val_psnr / n_val_actual
        avg_val_ssim = val_ssim / n_val_actual

        encoder.train()
        decoder.train()

        # Scheduler sleduje validační loss (ne train loss) — správnější přístup
        # ReduceLROnPlateau sníží LR pokud se val_loss nezlepší po 20 epochách
        scheduler.step(avg_val_loss)

        # Záznam metrik do historie pro grafy průběhu
        history_epochs.append(epoch + 1)
        history_steps.append((epoch + 1) * len(train_data))
        history_loss.append(avg_loss)
        history_bacc.append(avg_bacc)
        history_psnr.append(avg_psnr)
        history_ssim.append(avg_ssim)
        history_val_loss.append(avg_val_loss)
        history_val_bacc.append(avg_val_bacc)
        history_val_psnr.append(avg_val_psnr)
        history_val_ssim.append(avg_val_ssim)

        # Aktualizace globální historie pro uložení do checkpointu
        _checkpoint_history = {
            "epochs": history_epochs, "steps": history_steps,
            "loss": history_loss, "bacc": history_bacc,
            "psnr": history_psnr, "ssim": history_ssim,
            "val_loss": history_val_loss, "val_bacc": history_val_bacc,
            "val_psnr": history_val_psnr, "val_ssim": history_val_ssim
        }

        # Měření doby epochy — pro odhad zbývajícího času (ETA)
        epoch_dur = (datetime.now() - epoch_start).total_seconds()
        epoch_times.append(epoch_dur)
        if len(epoch_times) > 50:
            epoch_times.pop(0)  # klouzavé okno posledních 50 epoch (stabilnější ETA)

        # Sledování zlepšení podle VAL BitAcc (ne train) — korektní přístup
        improved = avg_val_bacc > best_bit_acc
        if improved:
            best_bit_acc       = avg_val_bacc
            stagnation_counter = 0
        else:
            stagnation_counter += 1

        if avg_val_psnr > best_psnr_val:
            best_psnr_val = avg_val_psnr

        # Uloží best checkpoint pokud se zlepšila VAL BitAcc
        if SAVE_CKPT and improved and avg_val_bacc > 0.60:
            save_checkpoint(CKPT_BEST_FILE, epoch + 1)

        # Periodické checkpointy každých CKPT_INTERVAL epoch
        if SAVE_CKPT and CKPT_INTERVAL > 0 and (epoch + 1) % CKPT_INTERVAL == 0:
            periodic_file = os.path.join(CKPT_DIR, f"ckpt_{RUN_ID}_ep{epoch+1:04d}.pt")
            save_checkpoint(periodic_file, epoch + 1)

        # -----------------------------------------------------------------
        # VÝPIS STATISTIK každých log_interval epoch
        # -----------------------------------------------------------------
        if (epoch + 1) % log_interval == 0:
            # Hustota masky: průměrná hodnota pixelů masky (0=nic aktivní, 1=vše aktivní)
            # Ideál ~0.15 = maska pokrývá ~15% snímku (anatomické hrany)
            mask_density = last_mask.mean().item()

            # δ_in/out ratio: průměrná absolutní perturbace v masce vs. mimo masku
            # ratio > 1.0 = encoder preferuje psát do masky (žádoucí chování)
            # ratio >> 5.0 = encoder píše téměř výhradně do masky
            delta_in     = (torch.abs(last_delta) * last_mask).mean().item()
            delta_out    = (torch.abs(last_delta) * (1 - last_mask)).mean().item()
            ratio        = delta_in / (delta_out + 1e-8)

            # Textový label fáze tréninku — zobrazuje aktuální sílu masky a útoku
            attack_progress = min(1.0, max(0.0, (epoch - ATTACK_START_EPOCH) /
                                   max(1, ATTACK_MAX_EPOCH - ATTACK_START_EPOCH)))
            attack_label = f" atk={attack_progress:.2f}" if epoch >= ATTACK_START_EPOCH else ""
            phase_label  = f"mask={warmup_progress:.2f}{attack_label}" if mask_active else "no-mask"

            # Počet chybně dekódovaných bitů (z průměrné BitAcc)
            avg_bit_errors = int((1.0 - avg_bacc) * 512)
            # Odhad ECC statusu: LDPC rate 0.5 zvládne ~35 bitových chyb (BER ~7%)
            ecc_status = "ECC OK" if avg_bit_errors <= 35 else "ECC FAIL"
            ecc_margin = 35 - avg_bit_errors  # kladné = rezerva, záporné = problém

            # ETA výpočet z klouzavého průměru doby epochy
            avg_epoch_time   = sum(epoch_times) / len(epoch_times)
            remaining_epochs = total_epochs - (epoch + 1)
            eta_seconds      = avg_epoch_time * remaining_epochs
            eta_h  = int(eta_seconds // 3600)
            eta_m  = int((eta_seconds % 3600) // 60)
            eta_str = f"{eta_h}h{eta_m:02d}m" if eta_h > 0 else f"{eta_m}m"

            elapsed     = (datetime.now() - train_start_time).total_seconds()
            el_h = int(elapsed // 3600)
            el_m = int((elapsed % 3600) // 60)
            elapsed_str = f"{el_h}h{el_m:02d}m" if el_h > 0 else f"{el_m}m"

            current_lr      = optimizer.param_groups[0]["lr"]  # aktuální LR (po případném snížení)
            improved_marker = " ★" if improved else ""         # hvězdička při zlepšení BitAcc

            # Detailní breakdown lossů na jednom vzorkovém snímku pro debugging
            # Provádí se v eval módu bez gradientů (jen pro výpis, ne do loss backpropu)
            with torch.no_grad():
                encoder.eval()
                decoder.eval()
                sample_img = train_data[0]["img"].to(device)
                sample_bit = train_data[0]["bits"].to(device)
                _stego, _mask = encoder(sample_img, sample_bit)
                _logits       = decoder(_stego)
                _delta        = _stego - sample_img
                _mean_delta   = torch.mean(torch.abs(_delta)).item()
                _delta_energy = torch.mean(_delta ** 2).item()
                _pl           = payload_criterion(_logits, sample_bit).item()
                _cp           = psnr(_stego, sample_img).item()
                # Výpočet jednotlivých složek loss pro první snímek (diagnostika)
                _pp           = (psnr_weight * max(0.0, target_psnr - _cp)) if epoch >= psnr_start else 0.0
                _wl           = write_loss_weight * max(0.0, write_loss_threshold - _mean_delta)
                _al           = 1000.0 * max(0.0, 1e-4 - _delta_energy)
                # Diagnostika logitů — jak moc je decoder jistý?
                # logit_std blízko 0 = decoder si není jistý, velká std = jistý
                _probs        = torch.sigmoid(_logits)
                _logit_std    = _logits.std().item()
                _logit_mean   = _logits.mean().item()
                _prob_mean    = _probs.mean().item()   # průměrná pravděpodobnost → ideálně ~0.5 nebo blíže 0/1
                encoder.train()
                decoder.train()

            # Řádek 1: hlavní metriky epochy — train i val
            print(
                f"  [Ep {epoch+1:04d}/{total_epochs} {phase_label}] "
                f"Loss {avg_loss:.4f} | "
                f"BitAcc {avg_bacc:.4f} | "
                f"PSNR {avg_psnr:.2f} dB | "
                f"MaskDens {mask_density:.3f} | "
                f"δ_in/out {ratio:.1f}x"
            )
            # Řádek 2: validační metriky (klíčové pro sledování generalizace)
            val_bit_errors = int((1.0 - avg_val_bacc) * 512)
            val_ecc_status = "ECC OK" if val_bit_errors <= 35 else "ECC FAIL"
            print(
                f"         [VAL]  BitAcc {avg_val_bacc:.4f}{improved_marker} | "
                f"PSNR {avg_val_psnr:.2f} dB | "
                f"Loss {avg_val_loss:.4f} | "
                f"Chyby ~{val_bit_errors}/512 | {val_ecc_status}"
            )
            # Řádek 3: ECC status, LR, čas
            print(
                f"         Chyby ~{avg_bit_errors}/512 | "
                f"{ecc_status} (rezerva {ecc_margin:+d} bitů) | "
                f"LR {current_lr:.2e} | "
                f"Běh {elapsed_str} | ETA {eta_str}"
            )
            # Řádek 3: breakdown lossů (pro identifikaci problémů)
            print(
                f"         [LOSS] payload={avg_pl:.4f} (train_avg)  "
                f"psnr_pen={_pp:.2f}  "
                f"write={_wl:.2f}  "
                f"mean_δ={_mean_delta:.4f}  "
                f"δ_energy={_delta_energy:.6f}"
            )
            # Řádek 4: diagnostika logitů decoderu
            print(
                f"         [DIAG] logit_mean={_logit_mean:.3f}  "
                f"logit_std={_logit_std:.3f}  "
                f"prob_mean={_prob_mean:.3f}"
            )
            # Řádek 5: gradient diagnostika (enc_grad≈0 = encoder se neučí)
            print(
                f"         [GRAD] dec_grad={_grad_dec:.2e}  "
                f"enc_grad={_grad_enc:.2e}  "
                f"(ideál: oba >1e-5)"
            )

            # Automatická varování při detekci problémů
            if stagnation_counter >= 50:
                print(f"  Stagnace {stagnation_counter} epoch — zvaž restart nebo změnu LR")
            if avg_psnr > 80.0:
                # PSNR > 80 dB = delta ≈ 0 (encoder se zhroutil, nepíše nic)
                print(f"  PSNR = {avg_psnr:.1f} dB — kolaps (delta≈0)! Zkontroluj write_loss")
            if avg_psnr < target_psnr - 10.0 and epoch >= psnr_start + 20:
                print(f"  PSNR = {avg_psnr:.1f} dB daleko pod targetem {target_psnr} dB")
            if avg_bacc < 0.52 and epoch > 20:
                # BitAcc ≤ 0.52 po 20 epochách = síť se vůbec neučí payload
                print(f"  BitAcc stagnuje na náhodné úrovni — síť se neučí!")
            if mask_active and ratio < 2.0 and epoch > mask_start + 10:
                # δ_in/out < 2 = maska nemá vliv na umístění perturbace
                print(f"  delta_in/out = {ratio:.1f}x — zvys mask_weight v yaml")

    # =========================================================================
    # GRAFY PRŮBĚHU TRÉNINKU — ukládají se do složky graphs/
    # Každá metrika jako samostatný soubor pojmenovaný: RUN_ID_nazevmetriky.png
    # =========================================================================
    print(f"\n--- Ukládám grafy průběhu tréninku ---")

    def save_metric_graph(steps, epochs, train_vals, val_vals, metric_name, ylabel, title):
        """Uloží graf průběhu jedné metriky (train + val) do graphs/.
        X osa = globální kroky (steps), sekundární horní osa = epochy."""
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(steps, train_vals, label=f"Train {metric_name}", color="steelblue", linewidth=1.5)
        if val_vals:
            ax.plot(steps, val_vals, label=f"Val {metric_name}", color="coral", linewidth=1.5, linestyle="--")
        ax.set_xlabel("Globální kroky (steps)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{title}  [{RUN_ID}]")
        ax.legend()
        ax.grid(True, alpha=0.3)
        # Sekundární horní osa s epochami
        ax_top = ax.twiny()
        ax_top.set_xlim(ax.get_xlim())
        _step = max(1, len(steps) // 10)
        epoch_ticks = steps[::_step]  # max 10 ticků
        epoch_labels = [str(epochs[i * _step]) for i in range(len(epoch_ticks))]
        ax_top.set_xticks(epoch_ticks)
        ax_top.set_xticklabels(epoch_labels)
        ax_top.set_xlabel("Epocha")
        path = os.path.join(GRAPHS_DIR, f"{RUN_ID}_{metric_name.lower().replace(' ', '_')}.png")
        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Graf průběhu uložen: {path}")

    if history_steps:
        save_metric_graph(history_steps, history_epochs, history_loss, history_val_loss,
                          "loss",    "Loss",       "Průběh Loss")
        save_metric_graph(history_steps, history_epochs, history_bacc, history_val_bacc,
                          "bitacc",  "BitAcc",     "Průběh BitAcc")
        save_metric_graph(history_steps, history_epochs, history_psnr, history_val_psnr,
                          "psnr",    "PSNR (dB)",  "Průběh PSNR")

        # PSNR + SSIM kombinovaný graf (dvě Y osy + epochy nahoře)
        fig, ax1 = plt.subplots(figsize=(10, 5))
        ax1.plot(history_steps, history_psnr, label="Train PSNR", color="steelblue", linewidth=1.5)
        if history_val_psnr:
            ax1.plot(history_steps, history_val_psnr, label="Val PSNR", color="steelblue", linewidth=1.5, linestyle="--")
        ax1.set_xlabel("Globální kroky (steps)")
        ax1.set_ylabel("PSNR (dB)", color="steelblue")
        ax1.tick_params(axis='y', labelcolor='steelblue')

        ax2 = ax1.twinx()
        ax2.plot(history_steps, history_ssim, label="Train SSIM", color="darkorange", linewidth=1.5)
        if history_val_ssim:
            ax2.plot(history_steps, history_val_ssim, label="Val SSIM", color="darkorange", linewidth=1.5, linestyle="--")
        ax2.set_ylabel("SSIM", color="darkorange")
        ax2.tick_params(axis='y', labelcolor='darkorange')

        # Sekundární horní osa s epochami
        ax_top = ax1.twiny()
        ax_top.set_xlim(ax1.get_xlim())
        _step2 = max(1, len(history_steps) // 10)
        epoch_ticks = history_steps[::_step2]
        epoch_labels = [str(history_epochs[i * _step2]) for i in range(len(epoch_ticks))]
        ax_top.set_xticks(epoch_ticks)
        ax_top.set_xticklabels(epoch_labels)
        ax_top.set_xlabel("Epocha")

        # Kombinovaná legenda
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower right")
        ax1.set_title(f"Průběh PSNR + SSIM  [{RUN_ID}]")
        ax1.grid(True, alpha=0.3)
        path = os.path.join(GRAPHS_DIR, f"{RUN_ID}_psnr_ssim.png")
        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Graf průběhu uložen: {path}")

    print(f"\n  ✓ Trénink hotovo | "
          f"Best BitAcc: {best_bit_acc:.4f} | Best PSNR: {best_psnr_val:.2f} dB")

    if SAVE_CKPT:
        # Uloží finální checkpoint po dokončení tréninku
        save_checkpoint(CKPT_FILE, total_epochs)
        print(f"  → Pro pokračování nastav v yaml:")
        print(f"     load_checkpoint: \"{CKPT_FILE}\"")
        print(f"  → Nejlepší checkpoint (BitAcc): {CKPT_BEST_FILE}")


# =============================================================================
# RUN — spuštění tréninku
# =============================================================================

train()

print("\n" + "=" * 60)
print("  Trénink dokončen")
print("=" * 60)

# =============================================================================
# FINAL EVALUATION
# Vyhodnocení natrénovaného modelu na testovacích snímcích
# (tyto snímky model během tréninku nikdy neviděl)
# =============================================================================

# Přepnutí do eval módu: vypne dropout, BatchNorm používá running statistics
encoder.eval()
decoder.eval()

print(f"\n--- Evaluace na {len(test_data)} testovacích snímcích ---\n")

# ── Předpočet LDPC bitů pro každý rate a každý test snímek ───────────────
# Každý rate má jiný LDPC kód (jiná H matice) → jiných 512 bitů
# Proto musíme pro každý rate znovu zakódovat hash a znovu embedovat
EVAL_RATES = [0.50, 0.33, 0.25]

# Předpočítání LDPC kodeků (aby se nevytvářely opakovaně)
print("  Předpočítávám LDPC kodeky pro rates:", EVAL_RATES)
for _rate in EVAL_RATES:
    get_ldpc_codec(n=512, rate=_rate)

# Pro každý snímek a rate: předpočítat 512-bitový payload
test_bits_by_rate = {r: [] for r in EVAL_RATES}
for sample in test_data:
    patient_hash = hash_patient_id(sample["patient_id"])
    for _rate in EVAL_RATES:
        codec = get_ldpc_codec(n=512, rate=_rate)
        hash_truncated = patient_hash[:(codec.k + 7) // 8]
        bits = ldpc_encode(hash_truncated, n=512, rate=_rate)
        bt = torch.tensor(bits, dtype=torch.float32).unsqueeze(0)  # CPU — na GPU per-snímek
        test_bits_by_rate[_rate].append(bt)

# ── Hlavní evaluace (rate=0.50, odpovídá tréninku) ───────────────────────
all_bacc = []
all_psnr = []
all_ssim = []
all_errs = []
all_ldpc_50 = []

for i, sample in enumerate(test_data):
    img_t = sample["img"].to(device)
    bit_t = sample["bits"].to(device)  # rate=0.50 z tréninku

    with torch.no_grad():
        stego, final_mask = encoder(img_t, bit_t)
        logits            = decoder(stego)

    pred  = (torch.sigmoid(logits) > 0.5).float()
    bacc  = (pred == bit_t).float().mean().item()
    errs  = int((pred != bit_t).sum().item())
    pv    = psnr(stego, img_t).item()
    sv    = (1 - ssim_loss(stego, img_t)).item()

    raw_logits = logits.squeeze().detach().cpu().numpy()
    try:
        ldpc_decode_soft(raw_logits, n=512, rate=0.50)
        ldpc_ok = True
    except ValueError:
        ldpc_ok = False

    all_bacc.append(bacc)
    all_psnr.append(pv)
    all_ssim.append(sv)
    all_errs.append(errs)
    all_ldpc_50.append(ldpc_ok)

    status = "✓ LDPC OK" if ldpc_ok else "✗ LDPC FAIL"
    print(f"  Snímek {i+1}: BitAcc {bacc:.4f} | Chyby {errs:3d} | "
          f"{status} | PSNR {pv:.2f} dB | SSIM {sv:.4f}")

# ── LDPC evaluace pro rate 0.33 a 0.25 (re-encode + re-embed) ───────────
all_ldpc_by_rate = {0.50: all_ldpc_50}
for _rate in [0.33, 0.25]:
    codec = get_ldpc_codec(n=512, rate=_rate)
    _ok_list = []
    print(f"\n  Evaluace LDPC rate={_rate:.2f} (k={codec.k}, "
          f"re-encode + re-embed {len(test_data)} snímků)...")
    for i, sample in enumerate(test_data):
        with torch.no_grad():
            stego_r, _ = encoder(sample["img"].to(device), test_bits_by_rate[_rate][i].to(device))
            logits_r   = decoder(stego_r)
        raw_r = logits_r.squeeze().detach().cpu().numpy()
        try:
            ldpc_decode_soft(raw_r, n=512, rate=_rate)
            _ok_list.append(True)
        except ValueError:
            _ok_list.append(False)
    all_ldpc_by_rate[_rate] = _ok_list
    print(f"    → LDPC OK: {sum(_ok_list)}/{len(_ok_list)} "
          f"({100*sum(_ok_list)/len(_ok_list):.1f}%)")

# ── Souhrn ────────────────────────────────────────────────────────────────
print(f"\n  === Průměr přes {len(test_data)} snímků ===")
print(f"  BitAcc   : {np.mean(all_bacc):.4f} ± {np.std(all_bacc):.4f}")
print(f"  Bit chyby: {np.mean(all_errs):.1f} ± {np.std(all_errs):.1f}")
for _rate in EVAL_RATES:
    _ok = sum(all_ldpc_by_rate[_rate])
    codec = get_ldpc_codec(n=512, rate=_rate)
    print(f"  LDPC OK (rate {_rate:.2f}, k={codec.k:3d}): {_ok}/{len(test_data)} snímků "
          f"({100*_ok/len(test_data):.1f}%)")
print(f"  PSNR     : {np.mean(all_psnr):.2f} ± {np.std(all_psnr):.2f} dB")
print(f"  SSIM     : {np.mean(all_ssim):.4f} ± {np.std(all_ssim):.4f}")

# =============================================================================
# ROBUSTNESS EVALUATION
# Testuje dekódovatelnost payloadu po aplikaci každého útoku zvlášť
# při plné intenzitě (progress=1.0). Ukazuje jak moc je model odolný.
# =============================================================================

print(f"\n--- Evaluace robustnosti (plná intenzita útoků) ---\n")

def eval_robustness(attack_name, attack_fn):
    """
    Vyhodnotí BitAcc a LDPC status na všech testovacích snímcích po daném útoku.
    Testuje LDPC dekódování při rate 0.50, 0.33, 0.25.

    Pro každý rate se provede:
      1. Re-encode hashe (truncated) do 512-bitového LDPC kódového slova
      2. Re-embed do snímku přes encoder (jiný codeword = jiná stego)
      3. Aplikace útoku na stego
      4. Dekódování přes decoder + LDPC soft-decision
    """
    r_bacc = []
    r_errs = []
    r_ldpc = {0.50: [], 0.33: [], 0.25: []}
    for i, sample in enumerate(test_data):
        _img = sample["img"].to(device)
        _bit = sample["bits"].to(device)
        # Rate 0.50 — primární rate (bits = sample["bits"])
        with torch.no_grad():
            stego, _ = encoder(_img, _bit)
            stego_attacked = attack_fn(stego)
            logits = decoder(stego_attacked)
        pred  = (torch.sigmoid(logits) > 0.5).float()
        bacc  = (pred == _bit).float().mean().item()
        errs  = int((pred != _bit).sum().item())
        raw_logits = logits.squeeze().detach().cpu().numpy()
        try:
            ldpc_decode_soft(raw_logits, n=512, rate=0.50)
            r_ldpc[0.50].append(True)
        except ValueError:
            r_ldpc[0.50].append(False)
        r_bacc.append(bacc)
        r_errs.append(errs)

        # Rates 0.33, 0.25 — re-encode + re-embed + attack + decode
        for _rate in [0.33, 0.25]:
            with torch.no_grad():
                stego_r, _ = encoder(_img, test_bits_by_rate[_rate][i].to(device))
                stego_r_attacked = attack_fn(stego_r)
                logits_r = decoder(stego_r_attacked)
            raw_r = logits_r.squeeze().detach().cpu().numpy()
            try:
                ldpc_decode_soft(raw_r, n=512, rate=_rate)
                r_ldpc[_rate].append(True)
            except ValueError:
                r_ldpc[_rate].append(False)

    ldpc_strs = []
    for _rate in [0.50, 0.33, 0.25]:
        _ok = sum(r_ldpc[_rate])
        ldpc_strs.append(f"r{_rate:.2f}={_ok}/{len(test_data)}")
    print(f"  {attack_name:<35s} BitAcc {np.mean(r_bacc):.4f} | "
          f"Chyby {np.mean(r_errs):.1f} | "
          f"LDPC OK: {' | '.join(ldpc_strs)}")

# Bez útoku (baseline)
eval_robustness("Bez útoku (baseline)",
    lambda s: s)

# JPEG komprese — tři úrovně kvality
if JPEG_ENABLED:
    eval_robustness("JPEG quality=95 (slabá komprese)",
        lambda s: jpeg_compress(s, 95))
    eval_robustness("JPEG quality=85 (střední komprese)",
        lambda s: jpeg_compress(s, 85))
    eval_robustness("JPEG quality=70 (silná komprese)",
        lambda s: jpeg_compress(s, 70))

# Gaussovský šum — plná intenzita
if NOISE_ENABLED:
    eval_robustness(f"Gaussovský šum σ={NOISE_SIGMA_MAX:.3f}",
        lambda s: torch.clamp(s + torch.randn_like(s) * NOISE_SIGMA_MAX, 0, 1))

# Gaussovské rozostření — plná intenzita
if BLUR_ENABLED:
    def blur_full(s):
        k = torch.ones(1, 1, 3, 3, device=s.device) / 9.0
        return s * (1 - BLUR_STRENGTH_MAX) + F.conv2d(s, k, padding=1) * BLUR_STRENGTH_MAX
    eval_robustness(f"Rozostření strength={BLUR_STRENGTH_MAX:.2f}",
        blur_full)

# Změna jasu — plná intenzita (nejhorší případ)
if BRIGHTNESS_ENABLED:
    eval_robustness(f"Jas +{BRIGHTNESS_DELTA_MAX:.2f}",
        lambda s: torch.clamp(s + BRIGHTNESS_DELTA_MAX, 0, 1))
    eval_robustness(f"Jas -{BRIGHTNESS_DELTA_MAX:.2f}",
        lambda s: torch.clamp(s - BRIGHTNESS_DELTA_MAX, 0, 1))

# =============================================================================
# VISUALIZATION
# Pro každý testovací snímek uloží panel 6 grafů:
#   [0,0] Original        — původní DICOM snímek
#   [0,1] Stego           — snímek s vloženým payloadem
#   [0,2] SE Adaptive Mask — anatomická maska (červená = kde encoder preferuje psát)
#   [1,0] Residual ×40    — perturbace zesílená 40× pro vizualizaci (seismic = +/-)
#   [1,1] Abs. rozdíl     — absolutní hodnota perturbace
#   [1,2] Residual × Maska — perturbace filtrovaná maskou (ukazuje kde encoder skutečně psal)
# =============================================================================

print("\n--- Ukládám grafy ---\n")

for i, sample in enumerate(test_data):

    # Ukládáme vizualizaci pouze každého 5. snímku (šetří disk a čas)
    if (i + 1) % 5 != 0:
        continue

    _img = sample["img"].to(device)
    _bit = sample["bits"].to(device)

    with torch.no_grad():
        stego, final_mask = encoder(_img, _bit)
        logits            = decoder(stego)

    pred     = (torch.sigmoid(logits) > 0.5).float()
    bacc     = (pred == _bit).float().mean()
    psnr_val = psnr(stego, _img)
    ssim_val = 1 - ssim_loss(stego, _img)
    errs     = int((pred != _bit).sum().item())
    # LDPC soft-decision status pro vizualizaci — všechny 3 rates
    raw_lg = logits.squeeze().detach().cpu().numpy()
    ecc_parts = []
    # Rate 0.50 — primární (bits = sample["bits"])
    try:
        ldpc_decode_soft(raw_lg, n=512, rate=0.50)
        ecc_parts.append("r0.50✓")
    except ValueError:
        ecc_parts.append("r0.50✗")
    # Rates 0.33, 0.25 — re-encode + re-embed + decode
    for _rate in [0.33, 0.25]:
        with torch.no_grad():
            stego_r, _ = encoder(_img, test_bits_by_rate[_rate][i].to(device))
            logits_r   = decoder(stego_r)
        raw_r = logits_r.squeeze().detach().cpu().numpy()
        try:
            ldpc_decode_soft(raw_r, n=512, rate=_rate)
            ecc_parts.append(f"r{_rate:.2f}✓")
        except ValueError:
            ecc_parts.append(f"r{_rate:.2f}✗")
    ecc_ok = "LDPC " + " | ".join(ecc_parts)

    # Převod tensorů na numpy pro matplotlib
    stego_np    = stego.squeeze().cpu().numpy()          # [512, 512]
    original_np = sample["img"].squeeze().cpu().numpy()  # [512, 512]
    residual_np = stego_np - original_np                 # perturbace (delta)
    mask_np     = final_mask.squeeze().cpu().numpy()     # anatomická maska [512, 512]

    # Výpočet δ_in/out ratio pro titulek grafu
    delta_in  = np.abs(residual_np[mask_np > 0.5]).mean() if (mask_np > 0.5).any() else 0
    delta_out = np.abs(residual_np[mask_np <= 0.5]).mean() if (mask_np <= 0.5).any() else 0
    ratio     = delta_in / (delta_out + 1e-8)

    img_path = f"result_{RUN_ID}_test{i+1}.png"

    # Vytvoří panel 2×3 grafů
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(
        f"MedStegViT [{RUN_ID}] · Snímek {i+1}/{len(test_data)} · "
        f"{os.path.basename(sample['path'])}\n"
        f"BitAcc={bacc.item():.4f} · Chyby={errs} · {ecc_ok} · "
        f"PSNR={psnr_val.item():.2f}dB · SSIM={ssim_val.item():.4f}",
        fontsize=10, fontweight="bold"
    )

    axes[0, 0].set_title("Original")
    axes[0, 0].imshow(original_np, cmap="gray")
    axes[0, 0].axis("off")

    axes[0, 1].set_title("Stego")
    axes[0, 1].imshow(stego_np, cmap="gray", vmin=0, vmax=1)
    axes[0, 1].axis("off")

    # Maska: hot colormap (černá = neaktivní, bílá/červená = aktivní)
    axes[0, 2].set_title("SE Adaptive Mask")
    im = axes[0, 2].imshow(mask_np, cmap="hot", vmin=0, vmax=1)
    plt.colorbar(im, ax=axes[0, 2])
    axes[0, 2].axis("off")

    # Residual ×40: seismic colormap — modrá = záporná delta, červená = kladná delta
    # Zesílení ×40 protože delta je velmi malá (neviditelná pouhým okem)
    axes[1, 0].set_title("Residual ×40")
    im2 = axes[1, 0].imshow(residual_np * 40, cmap="seismic", vmin=-1, vmax=1)
    plt.colorbar(im2, ax=axes[1, 0])
    axes[1, 0].axis("off")

    axes[1, 1].set_title("Abs. rozdíl")
    im3 = axes[1, 1].imshow(np.abs(residual_np), cmap="hot")
    plt.colorbar(im3, ax=axes[1, 1])
    axes[1, 1].axis("off")

    # Residual × Maska: ukazuje kde v anatomii encoder skutečně zapsal payload
    axes[1, 2].set_title(f"Residual × Maska  (δ_in/out={ratio:.1f}x)")
    im4 = axes[1, 2].imshow(np.abs(residual_np) * mask_np, cmap="hot")
    plt.colorbar(im4, ax=axes[1, 2])
    axes[1, 2].axis("off")

    plt.tight_layout()
    plt.savefig(img_path, dpi=150, bbox_inches="tight")
    plt.close()  # uvolní paměť (důležité při mnoha snímcích)
    print(f"  Graf uložen: {img_path}")

print(f"  Log uložen:  {LOG_FILE}")

# Obnoví původní stdout a uzavře log soubor
sys.stdout.close()
sys.stdout = sys.stdout.terminal