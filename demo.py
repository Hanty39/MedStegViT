# =============================================================================
# demo.py — Demonstrace celého MedStegViT pipeline
#
# Ukazuje kompletní workflow steganografického systému:
#   1. Načtení DICOM snímku a PatientID
#   2. Zakódování PatientID do snímku (encoder)
#   3. Dekódování PatientID zpět ze stego snímku (decoder)
#   4. Ověření správnosti pomocí LDPC / Reed-Solomon kódu
#   5. Test odolnosti vůči JPEG kompresi a dalším degradacím
#   6. Vizualizace výsledků
#
# =============================================================================

import os
import sys
import argparse
import hashlib
import io
import random
from datetime import datetime

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Vlastní moduly projektu
from medstegvit.config import load_config
from medstegvit.dicom_loader import load_dicom_512
from medstegvit.payload import hash_patient_id, bytes_to_bits, bits_to_bytes
from medstegvit.rs_codec import rs_encode, rs_decode
from medstegvit.ldpc_codec import ldpc_encode, ldpc_decode_soft, get_codec
from medstegvit.model_scif_net import SCIFEncoder, SCIFDecoder
from medstegvit.losses import ssim_loss


def psnr(img1, img2):
    mse = F.mse_loss(img1, img2)
    if mse < 1e-10:
        return 100.0
    return (10.0 * torch.log10(1.0 / mse)).item()


def jpeg_compress(stego_tensor, quality):
    img_np = stego_tensor.squeeze().detach().cpu().numpy()
    img_uint8 = (img_np * 255).clip(0, 255).astype(np.uint8)
    pil_img = Image.fromarray(img_uint8, mode="L")
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    pil_back = Image.open(buf)
    arr_back = np.array(pil_back).astype(np.float32) / 255.0
    return torch.tensor(arr_back, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(stego_tensor.device)


def main():
    parser = argparse.ArgumentParser(description="MedStegViT — Demo pipeline")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Cesta k checkpointu (.pt soubor)")
    parser.add_argument("--dicom", type=str, default=None,
                        help="Cesta k DICOM snímku (.dcm)")
    parser.add_argument("--patient_id", type=str, default=None,
                        help="Přepíše PatientID z DICOM tagu (pro testování)")
    parser.add_argument("--index", type=int, default=-1,
                        help="Index testovacího snímku (-1 = náhodný, 0 = první, ...)")
    parser.add_argument("--ecc", type=str, default="ldpc", choices=["rs", "ldpc"],
                        help="Opravný kód: 'ldpc' (soft-decision, default) nebo 'rs' (Reed-Solomon)")
    parser.add_argument("--rate", type=float, default=0.50,
                        help="LDPC code rate (0.50=default, 0.33, 0.25 = víc redundance)")
    parser.add_argument("--output", type=str, default=None,
                        help="Výstupní soubor s vizualizací (default: auto-generovaný)")
    args = parser.parse_args()

    # ── Konfigurace ──────────────────────────────────────────────────
    cfg = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 65)
    print("  MedStegViT — Demonstrace steganografického systému")
    print("=" * 65)
    print(f"  Device: {device}")

    # ── Nalezení checkpointu ─────────────────────────────────────────
    if args.checkpoint:
        ckpt_path = args.checkpoint
    else:
        # Hledá best checkpoint z nejnovějšího runu
        ckpt_dir = "checkpoints"
        candidates = sorted([
            f for f in os.listdir(ckpt_dir) if f.endswith("_best.pt")
        ])
        if not candidates:
            candidates = sorted([f for f in os.listdir(ckpt_dir) if f.endswith(".pt")])
        if not candidates:
            print("  ✗ Žádný checkpoint nenalezen v checkpoints/")
            print("    Nejdřív spusť trénink: python main.py")
            sys.exit(1)
        ckpt_path = os.path.join(ckpt_dir, candidates[-1])

    print(f"  Checkpoint: {ckpt_path}")

    # ── Nalezení DICOM snímku ────────────────────────────────────────
    idx = 0  # default pro případ --dicom
    if args.dicom:
        dicom_path = args.dicom
    else:
        # Vezme první testovací snímek
        test_dir = os.path.join(cfg["data"]["base_dir"], "test")
        dcm_files = sorted([
            os.path.join(test_dir, f) for f in os.listdir(test_dir)
            if f.endswith(".dcm")
        ])
        if not dcm_files:
            print(f"  ✗ Žádný .dcm soubor nenalezen v {test_dir}")
            sys.exit(1)
        if args.index < 0:
            idx = random.randint(0, len(dcm_files) - 1)
            print(f"  Náhodný testovací snímek [{idx}/{len(dcm_files)}]")
        else:
            idx = min(args.index, len(dcm_files) - 1)
            print(f"  Testovací snímek [{idx}/{len(dcm_files)}]")
        dicom_path = dcm_files[idx]

    print(f"  DICOM: {os.path.basename(dicom_path)}")

    # =====================================================================
    # KROK 1: Načtení DICOM snímku
    # =====================================================================
    print(f"\n{'─' * 65}")
    print("  KROK 1: Načtení DICOM snímku")
    print(f"{'─' * 65}")

    img, original_patient_id = load_dicom_512(dicom_path)
    if args.patient_id:
        original_patient_id = args.patient_id
        print(f"  PatientID přepsáno parametrem: {original_patient_id}")
    else:
        print(f"  PatientID (z DICOM tagu): {original_patient_id}")

    print(f"  Rozlišení snímku: {img.shape[0]}×{img.shape[1]} px")
    print(f"  Rozsah hodnot: [{img.min():.3f}, {img.max():.3f}]")

    # Převod na PyTorch tensor
    img_tensor = (torch.tensor(img, dtype=torch.float32)
                  .unsqueeze(0).unsqueeze(0).to(device))  # [1, 1, 512, 512]

    # =====================================================================
    # KROK 2: Vytvoření payloadu (PatientID → 512 bitů)
    # =====================================================================
    print(f"\n{'─' * 65}")
    print("  KROK 2: Vytvoření payloadu")
    print(f"{'─' * 65}")

    # SHA-256 hash
    patient_hash = hash_patient_id(original_patient_id)
    print(f"  SHA-256 hash: {patient_hash.hex()[:32]}...")
    print(f"               ({len(patient_hash)} bajtů = {len(patient_hash) * 8} bitů)")

    ecc_mode = args.ecc
    ldpc_rate = args.rate  # default pro výstupní filename
    print(f"  Opravný kód: {ecc_mode.upper()}")

    if ecc_mode == "rs":
        # Reed-Solomon: 32 B data → 64 B kódové slovo → 512 bitů
        rs_codeword = rs_encode(patient_hash)
        bits = bytes_to_bits(rs_codeword)
        print(f"  RS kódování: {len(patient_hash)} → {len(rs_codeword)} bajtů "
              f"(+{len(rs_codeword) - len(patient_hash)} opravných)")
        max_byte_errors = (len(rs_codeword) - len(patient_hash)) // 2
        print(f"  Opravná kapacita: {max_byte_errors} bajtů = {max_byte_errors * 8} bitů "
              f"(ale bitové chyby musí být soustředěné v ≤{max_byte_errors} bajtech)")
    else:
        # LDPC kódování s volitelným rate
        codec = get_codec(n=512, rate=ldpc_rate)
        # Zkrátit hash na k bitů (= codec.k) pokud je delší
        hash_bits_needed = (codec.k + 7) // 8  # počet bajtů potřebných
        hash_truncated = patient_hash[:hash_bits_needed]
        bits = ldpc_encode(hash_truncated, n=512, rate=ldpc_rate)
        print(f"  LDPC kódování: {len(hash_truncated)} B ({len(hash_truncated)*8} bitů) "
              f"→ {len(bits)} bitů (rate {codec.k/codec.n:.3f}, k={codec.k})")
        print(f"  Soft-decision dekódování: využívá magnitudu logitů z decoderu")
        if ldpc_rate < 0.50:
            print(f"  Truncated hash: {len(hash_truncated)*8} z 256 bitů SHA-256")
            print(f"    (kolizní odolnost 2^{len(hash_truncated)*8//2} — stále bezpečné)")

    bit_tensor = torch.tensor(bits, dtype=torch.float32).unsqueeze(0).to(device)  # [1, 512]
    print(f"  Payload: {len(bits)} bitů "
          f"(jedniček: {bits.sum()}, nul: {len(bits) - bits.sum()})")

    # =====================================================================
    # KROK 3: Načtení natrénovaného modelu
    # =====================================================================
    print(f"\n{'─' * 65}")
    print("  KROK 3: Načtení modelu")
    print(f"{'─' * 65}")

    encoder = SCIFEncoder(cfg).to(device)
    decoder = SCIFDecoder(cfg).to(device)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    print(f"  Klíče v checkpointu: {list(ckpt.keys())}")

    enc_res = encoder.load_state_dict(ckpt["encoder"], strict=False)
    dec_res = decoder.load_state_dict(ckpt["decoder"], strict=False)
    if enc_res.missing_keys:
        print(f"  Encoder — chybějící klíče: {enc_res.missing_keys}")
    if enc_res.unexpected_keys:
        print(f"  Encoder — ignorované klíče: {enc_res.unexpected_keys}")
    if dec_res.missing_keys:
        print(f"  Decoder — chybějící klíče: {dec_res.missing_keys}")
    if dec_res.unexpected_keys:
        print(f"  Decoder — ignorované klíče: {dec_res.unexpected_keys}")

    encoder.mask_strength = 1.0

    encoder.eval()
    decoder.eval()

    n_enc = sum(p.numel() for p in encoder.parameters())
    n_dec = sum(p.numel() for p in decoder.parameters())
    print(f"  Encoder: {n_enc:,} parametrů")
    print(f"  Decoder: {n_dec:,} parametrů")
    print(f"  ✓ Model načten z {os.path.basename(ckpt_path)}")

    # =====================================================================
    # KROK 4: Zakódování payloadu do snímku (steganografie)
    # =====================================================================
    print(f"\n{'─' * 65}")
    print("  KROK 4: Zakódování payloadu do snímku")
    print(f"{'─' * 65}")

    # Benchmark: měření doby enkódování (průměr přes 10 průchodů)
    import time
    with torch.no_grad():
        for _ in range(3):  # warm-up
            encoder(img_tensor, bit_tensor)
        t0 = time.perf_counter()
        for _ in range(10):
            stego, mask = encoder(img_tensor, bit_tensor)
        t_enc = (time.perf_counter() - t0) / 10 * 1000  # ms

    stego_psnr = psnr(stego, img_tensor)
    stego_ssim = (1 - ssim_loss(stego, img_tensor))
    delta = (stego - img_tensor).abs()
    mean_delta = delta.mean().item()
    max_delta = delta.max().item()

    print(f"  Čas enkódování: {t_enc:.1f} ms / snímek  (~{3600000/t_enc:.0f} snímků/hod)")
    print(f"  PSNR: {stego_psnr:.2f} dB (čím vyšší, tím méně viditelná změna)")
    print(f"  SSIM: {stego_ssim:.4f} (čím blíže 1, tím více zachována struktura)")
    print(f"  Průměrná perturbace: {mean_delta:.4f} (rozsah 0–1)")
    print(f"  Maximální perturbace: {max_delta:.4f}")
    print(f"  Maska — aktivních pixelů: {(mask > 0.5).float().mean().item() * 100:.1f}%")

    # =====================================================================
    # KROK 5: Dekódování payloadu ze stego snímku
    # =====================================================================
    print(f"\n{'─' * 65}")
    print("  KROK 5: Dekódování payloadu ze stego snímku")
    print(f"{'─' * 65}")

    with torch.no_grad():
        for _ in range(3):  # warm-up
            decoder(stego)
        t0 = time.perf_counter()
        for _ in range(10):
            logits = decoder(stego)
        t_dec = (time.perf_counter() - t0) / 10 * 1000  # ms

    # Prahování logitů → predikované bity
    pred_bits = (torch.sigmoid(logits) > 0.5).float()
    bit_errors = int((pred_bits != bit_tensor).sum().item())
    bit_acc = (pred_bits == bit_tensor).float().mean().item()

    print(f"  Čas dekódování: {t_dec:.1f} ms / snímek  (~{3600000/t_dec:.0f} snímků/hod)")
    print(f"  Dekódováno 512 logitů z decoderu")
    print(f"  Bit Accuracy (hard): {bit_acc:.4f} ({bit_acc * 100:.1f}%)")
    print(f"  Bitových chyb: {bit_errors}/512")

    # ECC dekódování
    ecc_success = False
    decoded_hash = None
    hash_match = False

    if ecc_mode == "rs":
        pred_bytes = bits_to_bytes(pred_bits.squeeze())
        orig_bytes = bits_to_bytes(bit_tensor.squeeze())
        byte_errors = sum(a != b for a, b in zip(pred_bytes, orig_bytes))
        max_byte_err = (cfg["payload"]["rs_n"] - cfg["payload"]["rs_k"]) // 2
        try:
            decoded_hash = bytes(rs_decode(pred_bytes))
            ecc_success = True
            hash_match = (decoded_hash == patient_hash)
        except Exception:
            ecc_success = False

        if ecc_success:
            print(f"  RS dekódování: ✓ Úspěch")
            print(f"    Bajtových chyb: {byte_errors}/64 (limit {max_byte_err})")
        else:
            print(f"  RS dekódování: ✗ Selhalo")
            print(f"    Bajtových chyb: {byte_errors}/64 (limit {max_byte_err})")

    else:  # LDPC
        # Soft-decision: předáme surové logity (ne tvrdé bity!)
        raw_logits = logits.squeeze().detach().cpu().numpy()
        print(f"  LDPC soft-decision dekódování (rate={ldpc_rate:.2f})...")
        try:
            decoded_bytes = ldpc_decode_soft(raw_logits, n=512, rate=ldpc_rate)
            decoded_hash = decoded_bytes[:hash_bits_needed]
            ecc_success = True
            hash_match = (decoded_hash == hash_truncated)
            print(f"  LDPC dekódování: ✓ BP konvergoval")
            print(f"    Bitových chyb před korekcí: {bit_errors}/512")
        except ValueError as e:
            ecc_success = False
            print(f"  LDPC dekódování: ✗ {e}")

    # =====================================================================
    # KROK 6: Ověření PatientID
    # =====================================================================
    print(f"\n{'─' * 65}")
    print("  KROK 6: Ověření PatientID")
    print(f"{'─' * 65}")

    if ecc_mode == "ldpc":
        hash_display = hash_truncated.hex()
        hash_label = f"hash ({len(hash_truncated)*8} bitů)"
    else:
        hash_display = patient_hash.hex()[:32]
        hash_label = "hash (256 bitů)"
    print(f"  Původní PatientID: {original_patient_id}")
    print(f"  Původní {hash_label}: {hash_display[:32]}{'...' if len(hash_display) > 32 else ''}")
    if ecc_success:
        print(f"  Dekódovaný hash:   {decoded_hash.hex()[:32]}...")
        if hash_match:
            print(f"\n  ╔══════════════════════════════════════════════════╗")
            print(f"  ║  ✓  PatientID OVĚŘENO — hash se shoduje         ║")
            print(f"  ╚══════════════════════════════════════════════════╝")
        else:
            print(f"\n  ╔══════════════════════════════════════════════════╗")
            print(f"  ║  ✗  PatientID NESHODUJE SE — hash je odlišný    ║")
            print(f"  ╚══════════════════════════════════════════════════╝")
    else:
        print(f"  Dekódovaný hash:   [nelze dekódovat — příliš mnoho chyb]")
        ecc_name = "LDPC" if ecc_mode == "ldpc" else "RS"
        print(f"\n  ╔══════════════════════════════════════════════════╗")
        print(f"  ║  ✗  PatientID NELZE OVĚŘIT — {ecc_name} dekódování selhalo ║")
        print(f"  ╚══════════════════════════════════════════════════╝")

    # =====================================================================
    # KROK 7: Test odolnosti vůči JPEG kompresi
    # =====================================================================
    print(f"\n{'─' * 65}")
    print("  KROK 7: Test odolnosti vůči JPEG kompresi")
    print(f"{'─' * 65}")

    for quality in [95, 85, 70]:
        with torch.no_grad():
            stego_jpeg = jpeg_compress(stego, quality)
            logits_jpeg = decoder(stego_jpeg)
        pred_jpeg = (torch.sigmoid(logits_jpeg) > 0.5).float()
        errs_jpeg = int((pred_jpeg != bit_tensor).sum().item())
        bacc_jpeg = (pred_jpeg == bit_tensor).float().mean().item()

        if ecc_mode == "rs":
            pred_bytes_jpeg = bits_to_bytes(pred_jpeg.squeeze())
            try:
                rs_decode(pred_bytes_jpeg)
                ecc_ok = "✓ RS OK"
            except Exception:
                ecc_ok = "✗ RS FAIL"
        else:
            raw_logits_jpeg = logits_jpeg.squeeze().detach().cpu().numpy()
            try:
                ldpc_decode_soft(raw_logits_jpeg, n=512, rate=ldpc_rate)
                ecc_ok = "✓ LDPC OK"
            except ValueError:
                ecc_ok = "✗ LDPC FAIL"

        print(f"  JPEG quality={quality:2d}: BitAcc {bacc_jpeg:.4f} | "
              f"Chyby {errs_jpeg:3d}/512 | {ecc_ok}")

    # =====================================================================
    # KROK 8: Vizualizace
    # =====================================================================
    print(f"\n{'─' * 65}")
    print("  KROK 8: Vizualizace")
    print(f"{'─' * 65}")

    img_np = img_tensor.squeeze().cpu().numpy()
    stego_np = stego.squeeze().cpu().numpy()
    mask_np = mask.squeeze().cpu().numpy()
    diff_np = np.abs(stego_np - img_np)

    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    fig.suptitle(
        f"MedStegViT Demo [{ecc_mode.upper()}] — PatientID: {original_patient_id[:36]}...\n"
        f"BitAcc={bit_acc:.4f} · Chyby={bit_errors}/512 · "
        f"{'✓ '+ecc_mode.upper()+' OK' if ecc_success else '✗ '+ecc_mode.upper()+' FAIL'} · "
        f"PSNR={stego_psnr:.1f} dB · SSIM={stego_ssim:.4f} · "
        f"{'✓ Hash shoda' if hash_match else '✗ Hash neshoda'}",
        fontsize=13, fontweight='bold'
    )

    # [0,0] Originální snímek
    axes[0, 0].imshow(img_np, cmap='gray', vmin=0, vmax=1)
    axes[0, 0].set_title("1. Originální DICOM snímek", fontsize=11)
    axes[0, 0].axis('off')

    # [0,1] Stego snímek
    axes[0, 1].imshow(stego_np, cmap='gray', vmin=0, vmax=1)
    axes[0, 1].set_title(f"2. Stego snímek (PSNR {stego_psnr:.1f} dB, SSIM {stego_ssim:.4f})", fontsize=11)
    axes[0, 1].axis('off')

    # [1,0] Anatomická maska
    im_mask = axes[1, 0].imshow(mask_np, cmap='hot', vmin=0, vmax=1)
    axes[1, 0].set_title("3. Anatomická maska (kde se píše)", fontsize=11)
    axes[1, 0].axis('off')
    plt.colorbar(im_mask, ax=axes[1, 0], fraction=0.046)

    # [1,1] Absolutní rozdíl (zesílený)
    im_diff = axes[1, 1].imshow(diff_np * 10, cmap='hot', vmin=0, vmax=diff_np.max() * 10)
    axes[1, 1].set_title("4. Rozdíl ×10 (perturbace)", fontsize=11)
    axes[1, 1].axis('off')
    plt.colorbar(im_diff, ax=axes[1, 1], fraction=0.046)


    plt.tight_layout()
    if args.output:
        output_path = args.output
    else:
        os.makedirs("demo_results", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        rate_str = f"r{ldpc_rate:.2f}" if ecc_mode == "ldpc" else "rs"
        output_path = f"demo_results/demo_{timestamp}_idx{idx}_{rate_str}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  ✓ Vizualizace uložena: {output_path}")

    print(f"\n{'=' * 65}")
    print(f"  Demo dokončeno")
    print(f"{'=' * 65}")


if __name__ == "__main__":
    main()
