# =============================================================================
# payload.py — Generování a konverze payloadu (PatientID → 512 bitů)
#
# Pipeline pro vytvoření payloadu:
#   PatientID (string) → SHA-256 hash (32 bajtů) → LDPC encode → 512 bitů
#
# Tento modul pokrývá: hash a konverzi bajtů ↔ bity.
# =============================================================================

import hashlib    # SHA-256 hashovací funkce ze standardní knihovny
import numpy as np
import torch


def hash_patient_id(patient_id: str) -> bytes:
    """
    Převede PatientID string na 32bajtový SHA-256 hash.
    """
    return hashlib.sha256(patient_id.encode("utf-8")).digest()


def bytes_to_bits(data: bytes) -> np.ndarray:
    """
    Převede bajty na pole bitů (0 nebo 1).
    """
    bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8))
    return bits  # pole 0/1 délky len(data)*8, tedy pro 64 bajtů → 512 prvků


def bits_to_bytes(bits) -> bytes:
    """
    Převede pole bitů zpět na bajty. Inverzní funkce k bytes_to_bits().
    """

    # Převede PyTorch tensor na numpy array pokud je potřeba
    if isinstance(bits, torch.Tensor):
        bits = bits.detach().cpu().numpy()

    bits = bits.flatten()  # zajistí 1D pole bez ohledu na tvar vstupu

    # Skládá bajty bit po bitu — každých 8 bitů tvoří jeden bajt
    # Bit shifting: byte = (byte << 1) | bit přidá bit zprava
    byte_array = []
    for i in range(0, len(bits), 8):
        byte = 0
        for j in range(8):
            byte = (byte << 1) | int(bits[i + j])  # posune doleva a přidá další bit
        byte_array.append(byte)

    return bytes(byte_array)