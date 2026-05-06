# =============================================================================
# dicom_loader.py — Načítání DICOM snímků z disku
#
# Tento modul načte .dcm soubor, normalizuje pixel data na [0, 1]
# a přečte PatientID které se použije pro generování payloadu.
# =============================================================================

import pydicom
import cv2
import numpy as np
import warnings


def load_dicom_512(path):
    ds = pydicom.dcmread(path)

    img = ds.pixel_array.astype(np.float32)

    # Per-snímek normalizace na [0, 1]
    img -= img.min()
    img /= img.max() + 1e-6

    # Přeškáluje snímek na pevné rozlišení 512×512 pixelů
    # Bilineární interpolace (INTER_LINEAR) je vhodná pro spojité snímky —
    # průměruje okolní pixely, nevytváří artefakty jako nearest-neighbor
    img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LINEAR)

    # Přečte PatientID z DICOM tagů
    patient_id = str(ds.get("PatientID", ""))

    if not patient_id:
        warnings.warn(
            f"PatientID chybí v {path} — použit prázdný string. "
            f"Všechny snímky bez PatientID budou mít stejný payload!",
            UserWarning
        )

    return img, patient_id