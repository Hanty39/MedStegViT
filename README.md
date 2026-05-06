# MedStegViT

Deep learning steganography system for embedding patient identity information into medical DICOM X-ray images.  
A Vision Transformer-based encoder-decoder architecture that hides a 512-bit LDPC-encoded SHA-256 hash of the patient ID inside chest X-ray images.

---

## Overview

MedStegViT embeds a cryptographic fingerprint of a patient's identity directly into the pixel data of a DICOM image. The modification is imperceptible to the human eye (PSNR ≥ 30 dB) and survives common image degradations such as JPEG compression and resampling.

The system consists of three neural networks trained end-to-end:

| Component | Role |
|---|---|
| **SCIFEncoder** | CNN + ViT encoder. Takes the cover image and 512-bit payload, outputs a stego image. |
| **StructureExtractor** | CNN that generates an anatomical mask (edges, ribs, heart border). The encoder is guided to write the payload only inside this mask. |
| **SCIFDecoder** | ViT decoder. Extracts the 512-bit payload from the stego image. |

The payload pipeline:

```
PatientID  →  SHA-256 (256 bits)  →  LDPC encoding (512 bits)  →  SCIFEncoder  →  stego DICOM
stego DICOM  →  SCIFDecoder  →  LDPC soft decoding  →  SHA-256 hash  →  verify PatientID
```

---

## Results

Evaluated on the [SIIM-ACR Pneumothorax dataset](https://www.kaggle.com/datasets/anisayari/siimacrpneumothoraxsegmentationzip-dataset) (1 377 test images, 512×512 px).

| Metric | Value |
|---|---|
| Bit accuracy | 94.74 % |
| Hash reconstruction (LDPC r = 0.50) | 99.9 % |
| Hash reconstruction (LDPC r = 0.33) | 100 % |
| Hash reconstruction (LDPC r = 0.25) | 100 % |
| PSNR | 30.07 dB |
| SSIM | 0.96 |
| Payload capacity | 512 bits |

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/Hanty39/MedStegViT.git
cd MedStegViT
```

### 2. Create a virtual environment

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux / macOS:
source .venv/bin/activate
```

### 3. Install PyTorch with CUDA

Go to [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/) and pick the right command for your GPU and CUDA version.

Example for **CUDA 12.4**:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

Example for **CUDA 13.0** (RTX 50xx series):
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
```

CPU-only (slow, not recommended for training):
```bash
pip install torch torchvision
```

### 4. Install remaining dependencies

```bash
pip install -r requirements.txt
```

---

## Dataset

The model was trained on the **SIIM-ACR Pneumothorax Segmentation** dataset (chest X-rays in DICOM format).

1. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/anisayari/siimacrpneumothoraxsegmentationzip-dataset).
2. Place the `.dcm` files into the following structure:

```
data/
└── raw_dicom/
    ├── train/   ← training DICOM files
    └── test/    ← test DICOM files
```

The loader accepts any 16-bit grayscale DICOM file and resizes it to 512×512 automatically.

---

## Training

All hyperparameters are set in `cfg/default.yaml`.

```bash
python main.py
```

Key parameters in `cfg/default.yaml`:

```yaml
data:
  n_train: 5070          # number of training images (0 = all)
  val_split: 0.15        # 15 % held out for validation

training:
  total_epochs: 50
  learning_rate: 0.0005

checkpoints:
  load_checkpoint: ""    # path to resume from checkpoint, "" = train from scratch
```

Checkpoints are saved to `checkpoints/`. The best model (highest validation BitAcc) is saved as `checkpoints/best_model.pt`.

Training logs and plots are written to `graphs/`.

### Pre-trained checkpoint

A pre-trained checkpoint is available in [Releases](https://github.com/Hanty39/MedStegViT/releases/tag/v1.0).  
Download [`ckpt_20260417_172607_best.pt`](https://github.com/Hanty39/MedStegViT/releases/download/v1.0/ckpt_20260417_172607_best.pt) (220 MB) and place it in the `checkpoints/` folder, then run:

```bash
python demo.py --checkpoint checkpoints/ckpt_20260417_172607_best.pt --rate 0.5
```

---

## Demo / Inference

Run the demo on a single DICOM file using a trained checkpoint:

```bash
# Simplest — random test image, LDPC rate 0.50
python demo.py

# LDPC rate 0.25 (stronger error correction, 100 % hash reconstruction)
python demo.py --rate 0.25

# Compare rates on the same image
python demo.py --rate 0.50 --index 42
python demo.py --rate 0.33 --index 42
python demo.py --rate 0.25 --index 42

# Specific checkpoint
python demo.py --checkpoint checkpoints/ckpt_20260417_172607_best.pt --rate 0.5

# Reed-Solomon instead of LDPC
python demo.py --ecc rs

# Custom DICOM file
python demo.py --dicom path/to/image.dcm --rate 0.25

# Custom output file
python demo.py --rate 0.25 --output my_test.png

# All options combined
python demo.py --checkpoint checkpoints/best_model.pt --index 100 --ecc ldpc --rate 0.50
```

| Argument | Default | Description |
|---|---|---|
| `--checkpoint` | auto-detected | Path to `.pt` checkpoint file |
| `--index` | random | Index of the test image (0 – 1376) |
| `--dicom` | — | Path to a custom `.dcm` file (overrides `--index`) |
| `--rate` | `0.50` | LDPC code rate: `0.25`, `0.33`, or `0.50` |
| `--ecc` | `ldpc` | Error-correction codec: `ldpc` or `rs` (Reed-Solomon) |
| `--output` | auto-named | Output PNG filename in `demo_results/` |
| `--patient_id` | from DICOM | Override PatientID for testing |


Results are saved to `demo_results/`.

---

## Project Structure

```
MedStegViT/
├── cfg/                          # model configuration
│   └── default.yaml              # hyperparameters, paths, augmentation
├── data/                         # input data
│   └── raw_dicom/                # DICOM images (SIIM-ACR dataset)
│       ├── test/                 # test images
│       │   └── *.dcm             # chest X-rays in DICOM format
│       └── train/                # training images
│           └── *.dcm             # chest X-rays in DICOM format
├── medstegvit/                   # main Python package
│   ├── config.py                 # YAML config loader
│   ├── dicom_loader.py           # DICOM loading and normalisation
│   ├── model_scif_net.py         # SCIFEncoder and SCIFDecoder
│   ├── structure_extractor.py    # StructureExtractor (U-Net + SE)
│   ├── payload_encoder.py        # MLP payload encoder (bits → token)
│   ├── payload_decoder.py        # MLP payload decoder (token → logits)
│   ├── patch_embed.py            # PatchEmbed layer for ViT
│   ├── transformer_block.py      # Transformer block (pre-LN, MSA)
│   ├── positional_encoding.py    # positional encoding for tokens
│   ├── ldpc_codec.py             # LDPC codec with soft-decision decoding
│   ├── losses.py                 # mask_aware_loss, sparsity_loss, SSIM
│   ├── payload.py                # SHA-256 hash of patient ID
│   └── rs_codec.py               # Reed-Solomon codec (used in demo.py)
├── checkpoints/                  # saved model checkpoints
│   └── *.pt                      # model weights (best / periodic)
├── graphs/                       # training progress plots
│   └── *.png                     # BitAcc, Loss, PSNR, SSIM per epoch
├── demo_results/                 # demo script outputs
│   └── *.png                     # cover / stego / mask visualisations
├── main.py                       # training and evaluation script
├── demo.py                       # inference demo (single image)
├── log_*.txt                     # training log files
└── requirements.txt
```


---

## Configuration Reference

See [`cfg/default.yaml`](cfg/default.yaml) for a fully commented list of all hyperparameters including loss weights, augmentation settings, LDPC code rates, and checkpoint options.

---

## License

This project was developed as a master's thesis at Brno University of Technology (VUT FEKT).  
For academic and research use only.
