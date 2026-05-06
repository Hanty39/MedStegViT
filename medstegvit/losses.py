# =============================================================================
# losses.py — Loss funkce pro trénink MedStegViT
# =============================================================================

import torch
import torch.nn.functional as F


def ssim_loss(x, y, window_size=11):
    # Konstanty pro numerickou stabilitu (zabraňují dělení nulou)
    # Hodnoty dle standardní definice SSIM
    C1 = 0.01 ** 2   # = 0.0001
    C2 = 0.03 ** 2   # = 0.0009

    # Lokální průměry — avg_pool2d počítá klouzavý průměr v okně window_size×window_size
    mu_x = F.avg_pool2d(x, window_size, stride=1, padding=window_size // 2)
    mu_y = F.avg_pool2d(y, window_size, stride=1, padding=window_size // 2)

    # Lokální rozptyly a kovariance
    # Vzorec: Var(X) = E[X²] - E[X]²
    sigma_x  = F.avg_pool2d(x * x, window_size, stride=1, padding=window_size // 2) - mu_x ** 2
    sigma_y  = F.avg_pool2d(y * y, window_size, stride=1, padding=window_size // 2) - mu_y ** 2
    # Kovariance: Cov(X,Y) = E[XY] - E[X]·E[Y]
    sigma_xy = F.avg_pool2d(x * y, window_size, stride=1, padding=window_size // 2) - mu_x * mu_y

    # SSIM vzorec — pro každý pixel lokálně
    # Čitatel: kombinuje podobnost jasu (mu) a kontrastu/struktury (sigma)
    # Jmenovatel: normalizační faktory
    ssim_map = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / (
        (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)
    )

    # Průměr přes všechny pixely → skalár, převod na loss (1 - SSIM)
    return 1 - ssim_map.mean()


def gradient_loss(x, y):

    sobel_x = torch.tensor(
        [[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]],
        dtype=torch.float32, device=x.device
    ).unsqueeze(0)

    sobel_y = torch.tensor(
        [[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]],
        dtype=torch.float32, device=x.device
    ).unsqueeze(0)

    # Aplikuje Sobelovy filtry konvolucí na oba snímky
    grad_x_x = F.conv2d(x, sobel_x, padding=1)   # hrany v x, směr x
    grad_y_x = F.conv2d(x, sobel_y, padding=1)   # hrany v x, směr y
    grad_x_y = F.conv2d(y, sobel_x, padding=1)   # hrany v y, směr x
    grad_y_y = F.conv2d(y, sobel_y, padding=1)   # hrany v y, směr y

    # L1 rozdíl gradientů — penalizuje odlišné hrany
    return F.l1_loss(grad_x_x, grad_x_y) + F.l1_loss(grad_y_x, grad_y_y)


def energy_loss(residual):
    return torch.mean(residual ** 2)


def mask_aware_loss(cover, stego, mask, lambda_outside=20.0):

    # Kvadratická perturbace (delta²) pro každý pixel
    delta = (stego - cover) ** 2

    loss_inside  = (delta * mask).mean()

    # Průměrná perturbace MIMO masku — silně penalizujeme
    # (1 - maska) invertuje masku: hrany → 0, pozadí → 1
    loss_outside = (delta * (1.0 - mask)).mean()

    return loss_inside + lambda_outside * loss_outside


def sparsity_loss(mask, target_density=0.15):

    return torch.abs(mask.mean() - target_density)