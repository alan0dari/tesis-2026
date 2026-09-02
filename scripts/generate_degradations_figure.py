"""
Genera la figura de las cinco degradaciones controladas del Capítulo 4:
la imagen original y las cinco degradaciones de contraste, cada una con
su histograma como inserto, para evidenciar la firma de cada degradación.

Salidas:
  docs/libro/Figures/capitulo4/degradaciones.pdf
  docs/libro/Figures/capitulo4/degradaciones.png

Uso:
    python scripts/generate_degradations_figure.py [--image data/original/114.jpg]
"""

import sys
import argparse
from pathlib import Path

PROJECT_ROOT = str(Path(__file__).parent.parent)
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.utils.degradation import (
    apply_low_contrast, apply_underexposure, apply_overexposure,
    apply_poor_local_contrast, apply_skewed_histogram
)

COL_HIST = '#20606e'


def main():
    parser = argparse.ArgumentParser(description='Figura de degradaciones (Cap. 4)')
    parser.add_argument('--image', default='data/original/114.jpg')
    args = parser.parse_args()

    original = cv2.imread(args.image, cv2.IMREAD_GRAYSCALE)
    if original is None:
        raise FileNotFoundError(f"No se pudo cargar {args.image}")

    # Parámetros moderados y fijos, representativos de los rangos aleatorios
    # que usa el experimento (ver apply_random_degradation)
    paneles = [
        ('(a) Original', original),
        ('(b) Bajo contraste global', apply_low_contrast(original, factor=0.45)),
        ('(c) Subexposición', apply_underexposure(original, gamma=2.2, offset=-25)),
        ('(d) Sobreexposición', apply_overexposure(original, gamma=0.55,
                                                   saturation_threshold=235)),
        ('(e) Bajo contraste local', apply_poor_local_contrast(
            original, blur_kernel=13, contrast_reduction=0.6)),
        ('(f) Histograma sesgado', apply_skewed_histogram(
            original, skew_direction='dark', intensity=0.7)),
    ]

    # Altura ajustada a la relación de aspecto de la panorámica (~2.9:1)
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 3.2),
                             gridspec_kw={'hspace': 0.35, 'wspace': 0.06})

    for k, (titulo, img) in enumerate(paneles):
        ax = axes[k // 3, k % 3]
        ax.imshow(img, cmap='gray', vmin=0, vmax=255, aspect='auto')
        ax.set_title(titulo, fontsize=10.5)
        ax.set_xticks([]); ax.set_yticks([])

        # Histograma como inserto (escala log para que los picos de
        # saturación no aplasten el resto de la distribución)
        axi = ax.inset_axes([0.66, 0.06, 0.32, 0.34])
        axi.patch.set_alpha(0.85)
        axi.hist(img.ravel(), bins=128, range=(0, 255), color=COL_HIST, log=True)
        axi.set_xlim(0, 255)
        axi.set_xticks([]); axi.set_yticks([])
        for s in axi.spines.values():
            s.set_linewidth(0.6)

    out_dir = Path(PROJECT_ROOT) / 'docs' / 'libro' / 'Figures' / 'capitulo4'
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / 'degradaciones.pdf', bbox_inches='tight')
    fig.savefig(out_dir / 'degradaciones.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Figura generada en {out_dir}")


if __name__ == '__main__':
    main()
