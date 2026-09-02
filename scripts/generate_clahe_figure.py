"""
Genera la figura comparativa de mejora de contraste del Capítulo 2:
imagen de entrada (bajo contraste) vs. HE global vs. CLAHE, con sus
histogramas y entropías. Sobre el panel CLAHE se superpone la grilla
de regiones contextuales (R_x x R_y).

Salidas:
  docs/libro/Figures/capitulo2/clahe_comparacion.pdf
  docs/libro/Figures/capitulo2/clahe_comparacion.png

Uso:
    python scripts/generate_clahe_figure.py [--image data/original/114.jpg]
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

from src.clahe.processor import CLAHEProcessor
from src.utils.degradation import apply_low_contrast
from src.metrics.entropy import calculate_entropy

COL_HIST = '#20606e'
COL_GRID = '#c8a415'

RX, RY, CLIP = 8, 8, 2.0


def main():
    parser = argparse.ArgumentParser(description='Figura HE vs CLAHE (Cap. 2)')
    parser.add_argument('--image', default='data/original/114.jpg')
    args = parser.parse_args()

    original = cv2.imread(args.image, cv2.IMREAD_GRAYSCALE)
    if original is None:
        raise FileNotFoundError(f"No se pudo cargar {args.image}")

    # Entrada con bajo contraste (escenario del problema)
    entrada = apply_low_contrast(original, factor=0.45)

    # Mejoras: HE global y CLAHE
    he = cv2.equalizeHist(entrada)
    clahe = CLAHEProcessor(rx=RX, ry=RY, clip_limit=CLIP).process(entrada)

    paneles = [
        ('Entrada (bajo contraste)', entrada, False),
        ('HE (ecualización global)', he, False),
        (f'CLAHE ($R_x = R_y = {RX}$, $C = {CLIP}$)', clahe, True),
    ]

    # La altura de la figura se ajusta a la relación de aspecto real de la
    # ortopantomografía (~2.9:1) para que no queden huecos entre filas
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 3.0),
                             gridspec_kw={'height_ratios': [1.15, 0.85],
                                          'hspace': 0.35, 'wspace': 0.12})

    for col, (titulo, img, con_grilla) in enumerate(paneles):
        # Fila superior: imagen
        ax = axes[0, col]
        ax.imshow(img, cmap='gray', vmin=0, vmax=255, aspect='auto')
        ax.set_title(titulo, fontsize=10.5)
        ax.set_xticks([]); ax.set_yticks([])

        # Grilla de regiones contextuales sobre el panel CLAHE
        if con_grilla:
            h, w = img.shape
            for i in range(1, RX):
                ax.axvline(w * i / RX, color=COL_GRID, lw=0.7, alpha=0.8)
            for j in range(1, RY):
                ax.axhline(h * j / RY, color=COL_GRID, lw=0.7, alpha=0.8)
            ax.text(0.02, 0.04, f'{RX}×{RY} regiones contextuales',
                    transform=ax.transAxes, fontsize=8, color=COL_GRID,
                    bbox=dict(boxstyle='round,pad=0.25', fc='black', alpha=0.55))

        # Fila inferior: histograma + entropía
        axh = axes[1, col]
        axh.hist(img.ravel(), bins=256, range=(0, 255), color=COL_HIST)
        axh.set_xlim(0, 255)
        axh.set_yticks([])
        axh.tick_params(labelsize=7)
        axh.set_xlabel('Nivel de gris', fontsize=8)
        if col == 0:
            axh.set_ylabel('Frecuencia', fontsize=8)
        H = calculate_entropy(img)
        axh.text(0.97, 0.88, f'$H = {H:.3f}$ bits', transform=axh.transAxes,
                 ha='right', va='top', fontsize=9)

    out_dir = Path(PROJECT_ROOT) / 'docs' / 'libro' / 'Figures' / 'capitulo2'
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / 'clahe_comparacion.pdf', bbox_inches='tight')
    fig.savefig(out_dir / 'clahe_comparacion.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Figura generada en {out_dir}")


if __name__ == '__main__':
    main()
