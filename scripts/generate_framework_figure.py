"""
Genera la figura ilustrativa del framework (Figura del Capítulo 4 del libro).

La figura se construye con elementos REALES del pipeline:
  1. Una ortopantomografía del dataset con degradación de contraste (entrada).
  2. Un Frente de Pareto 3D (H, SSIM, VIF) obtenido con SMPSO sobre esa imagen.
  3. La matriz de decisión derivada del frente.
  4. La selección por consenso de los 8 métodos MCDM.
  5. La imagen mejorada seleccionada (salida).

Salidas (vectorial para LaTeX + preview):
  docs/libro/Figures/capitulo4/framework.pdf
  docs/libro/Figures/capitulo4/framework.png

Uso:
    python scripts/generate_framework_figure.py [--image data/original/114.jpg]
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
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from src.optimization.smpso import SMPSOImageOptimizer
from src.utils.degradation import apply_low_contrast
from src.mcdm import (SMARTER, TOPSIS, BellmanZadeh, PROMETHEEII,
                      GRA, VIKOR, CODAS, MABAC)

# Paleta sobria (coherente con la figura original del artículo)
COL_ARROW = '#20606e'   # verde azulado de las flechas
COL_BOX = '#f5f5f0'     # fondo de los bloques de proceso
COL_EDGE = '#20606e'
COL_ACCENT = '#c8a415'  # dorado para la solución seleccionada
COL_TEXT = '#222222'


def run_pipeline(image_path: str, seed: int = 42):
    """Ejecuta el pipeline real sobre una imagen y retorna los artefactos."""
    original = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if original is None:
        raise FileNotFoundError(f"No se pudo cargar {image_path}")

    degraded = apply_low_contrast(original, factor=0.45)

    print("Ejecutando SMPSO (30 particulas x 30 iteraciones)...")
    optimizer = SMPSOImageOptimizer(
        image=degraded, n_particles=30, max_iterations=30,
        verbose=False, seed=seed
    )
    front = optimizer.run()

    decision_matrix = front.get_decision_matrix()
    params_matrix = front.get_parameters_matrix()

    # Selección por consenso de los 8 métodos MCDM, con el esquema de pesos
    # del experimento: ROC para el orden de importancia VIF > H > SSIM
    weights = np.array([0.2778, 0.1111, 0.6111])
    criteria_rank = [2, 0, 1]
    criteria = ['benefit'] * 3
    methods = [SMARTER, TOPSIS, BellmanZadeh, PROMETHEEII, GRA, VIKOR, CODAS, MABAC]
    votes = {}
    selections = {}
    for MethodClass in methods:
        if MethodClass is SMARTER:
            method = MethodClass(criteria_types=criteria.copy(),
                                 use_rank_order_weights=True,
                                 criteria_rank=criteria_rank)
        else:
            method = MethodClass(weights=weights.copy(), criteria_types=criteria.copy())
        best_idx, _ = method.select(decision_matrix.copy())
        selections[MethodClass.__name__] = int(best_idx)
        votes[best_idx] = votes.get(best_idx, 0) + 1
    consensus_idx = max(votes, key=votes.get)

    enhanced = optimizer.get_enhanced_image(front[consensus_idx])
    print(f"Frente: {len(front)} soluciones | consenso: alternativa "
          f"{consensus_idx} con {votes[consensus_idx]} votos")
    return degraded, enhanced, decision_matrix, params_matrix, consensus_idx, selections


def _panel_label(fig, x, y, num, text):
    """Etiqueta numerada bajo cada panel, en coordenadas de figura."""
    fig.text(x, y, f"{num}", ha='center', va='center', fontsize=11, color='white',
             fontweight='bold', zorder=6,
             bbox=dict(boxstyle='circle,pad=0.35', fc=COL_ARROW, ec='none'))
    fig.text(x + 0.017, y, text, ha='left', va='center', fontsize=9.5,
             color=COL_TEXT, zorder=6)


def _arrow(fig, xy_from, xy_to):
    arrow = FancyArrowPatch(
        xy_from, xy_to, transform=fig.transFigure,
        arrowstyle='simple,head_width=12,head_length=10,tail_width=5',
        fc=COL_ARROW, ec='none', zorder=5
    )
    fig.add_artist(arrow)


def build_figure(degraded, enhanced, decision_matrix, params_matrix,
                 consensus_idx, selections, out_dir: Path):
    fig = plt.figure(figsize=(11.5, 7.0))

    # ---------- Fila superior ----------
    # (1) Entrada: imagen degradada
    ax_in = fig.add_axes([0.030, 0.60, 0.225, 0.30])
    ax_in.imshow(degraded, cmap='gray', vmin=0, vmax=255)
    ax_in.set_xticks([]); ax_in.set_yticks([])
    for s in ax_in.spines.values():
        s.set_color(COL_EDGE)

    # (2) Optimización: bloque SMPSO-CLAHE
    box_opt = FancyBboxPatch((0.335, 0.62), 0.185, 0.26,
                             boxstyle='round,pad=0.012',
                             transform=fig.transFigure,
                             fc=COL_BOX, ec=COL_EDGE, lw=1.6, zorder=3)
    fig.add_artist(box_opt)
    fig.text(0.4275, 0.815, 'SMPSO + CLAHE', ha='center', fontsize=11,
             fontweight='bold', color=COL_TEXT, zorder=4)
    fig.text(0.4275, 0.745,
             'Optimización multiobjetivo\nde parámetros $(R_x, R_y, C)$',
             ha='center', va='center', fontsize=9, color=COL_TEXT, zorder=4)
    fig.text(0.4275, 0.665,
             r'max $\left[\, H,\ \mathrm{SSIM},\ \mathrm{VIF} \,\right]$',
             ha='center', va='center', fontsize=9.5, color=COL_ARROW, zorder=4)

    # (3) Frente de Pareto 3D real
    ax_p = fig.add_axes([0.600, 0.595, 0.385, 0.385], projection='3d')
    H, S, V = decision_matrix[:, 0], decision_matrix[:, 1], decision_matrix[:, 2]
    ax_p.scatter(H, S, V, c=COL_ARROW, s=20, depthshade=True, alpha=0.75)
    ax_p.scatter([H[consensus_idx]], [S[consensus_idx]], [V[consensus_idx]],
                 c=COL_ACCENT, s=180, marker='*', edgecolor='black',
                 linewidth=0.8, zorder=5, label='Solución seleccionada')
    ax_p.set_xlabel('Entropía (H)', fontsize=8, labelpad=-6)
    ax_p.set_ylabel('SSIM', fontsize=8, labelpad=-6)
    ax_p.set_zlabel('VIF', fontsize=8, labelpad=-6)
    ax_p.tick_params(labelsize=5.5, pad=-3)
    ax_p.view_init(elev=20, azim=-50)
    ax_p.legend(loc='upper center', fontsize=7.5, frameon=False,
                bbox_to_anchor=(0.5, 1.04))

    # ---------- Fila inferior ----------
    # (4) Matriz de decisión (valores reales, primeras filas)
    ax_m = fig.add_axes([0.660, 0.13, 0.30, 0.28])
    ax_m.axis('off')
    n_rows = min(5, len(decision_matrix))
    col_labels = ['$A_i$', 'H', 'SSIM', 'VIF']
    cell_text = [[f"$A_{{{i}}}$", f"{decision_matrix[i,0]:.3f}",
                  f"{decision_matrix[i,1]:.3f}", f"{decision_matrix[i,2]:.3f}"]
                 for i in range(n_rows)]
    cell_text.append(['⋮', '⋮', '⋮', '⋮'])
    table = ax_m.table(cellText=cell_text, colLabels=col_labels,
                       cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.25)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor('#999999')
        if row == 0:
            cell.set_facecolor(COL_ARROW)
            cell.set_text_props(color='white', fontweight='bold')
        elif row - 1 == consensus_idx and row <= n_rows:
            cell.set_facecolor('#f4e9c3')

    # (5) Decisión: bloque MCDM
    box_dec = FancyBboxPatch((0.335, 0.115), 0.235, 0.30,
                             boxstyle='round,pad=0.012',
                             transform=fig.transFigure,
                             fc=COL_BOX, ec=COL_EDGE, lw=1.6, zorder=3)
    fig.add_artist(box_dec)
    fig.text(0.4525, 0.385, '8 métodos MCDM', ha='center', fontsize=10.5,
             fontweight='bold', color=COL_TEXT, zorder=4)
    method_names = ['SMARTER', 'TOPSIS', 'B.-Zadeh', 'PROM. II',
                    'GRA', 'VIKOR', 'CODAS', 'MABAC']
    for k, name in enumerate(method_names):
        cx = 0.395 + (k % 2) * 0.115
        cy = 0.345 - (k // 2) * 0.042
        sel = selections[['SMARTER', 'TOPSIS', 'BellmanZadeh', 'PROMETHEEII',
                          'GRA', 'VIKOR', 'CODAS', 'MABAC'][k]]
        marker = '#c8a415' if sel == consensus_idx else '#aaaaaa'
        fig.text(cx, cy, name, ha='center', va='center', fontsize=8,
                 color=COL_TEXT, zorder=4,
                 bbox=dict(boxstyle='round,pad=0.30', fc='white', ec=marker, lw=1.4))
    fig.text(0.4525, 0.155, 'Selección a posteriori por consenso',
             ha='center', fontsize=8.5, style='italic', color=COL_ARROW, zorder=4)

    # (6) Salida: imagen mejorada
    ax_out = fig.add_axes([0.030, 0.125, 0.225, 0.30])
    ax_out.imshow(enhanced, cmap='gray', vmin=0, vmax=255)
    ax_out.set_xticks([]); ax_out.set_yticks([])
    for s in ax_out.spines.values():
        s.set_color(COL_ACCENT)
        s.set_linewidth(2.2)

    # ---------- Flechas del flujo ----------
    _arrow(fig, (0.262, 0.755), (0.325, 0.755))   # entrada -> optimización
    _arrow(fig, (0.532, 0.755), (0.592, 0.755))   # optimización -> pareto
    _arrow(fig, (0.810, 0.505), (0.810, 0.435))   # pareto -> matriz (baja)
    _arrow(fig, (0.648, 0.265), (0.585, 0.265))   # matriz -> mcdm
    _arrow(fig, (0.323, 0.265), (0.262, 0.265))   # mcdm -> salida

    # ---------- Etiquetas numeradas ----------
    _panel_label(fig, 0.045, 0.555, 1, 'Entrada: radiografía con\ncontraste degradado')
    _panel_label(fig, 0.350, 0.555, 2, 'Optimización evolutiva\nmultiobjetivo')
    _panel_label(fig, 0.680, 0.545, 3, 'Frente de Pareto 3D:\nsoluciones candidatas')
    _panel_label(fig, 0.675, 0.075, 4, 'Matriz de decisión\n(alternativas × criterios)')
    _panel_label(fig, 0.350, 0.075, 5, 'Decisión multicriterio\na posteriori')
    _panel_label(fig, 0.045, 0.075, 6, 'Salida: imagen mejorada\nseleccionada')

    # ---------- Exportar ----------
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / 'framework.pdf'
    png_path = out_dir / 'framework.png'
    fig.savefig(pdf_path, bbox_inches='tight')
    fig.savefig(png_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Figura generada:\n  {pdf_path}\n  {png_path}")


def main():
    parser = argparse.ArgumentParser(description='Genera la figura del framework')
    parser.add_argument('--image', default='data/original/114.jpg',
                        help='Imagen del dataset a usar (default: 114.jpg)')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    artifacts = run_pipeline(args.image, seed=args.seed)
    out_dir = Path(PROJECT_ROOT) / 'docs' / 'libro' / 'Figures' / 'capitulo4'
    build_figure(*artifacts, out_dir=out_dir)


if __name__ == '__main__':
    main()
