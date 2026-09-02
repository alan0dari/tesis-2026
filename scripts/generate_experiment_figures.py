"""
Genera figuras y tablas a partir de un experimento ejecutado con
scripts/run_experiment.py.

Por cada imagen procesada (en <experimento>/images/<id>/):
  - trio.png/.pdf: original | degradada | mejorada, con métricas de validación.
  - pareto_mcdm.png/.pdf: Frente de Pareto 3D con las selecciones de los
    8 métodos MCDM y la solución de consenso marcadas.

Agregados (en <experimento>/figures/):
  - acuerdo_mcdm.png/.pdf: mapa de calor de la matriz de acuerdo entre métodos.
  - votos_consenso.png/.pdf: distribución de votos de la solución de consenso.
  - recuperacion_vif.png/.pdf: VIF vs original antes y después del realce.
  - parametros_clahe.png/.pdf: distribución de los parámetros seleccionados.
  - recuperacion_por_degradacion.png/.pdf: recuperación según degradación.

Tablas LaTeX (en <experimento>/tables/ y, con --update-book, copiadas a
docs/libro/Tables/ para ser incluidas con \\input en el Capítulo 5):
  - tab_resumen_ejecucion.tex, tab_metricas_seleccion.tex,
    tab_parametros_clahe.tex, tab_validacion.tex, tab_matriz_acuerdo.tex,
    tab_votos_por_metodo.tex, tab_correlacion_spearman.tex

Uso:
    python scripts/generate_experiment_figures.py [--experiment results/experiment_X]
                                                  [--update-book] [--skip-per-image]
Si no se indica --experiment, usa el más reciente en results/.
"""

import sys
import json
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

COL = '#20606e'
COL_ACCENT = '#c8a415'
COL_FRONT = '#9db8bf'
METHODS = ['SMARTER', 'TOPSIS', 'BellmanZadeh', 'PROMETHEEII',
           'GRA', 'VIKOR', 'CODAS', 'MABAC']
# Paleta Okabe-Ito (segura para daltonismo): mismo marcador, distinto color
METHOD_COLORS = {
    'SMARTER': '#0072B2', 'TOPSIS': '#E69F00', 'BellmanZadeh': '#009E73',
    'PROMETHEEII': '#D55E00', 'GRA': '#CC79A7', 'VIKOR': '#56B4E9',
    'CODAS': '#B8A800', 'MABAC': '#333333',
}
METHOD_LABELS = {'SMARTER': 'SMARTER', 'TOPSIS': 'TOPSIS',
                 'BellmanZadeh': 'Bellman-Zadeh', 'PROMETHEEII': 'PROMETHEE II',
                 'GRA': 'GRA', 'VIKOR': 'VIKOR', 'CODAS': 'CODAS', 'MABAC': 'MABAC'}
DEGRADATION_LABELS = {
    'low_contrast': 'Bajo contraste',
    'underexposure': 'Subexposición',
    'overexposure': 'Sobreexposición',
    'poor_local_contrast': 'Bajo contraste local',
    'skewed_histogram': 'Histograma sesgado',
}


def find_latest_experiment(results_dir: Path) -> Path:
    candidates = sorted(results_dir.glob('experiment_*'))
    candidates = [c for c in candidates if (c / 'experiment_data.csv').exists()]
    if not candidates:
        raise FileNotFoundError(
            f"No hay experimentos con experiment_data.csv en {results_dir}")
    return candidates[-1]


def _save(fig, out_base: Path):
    fig.savefig(out_base.with_suffix('.pdf'), bbox_inches='tight')
    fig.savefig(out_base.with_suffix('.png'), dpi=180, bbox_inches='tight')
    plt.close(fig)


# ============================================================================
# FIGURAS POR IMAGEN
# ============================================================================

def figure_trio(img_dir: Path, result: dict):
    """Original | degradada | mejorada con métricas de validación."""
    import cv2
    images = {}
    for name in ['original', 'degraded', 'enhanced']:
        img = cv2.imread(str(img_dir / f'{name}.png'), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return
        images[name] = img

    val = result.get('validation_vs_original', {})
    deg_label = DEGRADATION_LABELS.get(result.get('degradation_type', ''),
                                       result.get('degradation_type', ''))
    titles = [
        '(a) Original',
        f"(b) Degradada: {deg_label.lower()}\n"
        f"SSIM$_o$={val.get('ssim_degraded', float('nan')):.3f}, "
        f"VIF$_o$={val.get('vif_degraded', float('nan')):.3f}",
        f"(c) Mejorada (consenso MCDM)\n"
        f"SSIM$_o$={val.get('ssim_enhanced', float('nan')):.3f}, "
        f"VIF$_o$={val.get('vif_enhanced', float('nan')):.3f}",
    ]

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 2.6),
                             gridspec_kw={'wspace': 0.04})
    for ax, (name, title) in zip(axes, zip(['original', 'degraded', 'enhanced'], titles)):
        ax.imshow(images[name], cmap='gray', vmin=0, vmax=255, aspect='auto')
        ax.set_title(title, fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    _save(fig, img_dir / 'trio')


def _axis_limits(values: np.ndarray, margin: float = 0.05):
    """Límites ajustados al rango de los datos con un margen relativo."""
    lo, hi = float(values.min()), float(values.max())
    pad = (hi - lo) * margin or 0.01
    return lo - pad, hi + pad


# Tamaños (área) de los anillos concéntricos para selecciones coincidentes
_RING_SIZES = [80, 190, 340, 530, 760, 1030, 1340, 1690]


def _draw_selections_2d(ax, X, Y, seen: dict, consensus):
    """
    Dibuja las selecciones MCDM en una proyección 2D.

    Todos los métodos usan el mismo marcador (círculo) y tamaño, con color
    distintivo. Cuando varios métodos eligen la misma alternativa, se dibujan
    anillos concéntricos: el círculo relleno corresponde al primer método y
    cada anillo adicional a otro método coincidente.
    """
    for idx, names in seen.items():
        ax.scatter([X[idx]], [Y[idx]], marker='o', s=_RING_SIZES[0],
                   color=METHOD_COLORS[names[0]], edgecolor='white',
                   linewidth=0.8, zorder=4)
        for k, name in enumerate(names[1:], start=1):
            ax.scatter([X[idx]], [Y[idx]], marker='o', s=_RING_SIZES[k],
                       facecolors='none', edgecolors=METHOD_COLORS[name],
                       linewidth=2.0, zorder=4)
    if consensus is not None:
        ax.scatter([X[consensus]], [Y[consensus]], marker='*', s=300,
                   color=COL_ACCENT, edgecolor='black', linewidth=0.8,
                   zorder=6)


def figure_pareto_mcdm(img_dir: Path, result: dict, with_surface: bool = True):
    """
    Frente de Pareto con las selecciones MCDM y el consenso.

    Figura compuesta: panel 3D (con superficie ilustrativa por triangulación)
    y las tres proyecciones 2D (H-SSIM, H-VIF, SSIM-VIF), que permiten
    apreciar la forma y distribución del frente sin la oclusión del 3D.
    """
    pareto_csv = img_dir / 'pareto.csv'
    if not pareto_csv.exists():
        return
    df = pd.read_csv(pareto_csv)
    H = df['objective_0'].values
    S = df['objective_1'].values
    V = df['objective_2'].values

    mcdm = result.get('mcdm_results', {})
    consensus = result.get('consensus', {}).get('index')
    if consensus is not None and consensus >= len(H):
        consensus = None

    # Alternativas seleccionadas y por qué métodos (en orden fijo)
    seen: dict = {}
    for name in METHODS:
        idx = mcdm.get(name, {}).get('best_index')
        if idx is not None:
            seen.setdefault(idx, []).append(name)

    lim_h, lim_s, lim_v = _axis_limits(H), _axis_limits(S), _axis_limits(V)

    fig = plt.figure(figsize=(11.5, 9.2))
    gs = fig.add_gridspec(2, 2, hspace=0.24, wspace=0.20,
                          left=0.07, right=0.97, top=0.90, bottom=0.07)

    # --- Panel 3D ---
    ax3d = fig.add_subplot(gs[0, 0], projection='3d')
    if with_surface and len(H) >= 4:
        try:
            ax3d.plot_trisurf(H, S, V, color=COL_FRONT, alpha=0.30,
                              linewidth=0.15, edgecolor='#7a969c')
        except Exception:
            pass  # triangulación degenerada (puntos colineales)
    ax3d.scatter(H, S, V, c=COL, s=14, alpha=0.65, depthshade=False)
    for idx, names in seen.items():
        ax3d.scatter([H[idx]], [S[idx]], [V[idx]], marker='o', s=55,
                     color=METHOD_COLORS[names[0]], edgecolor='white',
                     linewidth=0.6, zorder=5)
    if consensus is not None:
        ax3d.scatter([H[consensus]], [S[consensus]], [V[consensus]],
                     marker='*', s=300, color=COL_ACCENT, edgecolor='black',
                     linewidth=0.8, zorder=6)
    ax3d.set_xlim(lim_h); ax3d.set_ylim(lim_s); ax3d.set_zlim(lim_v)
    ax3d.set_box_aspect((1.0, 1.0, 0.85), zoom=1.12)
    ax3d.set_xlabel('Entropía (H)', fontsize=8.5, labelpad=-3)
    ax3d.set_ylabel('SSIM', fontsize=8.5, labelpad=-3)
    ax3d.set_zlabel('VIF', fontsize=8.5, labelpad=-3)
    ax3d.tick_params(labelsize=6.5, pad=-2)
    ax3d.view_init(elev=18, azim=-48)
    ax3d.set_title('Frente de Pareto 3D (superficie ilustrativa)', fontsize=10)

    # --- Proyecciones 2D ---
    proyecciones = [
        (gs[0, 1], H, S, 'Entropía (H)', 'SSIM', lim_h, lim_s),
        (gs[1, 0], H, V, 'Entropía (H)', 'VIF', lim_h, lim_v),
        (gs[1, 1], S, V, 'SSIM', 'VIF', lim_s, lim_v),
    ]
    for spec, X, Y, xl, yl, lx, ly in proyecciones:
        ax = fig.add_subplot(spec)
        ax.scatter(X, Y, s=18, c=COL_FRONT, alpha=0.6, zorder=2)
        _draw_selections_2d(ax, X, Y, seen, consensus)
        ax.set_xlim(lx); ax.set_ylim(ly)
        ax.set_xlabel(xl, fontsize=9)
        ax.set_ylabel(yl, fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(alpha=0.25, zorder=0)

    # --- Leyenda unificada ---
    from matplotlib.lines import Line2D
    n_votes = result.get('consensus', {}).get('votes', 0)
    handles = [Line2D([], [], marker='o', ls='', color=COL_FRONT, ms=7,
                      label='Frente de Pareto')]
    handles += [Line2D([], [], marker='o', ls='', color=METHOD_COLORS[m],
                       ms=9, markeredgecolor='white',
                       label=METHOD_LABELS[m]) for m in METHODS]
    handles.append(Line2D([], [], marker='*', ls='', color=COL_ACCENT, ms=15,
                          markeredgecolor='black',
                          label=f'Consenso ({n_votes} votos)'))
    fig.legend(handles=handles, loc='upper center', ncol=5, fontsize=8.5,
               frameon=False, bbox_to_anchor=(0.5, 0.985),
               title='Anillos concéntricos: métodos que seleccionan la misma alternativa',
               title_fontsize=8)
    _save(fig, img_dir / 'pareto_mcdm')


# ============================================================================
# FIGURAS AGREGADAS
# ============================================================================

def figure_agreement_heatmap(exp_dir: Path, fig_dir: Path):
    csv = exp_dir / 'mcdm_agreement_matrix.csv'
    if not csv.exists():
        return
    m = pd.read_csv(csv, index_col=0)
    labels = [METHOD_LABELS.get(c, c) for c in m.columns]

    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    im = ax.imshow(m.values, cmap='YlGnBu', vmin=0, vmax=100)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=40, ha='right', fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)
    for i in range(len(labels)):
        for j in range(len(labels)):
            v = m.values[i, j]
            ax.text(j, i, f'{v:.0f}', ha='center', va='center', fontsize=8,
                    color='white' if v > 55 else '#333333')
    fig.colorbar(im, ax=ax, shrink=0.8, label='% de imágenes con la misma selección')
    _save(fig, fig_dir / 'acuerdo_mcdm')


def compute_mean_spearman(exp_dir: Path):
    """
    Correlación de Spearman promedio entre los rankings de los métodos MCDM.

    A diferencia de la matriz de acuerdo (coincidencia exacta de la selección,
    que con frentes grandes es naturalmente baja), la correlación de rangos
    compara los rankings completos: dos métodos pueden elegir alternativas
    distintas pero ordenar el frente de forma casi idéntica.
    """
    from scipy.stats import spearmanr
    mats = []
    for rj in sorted((exp_dir / 'images').glob('*/result.json')):
        r = json.loads(rj.read_text(encoding='utf-8'))
        scores = []
        for m in METHODS:
            info = r.get('mcdm_results', {}).get(m, {})
            if 'scores' not in info:
                scores = None
                break
            scores.append(info['scores'])
        if scores is None:
            continue
        rho = spearmanr(np.array(scores), axis=1)[0]
        mats.append(rho)
    if not mats:
        return None
    return np.nanmean(np.array(mats), axis=0)


def figure_spearman_heatmap(spearman, fig_dir: Path):
    if spearman is None:
        return
    labels = [METHOD_LABELS[m] for m in METHODS]
    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    im = ax.imshow(spearman, cmap='RdYlGn', vmin=-1, vmax=1)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=40, ha='right', fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f'{spearman[i, j]:.2f}', ha='center', va='center',
                    fontsize=8, color='#333333')
    fig.colorbar(im, ax=ax, shrink=0.8,
                 label='Correlación de Spearman promedio entre rankings')
    _save(fig, fig_dir / 'correlacion_spearman')


def figure_consensus_votes(df: pd.DataFrame, fig_dir: Path):
    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    counts = df['consensus_votes'].value_counts().sort_index()
    ax.bar(counts.index, counts.values, color=COL)
    ax.set_xlabel('Votos de la solución de consenso (de 8 métodos)', fontsize=10)
    ax.set_ylabel('Cantidad de imágenes', fontsize=10)
    ax.set_xticks(range(int(df['consensus_votes'].min()),
                        int(df['consensus_votes'].max()) + 1))
    ax.spines[['top', 'right']].set_visible(False)
    _save(fig, fig_dir / 'votos_consenso')


def figure_vif_recovery(df: pd.DataFrame, fig_dir: Path):
    if 'val_vif_degraded' not in df.columns:
        return
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    lim_min = min(df['val_vif_degraded'].min(), df['val_vif_enhanced'].min()) - 0.05
    lim_max = max(df['val_vif_degraded'].max(), df['val_vif_enhanced'].max()) + 0.05
    ax.plot([lim_min, lim_max], [lim_min, lim_max], '--', color='#999999', lw=1,
            label='Sin cambio')
    for deg, g in df.groupby('degradation_type'):
        ax.scatter(g['val_vif_degraded'], g['val_vif_enhanced'], s=30, alpha=0.8,
                   label=DEGRADATION_LABELS.get(deg, deg))
    ax.set_xlabel('VIF(original, degradada)', fontsize=10)
    ax.set_ylabel('VIF(original, mejorada)', fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    ax.spines[['top', 'right']].set_visible(False)
    _save(fig, fig_dir / 'recuperacion_vif')


def figure_params_distribution(df: pd.DataFrame, fig_dir: Path):
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.0))
    specs = [('compromise_Rx', '$R_x$', np.arange(2, 66, 4)),
             ('compromise_Ry', '$R_y$', np.arange(2, 66, 4)),
             ('compromise_Clip', '$C$ (clip limit)', np.linspace(1.0, 4.0, 16))]
    for ax, (col, label, bins) in zip(axes, specs):
        ax.hist(df[col], bins=bins, color=COL)
        ax.set_xlabel(label, fontsize=10)
        ax.spines[['top', 'right']].set_visible(False)
    axes[0].set_ylabel('Cantidad de imágenes', fontsize=10)
    _save(fig, fig_dir / 'parametros_clahe')


def figure_recovery_by_degradation(df: pd.DataFrame, fig_dir: Path):
    if 'val_vif_recovery' not in df.columns:
        return
    order = [d for d in DEGRADATION_LABELS if d in set(df['degradation_type'])]
    data = [df.loc[df['degradation_type'] == d, 'val_vif_recovery'] for d in order]
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    bp = ax.boxplot(data, tick_labels=[DEGRADATION_LABELS[d] for d in order],
                    patch_artist=True)
    for box in bp['boxes']:
        box.set_facecolor('#9db8bf')
    ax.axhline(0, color='#999999', lw=1, ls='--')
    ax.set_ylabel('Recuperación de VIF\n(mejorada $-$ degradada, vs. original)',
                  fontsize=9)
    ax.tick_params(axis='x', labelsize=8.5)
    ax.spines[['top', 'right']].set_visible(False)
    _save(fig, fig_dir / 'recuperacion_por_degradacion')


# ============================================================================
# TABLAS LATEX
# ============================================================================

def _fmt(x, nd=4):
    return f'{x:.{nd}f}'


def latex_table_spearman(spearman, tab_dir: Path):
    if spearman is None:
        return
    short = {'SMARTER': 'SMT', 'TOPSIS': 'TOP', 'BellmanZadeh': 'BZ',
             'PROMETHEEII': 'PR2', 'GRA': 'GRA', 'VIKOR': 'VIK',
             'CODAS': 'COD', 'MABAC': 'MAB'}
    labels = [short[m] for m in METHODS]
    lines = ['\\begin{tabular}{l' + 'c' * len(labels) + '}', '\\hline',
             ' & '.join([''] + labels) + ' \\\\', '\\hline']
    for i, m in enumerate(METHODS):
        vals = ' & '.join(f'{spearman[i, j]:.2f}' for j in range(len(labels)))
        lines.append(f'{short[m]} & {vals} \\\\')
    lines += ['\\hline', '\\end{tabular}']
    (tab_dir / 'tab_correlacion_spearman.tex').write_text('\n'.join(lines), encoding='utf-8')


def formatear_duracion(segundos: float) -> str:
    """
    Formatea una duración en la unidad más legible:
    horas si supera los 90 minutos, minutos si supera el minuto,
    segundos en caso contrario.
    """
    if segundos > 90 * 60:
        return f"{segundos / 3600:.1f} h"
    if segundos > 60:
        return f"{segundos / 60:.1f} min"
    return f"{segundos:.1f} s"


def _duracion_con_de(media_s: float, de_s: float) -> str:
    """Media ± DE en la unidad que corresponde a la media."""
    if media_s > 90 * 60:
        return f"{media_s / 3600:.1f} $\\pm$ {de_s / 3600:.1f} h"
    if media_s > 60:
        return f"{media_s / 60:.1f} $\\pm$ {de_s / 60:.1f} min"
    return f"{media_s:.1f} $\\pm$ {de_s:.1f} s"


def latex_tables(exp_dir: Path, df: pd.DataFrame, summary: dict, tab_dir: Path):
    tab_dir.mkdir(parents=True, exist_ok=True)
    cfg = summary.get('config', {})

    # --- Resumen de ejecución ---
    deg_counts = df['degradation_type'].value_counts()
    deg_str = ', '.join(f"{DEGRADATION_LABELS.get(k, k)}: {v}"
                        for k, v in deg_counts.items())
    rows = [
        ('Imágenes procesadas', f"{len(df)} de {cfg.get('sample_size', '?')} muestreadas"),
        ('Partículas / iteraciones SMPSO', f"{cfg.get('particles')} / {cfg.get('iterations')}"),
        ('Semilla', str(cfg.get('seed'))),
        ('Tiempo total (5 procesos en paralelo)',
         formatear_duracion(summary.get('total_time_seconds', 0))),
        ('Tiempo promedio por imagen',
         _duracion_con_de(df['processing_time'].mean(), df['processing_time'].std())),
        ('Tamaño promedio del Frente de Pareto',
         f"{df['pareto_size'].mean():.1f} $\\pm$ {df['pareto_size'].std():.1f}"),
        ('Degradaciones aplicadas', deg_str),
    ]
    lines = ['\\begin{tabular}{ll}', '\\hline',
             '\\textbf{Parámetro} & \\textbf{Valor} \\\\', '\\hline']
    lines += [f'{k} & {v} \\\\' for k, v in rows]
    lines += ['\\hline', '\\end{tabular}']
    (tab_dir / 'tab_resumen_ejecucion.tex').write_text('\n'.join(lines), encoding='utf-8')

    # --- Métricas de la solución de compromiso ---
    from scipy import stats as sstats
    lines = ['\\begin{tabular}{lccc}', '\\hline',
             '\\textbf{Métrica} & \\textbf{Media $\\pm$ DE} & \\textbf{IC 95\\%} & \\textbf{Rango} \\\\',
             '\\hline']
    for col, label in [('compromise_H', 'Entropía (H)'),
                       ('compromise_SSIM', 'SSIM'),
                       ('compromise_VIF', 'VIF')]:
        x = df[col]
        se = sstats.sem(x)
        h = se * sstats.t.ppf(0.975, len(x) - 1)
        lines.append(
            f'{label} & {_fmt(x.mean())} $\\pm$ {_fmt(x.std())} & '
            f'[{_fmt(x.mean() - h)}, {_fmt(x.mean() + h)}] & '
            f'[{_fmt(x.min())}, {_fmt(x.max())}] \\\\')
    lines += ['\\hline', '\\end{tabular}']
    (tab_dir / 'tab_metricas_seleccion.tex').write_text('\n'.join(lines), encoding='utf-8')

    # --- Parámetros CLAHE ---
    lines = ['\\begin{tabular}{lcccc}', '\\hline',
             '\\textbf{Parámetro} & \\textbf{Media} & \\textbf{DE} & \\textbf{Moda} & \\textbf{Rango} \\\\',
             '\\hline']
    for col, label, nd in [('compromise_Rx', '$R_x$', 1),
                           ('compromise_Ry', '$R_y$', 1),
                           ('compromise_Clip', '$C$', 3)]:
        x = df[col]
        mode = x.mode().iloc[0] if not x.mode().empty else float('nan')
        lines.append(
            f'{label} & {_fmt(x.mean(), nd)} & {_fmt(x.std(), nd)} & {_fmt(mode, nd)} & '
            f'[{_fmt(x.min(), nd)}, {_fmt(x.max(), nd)}] \\\\')
    lines += ['\\hline', '\\end{tabular}']
    (tab_dir / 'tab_parametros_clahe.tex').write_text('\n'.join(lines), encoding='utf-8')

    # --- Validación contra la original ---
    if 'val_ssim_degraded' in df.columns:
        lines = ['\\begin{tabular}{lccc}', '\\hline',
                 '\\textbf{Métrica vs. original} & \\textbf{Degradada} & '
                 '\\textbf{Mejorada} & \\textbf{Recuperación} \\\\', '\\hline']
        for pre, label in [('val_ssim', 'SSIM'), ('val_vif', 'VIF')]:
            d, e = df[f'{pre}_degraded'], df[f'{pre}_enhanced']
            r = df[f'{pre}_recovery']
            lines.append(
                f'{label} & {_fmt(d.mean())} $\\pm$ {_fmt(d.std())} & '
                f'{_fmt(e.mean())} $\\pm$ {_fmt(e.std())} & '
                f'{"+" if r.mean() >= 0 else ""}{_fmt(r.mean())} \\\\')
        lines += ['\\hline', '\\end{tabular}']
        (tab_dir / 'tab_validacion.tex').write_text('\n'.join(lines), encoding='utf-8')

    # --- Matriz de acuerdo ---
    csv = exp_dir / 'mcdm_agreement_matrix.csv'
    if csv.exists():
        m = pd.read_csv(csv, index_col=0)
        short = {'SMARTER': 'SMT', 'TOPSIS': 'TOP', 'BellmanZadeh': 'BZ',
                 'PROMETHEEII': 'PR2', 'GRA': 'GRA', 'VIKOR': 'VIK',
                 'CODAS': 'COD', 'MABAC': 'MAB'}
        cols = list(m.columns)
        header = ' & '.join([''] + [short.get(c, c) for c in cols])
        lines = ['\\begin{tabular}{l' + 'c' * len(cols) + '}', '\\hline',
                 header + ' \\\\', '\\hline']
        for i, row_name in enumerate(m.index):
            vals = ' & '.join(f'{m.values[i, j]:.0f}' for j in range(len(cols)))
            lines.append(f'{short.get(row_name, row_name)} & {vals} \\\\')
        lines += ['\\hline', '\\end{tabular}']
        (tab_dir / 'tab_matriz_acuerdo.tex').write_text('\n'.join(lines), encoding='utf-8')

    # --- Frecuencia con que cada método coincide con el consenso ---
    sel_cols = [f'{m}_selection' for m in METHODS if f'{m}_selection' in df.columns]
    if sel_cols and 'consensus_index' not in df.columns:
        # consensus_index no está en el CSV: derivarlo del voto mayoritario por fila
        consensus = df[sel_cols].mode(axis=1)[0]
    elif 'consensus_index' in df.columns:
        consensus = df['consensus_index']
    else:
        consensus = None
    if consensus is not None and sel_cols:
        lines = ['\\begin{tabular}{lc}', '\\hline',
                 '\\textbf{Método} & \\textbf{Coincidencia con el consenso (\\%)} \\\\',
                 '\\hline']
        for m_name in METHODS:
            col = f'{m_name}_selection'
            if col in df.columns:
                pct = (df[col] == consensus).mean() * 100
                lines.append(f'{METHOD_LABELS[m_name]} & {pct:.1f} \\\\')
        lines += ['\\hline', '\\end{tabular}']
        (tab_dir / 'tab_votos_por_metodo.tex').write_text('\n'.join(lines), encoding='utf-8')


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Figuras y tablas de un experimento')
    parser.add_argument('--experiment', type=str, default=None,
                        help='Directorio del experimento (default: el más reciente)')
    parser.add_argument('--update-book', action='store_true',
                        help='Copiar tablas a docs/libro/Tables/')
    parser.add_argument('--skip-per-image', action='store_true',
                        help='No generar figuras por imagen (más rápido)')
    args = parser.parse_args()

    exp_dir = Path(args.experiment) if args.experiment \
        else find_latest_experiment(PROJECT_ROOT / 'results')
    print(f'Experimento: {exp_dir}')

    df = pd.read_csv(exp_dir / 'experiment_data.csv')
    summary = json.loads((exp_dir / 'experiment_summary.json').read_text(encoding='utf-8'))

    # Figuras por imagen
    if not args.skip_per_image:
        img_dirs = sorted((exp_dir / 'images').iterdir()) if (exp_dir / 'images').exists() else []
        for img_dir in img_dirs:
            result_json = img_dir / 'result.json'
            if not result_json.exists():
                continue
            result = json.loads(result_json.read_text(encoding='utf-8'))
            figure_trio(img_dir, result)
            figure_pareto_mcdm(img_dir, result)
        print(f'Figuras por imagen: {len(img_dirs)} directorios procesados')

    # Figuras agregadas
    fig_dir = exp_dir / 'figures'
    fig_dir.mkdir(exist_ok=True)
    figure_agreement_heatmap(exp_dir, fig_dir)
    figure_consensus_votes(df, fig_dir)
    figure_vif_recovery(df, fig_dir)
    figure_params_distribution(df, fig_dir)
    figure_recovery_by_degradation(df, fig_dir)
    spearman = compute_mean_spearman(exp_dir)
    figure_spearman_heatmap(spearman, fig_dir)
    print(f'Figuras agregadas en {fig_dir}')

    # Tablas LaTeX
    tab_dir = exp_dir / 'tables'
    latex_tables(exp_dir, df, summary, tab_dir)
    latex_table_spearman(spearman, tab_dir)
    print(f'Tablas LaTeX en {tab_dir}')

    if args.update_book:
        import shutil
        book_tables = PROJECT_ROOT / 'docs' / 'libro' / 'Tables'
        book_tables.mkdir(parents=True, exist_ok=True)
        for tex in tab_dir.glob('*.tex'):
            shutil.copy2(tex, book_tables / tex.name)
        print(f'Tablas copiadas a {book_tables}')

        # Figuras agregadas al libro
        book_figs = PROJECT_ROOT / 'docs' / 'libro' / 'Figures' / 'capitulo5'
        book_figs.mkdir(parents=True, exist_ok=True)
        for f in fig_dir.glob('*.*'):
            shutil.copy2(f, book_figs / f.name)

        # Imagen representativa por tipo de degradación: la de recuperación
        # de VIF mediana, con nombre determinístico para el \includegraphics
        if 'val_vif_recovery' in df.columns:
            for deg, g in df.groupby('degradation_type'):
                rep = g.iloc[(g['val_vif_recovery'] - g['val_vif_recovery'].median())
                             .abs().argsort().iloc[0]]
                img_id = str(rep['image_id'])
                src_dir = exp_dir / 'images' / img_id
                for base, dst in [('trio', f'trio_{deg}'),
                                  ('pareto_mcdm', f'pareto_{deg}')]:
                    for ext in ['.png', '.pdf']:
                        src = src_dir / f'{base}{ext}'
                        if src.exists():
                            shutil.copy2(src, book_figs / f'{dst}{ext}')
                print(f'  Representativa de {deg}: imagen {img_id}')
        print(f'Figuras copiadas a {book_figs}')


if __name__ == '__main__':
    main()
