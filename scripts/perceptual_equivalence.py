"""
Mide cuán distinguibles son entre sí las soluciones que eligen los distintos MCDM.

Motivación
----------
El acuerdo exacto entre pares de métodos MCDM es del 17.3%: numéricamente eligen
alternativas distintas del frente de Pareto casi siempre. Pero "distinta fila de
la matriz de decisión" no implica "imagen distinta a los ojos de un odontólogo".
Antes de montar un estudio perceptual hay que saber cuántos de esos desacuerdos
sobreviven como diferencia visible: si no sobreviven, comparar esos pares sólo
agrega ruido y fatiga al evaluador.

Qué hace
--------
Para cada imagen del experimento regenera la imagen realzada de cada candidato
MCDM distinto (aplicando CLAHE con sus parámetros sobre la degradada) y calcula
SSIM y MAE entre cada par. Luego agrupa por equivalencia perceptual con
union-find a varios umbrales y reporta cuántas comparaciones sobreviven.

Uso:
    python scripts/perceptual_equivalence.py
    python scripts/perceptual_equivalence.py --experiment results/experiment_XXX --limit 20
"""

import sys
import argparse
from pathlib import Path
from itertools import combinations

PROJECT_ROOT = str(Path(__file__).parent.parent)
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
import cv2

from src.clahe.processor import apply_clahe_simple
from src.metrics.ssim import calculate_ssim

METHODS = ['SMARTER', 'TOPSIS', 'BellmanZadeh', 'PROMETHEEII',
           'GRA', 'VIKOR', 'CODAS', 'MABAC']

# Umbrales de equivalencia perceptual a explorar. El valor operativo (0.98) es
# una hipótesis a calibrar con los propios odontólogos: el estudio incluye pares
# de distintas bandas de SSIM para estimar dónde cae el umbral de discriminación.
THRESHOLDS = (0.95, 0.97, 0.98, 0.99)


def candidate_images(exp_dir, row):
    """Regenera la imagen realzada de cada candidato MCDM distinto de una imagen."""
    image_dir = exp_dir / 'images' / str(row['image_id'])
    degraded = cv2.imread(str(image_dir / 'degraded.png'), cv2.IMREAD_GRAYSCALE)
    if degraded is None:
        return None, None
    pareto = pd.read_csv(image_dir / 'pareto.csv')

    voters = {}
    for method in METHODS:
        idx = row.get(f'{method}_selection')
        if pd.notna(idx):
            voters.setdefault(int(idx), []).append(method)

    images = {}
    for idx in voters:
        p = pareto[pareto.solution_id == idx].iloc[0]
        images[idx] = apply_clahe_simple(degraded, int(p.param_0),
                                         int(p.param_1), float(p.param_2))
    return images, voters


def analyze_mechanism(exp_dir, df, n_images, rng):
    """
    Distingue las dos explicaciones posibles del colapso perceptual.

    ¿Los candidatos MCDM se parecen porque el frente entero es perceptualmente
    plano, o porque los métodos convergen a una misma región de un frente que sí
    tiene rango? Se decide comparando la similitud entre las selecciones MCDM
    contra (i) pares al azar del mismo frente y (ii) los extremos mono-objetivo.
    """
    from scipy import stats

    rows = []
    for _, row in df.head(n_images).iterrows():
        image_dir = exp_dir / 'images' / str(row['image_id'])
        degraded = cv2.imread(str(image_dir / 'degraded.png'), cv2.IMREAD_GRAYSCALE)
        if degraded is None:
            continue
        pareto = pd.read_csv(image_dir / 'pareto.csv')

        def render(idx):
            p = pareto[pareto.solution_id == idx].iloc[0]
            return apply_clahe_simple(degraded, int(p.param_0),
                                      int(p.param_1), float(p.param_2))

        def mean_ssim(indices):
            imgs = {i: render(i) for i in indices}
            vals = [calculate_ssim(imgs[x], imgs[y])
                    for x, y in combinations(sorted(indices), 2)]
            return (np.mean(vals) if vals else np.nan,
                    np.mean([v >= 0.98 for v in vals]) if vals else np.nan)

        selected = sorted({int(row[f'{m}_selection']) for m in METHODS
                           if pd.notna(row.get(f'{m}_selection'))})
        ids = pareto.solution_id.tolist()
        random_pick = rng.choice(ids, size=min(len(selected), len(ids)),
                                 replace=False).tolist()
        # Extremos: la mejor solución en cada objetivo por separado
        extremes = sorted({int(pareto.loc[pareto[f'objective_{k}'].idxmax(),
                                          'solution_id']) for k in range(3)})

        m_ssim, m_ind = mean_ssim(selected)
        r_ssim, r_ind = mean_ssim(random_pick)
        e_ssim, e_ind = mean_ssim(extremes)
        rows.append({'mcdm': m_ssim, 'mcdm_ind': m_ind,
                     'azar': r_ssim, 'azar_ind': r_ind,
                     'extremos': e_ssim, 'extremos_ind': e_ind})

    t = pd.DataFrame(rows)
    print(f'\n{"=" * 70}\nMECANISMO DEL COLAPSO ({len(t)} imágenes)\n{"=" * 70}')
    print(f'\n{"Qué se compara":<46}{"SSIM":>8}{"% indist.":>12}')
    for key, label in [('extremos', 'Extremos del frente (mejor H / SSIM / VIF)'),
                       ('azar', 'Pares al azar del mismo frente'),
                       ('mcdm', 'Soluciones elegidas por los MCDM')]:
        print(f'  {label:<44}{t[key].mean():8.4f}{t[f"{key}_ind"].mean() * 100:11.0f}%')

    w = stats.wilcoxon(t.mcdm.dropna(), t.azar.dropna())
    print(f'\nMCDM vs azar: Wilcoxon W={w.statistic:.0f}, p={w.pvalue:.2e}'
          f'  ({(t.mcdm > t.azar).sum()}/{len(t)} imágenes)')
    print('\nLectura: si los extremos SÍ se distinguen, el frente no es plano y el')
    print('colapso viene de su densidad más la convergencia de los métodos.')


def merge_equivalent(pairs, threshold):
    """Agrupa candidatos en clases de equivalencia perceptual (union-find)."""
    nodes = sorted(set(pairs.a) | set(pairs.b))
    parent = {n: n for n in nodes}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for _, e in pairs[pairs.ssim_ab >= threshold].iterrows():
        ra, rb = find(e.a), find(e.b)
        if ra != rb:
            parent[ra] = rb
    return len({find(n) for n in nodes})


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--experiment', default='results/experiment_20260709_215441')
    parser.add_argument('--limit', type=int, help='Analizar sólo las primeras N imágenes')
    parser.add_argument('--target-images', type=int, default=50,
                        help='Tamaño del estudio a proyectar (default: 50)')
    parser.add_argument('--mechanism', action='store_true',
                        help='Analizar por qué colapsan: frente plano vs. '
                             'convergencia de los métodos (ver docs/evaluacion/'
                             'colapso_perceptual_mcdm.md)')
    parser.add_argument('--mechanism-images', type=int, default=25,
                        help='Imágenes para el análisis de mecanismo (default: 25)')
    args = parser.parse_args()

    exp_dir = Path(args.experiment)
    df = pd.read_csv(exp_dir / 'experiment_data.csv')
    if args.limit:
        df = df.head(args.limit)

    rows = []
    for _, row in df.iterrows():
        images, voters = candidate_images(exp_dir, row)
        if not images:
            continue
        for a, b in combinations(sorted(images), 2):
            rows.append({
                'image_id': str(row['image_id']), 'a': a, 'b': b,
                'methods_a': '+'.join(voters[a]), 'methods_b': '+'.join(voters[b]),
                'ssim_ab': calculate_ssim(images[a], images[b]),
                'mae': float(np.mean(np.abs(images[a].astype(float) - images[b].astype(float)))),
            })

    pairs = pd.DataFrame(rows)
    out = exp_dir / 'perceptual_equivalence.csv'
    pairs.to_csv(out, index=False)

    n_img = pairs.image_id.nunique()
    print(f'Imágenes: {n_img}   Pares de candidatos: {len(pairs)}')
    print(f'\nSSIM entre candidatos MCDM del mismo frente de Pareto:')
    print(f'  media {pairs.ssim_ab.mean():.4f}   mediana {pairs.ssim_ab.median():.4f}'
          f'   rango [{pairs.ssim_ab.min():.4f}, {pairs.ssim_ab.max():.4f}]')
    print(f'  MAE (niveles de gris): media {pairs.mae.mean():.2f}   '
          f'mediana {pairs.mae.median():.2f}')

    print('\nDistribución de SSIM por banda:')
    bands = [(0.0, 0.90), (0.90, 0.95), (0.95, 0.98), (0.98, 0.99), (0.99, 1.01)]
    for lo, hi in bands:
        sel = (pairs.ssim_ab >= lo) & (pairs.ssim_ab < hi)
        if sel.sum():
            print(f'  [{lo:.2f}, {hi:.2f}): {sel.sum():4d} pares ({sel.mean() * 100:5.1f}%)'
                  f'   MAE medio {pairs.loc[sel, "mae"].mean():5.2f}')

    print(f'\nCandidatos perceptualmente distintos por imagen, según el corte:')
    print(f"{'corte':>13} {'k medio':>9} {'imgs k=1':>10} {'pares/img':>11}"
          f" {'comparaciones en ' + str(args.target_images):>22}")
    k_raw = pairs.groupby('image_id').apply(
        lambda g: len(set(g.a) | set(g.b)), include_groups=False)
    for thr in THRESHOLDS:
        ks = pd.Series([merge_equivalent(g, thr) for _, g in pairs.groupby('image_id')])
        per_img = (ks * (ks - 1) / 2).mean()
        print(f'{thr:13.2f} {ks.mean():9.2f} {f"{(ks == 1).sum()}/{len(ks)}":>10}'
              f' {per_img:11.2f} {per_img * args.target_images:22.0f}')
    per_img_raw = (k_raw * (k_raw - 1) / 2).mean()
    print(f'{"sin fusionar":>13} {k_raw.mean():9.2f} {"0/" + str(len(k_raw)):>10}'
          f' {per_img_raw:11.2f} {per_img_raw * args.target_images:22.0f}')

    if args.mechanism:
        analyze_mechanism(exp_dir, df, args.mechanism_images,
                          np.random.default_rng(0))

    print(f'\n-> {out}')


if __name__ == '__main__':
    main()
