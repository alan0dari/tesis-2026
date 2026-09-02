"""
Indicadores de calidad del Frente de Pareto (hipervolumen y spacing) sobre los
frentes ya almacenados de un experimento, sin volver a optimizar.

Lee <experimento>/images/<id>/pareto.csv (frente final de cada imagen) y calcula
por imagen:

  - HV bruto: hipervolumen exacto en el espacio de objetivos original
    (H, SSIM, VIF), con punto de referencia igual al nadir del propio frente
    menos un margen del 5% del rango (misma convención que
    scripts/convergence_study.py). Comparable dentro de una imagen, no entre
    imágenes, porque el punto de referencia depende de cada frente.

  - HV normalizado: los objetivos se llevan a [0,1] con los extremos del propio
    frente y se reporta la fracción de la caja de referencia dominada. Al ser
    adimensional sí es comparable entre imágenes: mide qué tan "lleno" es el
    frente (cuánto de la caja definida por sus propios extremos domina), no cuán
    bueno es en términos absolutos.

  - Spacing bruto y normalizado: uniformidad de la distribución de soluciones.
    El normalizado es el que tiene sentido agregar entre imágenes, porque el
    bruto depende de la escala de cada objetivo.

Salidas:
  - <experimento>/pareto_indicators.csv  (una fila por imagen)
  - <experimento>/tables/tab_indicadores_frente.tex  (agregado por degradación)
  - con --update-book, la tabla se copia a docs/libro/Tables/

Uso:
    python scripts/pareto_indicators.py [--experiment results/experiment_X]
                                        [--update-book]
Si no se indica --experiment, usa el más reciente en results/.
"""

import sys
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.optimization.pareto import calculate_hypervolume

OBJ_COLS = ['objective_0', 'objective_1', 'objective_2']
OBJ_LABELS = ['H', 'SSIM', 'VIF']
MARGIN = 0.05  # margen del punto de referencia, como fracción del rango

DEGRADATION_LABELS = {
    'low_contrast': 'Bajo contraste',
    'underexposure': 'Subexposición',
    'overexposure': 'Sobreexposición',
    'poor_local_contrast': 'Bajo contraste local',
    'skewed_histogram': 'Histograma sesgado',
    'noise': 'Ruido',
    'blur': 'Desenfoque',
}


def find_latest_experiment(results_dir: Path) -> Path:
    """Devuelve el experimento más reciente con experiment_data.csv."""
    candidates = [d for d in results_dir.iterdir()
                  if d.is_dir() and (d / 'experiment_data.csv').exists()]
    if not candidates:
        raise SystemExit(f'No hay experimentos en {results_dir}')
    return max(candidates, key=lambda d: d.name)


def spacing(M: np.ndarray) -> float:
    """Spacing (desvío de las distancias al vecino más cercano)."""
    if len(M) < 2:
        return 0.0
    d = np.linalg.norm(M[:, None, :] - M[None, :, :], axis=2)
    np.fill_diagonal(d, np.inf)
    mins = d.min(axis=1)
    return float(np.sqrt(np.mean((mins - mins.mean()) ** 2)))


def hypervolume(M: np.ndarray, ref: np.ndarray) -> float:
    front = [{'objectives': row} for row in M]
    return calculate_hypervolume(front, ref)


def indicators_for_front(M: np.ndarray) -> dict:
    """Indicadores brutos y normalizados de un frente (matriz m x 3)."""
    lo, hi = M.min(0), M.max(0)
    rng = np.where(hi > lo, hi - lo, 1.0)

    ref_raw = lo - MARGIN * rng
    hv_raw = hypervolume(M, ref_raw)

    Z = (M - lo) / rng                      # extremos del frente -> [0,1]
    ref_norm = np.full(3, -MARGIN)
    hv_norm = hypervolume(Z, ref_norm)
    box = (1.0 + MARGIN) ** 3               # volumen máximo alcanzable

    out = {
        'front_size': len(M),
        'hv_raw': hv_raw,
        'hv_norm': hv_norm / box,
        'spacing_raw': spacing(M),
        'spacing_norm': spacing(Z),
    }
    for k, label in enumerate(OBJ_LABELS):
        out[f'range_{label}'] = float(hi[k] - lo[k])
    return out


def _fmt(x, nd=4):
    return f'{x:.{nd}f}'


def latex_table(df: pd.DataFrame, tab_dir: Path) -> Path:
    """Tabla agregada por tipo de degradación, más la fila global."""
    tab_dir.mkdir(parents=True, exist_ok=True)
    lines = ['\\begin{tabular}{lcccc}', '\\hline',
             '\\textbf{Degradación} & \\textbf{$n$} & \\textbf{HV normalizado} & '
             '\\textbf{Spacing normalizado} & \\textbf{Tamaño del frente} \\\\',
             '\\hline']

    def row(label, sub):
        return (f'{label} & {len(sub)} & '
                f'{_fmt(sub.hv_norm.mean(), 3)} $\\pm$ {_fmt(sub.hv_norm.std(), 3)} & '
                f'{_fmt(sub.spacing_norm.mean(), 4)} $\\pm$ {_fmt(sub.spacing_norm.std(), 4)} & '
                f'{_fmt(sub.front_size.mean(), 1)} $\\pm$ {_fmt(sub.front_size.std(), 1)} \\\\')

    if 'degradation_type' in df.columns:
        for deg, sub in df.groupby('degradation_type'):
            lines.append(row(DEGRADATION_LABELS.get(deg, deg), sub))
        lines.append('\\hline')
    lines.append(row('\\textbf{Global}', df))
    lines += ['\\hline', '\\end{tabular}']

    out = tab_dir / 'tab_indicadores_frente.tex'
    out.write_text('\n'.join(lines), encoding='utf-8')
    return out


def describe(x: pd.Series, nd=4) -> str:
    q1, q3 = x.quantile(0.25), x.quantile(0.75)
    return (f'media {x.mean():.{nd}f} +/- {x.std():.{nd}f} | '
            f'mediana {x.median():.{nd}f} [IQR {q1:.{nd}f}-{q3:.{nd}f}] | '
            f'rango [{x.min():.{nd}f}, {x.max():.{nd}f}]')


def main():
    parser = argparse.ArgumentParser(
        description='Hipervolumen y spacing por imagen sobre frentes ya calculados')
    parser.add_argument('--experiment', type=str, default=None,
                        help='Directorio del experimento (default: el más reciente)')
    parser.add_argument('--update-book', action='store_true',
                        help='Copiar la tabla a docs/libro/Tables/')
    args = parser.parse_args()

    exp_dir = Path(args.experiment) if args.experiment \
        else find_latest_experiment(PROJECT_ROOT / 'results')
    print(f'Experimento: {exp_dir}')

    deg_by_image = {}
    data_csv = exp_dir / 'experiment_data.csv'
    if data_csv.exists():
        df_exp = pd.read_csv(data_csv)
        deg_by_image = dict(zip(df_exp.image_id.astype(str),
                                df_exp.degradation_type))

    rows = []
    img_dirs = sorted((exp_dir / 'images').iterdir())
    for img_dir in img_dirs:
        csv = img_dir / 'pareto.csv'
        if not csv.exists():
            print(f'  sin pareto.csv: {img_dir.name}')
            continue
        M = pd.read_csv(csv)[OBJ_COLS].to_numpy(dtype=float)
        row = {'image_id': img_dir.name,
               'degradation_type': deg_by_image.get(img_dir.name, '')}
        row.update(indicators_for_front(M))
        rows.append(row)

    df = pd.DataFrame(rows)
    out_csv = exp_dir / 'pareto_indicators.csv'
    df.to_csv(out_csv, index=False)

    print(f'\n{len(df)} imagenes procesadas -> {out_csv}\n')
    print(f'  HV normalizado     : {describe(df.hv_norm, 3)}')
    print(f'  Spacing normalizado: {describe(df.spacing_norm, 4)}')
    print(f'  Tamano del frente  : {describe(df.front_size, 1)}')
    print(f'  HV bruto (no comparable entre imagenes): {describe(df.hv_raw, 4)}')
    print('\n  Por degradacion (HV normalizado):')
    if 'degradation_type' in df.columns:
        for deg, sub in df.groupby('degradation_type'):
            print(f'    {DEGRADATION_LABELS.get(deg, deg):<22} n={len(sub):<3} '
                  f'{sub.hv_norm.mean():.3f} +/- {sub.hv_norm.std():.3f}')

    tex = latex_table(df, exp_dir / 'tables')
    print(f'\nTabla LaTeX: {tex}')

    if args.update_book:
        import shutil
        book_tables = PROJECT_ROOT / 'docs' / 'libro' / 'Tables'
        book_tables.mkdir(parents=True, exist_ok=True)
        shutil.copy2(tex, book_tables / tex.name)
        print(f'Copiada a {book_tables}')


if __name__ == '__main__':
    main()
