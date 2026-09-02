"""
Estudio de variabilidad estocástica de SMPSO: corridas independientes con
distintas semillas sobre las mismas imágenes.

El experimento principal ejecuta una única corrida por imagen con semilla fija,
de modo que la variabilidad reportada es *entre imágenes*, no entre repeticiones
de la misma imagen. Este estudio cubre esa segunda fuente de variabilidad, que
es la convención en la literatura de algoritmos evolutivos multiobjetivo
(~30 corridas independientes por instancia, con media, desvío y dispersión de
los indicadores de calidad).

Sobre las mismas 3 imágenes representativas del estudio de convergencia
(reutilizando sus imágenes degradadas del experimento principal, para plena
comparabilidad) se ejecuta SMPSO con la configuración del experimento principal
(100 partículas, 100 iteraciones, archivo externo de 100) una vez por semilla,
y se guarda el frente final de cada corrida.

Del análisis posterior se obtiene, por imagen:

  - Hipervolumen normalizado por corrida. Los objetivos se normalizan a [0,1]
    con los extremos observados en el conjunto de todas las corridas de esa
    imagen (bounds comunes), de modo que los HV de distintas semillas son
    directamente comparables entre sí. Se reporta media, desvío, coeficiente de
    variación, mediana, IQR y rango.

  - IGD+ por corrida respecto de un frente de referencia construido como el
    conjunto no dominado de la unión de los frentes de todas las corridas
    (práctica estándar cuando el frente de Pareto verdadero es desconocido,
    como ocurre en un problema real y no en un benchmark analítico). IGD+ es la
    variante débilmente Pareto-compatible de IGD (Ishibuchi et al., 2015): solo
    penaliza las componentes en las que la aproximación es peor que el punto de
    referencia.

  - Contribución al frente de referencia: cuántas soluciones de cada corrida
    sobreviven en el conjunto no dominado de la unión.

Fases:
    python scripts/multiseed_study.py --run       # ~8 h con 5 workers (90 corridas)
    python scripts/multiseed_study.py --analyze   # segundos

La fase --run es reanudable: omite las corridas cuyo .npz ya existe.

Uso:
    python scripts/multiseed_study.py --run --seeds 30 --workers 5
    python scripts/multiseed_study.py --analyze [--update-book]
"""

import sys
import json
import time
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

EXPERIMENT = PROJECT_ROOT / 'results' / 'experiment_20260709_215441'
OUT_DIR = PROJECT_ROOT / 'results' / 'multiseed_study'

# Las mismas imágenes del estudio de convergencia (Capítulo 5)
IMAGES = {
    '167': 'low_contrast',
    '194': 'underexposure',
    '308': 'poor_local_contrast',
}
DEG_LABELS = {
    'low_contrast': 'Bajo contraste global',
    'underexposure': 'Subexposición',
    'poor_local_contrast': 'Bajo contraste local',
}

# Configuración del experimento principal (scripts/run_experiment.py)
N_PARTICLES = 100
N_ITERATIONS = 100
ARCHIVE_SIZE = 100
MARGIN = 0.05  # margen del punto de referencia, como fracción del rango

COL = '#20606e'
COL_ACCENT = '#c8a415'


# ---------------------------------------------------------------------------
# Fase de ejecución
# ---------------------------------------------------------------------------

def run_one(args: tuple) -> dict:
    """Ejecuta una corrida independiente y guarda su frente final."""
    image_id, seed = args
    out = OUT_DIR / f'run_{image_id}_s{seed:03d}.npz'
    if out.exists():
        return {'image_id': image_id, 'seed': seed, 'status': 'cached'}

    import cv2
    from src.optimization.smpso import SMPSOImageOptimizer

    degraded = cv2.imread(
        str(EXPERIMENT / 'images' / image_id / 'degraded.png'),
        cv2.IMREAD_GRAYSCALE
    )
    if degraded is None:
        return {'image_id': image_id, 'seed': seed, 'status': 'error',
                'error': 'no se pudo cargar degraded.png'}

    t0 = time.time()
    optimizer = SMPSOImageOptimizer(
        image=degraded,
        n_particles=N_PARTICLES,
        max_iterations=N_ITERATIONS,
        archive_size=ARCHIVE_SIZE,
        verbose=False,
        seed=seed,
    )
    optimizer.run()
    elapsed = time.time() - t0

    front = optimizer.pareto_front.get_decision_matrix()
    params = optimizer.pareto_front.get_parameters_matrix()
    np.savez_compressed(out, front=front, params=params, elapsed=elapsed)

    return {'image_id': image_id, 'seed': seed, 'status': 'success',
            'elapsed': elapsed, 'front_size': len(front)}


def phase_run(seeds: int, workers: int):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    jobs = [(img, s) for img in IMAGES for s in range(1, seeds + 1)]
    pending = [j for j in jobs
               if not (OUT_DIR / f'run_{j[0]}_s{j[1]:03d}.npz').exists()]
    print(f'{len(jobs)} corridas ({N_PARTICLES} particulas x {N_ITERATIONS} '
          f'iteraciones), {len(pending)} pendientes, {workers} workers')
    if not pending:
        print('Nada que ejecutar.')
        return

    t0 = time.time()
    results = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(run_one, j): j for j in pending}
        for fut in as_completed(futures):
            r = fut.result()
            results.append(r)
            done, total = len(results), len(pending)
            eta = (time.time() - t0) / done * (total - done) / 3600
            print(f"  [{done}/{total}] imagen {r['image_id']} "
                  f"semilla {r['seed']}: {r['status']} "
                  f"({r.get('elapsed', 0) / 60:.1f} min, "
                  f"frente={r.get('front_size')}) ETA {eta:.1f} h")

    (OUT_DIR / 'runs_summary.json').write_text(
        json.dumps(results, indent=2), encoding='utf-8')
    print(f'Fase de ejecucion completa en {(time.time() - t0) / 3600:.1f} h')


# ---------------------------------------------------------------------------
# Indicadores
# ---------------------------------------------------------------------------

def non_dominated(M: np.ndarray) -> np.ndarray:
    """Máscara de los puntos no dominados de M (maximización, 3 objetivos)."""
    n = len(M)
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        if not keep[i]:
            continue
        # j domina a i si es >= en todo y > en algo
        ge = np.all(M >= M[i], axis=1)
        gt = np.any(M > M[i], axis=1)
        if np.any(ge & gt):
            keep[i] = False
    return keep


def igd_plus(A: np.ndarray, R: np.ndarray) -> float:
    """
    IGD+ de la aproximación A respecto del frente de referencia R
    (maximización): para cada z de R, la distancia modificada al punto más
    cercano de A contando solo las componentes en que A es peor que z.
    """
    if len(A) == 0 or len(R) == 0:
        return float('nan')
    diff = np.maximum(R[:, None, :] - A[None, :, :], 0.0)   # |R| x |A| x k
    d = np.linalg.norm(diff, axis=2)
    return float(d.min(axis=1).mean())


def phase_analyze():
    from src.optimization.pareto import calculate_hypervolume

    def hypervolume(M, ref):
        return calculate_hypervolume([{'objectives': row} for row in M], ref)

    data = {}
    for image_id, deg in IMAGES.items():
        files = sorted(OUT_DIR.glob(f'run_{image_id}_s*.npz'))
        if not files:
            print(f'Sin corridas para la imagen {image_id}')
            continue

        runs = {}
        for f in files:
            z = np.load(f)
            seed = int(f.stem.split('_s')[1])
            runs[seed] = {'front': z['front'], 'elapsed': float(z['elapsed'])}

        # Bounds comunes a todas las corridas de la imagen: hacen comparables
        # el HV entre semillas y ponen el IGD+ en escala adimensional
        all_pts = np.vstack([r['front'] for r in runs.values()])
        lo, hi = all_pts.min(0), all_pts.max(0)
        rng = np.where(hi > lo, hi - lo, 1.0)
        ref = np.full(3, -MARGIN)
        box = (1.0 + MARGIN) ** 3

        norm = {s: (r['front'] - lo) / rng for s, r in runs.items()}

        # Frente de referencia: no dominados de la unión de todas las corridas
        union = np.vstack([norm[s] for s in sorted(norm)])
        owner = np.concatenate([[s] * len(norm[s]) for s in sorted(norm)])
        mask = non_dominated(union)
        R = union[mask]
        contrib = {int(s): int(np.sum(owner[mask] == s)) for s in sorted(norm)}

        per_seed = {}
        for s in sorted(norm):
            per_seed[int(s)] = {
                'hv': hypervolume(norm[s], ref) / box,
                'igd_plus': igd_plus(norm[s], R),
                'front_size': int(len(norm[s])),
                'elapsed': runs[s]['elapsed'],
                'contribution': contrib[int(s)],
            }

        hv = np.array([v['hv'] for v in per_seed.values()])
        igd = np.array([v['igd_plus'] for v in per_seed.values()])
        data[image_id] = {
            'degradation': deg,
            'n_runs': len(per_seed),
            'reference_front_size': int(len(R)),
            'bounds_lo': lo.tolist(),
            'bounds_hi': hi.tolist(),
            'per_seed': per_seed,
            'hv_stats': _stats(hv),
            'igd_stats': _stats(igd),
        }

        print(f"imagen {image_id} ({DEG_LABELS[deg]}), {len(per_seed)} corridas:")
        print(f"  HV    media {hv.mean():.4f} +/- {hv.std(ddof=1):.4f} "
              f"(CV {hv.std(ddof=1) / hv.mean() * 100:.2f}%), "
              f"rango [{hv.min():.4f}, {hv.max():.4f}]")
        print(f"  IGD+  media {igd.mean():.4f} +/- {igd.std(ddof=1):.4f}, "
              f"rango [{igd.min():.4f}, {igd.max():.4f}]")
        print(f"  frente de referencia: {len(R)} soluciones, "
              f"aportadas por {sum(1 for c in contrib.values() if c > 0)} corridas")

    (OUT_DIR / 'multiseed_data.json').write_text(
        json.dumps(data, indent=2), encoding='utf-8')
    print(f'\nAnalisis guardado en {OUT_DIR / "multiseed_data.json"}')


def _stats(x: np.ndarray) -> dict:
    return {
        'mean': float(x.mean()),
        'std': float(x.std(ddof=1)) if len(x) > 1 else 0.0,
        'cv': float(x.std(ddof=1) / x.mean() * 100) if len(x) > 1 and x.mean() else 0.0,
        'median': float(np.median(x)),
        'q1': float(np.percentile(x, 25)),
        'q3': float(np.percentile(x, 75)),
        'min': float(x.min()),
        'max': float(x.max()),
    }


# ---------------------------------------------------------------------------
# Figura y tabla
# ---------------------------------------------------------------------------

def phase_figures(update_book: bool = False):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    data = json.loads((OUT_DIR / 'multiseed_data.json').read_text(encoding='utf-8'))
    if not data:
        raise SystemExit('No hay datos analizados; corra --analyze primero.')

    ids = [i for i in IMAGES if i in data]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6), constrained_layout=True)

    for ax, key, label in [(axes[0], 'hv', 'Hipervolumen normalizado'),
                           (axes[1], 'igd_plus', 'IGD$^+$')]:
        vals = [[data[i]['per_seed'][s][key] for s in data[i]['per_seed']]
                for i in ids]
        bp = ax.boxplot(vals, widths=0.55, patch_artist=True,
                        medianprops=dict(color=COL_ACCENT, lw=1.6))
        for box in bp['boxes']:
            box.set(facecolor='#dfe9eb', edgecolor=COL, lw=1.1)
        for k, v in enumerate(vals, start=1):
            ax.scatter(np.random.normal(k, 0.045, len(v)), v, s=9,
                       color=COL, alpha=0.55, zorder=3)
        ax.set_xticks(range(1, len(ids) + 1))
        ax.set_xticklabels([f"{i}\n({DEG_LABELS[data[i]['degradation']].lower()})"
                            for i in ids], fontsize=8)
        ax.set_ylabel(label, fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(alpha=0.25, axis='y')

    n = data[ids[0]]['n_runs']
    fig.suptitle(f'Variabilidad entre {n} corridas independientes por imagen',
                 fontsize=10)
    for ext in ['.pdf', '.png']:
        fig.savefig(OUT_DIR / f'variabilidad_semillas{ext}', dpi=200)
    plt.close(fig)

    # Tabla LaTeX
    lines = ['\\begin{tabular}{lccccc}', '\\hline',
             '\\textbf{Imagen} & \\textbf{Corridas} & \\textbf{HV (media $\\pm$ DE)} & '
             '\\textbf{CV} & \\textbf{IGD$^+$ (media $\\pm$ DE)} & '
             '\\textbf{Rango de HV} \\\\', '\\hline']
    for i in ids:
        d = data[i]
        h, g = d['hv_stats'], d['igd_stats']
        lines.append(
            f"{i} ({DEG_LABELS[d['degradation']].lower()}) & {d['n_runs']} & "
            f"{h['mean']:.4f} $\\pm$ {h['std']:.4f} & {h['cv']:.2f}\\% & "
            f"{g['mean']:.4f} $\\pm$ {g['std']:.4f} & "
            f"[{h['min']:.4f}, {h['max']:.4f}] \\\\")
    lines += ['\\hline', '\\end{tabular}']
    tex = OUT_DIR / 'tab_multiseed.tex'
    tex.write_text('\n'.join(lines), encoding='utf-8')
    print(f'Figura y tabla en {OUT_DIR}')

    if update_book:
        import shutil
        book_tables = PROJECT_ROOT / 'docs' / 'libro' / 'Tables'
        book_figs = PROJECT_ROOT / 'docs' / 'libro' / 'Figures' / 'capitulo5'
        book_tables.mkdir(parents=True, exist_ok=True)
        book_figs.mkdir(parents=True, exist_ok=True)
        shutil.copy2(tex, book_tables / tex.name)
        for ext in ['.pdf', '.png']:
            f = OUT_DIR / f'variabilidad_semillas{ext}'
            shutil.copy2(f, book_figs / f.name)
        print(f'Copiados a {book_tables} y {book_figs}')


def main():
    parser = argparse.ArgumentParser(
        description='Variabilidad de SMPSO entre corridas independientes')
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--analyze', action='store_true')
    parser.add_argument('--figures', action='store_true')
    parser.add_argument('--seeds', type=int, default=30,
                        help='Número de semillas por imagen (default: 30)')
    parser.add_argument('--workers', type=int, default=5)
    parser.add_argument('--update-book', action='store_true',
                        help='Copiar figura y tabla a docs/libro/')
    args = parser.parse_args()

    if args.run:
        phase_run(args.seeds, args.workers)
    if args.analyze:
        phase_analyze()
    if args.figures or args.update_book:
        phase_figures(args.update_book)
    if not (args.run or args.analyze or args.figures or args.update_book):
        parser.print_help()


if __name__ == '__main__':
    main()
