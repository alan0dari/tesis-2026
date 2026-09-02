"""
Estudio de convergencia y cobertura del frente para justificar la
configuración de SMPSO (Capítulos 4 y 5 del libro).

Sobre 3 imágenes representativas del experimento principal (reutilizando sus
imágenes degradadas para plena comparabilidad) se ejecuta SMPSO con 100
partículas durante 250 iteraciones, con dos tamaños de archivo externo
(100 y 200), registrando en cada iteración una instantánea del frente.

Del análisis posterior se obtiene, por iteración: hipervolumen exacto
(respecto a un punto de referencia fijo por imagen), spacing y tamaño del
frente. Esto permite:
  1. Justificar empíricamente el presupuesto de 100 iteraciones usado en el
     experimento principal (meseta del hipervolumen).
  2. Cuantificar cuánta cobertura adicional del frente aporta un archivo
     mayor (100 vs. 200).

Fases:
    python scripts/convergence_study.py --run       # ~2 h (paralelo)
    python scripts/convergence_study.py --analyze   # segundos
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
OUT_DIR = PROJECT_ROOT / 'results' / 'convergence_study'

# Imágenes representativas (las mismas del Capítulo 5) con degradaciones
# de dificultad distinta para el framework
IMAGES = {
    '167': 'low_contrast',
    '194': 'underexposure',
    '308': 'poor_local_contrast',
}
ARCHIVE_SIZES = [100, 200]
N_PARTICLES = 100
N_ITERATIONS = 250
SEED = 42


def run_one(args: tuple) -> dict:
    """Ejecuta una corrida instrumentada y guarda las instantáneas del frente."""
    image_id, archive_size = args

    import cv2
    from src.optimization.smpso import SMPSOImageOptimizer

    degraded = cv2.imread(
        str(EXPERIMENT / 'images' / image_id / 'degraded.png'),
        cv2.IMREAD_GRAYSCALE
    )
    if degraded is None:
        return {'image_id': image_id, 'archive_size': archive_size,
                'status': 'error', 'error': 'no se pudo cargar degraded.png'}

    snapshots = []

    class InstrumentedOptimizer(SMPSOImageOptimizer):
        def _record_iteration(self, iteration):
            super()._record_iteration(iteration)
            snapshots.append(self.pareto_front.get_decision_matrix().copy())

    t0 = time.time()
    optimizer = InstrumentedOptimizer(
        image=degraded,
        n_particles=N_PARTICLES,
        max_iterations=N_ITERATIONS,
        archive_size=archive_size,
        verbose=False,
        seed=SEED,
    )
    optimizer.run()
    elapsed = time.time() - t0

    out = OUT_DIR / f'run_{image_id}_a{archive_size}.npz'
    np.savez_compressed(
        out,
        **{f'iter_{i}': snap for i, snap in enumerate(snapshots)},
        elapsed=elapsed,
    )
    return {'image_id': image_id, 'archive_size': archive_size,
            'status': 'success', 'elapsed': elapsed,
            'final_front': len(snapshots[-1])}


def phase_run(workers: int):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    jobs = [(img, a) for img in IMAGES for a in ARCHIVE_SIZES]
    print(f'{len(jobs)} corridas ({N_PARTICLES} particulas x {N_ITERATIONS} '
          f'iteraciones), {workers} workers')
    t0 = time.time()
    results = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(run_one, j): j for j in jobs}
        for fut in as_completed(futures):
            r = fut.result()
            results.append(r)
            print(f"  [{len(results)}/{len(jobs)}] {r['image_id']} "
                  f"archivo={r['archive_size']}: {r['status']} "
                  f"({r.get('elapsed', 0)/60:.1f} min, "
                  f"frente final={r.get('final_front')})")
    (OUT_DIR / 'runs_summary.json').write_text(
        json.dumps(results, indent=2), encoding='utf-8')
    print(f'Fase de ejecucion completa en {(time.time()-t0)/60:.1f} min')


def phase_analyze():
    from src.optimization.pareto import calculate_hypervolume

    def spacing(M: np.ndarray) -> float:
        if len(M) < 2:
            return 0.0
        d = np.linalg.norm(M[:, None, :] - M[None, :, :], axis=2)
        np.fill_diagonal(d, np.inf)
        mins = d.min(axis=1)
        return float(np.sqrt(np.mean((mins - mins.mean()) ** 2)))

    # Punto de referencia fijo por imagen: mínimo global observado en todas
    # las corridas de esa imagen, menos un margen del 5% del rango
    data = {}
    for image_id in IMAGES:
        runs = {}
        for a in ARCHIVE_SIZES:
            f = OUT_DIR / f'run_{image_id}_a{a}.npz'
            if not f.exists():
                print(f'FALTA {f}')
                continue
            z = np.load(f)
            snaps = [z[f'iter_{i}'] for i in range(N_ITERATIONS)]
            runs[a] = snaps
        if not runs:
            continue

        all_pts = np.vstack([s for snaps in runs.values() for s in snaps])
        lo, hi = all_pts.min(0), all_pts.max(0)
        ref = lo - 0.05 * (hi - lo)

        data[image_id] = {'reference_point': ref.tolist(), 'runs': {}}
        for a, snaps in runs.items():
            hv, sp, size = [], [], []
            for M in snaps:
                front = [{'objectives': row} for row in M]
                hv.append(calculate_hypervolume(front, ref))
                sp.append(spacing(M))
                size.append(len(M))
            data[image_id]['runs'][str(a)] = {
                'hypervolume': hv, 'spacing': sp, 'front_size': size}
            print(f'imagen {image_id} archivo {a}: '
                  f'HV@100={hv[99]:.4f} HV@250={hv[-1]:.4f} '
                  f'(HV@100/HV@250={hv[99]/hv[-1]*100:.1f}%) '
                  f'frente@250={size[-1]}')

    (OUT_DIR / 'convergence_data.json').write_text(
        json.dumps(data, indent=2), encoding='utf-8')
    print(f'Analisis guardado en {OUT_DIR / "convergence_data.json"}')


def phase_figures():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    data = json.loads((OUT_DIR / 'convergence_data.json').read_text(encoding='utf-8'))

    deg_labels = {'low_contrast': 'Bajo contraste global',
                  'underexposure': 'Subexposición',
                  'poor_local_contrast': 'Bajo contraste local'}
    colors = {'100': '#20606e', '200': '#c8a415'}

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.6), sharex=True,
                             sharey=True, constrained_layout=True)
    for ax, (image_id, deg) in zip(axes, IMAGES.items()):
        d = data.get(image_id)
        if d is None:
            continue
        for a, series in d['runs'].items():
            hv = np.array(series['hypervolume'])
            hv_norm = hv / hv.max()
            ax.plot(range(1, len(hv) + 1), hv_norm, color=colors[a],
                    lw=1.7, label=f'Archivo {a}')
        ax.axvline(100, color='#888888', ls='--', lw=1.1)
        if image_id == list(IMAGES)[0]:
            ax.text(0.44, 0.10, 'presupuesto del\nexperimento (100 iter.)',
                    transform=ax.transAxes, fontsize=8, color='#555555')
        ax.set_title(f'Imagen {image_id} ({deg_labels[deg].lower()})',
                     fontsize=10)
        ax.set_xlabel('Iteración', fontsize=9)
        ax.tick_params(labelsize=8)
        ax.set_ylim(0.55, 1.03)
        ax.set_xlim(0, 252)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel('Hipervolumen normalizado', fontsize=9)
    axes[2].legend(fontsize=8.5, loc='lower right', frameon=False)
    for ext in ['.pdf', '.png']:
        fig.savefig(OUT_DIR / f'convergencia_hv{ext}', dpi=200)
    plt.close(fig)

    # Tabla LaTeX para el libro
    lines = ['\\begin{tabular}{llcccc}', '\\hline',
             '\\textbf{Imagen} & \\textbf{Archivo} & \\textbf{HV$_{100}$/HV$_{250}$} & '
             '\\textbf{Frente$_{250}$} & \\textbf{Spacing$_{100}$} & \\textbf{Spacing$_{250}$} \\\\',
             '\\hline']
    for image_id, deg in IMAGES.items():
        d = data.get(image_id)
        if d is None:
            continue
        for a, s in d['runs'].items():
            hv = s['hypervolume']
            lines.append(
                f"{image_id} ({deg_labels[deg].lower()}) & {a} & "
                f"{hv[99] / hv[-1] * 100:.1f}\\% & {s['front_size'][-1]} & "
                f"{s['spacing'][99]:.4f} & {s['spacing'][-1]:.4f} \\\\")
    lines += ['\\hline', '\\end{tabular}']
    (OUT_DIR / 'tab_convergencia.tex').write_text('\n'.join(lines), encoding='utf-8')
    print(f'Figura y tabla en {OUT_DIR}')


def main():
    parser = argparse.ArgumentParser(description='Estudio de convergencia SMPSO')
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--analyze', action='store_true')
    parser.add_argument('--workers', type=int, default=5)
    args = parser.parse_args()

    if args.run:
        phase_run(args.workers)
    if args.analyze or not args.run:
        phase_analyze()
        phase_figures()


if __name__ == '__main__':
    main()
