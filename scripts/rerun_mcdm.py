"""
Recomputa la etapa de decisión (MCDM) de un experimento ya ejecutado,
reutilizando los Frentes de Pareto guardados en cada images/<id>/pareto.csv.

Esto permite corregir o variar la etapa de decisión sin repetir la costosa
optimización SMPSO, y realizar el análisis de sensibilidad de los pesos.

Esquemas de pesos evaluados (orden de importancia: VIF > H > SSIM, ver §4):
  - roc     : Rank Order Centroid (0.611, 0.278, 0.111) -- esquema principal
  - rs      : Rank Sum (0.500, 0.333, 0.167)
  - equal   : pesos iguales (0.333, 0.333, 0.333)
  - legacy  : (0.400, 0.350, 0.250) sobre (H, SSIM, VIF), usado en la corrida original

Acciones:
  1. Actualiza result.json de cada imagen con las selecciones del esquema
     principal (roc) y guarda todas las selecciones por esquema en
     sensitivity.json.
  2. Reescribe experiment_data.csv y mcdm_agreement_matrix.csv coherentes
     con el esquema principal.
  3. Escribe weights_sensitivity.csv con la selección de cada método bajo
     cada esquema de pesos.

Uso:
    python scripts/rerun_mcdm.py [--experiment results/experiment_X] [--scheme roc]
"""

import sys
import json
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.mcdm import (SMARTER, TOPSIS, BellmanZadeh, PROMETHEEII,
                      GRA, VIKOR, CODAS, MABAC)

METHODS = [('SMARTER', SMARTER), ('TOPSIS', TOPSIS),
           ('BellmanZadeh', BellmanZadeh), ('PROMETHEEII', PROMETHEEII),
           ('GRA', GRA), ('VIKOR', VIKOR),
           ('CODAS', CODAS), ('MABAC', MABAC)]
METHOD_NAMES = [n for n, _ in METHODS]

# Columnas de la matriz de decisión: 0=H, 1=SSIM, 2=VIF
# Orden de importancia declarado: VIF (2) > H (0) > SSIM (1)
CRITERIA_RANK = [2, 0, 1]


def roc_weights(n: int) -> np.ndarray:
    return np.array([(1.0 / n) * np.sum(1.0 / np.arange(j + 1, n + 1)) for j in range(n)])


def rs_weights(n: int) -> np.ndarray:
    return np.array([2.0 * (n + 1 - (j + 1)) / (n * (n + 1)) for j in range(n)])


def weights_by_rank(position_weights: np.ndarray, rank) -> np.ndarray:
    """Mapea pesos por posición de importancia a pesos por columna."""
    w = np.zeros(len(position_weights))
    for position, criterion_index in enumerate(rank):
        w[criterion_index] = position_weights[position]
    return w


def build_schemes() -> dict:
    n = 3
    return {
        'roc': weights_by_rank(roc_weights(n), CRITERIA_RANK),
        'rs': weights_by_rank(rs_weights(n), CRITERIA_RANK),
        'equal': np.ones(n) / n,
        'legacy': np.array([0.40, 0.35, 0.25]),
    }


def apply_mcdm(decision_matrix: np.ndarray, weights: np.ndarray) -> dict:
    """Aplica los 8 métodos y devuelve selecciones, puntajes y órdenes."""
    criteria_types = ['benefit'] * decision_matrix.shape[1]
    out = {}
    for name, MethodClass in METHODS:
        try:
            if name == 'SMARTER':
                # SMARTER canónico: pesos ROC derivados del orden de importancia
                method = MethodClass(criteria_types=criteria_types.copy(),
                                     use_rank_order_weights=True,
                                     criteria_rank=CRITERIA_RANK)
            else:
                method = MethodClass(weights=weights.copy(),
                                     criteria_types=criteria_types.copy())
            best_idx, rankings = method.select(decision_matrix.copy())
            scores = np.asarray(rankings).ravel()
            out[name] = {
                'best_index': int(best_idx),
                'scores': [round(float(s), 6) for s in scores],
                'ranking_order': [int(i) for i in method.get_ranking_order(scores)],
            }
        except Exception as e:  # noqa: BLE001
            out[name] = {'error': str(e)}
    return out


def consensus_of(mcdm_results: dict) -> dict:
    votes = {}
    for info in mcdm_results.values():
        if 'best_index' in info:
            votes[info['best_index']] = votes.get(info['best_index'], 0) + 1
    if not votes:
        return {'index': None, 'votes': 0, 'total_methods': 0}
    best = max(votes, key=votes.get)
    return {'index': int(best), 'votes': int(votes[best]),
            'total_methods': sum(1 for i in mcdm_results.values() if 'best_index' in i)}


def regenerate_enhanced(img_dir: Path, result: dict, params: np.ndarray) -> dict:
    """
    Regenera enhanced.png aplicando CLAHE con los parámetros de la solución de
    consenso, y recalcula las métricas de validación contra la imagen original.

    La salida del framework es la solución elegida en la etapa de decisión, de
    modo que la imagen guardada debe corresponder al consenso de los métodos
    MCDM (y no a la solución de compromiso geométrica del frente).
    """
    import cv2
    from src.clahe.processor import CLAHEProcessor
    from src.metrics.ssim import calculate_ssim
    from src.metrics.vif import calculate_vif
    from src.utils.degradation import get_image_quality_metrics

    original = cv2.imread(str(img_dir / 'original.png'), cv2.IMREAD_GRAYSCALE)
    degraded = cv2.imread(str(img_dir / 'degraded.png'), cv2.IMREAD_GRAYSCALE)
    if original is None or degraded is None:
        return result

    idx = result['consensus']['index']
    rx, ry, clip = params[idx]
    processor = CLAHEProcessor(rx=int(round(rx)), ry=int(round(ry)), clip_limit=float(clip))
    enhanced = processor.process(degraded)
    cv2.imwrite(str(img_dir / 'enhanced.png'), enhanced)

    result['metrics']['enhanced'] = get_image_quality_metrics(enhanced)
    result['validation_vs_original'] = {
        'ssim_degraded': float(calculate_ssim(original, degraded)),
        'ssim_enhanced': float(calculate_ssim(original, enhanced)),
        'vif_degraded': float(calculate_vif(original, degraded)),
        'vif_enhanced': float(calculate_vif(original, enhanced)),
    }
    return result


def main():
    parser = argparse.ArgumentParser(description='Recomputa la etapa MCDM')
    parser.add_argument('--experiment', type=str, required=True)
    parser.add_argument('--scheme', type=str, default='roc',
                        choices=['roc', 'rs', 'equal', 'legacy'],
                        help='Esquema de pesos principal (default: roc)')
    parser.add_argument('--skip-images', action='store_true',
                        help='No regenerar enhanced.png ni métricas de validación')
    args = parser.parse_args()

    exp_dir = Path(args.experiment)
    schemes = build_schemes()
    main_scheme = args.scheme
    print(f'Experimento: {exp_dir}')
    print('Esquemas de pesos (columnas H, SSIM, VIF):')
    for k, v in schemes.items():
        marca = ' <- principal' if k == main_scheme else ''
        print(f'  {k:7s}: {np.round(v, 4)}{marca}')

    sensitivity_rows = []
    updated = 0

    for img_dir in sorted((exp_dir / 'images').iterdir()):
        pareto_csv = img_dir / 'pareto.csv'
        result_json = img_dir / 'result.json'
        if not (pareto_csv.exists() and result_json.exists()):
            continue

        df = pd.read_csv(pareto_csv)
        dm = df[['objective_0', 'objective_1', 'objective_2']].values
        params = df[['param_0', 'param_1', 'param_2']].values
        result = json.loads(result_json.read_text(encoding='utf-8'))
        image_id = result['image_id']

        per_scheme = {}
        for scheme_name, w in schemes.items():
            res = apply_mcdm(dm, w)
            cons = consensus_of(res)
            per_scheme[scheme_name] = {'mcdm_results': res, 'consensus': cons}

            row = {'image_id': image_id, 'scheme': scheme_name,
                   'consensus_index': cons['index'], 'consensus_votes': cons['votes']}
            for m in METHOD_NAMES:
                row[f'{m}_selection'] = res.get(m, {}).get('best_index')
            sensitivity_rows.append(row)

        # Esquema principal → result.json
        main_res = per_scheme[main_scheme]
        for name, info in main_res['mcdm_results'].items():
            if 'best_index' in info:
                idx = info['best_index']
                info['best_params'] = params[idx].tolist()
                info['best_objectives'] = dm[idx].tolist()
        result['mcdm_results'] = main_res['mcdm_results']
        result['consensus'] = main_res['consensus']
        result['weights_scheme'] = main_scheme
        result['weights'] = [float(x) for x in schemes[main_scheme]]

        # Solución seleccionada por el framework = consenso de los métodos MCDM
        cons_idx = result['consensus']['index']
        if cons_idx is not None:
            result['selected'] = {
                'source': 'consensus',
                'index': int(cons_idx),
                'params': [float(x) for x in params[cons_idx]],
                'objectives': [float(x) for x in dm[cons_idx]],
            }

        # La imagen de salida debe corresponder a la solución de consenso
        if not args.skip_images and cons_idx is not None:
            result = regenerate_enhanced(img_dir, result, params)

        result_json.write_text(json.dumps(result, indent=2), encoding='utf-8')

        # sensitivity.json por imagen
        (img_dir / 'sensitivity.json').write_text(
            json.dumps({k: {'consensus': v['consensus'],
                            'selections': {m: i.get('best_index')
                                           for m, i in v['mcdm_results'].items()}}
                        for k, v in per_scheme.items()}, indent=2),
            encoding='utf-8')
        updated += 1

    print(f'\nImágenes actualizadas: {updated}')

    sens = pd.DataFrame(sensitivity_rows)
    sens.to_csv(exp_dir / 'weights_sensitivity.csv', index=False)
    print(f'Sensibilidad de pesos: {exp_dir / "weights_sensitivity.csv"}')

    # Regenerar experiment_data.csv y matriz de acuerdo con el esquema principal
    from scripts.run_experiment import generate_statistical_analysis
    results = []
    for img_dir in sorted((exp_dir / 'images').iterdir()):
        rj = img_dir / 'result.json'
        if rj.exists():
            r = json.loads(rj.read_text(encoding='utf-8'))
            if r.get('status') == 'success':
                results.append(r)
    generate_statistical_analysis(results, exp_dir)
    print(f'\nexperiment_data.csv y mcdm_agreement_matrix.csv regenerados '
          f'({len(results)} imágenes, esquema {main_scheme})')


if __name__ == '__main__':
    main()
