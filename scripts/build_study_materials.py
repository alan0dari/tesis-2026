"""
Fase 2 del estudio perceptual: genera los materiales que ven los odontólogos.

A partir de un experimento ya corrido produce, de forma determinista:

  1. Las imágenes candidatas de cada radiografía (original, degradada, las
     selecciones de los 8 métodos MCDM y los extremos mono-objetivo del frente).
  2. Deduplicación perceptual: agrupa candidatas indistinguibles (SSIM ≥ umbral)
     para no gastar pantallas en comparaciones sin señal. Cubre el requisito de
     no duplicar las soluciones de consenso.
  3. Asignación de un sextante por radiografía, balanceada sobre los 6.
  4. Recortes del sextante, idénticos en geometría para todas las condiciones de
     una misma radiografía, así cada par queda exactamente pareado.
  5. Los ensayos (Q1, Q2, Q3, Q4) según docs/evaluacion/protocolo_evaluacion.md §5.
  6. Reparto en bloques incompletos balanceados entre los evaluadores.

Las radiografías NO se modifican: el recorte del sextante es de visualización,
por la razón clínica de que el diagnóstico se hace por regiones. Los archivos de
`data/evaluacion_50/` y las salidas del experimento quedan intactos, con su
anotación quemada.

Uso:
    python scripts/build_study_materials.py \
        --experiment results/experiment_20260803_203341 --raters 8
"""

import sys
import json
import argparse
import hashlib
from pathlib import Path
from itertools import combinations

PROJECT_ROOT = str(Path(__file__).parent.parent)
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
import cv2
from scipy.signal import find_peaks

from src.clahe.processor import apply_clahe_simple
from src.metrics.entropy import calculate_entropy
from src.metrics.ssim import calculate_ssim
from src.metrics.vif import calculate_vif

METHODS = ['SMARTER', 'TOPSIS', 'BellmanZadeh', 'PROMETHEEII',
           'GRA', 'VIKOR', 'CODAS', 'MABAC']

# Umbral de equivalencia perceptual. Ver docs/evaluacion/colapso_perceptual_mcdm.md:
# es hipótesis operativa, y el propio estudio la calibra incluyendo pares de
# distintas bandas de SSIM.
EQUIV_THRESHOLD = 0.98

# Sextantes. En una panorámica la derecha del paciente cae a la IZQUIERDA de la
# imagen (por eso la marca "R" está a la izquierda).
SEXTANT_X = {
    'posterior_derecho': (0.18, 0.40),
    'anterior': (0.39, 0.61),
    'posterior_izquierdo': (0.60, 0.82),
}
# Medio alto del recorte, en fracción de la altura: cubre corona más raíz.
SEXTANT_HALF_HEIGHT = 0.145
SEXTANTS = [(a, b) for b in ('superior', 'inferior') for a in SEXTANT_X]

# Presupuesto de ensayos (protocolo §5)
N_Q2 = 50          # original vs. consenso: una por imagen
N_Q4_IMAGES = 15   # imágenes que aportan pares de extremos
N_Q3 = 20          # pares MCDM de mayor contraste
N_Q1 = 15          # degradada vs. consenso, control de atención


# ---------------------------------------------------------------- geometría

TOOTH_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (81, 81))

# Franja donde puede caer la dentición, y bandas de anotación quemada a evitar
# al elegir el encuadre (las imágenes no se editan: sólo se evita apuntar ahí).
# El corte inferior en 0.76 deja fuera el borde cortical de la mandíbula, que es
# brillante y compacto y por lo tanto también responde al top-hat: al buscar "el
# lóbulo más bajo" se lo confundía con la arcada y el recorte caía sobre hueso.
SEARCH_Y = (0.28, 0.76)
SAFE_Y = (0.07, 0.93)


def arch_anchor(image, x0, x1, arch):
    """
    Centro vertical de la arcada pedida dentro de una franja de columnas.

    Un top-hat con núcleo del tamaño de una pieza deja sólo las estructuras
    compactas y brillantes -- los dientes -- sobre el fondo suave del hueso. El
    perfil por fila tiene un lóbulo por arcada, y el recorte se centra en el
    lóbulo, no en un desplazamiento fijo desde el plano oclusal.

    Dos decisiones que costaron iteraciones, ambas por el sector anterior:

    - La mandíbula es **el lóbulo más bajo que supere la mitad del máximo**, no
      el segundo más alto. En el anterior el paladar duro y la espina nasal
      responden más que los incisivos, así que "los dos lóbulos más altos" podían
      ser paladar y maxilar, dejando la mandíbula sin detectar.
    - El plano oclusal se busca sólo en una ventana estrecha encima de la
      mandíbula. Sin acotar, el mínimo de intensidad se escapa al fondo negro
      fuera del maxilar, al seno maxilar o a la apertura nasal, todos más oscuros
      que el espacio interoclusal.

    El maxilar es entonces el lóbulo inmediatamente por encima del plano oclusal;
    si no hay ninguno, se recurre a un desplazamiento anatómico por defecto.
    """
    h = image.shape[0]
    ya, yb = int(SEARCH_Y[0] * h), int(SEARCH_Y[1] * h)
    band = image[ya:yb, x0:x1]

    lo, hi = np.percentile(band, [5, 99])
    norm = np.clip((band.astype(np.float32) - lo) / max(hi - lo, 1), 0, 1)
    tophat = cv2.morphologyEx((norm * 255).astype(np.uint8),
                              cv2.MORPH_TOPHAT, TOOTH_KERNEL)
    lobes = np.convolve(tophat.mean(axis=1), np.ones(31) / 31, mode='same')

    # `width` exige que el lóbulo tenga grosor de pieza dental: descarta líneas
    # brillantes finas (bordes corticales, tabiques) que sí superan el umbral.
    peaks, _ = find_peaks(lobes, height=0.5 * lobes.max(),
                          distance=max(1, int(0.05 * h)),
                          width=max(1, int(0.025 * h)))
    if len(peaks) == 0:
        peaks, _ = find_peaks(lobes, height=0.5 * lobes.max(),
                              distance=max(1, int(0.05 * h)))
    if len(peaks) == 0:
        peaks = np.array([int(lobes.argmax())])
    mandible = int(peaks[-1])

    intensity = np.convolve(band.mean(axis=1).astype(np.float32),
                            np.ones(15) / 15, mode='same')
    top = max(0, mandible - int(0.14 * h))
    occlusal = top + int(intensity[top:mandible + 1].argmin())

    if arch == 'inferior':
        return ya + mandible
    above = [p for p in peaks if p < occlusal]
    return ya + (int(above[-1]) if above else max(0, occlusal - int(0.10 * h)))


def tooth_response(image, x0, x1):
    """Respuesta al top-hat dental en una franja de columnas, para todo el alto."""
    strip = image[:, x0:x1]
    lo, hi = np.percentile(strip, [5, 99])
    norm = np.clip((strip.astype(np.float32) - lo) / max(hi - lo, 1), 0, 1)
    return cv2.morphologyEx((norm * 255).astype(np.uint8),
                            cv2.MORPH_TOPHAT, TOOTH_KERNEL).mean(axis=1)


def sextant_box(image, sextant):
    """
    Rectángulo del sextante, centrado en la arcada que corresponda.

    Tras anclar en el lóbulo dental se hace un ajuste fino: se prueban
    desplazamientos verticales y se elige el que deja más estructura dental
    dentro del cuadro. Las reglas geométricas solas no cubren la variación
    anatómica -- en algunos pacientes el sector posterior mandibular cae en
    diagonal pronunciada y un rectángulo de alto fijo se llena de fondo --, y
    optimizar directamente el contenido dental es más simple que seguir
    afinando reglas.
    """
    region, arch = sextant
    h, w = image.shape
    x0, x1 = (int(f * w) for f in SEXTANT_X[region])

    centre = arch_anchor(image, x0, x1, arch)
    height = int(2 * SEXTANT_HALF_HEIGHT * h)
    response = tooth_response(image, x0, x1)
    top_limit, bottom_limit = SAFE_Y[0] * h, SAFE_Y[1] * h

    def clamp(y0):
        return int(np.clip(y0, max(0, top_limit), min(h, bottom_limit) - height))

    best = max((float(response[y:y + height].mean()), y)
               for y in (clamp(centre - height // 2 + int(d * h))
                         for d in np.arange(-0.06, 0.061, 0.01)))[1]
    return x0, best, x1, best + height


def crop(image, box):
    x0, y0, x1, y1 = box
    return image[y0:y1, x0:x1]


# ---------------------------------------------------------------- candidatas

def build_candidates(image_dir):
    """
    Devuelve {condición: imagen} con todas las alternativas de una radiografía.

    Las realzadas se regeneran aplicando CLAHE sobre la degradada con los
    parámetros de cada solución, en lugar de leer `enhanced.png`, para que todas
    las condiciones pasen exactamente por el mismo camino de procesamiento.
    """
    result = json.loads((image_dir / 'result.json').read_text(encoding='utf-8'))
    pareto = pd.read_csv(image_dir / 'pareto.csv')
    original = cv2.imread(str(image_dir / 'original.png'), cv2.IMREAD_GRAYSCALE)
    degraded = cv2.imread(str(image_dir / 'degraded.png'), cv2.IMREAD_GRAYSCALE)
    if original is None or degraded is None:
        return None

    def render(idx):
        row = pareto[pareto.solution_id == idx].iloc[0]
        return apply_clahe_simple(degraded, int(row.param_0),
                                  int(row.param_1), float(row.param_2))

    images = {'original': original, 'degraded': degraded}
    voters = {}

    for method in METHODS:
        info = result['mcdm_results'].get(method, {})
        if 'best_index' in info:
            voters.setdefault(int(info['best_index']), []).append(method)
    for idx in voters:
        images[f'sol:{idx}'] = render(idx)

    # Extremos mono-objetivo: lo más distinguible que produce el framework
    extremes = {}
    for k, name in enumerate(('H', 'SSIM', 'VIF')):
        idx = int(pareto.loc[pareto[f'objective_{k}'].idxmax(), 'solution_id'])
        extremes[name] = idx
        images[f'ext:{name}'] = render(idx)

    return {
        'result': result, 'images': images, 'voters': voters,
        'extremes': extremes,
        'consensus': int(result['consensus']['index']),
        'objectives': {int(r.solution_id): (r.objective_0, r.objective_1, r.objective_2)
                       for r in pareto.itertuples()},
    }


def merge_equivalent(images, keys, threshold):
    """Agrupa condiciones perceptualmente indistinguibles (union-find sobre SSIM)."""
    parent = {k: k for k in keys}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    pairwise = {}
    for a, b in combinations(keys, 2):
        s = calculate_ssim(images[a], images[b])
        pairwise[(a, b)] = s
        if s >= threshold:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

    classes = {}
    for k in keys:
        classes.setdefault(find(k), []).append(k)
    return classes, pairwise


# ---------------------------------------------------------------- ensayos

def opaque(*parts):
    """Identificador sin información: el evaluador no debe poder inferir nada."""
    return hashlib.sha1('|'.join(str(p) for p in parts).encode()).hexdigest()[:10]


def build_trials(per_image, rng):
    """
    Arma los ensayos de los cuatro bloques.

    Q3 y Q4 se filtran por SSIM: los pares indistinguibles no se presentan,
    porque medirían fatiga en lugar de percepción. Q4 también se filtra, aunque
    los extremos son mucho más distinguibles: en ~17% de las imágenes dos óptimos
    mono-objetivo caen muy cerca.
    """
    trials = []

    def add(block, image_id, cond_a, cond_b, ssim):
        trials.append({
            'trial_id': opaque(block, image_id, cond_a, cond_b),
            'block': block, 'image_id': image_id,
            'cond_a': cond_a, 'cond_b': cond_b,
            'ssim': None if ssim is None else round(float(ssim), 4),
        })

    ids = sorted(per_image)

    # Q2 -- original vs. consenso, una por imagen
    for image_id in ids:
        d = per_image[image_id]
        add('Q2', image_id, 'original', f'sol:{d["consensus"]}',
            calculate_ssim(d['images']['original'],
                           d['images'][f'sol:{d["consensus"]}']))

    # Q1 -- degradada vs. consenso, control de atención
    for image_id in rng.choice(ids, size=min(N_Q1, len(ids)), replace=False):
        d = per_image[image_id]
        add('Q1', image_id, 'degraded', f'sol:{d["consensus"]}', None)

    # Q4 -- extremos mono-objetivo, sólo pares distinguibles
    q4_pool = []
    for image_id in ids:
        d = per_image[image_id]
        for a, b in combinations(('H', 'SSIM', 'VIF'), 2):
            if d['extremes'][a] == d['extremes'][b]:
                continue
            s = calculate_ssim(d['images'][f'ext:{a}'], d['images'][f'ext:{b}'])
            if s < EQUIV_THRESHOLD:
                q4_pool.append((image_id, a, b, s))
    # Repartir entre imágenes distintas antes que concentrar en pocas
    by_image = {}
    for image_id, a, b, s in q4_pool:
        by_image.setdefault(image_id, []).append((a, b, s))
    chosen_images = sorted(by_image, key=lambda i: -len(by_image[i]))[:N_Q4_IMAGES]
    for image_id in chosen_images:
        for a, b, s in by_image[image_id]:
            add('Q4', image_id, f'ext:{a}', f'ext:{b}', s)

    # Q3 -- pares MCDM de mayor contraste entre clases perceptuales distintas
    q3_pool = []
    for image_id in ids:
        d = per_image[image_id]
        reps = sorted(d['classes'])
        for a, b in combinations(reps, 2):
            s = d['pairwise'].get((a, b)) or d['pairwise'].get((b, a))
            if s is not None and s < EQUIV_THRESHOLD:
                q3_pool.append((s, image_id, a, b))
    q3_pool.sort()
    for s, image_id, a, b in q3_pool[:N_Q3]:
        add('Q3', image_id, a, b, s)

    return trials


def assign_to_raters(trials, n_raters, per_trial, rng):
    """
    Bloques incompletos balanceados: cada ensayo va a `per_trial` evaluadores y
    todos los evaluadores terminan con la misma carga.
    """
    load = {r: 0 for r in range(n_raters)}
    assignment = {r: [] for r in range(n_raters)}
    for trial in trials:
        # Los menos cargados primero; desempate al azar para no crear patrones
        order = sorted(range(n_raters), key=lambda r: (load[r], rng.random()))
        for r in order[:per_trial]:
            assignment[r].append(trial['trial_id'])
            load[r] += 1
    return assignment


# ---------------------------------------------------------------- main

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--experiment', default='results/experiment_20260803_203341')
    parser.add_argument('--out-dir', default='docs/evaluacion/estudio')
    parser.add_argument('--raters', type=int, default=8)
    parser.add_argument('--raters-per-trial', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    exp_dir = Path(args.experiment)
    out_dir = Path(args.out_dir)
    img_dir = out_dir / 'img'
    img_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    image_dirs = sorted((exp_dir / 'images').iterdir(), key=lambda p: p.name)
    print(f'Radiografías: {len(image_dirs)}')

    # Sextante por imagen, balanceado sobre los 6
    order = np.array([SEXTANTS[i % len(SEXTANTS)] for i in range(len(image_dirs))],
                     dtype=object)
    rng.shuffle(order)

    per_image = {}
    print('\nGenerando candidatas y recortes...')
    for i, d in enumerate(image_dirs):
        data = build_candidates(d)
        if data is None:
            print(f'  [X] {d.name}: no se pudo leer')
            continue
        image_id = d.name
        sextant = tuple(order[i])
        box = sextant_box(data['images']['original'], sextant)

        sol_keys = [k for k in data['images'] if k.startswith('sol:')]
        classes, pairwise = merge_equivalent(data['images'], sol_keys, EQUIV_THRESHOLD)

        data.update({'sextant': sextant, 'box': box,
                     'classes': classes, 'pairwise': pairwise})
        per_image[image_id] = data

        if (i + 1) % 10 == 0:
            print(f'  {i + 1}/{len(image_dirs)}')

    # Ensayos
    print('\nArmando ensayos...')
    trials = build_trials(per_image, rng)
    counts = pd.Series([t['block'] for t in trials]).value_counts()
    for block in ('Q1', 'Q2', 'Q3', 'Q4'):
        print(f'  {block}: {counts.get(block, 0)}')
    print(f'  TOTAL: {len(trials)}')

    # Recortes: sólo los que algún ensayo usa
    print('\nEscribiendo recortes...')
    needed = {(t['image_id'], c) for t in trials for c in (t['cond_a'], t['cond_b'])}
    filenames = {}
    for image_id, cond in sorted(needed):
        d = per_image[image_id]
        patch = crop(d['images'][cond], d['box'])
        name = f'{opaque(image_id, cond)}.png'
        cv2.imwrite(str(img_dir / name), patch)
        filenames[(image_id, cond)] = name
    print(f'  {len(needed)} recortes -> {img_dir}')

    report_framing(per_image, out_dir)

    # Lado izquierdo/derecho al azar por ensayo
    for t in trials:
        if rng.random() < 0.5:
            t['cond_a'], t['cond_b'] = t['cond_b'], t['cond_a']
        t['file_a'] = filenames[(t['image_id'], t['cond_a'])]
        t['file_b'] = filenames[(t['image_id'], t['cond_b'])]

    # Reparto entre evaluadores
    assignment = assign_to_raters(trials, args.raters, args.raters_per_trial, rng)
    loads = {r: len(v) for r, v in assignment.items()}
    print(f'\nReparto: {args.raters} evaluadores, '
          f'{args.raters_per_trial} por ensayo, '
          f'{min(loads.values())}-{max(loads.values())} ensayos c/u')

    # Salidas. `trials.json` va a la app; `clave.csv` se queda con el equipo.
    by_id = {t['trial_id']: t for t in trials}
    app_trials = {t['trial_id']: {'file_a': t['file_a'], 'file_b': t['file_b']}
                  for t in trials}
    (out_dir / 'trials.json').write_text(json.dumps(
        {'trials': app_trials,
         'assignment': {f'{r + 1:02d}': v for r, v in assignment.items()}},
        indent=1), encoding='utf-8')

    key = pd.DataFrame([{
        'trial_id': t['trial_id'], 'block': t['block'], 'image_id': t['image_id'],
        'cond_izquierda': t['cond_a'], 'cond_derecha': t['cond_b'],
        'ssim': t['ssim'],
        'sextante': '_'.join(per_image[t['image_id']]['sextant']),
        'degradacion': per_image[t['image_id']]['result']['degradation_type'],
        'metodos_izq': '+'.join(_methods_of(per_image[t['image_id']], t['cond_a'])),
        'metodos_der': '+'.join(_methods_of(per_image[t['image_id']], t['cond_b'])),
        **_delta_objectives(per_image[t['image_id']], t['cond_a'], t['cond_b']),
    } for t in trials])
    key.to_csv(out_dir / 'clave.csv', index=False)

    print(f'\n  {out_dir / "trials.json"}')
    print(f'  {out_dir / "clave.csv"}   <- NO enviar a los evaluadores')
    print(f'\nSextantes: '
          + ', '.join(f'{k}={v}' for k, v in
                      pd.Series(['_'.join(d['sextant']) for d in per_image.values()])
                      .value_counts().sort_index().items()))
    return by_id


def report_framing(per_image, out_dir):
    """
    Verifica el encuadre de los 50 recortes sin depender de mirarlos uno por uno.

    Mide la respuesta media al top-hat dental dentro del recorte, relativa a la
    mejor respuesta disponible en esa columna de la radiografía. Un valor bajo
    significa que el encuadre agarró poco diente -- se fue a hueso, a seno o
    fuera del maxilar --, que es exactamente el modo de fallo a vigilar.

    Contar píxeles oscuros no sirve: en estas radiografías el fondo ronda los 30-60
    niveles, no el negro, y un umbral absoluto marcaba 0.00 en recortes que a
    ojo estaban claramente mal encuadrados.
    """
    rows = []
    for image_id, d in per_image.items():
        x0, y0, x1, y1 = d['box']
        response = tooth_response(d['images']['original'], x0, x1)
        height = y1 - y0
        best = max(float(response[y:y + height].mean())
                   for y in range(0, len(response) - height, 8))
        rows.append({
            'image_id': image_id,
            'sextante': '_'.join(d['sextant']),
            'contenido_dental': round(float(response[y0:y1].mean()), 1),
            'relativo_al_mejor': round(float(response[y0:y1].mean()) / max(best, 1e-6), 3),
        })
    df = pd.DataFrame(rows).sort_values('relativo_al_mejor')
    df.to_csv(out_dir / 'encuadre.csv', index=False)

    print(f'\nEncuadre -- contenido dental relativo al mejor encuadre posible:')
    print(f'  mediana {df.relativo_al_mejor.median():.2f}, '
          f'p10 {df.relativo_al_mejor.quantile(0.1):.2f}, '
          f'mínimo {df.relativo_al_mejor.min():.2f}')
    print('  Peores 5 (revisar a ojo si bajan de ~0.85):')
    for _, r in df.head(5).iterrows():
        print(f'    img {r.image_id:>4}  {r.sextante:<28} {r.relativo_al_mejor:.2f}')


def _methods_of(data, cond):
    """Qué métodos MCDM eligieron esta condición (vacío si no es una selección)."""
    if not cond.startswith('sol:'):
        return []
    idx = int(cond.split(':')[1])
    members = data['classes'].get(cond, [cond])
    out = []
    for member in members:
        out.extend(data['voters'].get(int(member.split(':')[1]), []))
    return sorted(set(out)) or (['(no elegida)'] if idx not in data['voters'] else [])


def _delta_objectives(data, cond_a, cond_b):
    """
    ΔH, ΔSSIM y ΔVIF entre las dos imágenes: los predictores del análisis final.

    `original` y `degraded` no son soluciones del frente y por lo tanto no traen
    objetivos precomputados, pero se miden en el mismo sistema de coordenadas que
    el resto -- entropía propia, y SSIM y VIF contra la degradada, que es la
    entrada del optimizador -- así que se calculan al vuelo. Sin esto los deltas
    existirían sólo en la mitad de los ensayos, y la regresión que estima los
    pesos empíricos perdería justamente los pares de mayor contraste.
    """
    def obj(cond):
        if cond in ('original', 'degraded'):
            image = data['images'][cond]
            degraded = data['images']['degraded']
            return (calculate_entropy(image),
                    calculate_ssim(image, degraded),
                    calculate_vif(image, degraded))
        if cond.startswith('ext:'):
            return data['objectives'].get(data['extremes'][cond.split(':')[1]])
        return data['objectives'].get(int(cond.split(':')[1]))

    a, b = obj(cond_a), obj(cond_b)
    if a is None or b is None:
        return {'dH': None, 'dSSIM': None, 'dVIF': None}
    return {'dH': round(a[0] - b[0], 5), 'dSSIM': round(a[1] - b[1], 5),
            'dVIF': round(a[2] - b[2], 5)}


if __name__ == '__main__':
    main()
