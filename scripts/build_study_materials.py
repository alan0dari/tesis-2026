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
# Alto del recorte medido desde el plano oclusal, en fracción de la altura de la
# imagen: (arriba, abajo). Es asimétrico a propósito. Las raíces del maxilar
# suben hacia el piso del seno y las de la mandíbula bajan hacia el conducto, así
# que el lado del ápice necesita bastante más lugar que el lado de la corona.
SEXTANT_SPAN = {
    'superior': (0.27, 0.07),
    'inferior': (0.07, 0.35),
}
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


def occlusal_plane(image, x0, x1):
    """
    Plano oclusal dentro de una franja de columnas.

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

    Se ancla acá y no en el lóbulo de cada arcada porque el lóbulo del maxilar es
    inestable: el seno y el malar responden al top-hat tanto como los molares, y
    según la imagen el anclaje se iba al seno o caía sobre el propio oclusal. El
    mínimo de intensidad entre las dos arcadas, en cambio, se detecta parejo en
    las 50.
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

    return ya + occlusal


def tooth_response(image, x0, x1):
    """Respuesta al top-hat dental en una franja de columnas, para todo el alto."""
    strip = image[:, x0:x1]
    lo, hi = np.percentile(strip, [5, 99])
    norm = np.clip((strip.astype(np.float32) - lo) / max(hi - lo, 1), 0, 1)
    return cv2.morphologyEx((norm * 255).astype(np.uint8),
                            cv2.MORPH_TOPHAT, TOOTH_KERNEL).mean(axis=1)


def sextant_box(image, sextant):
    """
    Rectángulo del sextante, anclado en el plano oclusal.

    La versión anterior centraba una caja de alto fijo donde más respondía el
    top-hat dental. Eso optimiza en contra de lo que hace falta ver: el esmalte
    responde mucho más que el hueso periapical, así que la caja se centraba en
    las coronas y dejaba los ápices fuera de cuadro. Los odontólogos que
    revisaron el material lo marcaron sobre todo en los sectores posteriores.

    Ahora se ancla en el plano oclusal y se extiende hacia donde van las raíces
    de la arcada pedida, con los márgenes de SEXTANT_SPAN. Así el recorte incluye
    corona, raíz completa y el hueso de alrededor -- el piso del seno en los
    posteriores superiores, la zona del conducto en los inferiores.
    """
    region, arch = sextant
    h, w = image.shape
    x0, x1 = (int(f * w) for f in SEXTANT_X[region])

    arriba, abajo = SEXTANT_SPAN[arch]
    occlusal = occlusal_plane(image, x0, x1)
    y0, y1 = occlusal - int(arriba * h), occlusal + int(abajo * h)
    height = y1 - y0

    # Sin invadir las bandas de anotación quemada: se desplaza la caja entera en
    # vez de recortarla, para que el alto no cambie entre imágenes.
    top_limit, bottom_limit = int(SAFE_Y[0] * h), int(SAFE_Y[1] * h)
    if y0 < top_limit:
        y0, y1 = top_limit, top_limit + height
    if y1 > bottom_limit:
        y0, y1 = bottom_limit - height, bottom_limit
    return x0, max(0, y0), x1, min(h, y1)


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
    write_index(key, per_image, out_dir)

    print(f'\n  {out_dir / "trials.json"}')
    print(f'  {out_dir / "clave.csv"}   <- NO enviar a los evaluadores')
    print(f'  {out_dir / "img" / "INDICE.md"}   <- NO enviar a los evaluadores')
    print(f'\nSextantes: '
          + ', '.join(f'{k}={v}' for k, v in
                      pd.Series(['_'.join(d['sextant']) for d in per_image.values()])
                      .value_counts().sort_index().items()))
    return by_id


SECTOR_LARGO = {
    'posterior_derecho_superior': 'Posterior superior derecho — del paciente, o sea a la izquierda de la imagen (lado «R»)',
    'posterior_derecho_inferior': 'Posterior inferior derecho — del paciente, o sea a la izquierda de la imagen (lado «R»)',
    'anterior_superior': 'Anterior superior — incisivos y caninos del maxilar',
    'anterior_inferior': 'Anterior inferior — incisivos y caninos de la mandíbula',
    'posterior_izquierdo_superior': 'Posterior superior izquierdo — del paciente, o sea a la derecha de la imagen (lado «L»)',
    'posterior_izquierdo_inferior': 'Posterior inferior izquierdo — del paciente, o sea a la derecha de la imagen (lado «L»)',
}


def write_index(key, per_image, out_dir):
    """
    Índice navegable dentro de `img/`, para poder ir del recorte a su radiografía.

    Los recortes se llaman con un hash a propósito -- que el evaluador no pueda
    inferir la condición mirando el sitio --, y eso deja al equipo sin poder
    hacerlo tampoco. Esta tabla es la traducción, con enlaces relativos que
    GitHub y cualquier visor de Markdown resuelven solos.

    **Revela qué condición es cada recorte, así que es interno igual que
    `clave.csv`.** No viaja a la web: `build_web_study.py` copia sólo `*.png`.
    """
    img_dir = out_dir / 'img'

    # condición -> dónde se usa, por imagen
    usos = {}
    for r in key.itertuples():
        for cond, lado in ((r.cond_izquierda, 'izq'), (r.cond_derecha, 'der')):
            usos.setdefault((r.image_id, cond), []).append(f'{r.trial_id} ({r.block}, {lado})')

    lineas = [
        '# Índice de recortes',
        '',
        'Generado por `scripts/build_study_materials.py`. Cada recorte se llama',
        '`sha1("<id de la radiografía>|<condición>")` truncado a 10 caracteres, para que',
        'el evaluador no pueda inferir nada mirando el sitio.',
        '',
        '> **Interno, igual que `clave.csv`.** Dice qué condición experimental es cada',
        '> recorte: si un evaluador lo ve, se rompe el ciego. No se sube a la web —',
        '> `build_web_study.py` copia únicamente los `.png`.',
        '',
        'Los enlaces son relativos y funcionan desde GitHub o desde cualquier visor de',
        'Markdown, siempre que `data/` y `results/` estén en su lugar.',
        '',
        '## Por radiografía',
        '',
        '| Radiografía | Sector evaluado | Recortes usados en la evaluación |',
        '|---|---|---|',
    ]

    for image_id in sorted(per_image, key=lambda x: int(x)):
        d = per_image[image_id]
        sextante = '_'.join(d['sextant'])
        pano = f'../../../../data/evaluacion_50/{image_id}.jpg'
        conds = sorted({c for (i, c) in usos if i == image_id})
        recortes = '<br>'.join(
            f'[`{opaque(image_id, c)}.png`]({opaque(image_id, c)}.png) — `{c}`'
            for c in conds)
        lineas.append(f'| **[{image_id}]({pano})** | {sextante} | {recortes} |')

    lineas += [
        '',
        '## Por recorte',
        '',
        'Para el camino inverso: de un archivo suelto a su radiografía.',
        '',
        '| Recorte | Radiografía | Sector | Condición | Ensayos |',
        '|---|---|---|---|---|',
    ]

    filas = []
    for (image_id, cond), donde in usos.items():
        sextante = '_'.join(per_image[image_id]['sextant'])
        filas.append((f'{opaque(image_id, cond)}.png', image_id, sextante, cond,
                      ', '.join(sorted(donde))))
    for archivo, image_id, sextante, cond, donde in sorted(filas):
        pano = f'../../../../data/evaluacion_50/{image_id}.jpg'
        lineas.append(f'| [`{archivo}`]({archivo}) | [{image_id}]({pano}) '
                      f'| {sextante} | `{cond}` | {donde} |')

    lineas += ['', '## Los seis sectores', '',
               '| Clave | Qué es |', '|---|---|']
    for k, v in SECTOR_LARGO.items():
        lineas.append(f'| `{k}` | {v} |')
    lineas += ['',
               'Ojo con la lateralidad: **derecho e izquierdo son del paciente**, así que el',
               'sector posterior derecho cae a la izquierda de la imagen, donde está la',
               'marca «R».', '']

    (img_dir / 'INDICE.md').write_text('\n'.join(lineas), encoding='utf-8')


def report_framing(per_image, out_dir):
    """
    Verifica el encuadre de los 50 recortes sin depender de mirarlos uno por uno.

    Dos medidas. `relativo_al_mejor` es la respuesta media al top-hat dental
    dentro del recorte, contra la mejor disponible en esa columna: sirve para
    detectar el recorte que se fue del todo a hueso o fuera del maxilar. Ojo con
    interpretarla como antes: el encuadre nuevo incluye a propósito ápices y
    hueso periapical, que responden poco, así que los valores bajaron para todos
    y lo que importa es el orden relativo, no el valor absoluto.

    `oclusal_rel` es dónde cae el plano oclusal dentro del recorte, de 0 (borde
    superior) a 1 (inferior). Es el control directo del encuadre nuevo: tiene que
    dar cerca de 0.79 en los superiores y 0.17 en los inferiores. Un valor lejos
    de eso delata una detección de oclusal fallada.

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
        occlusal = occlusal_plane(d['images']['original'], x0, x1)
        rows.append({
            'image_id': image_id,
            'sextante': '_'.join(d['sextant']),
            'oclusal_rel': round((occlusal - y0) / height, 3),
            'contenido_dental': round(float(response[y0:y1].mean()), 1),
            'relativo_al_mejor': round(float(response[y0:y1].mean()) / max(best, 1e-6), 3),
        })
    df = pd.DataFrame(rows).sort_values('relativo_al_mejor')
    df.to_csv(out_dir / 'encuadre.csv', index=False)

    # El control que importa ahora: donde cae el plano oclusal dentro del
    # recorte. Si se aleja de lo que pide SEXTANT_SPAN es que fallo la deteccion
    # del oclusal, o que la caja topo contra la banda de anotacion y se corrio.
    print("\nEncuadre -- posicion del plano oclusal dentro del recorte:")
    for arch, (arriba, abajo) in SEXTANT_SPAN.items():
        g = df[df.sextante.str.endswith(arch)]
        esperado = arriba / (arriba + abajo)
        fuera = g[(g.oclusal_rel - esperado).abs() > 0.08]
        print(f"  {arch:<9} esperado {esperado:.2f}  "
              f"mediana {g.oclusal_rel.median():.2f}  fuera de rango: {len(fuera)}")
        for _, r in fuera.iterrows():
            print(f"    img {r.image_id:>4}  {r.sextante:<28} {r.oclusal_rel:.2f}")

    print("\nContenido dental relativo al mejor encuadre posible:")
    print(f"  mediana {df.relativo_al_mejor.median():.2f}, "
          f"p10 {df.relativo_al_mejor.quantile(0.1):.2f}, "
          f"minimo {df.relativo_al_mejor.min():.2f}")
    print("  Sirve para comparar entre recortes, no contra un umbral fijo: el "
          "encuadre incluye a proposito hueso, que responde poco al top-hat.")


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
