"""
Triaje de ortopantomografías para el estudio perceptual con odontólogos.

Reduce las 598 imágenes de `data/original` a una lista corta que se aproxime al
criterio del tutor -- dentición completa y sin trabajos dentales, o con la menor
modificación posible -- para que dos revisores humanos confirmen la selección
final de 50 sobre hojas de contacto.

QUÉ ES FIABLE Y QUÉ NO
----------------------
Este script es un PRE-FILTRO, no un clasificador. Se validó contra un puñado de
imágenes inspeccionadas a ojo:

  - `dentition` (top-hat morfológico sobre la banda de la arcada) ordena bien la
    completitud: edéntula 107 -> 86, casi completa 294 -> 159, completa 256 -> 168.
  - `metal_score` detecta con seguridad amalgamas y coronas (saturan muy por
    encima del esmalte) y, vía top-hat de núcleo pequeño, brackets y arcos de
    ortodoncia, que en imágenes subexpuestas NO cruzan un umbral absoluto:
    la primera versión con umbral fijo 240 dejó pasar 291 y 151, dos casos de
    ortodoncia completa. Aun así puede escapársele una obturación pequeña.

Detectar "todas las piezas presentes" con garantías exige un detector de dientes
entrenado; acá sólo se ordena por verosimilitud. La decisión es de los revisores.

Uso:
    # 1. Puntuar las 598 imágenes
    python scripts/screen_evaluation_set.py --data-dir data/original

    # 2. Hojas de contacto de las mejores candidatas para revisión humana
    python scripts/screen_evaluation_set.py --contact-sheets --top 120

    # 3. Materializar el conjunto final desde los IDs aprobados
    python scripts/screen_evaluation_set.py --build-set docs/evaluacion/ids_aprobados.txt
"""

import sys
import argparse
import shutil
from pathlib import Path

PROJECT_ROOT = str(Path(__file__).parent.parent)
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
import cv2

# Bandas de anotación quemada por el equipo (nombre, fecha, kV/mA, marcas R/L).
# Medidas sobre el dataset: filas 0-64 y 960-1024 de 1024 -> ~6% arriba y abajo.
TEXT_MARGIN_TOP = 0.07
TEXT_MARGIN_BOTTOM = 0.07
SIDE_MARGIN = 0.06

# Banda que contiene coronas y raíces de ambas arcadas.
ARCH_Y_MIN, ARCH_Y_MAX = 0.38, 0.75
ARCH_X_MIN, ARCH_X_MAX = 0.18, 0.82

# Amalgamas y coronas saturan; el esmalte sano de este dataset llega a ~205.
METAL_ABS_THRESHOLD = 240
MIN_BLOB_AREA = 40

# Núcleo del tamaño de una pieza: aísla dientes del hueso de fondo.
TOOTH_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (81, 81))
# Núcleo pequeño: aísla brackets, arcos y obturaciones del esmalte que los rodea.
APPLIANCE_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))

# Corte de `appliance_frac`. Calibrado sobre casos verificados a ojo:
#   256 limpia impecable   0.027
#   294 dos obturaciones   0.114
#   291 ortodoncia         1.361
#   151 ortodoncia         1.613
# Separa ortodoncia y restauración franca (descalificantes) de la dentición
# sana, dejando pasar el ruido de compresión. Las obturaciones pequeñas caen
# en la zona gris a propósito: las juzgan los revisores sobre la hoja.
APPLIANCE_FLAG = 0.30


def content_roi(image):
    """Recorta las bandas de texto quemado y los márgenes laterales."""
    h, w = image.shape
    return image[int(TEXT_MARGIN_TOP * h):int((1 - TEXT_MARGIN_BOTTOM) * h),
                 int(SIDE_MARGIN * w):int((1 - SIDE_MARGIN) * w)]


def arch_band(image):
    h, w = image.shape
    return image[int(ARCH_Y_MIN * h):int(ARCH_Y_MAX * h),
                 int(ARCH_X_MIN * w):int(ARCH_X_MAX * w)]


def _normalize(band):
    """Lleva la banda a contraste comparable entre estudios con distinta exposición."""
    lo, hi = np.percentile(band, [5, 99])
    return np.clip((band.astype(np.float32) - lo) / max(hi - lo, 1), 0, 1)


def measure_dentition(image):
    """
    Estima la completitud de la dentición.

    Un top-hat con núcleo del tamaño de una pieza deja sólo las estructuras
    compactas y brillantes -- los dientes -- sobre el fondo suave del hueso.
    La mediana del perfil por columna resume cuánta arcada tiene pieza encima:
    en una arcada edéntula el top-hat apenas responde.
    """
    band = _normalize(arch_band(image))
    tophat = cv2.morphologyEx((band * 255).astype(np.uint8),
                              cv2.MORPH_TOPHAT, TOOTH_KERNEL)
    profile = np.convolve(tophat.max(axis=0).astype(np.float32),
                          np.ones(25) / 25, mode='same')
    return {
        'dentition': float(np.percentile(profile, 50)),
        'dentition_p25': float(np.percentile(profile, 25)),
        'vstd': float(np.percentile(band.std(axis=0), 50)),
    }


def measure_radiopaque(image):
    """
    Cuantifica material radiopaco: restauraciones, coronas, endodoncias, ortodoncia.

    Dos detectores complementarios, porque ninguno solo alcanza:
      - absoluto: amalgamas y coronas saturan por encima de 240.
      - top-hat pequeño: brackets y arcos destacan sobre el esmalte vecino
        aunque la imagen esté subexpuesta y nunca lleguen a 240.
    """
    band = arch_band(image)

    mask = (band >= METAL_ABS_THRESHOLD).astype(np.uint8)
    n_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    areas = stats[1:, cv2.CC_STAT_AREA] if n_labels > 1 else np.array([])
    blobs = areas[areas >= MIN_BLOB_AREA]

    norm = _normalize(band)
    appliance = cv2.morphologyEx((norm * 255).astype(np.uint8),
                                 cv2.MORPH_TOPHAT, APPLIANCE_KERNEL)

    return {
        'metal_abs_frac': float(blobs.sum() / band.size) * 100 if len(blobs) else 0.0,
        'metal_abs_blobs': int(len(blobs)),
        'appliance_frac': float((appliance > 60).mean()) * 100,
        'appliance_peak': float(np.percentile(appliance, 99.9)),
    }


def measure_technical_quality(image):
    """La imagen fuente debe ser de buena calidad: la degradación la aplicamos nosotros."""
    roi = content_roi(image)
    hist = np.bincount(roi.ravel(), minlength=256).astype(float)
    hist /= hist.sum()
    nz = hist[hist > 0]
    return {
        'entropy': float(-(nz * np.log2(nz)).sum()),
        'contrast_std': float(roi.std()),
        'mean_intensity': float(roi.mean()),
        'clipped_high': float((roi >= 253).mean()) * 100,
    }


def screen_image(path):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None
    row = {'image_id': path.stem, 'file': path.name,
           'height': image.shape[0], 'width': image.shape[1]}
    row.update(measure_dentition(image))
    row.update(measure_radiopaque(image))
    row.update(measure_technical_quality(image))
    return row


def rank_candidates(df):
    """
    Marca las que llevan material radiopaco y ordena el resto por dentición.

    Ver `APPLIANCE_FLAG` para la calibración del corte. La marca no descarta:
    sólo baja la imagen en el orden de revisión.
    """
    df = df.copy()
    df['flag_metal'] = (df.metal_abs_blobs > 0) | (df.appliance_frac > APPLIANCE_FLAG)
    df = df.sort_values(['flag_metal', 'dentition'], ascending=[True, False])
    return df.reset_index(drop=True)


def make_contact_sheets(df, data_dir, out_dir, top, per_sheet=9, thumb_w=900):
    """Hojas de contacto con ID y puntajes sobreimpresos, para la revisión humana."""
    out_dir.mkdir(parents=True, exist_ok=True)
    subset = df.head(top)
    cols = 2
    thumb_h = thumb_w // 2  # las panorámicas son ~2:1
    rows = (per_sheet + cols - 1) // cols

    paths = []
    for start in range(0, len(subset), per_sheet):
        chunk = subset.iloc[start:start + per_sheet]
        canvas = np.zeros((rows * (thumb_h + 30), cols * thumb_w), np.uint8)

        for i, (_, r) in enumerate(chunk.iterrows()):
            img = cv2.imread(str(data_dir / r['file']), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            thumb = cv2.resize(content_roi(img), (thumb_w, thumb_h))
            gy, gx = divmod(i, cols)
            y0 = gy * (thumb_h + 30)
            canvas[y0:y0 + thumb_h, gx * thumb_w:(gx + 1) * thumb_w] = thumb
            label = (f"#{start + i + 1}  ID {r['image_id']}   "
                     f"denticion {r['dentition']:.0f}   "
                     f"metal {r['metal_abs_blobs']:.0f}/{r['appliance_frac']:.3f}"
                     f"{'  [MARCADA]' if r['flag_metal'] else ''}")
            cv2.putText(canvas, label, (gx * thumb_w + 8, y0 + thumb_h + 21),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, 255, 1, cv2.LINE_AA)

        path = out_dir / f"hoja_{start // per_sheet + 1:02d}.png"
        cv2.imwrite(str(path), canvas)
        paths.append(path)
        print(f'  {path}')
    return paths


def build_set(ids_file, data_dir, out_dir):
    """Copia a `out_dir` las imágenes aprobadas por los revisores."""
    ids = [ln.split('#')[0].strip() for ln in Path(ids_file).read_text().splitlines()]
    ids = [i for i in ids if i]
    out_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    for image_id in ids:
        matches = list(data_dir.glob(f'{image_id}.*'))
        if not matches:
            print(f'  [X] no encontrada: {image_id}')
            continue
        shutil.copy2(matches[0], out_dir / matches[0].name)
        copied += 1
    print(f'\n{copied}/{len(ids)} imágenes copiadas a {out_dir}')
    return copied


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data-dir', default='data/original')
    parser.add_argument('--out-dir', default='docs/evaluacion/screening')
    parser.add_argument('--contact-sheets', action='store_true')
    parser.add_argument('--top', type=int, default=120)
    parser.add_argument('--build-set', metavar='IDS_FILE')
    parser.add_argument('--set-dir', default='data/evaluacion_50')
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)

    if args.build_set:
        build_set(args.build_set, data_dir, Path(args.set_dir))
        return

    scores_path = out_dir / 'screening_scores.csv'

    if scores_path.exists() and args.contact_sheets:
        # Se reordena al vuelo: cambiar los umbrales no obliga a reanalizar.
        df = rank_candidates(pd.read_csv(scores_path, dtype={'image_id': str}))
        print(f'Puntajes cargados de {scores_path} ({(~df.flag_metal).sum()} sin marcar)')
    else:
        files = sorted(data_dir.glob('*.jpg')) + sorted(data_dir.glob('*.png'))
        print(f'Analizando {len(files)} imágenes de {data_dir}...')
        rows = []
        for i, f in enumerate(files, 1):
            if (row := screen_image(f)):
                rows.append(row)
            if i % 100 == 0:
                print(f'  {i}/{len(files)}')
        df = rank_candidates(pd.DataFrame(rows))
        out_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(scores_path, index=False)

        n_clean = (~df.flag_metal).sum()
        print(f'\nPuntajes -> {scores_path}')
        print(f'\nSin material radiopaco detectado: {n_clean}/{len(df)} '
              f'({n_clean / len(df) * 100:.1f}%)')
        print(f'  marcadas por umbral absoluto (amalgama/corona): '
              f'{(df.metal_abs_blobs > 0).sum()}')
        print(f'  marcadas por top-hat pequeño (ortodoncia/obturación): '
              f'{(df.appliance_frac > 0.02).sum()}')
        print(f'\nDentición -- mediana {df.dentition.median():.0f}, '
              f'p90 {df.dentition.quantile(0.9):.0f}, máx {df.dentition.max():.0f}')
        print(f'  (referencia validada a ojo: edéntula 107={df.loc[df.image_id == "107", "dentition"].iloc[0]:.0f}, '
              f'completa 256={df.loc[df.image_id == "256", "dentition"].iloc[0]:.0f})')
        print('\nTop 25 candidatas (sin metal, mejor dentición):')
        print(df.head(25)[['image_id', 'dentition', 'appliance_frac',
                           'metal_abs_blobs', 'entropy']].to_string(index=False))

    if args.contact_sheets:
        print(f'\nGenerando hojas de contacto (top {args.top})...')
        make_contact_sheets(df, data_dir, out_dir / 'hojas_contacto', args.top)
        print('\nRevisar las hojas, anotar los IDs aprobados en un .txt (uno por línea),'
              '\ny luego: python scripts/screen_evaluation_set.py --build-set <archivo>')


if __name__ == '__main__':
    main()
