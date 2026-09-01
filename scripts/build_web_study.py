"""
Fase 3-bis: materiales estáticos del sitio web de evaluación.

Toma lo que ya produjo `build_study_materials.py` y deja en `webapp/public/`
todo lo que hay que subir al hosting:

  - `assets/sets.json`: los 8 sets de ensayos y el mapa ensayo -> par de archivos.
  - `img/`: los 193 recortes, copiados **sin recomprimir**. Van en PNG sin
    pérdida a propósito (ver `docs/evaluacion/estudio/LEEME.md`): recomprimir con
    pérdida metería artefactos justo en un estudio que mide percepción de calidad.
  - `assets/concepto/`: las figuras de la sección informativa, reescaladas para web.

No decide nada del diseño experimental. El reparto de ensayos entre evaluadores
sale tal cual de `trials.json`. Las dos diferencias con la app offline son:

  1. No hay ensayos de práctica. El sitio arranca directo con los 65 propios.
  2. El orden de los ensayos y los 5 repetidos del final los arma el navegador
     con una semilla que le da la base de datos, distinta por participante. Así
     dos personas que reciben el mismo set (el 9no participante repite el set 1)
     no ven la misma secuencia.

Uso:
    .venv/Scripts/python scripts/build_web_study.py
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

N_REPETIDOS = 5          # repetidos al final, miden consistencia intra-evaluador
ANCHO_WEB = 1400         # ancho máximo de las figuras conceptuales, en px
ILUSTRACION = '517.jpg'  # panorámica de ejemplo: adulta, fuera de las 50 del estudio


# sets.json

def escribir_sets(estudio: Path, destino: Path) -> dict:
    """
    Vuelca `trials.json` al formato que consume el navegador.

    Se queda solo con lo que el cliente necesita: qué ensayos le tocan a cada
    set y qué dos archivos muestra cada ensayo. La condición experimental de
    cada lado no viaja nunca al navegador; vive solo en `clave.csv`.
    """
    datos = json.loads((estudio / 'trials.json').read_text(encoding='utf-8'))
    ensayos, asignacion = datos['trials'], datos['assignment']

    sets = {rater: sorted(ids) for rater, ids in sorted(asignacion.items())}
    usados = {tid for ids in sets.values() for tid in ids}
    mapa = {tid: [ensayos[tid]['file_a'], ensayos[tid]['file_b']]
            for tid in sorted(usados)}

    salida = {
        'generado_por': 'scripts/build_web_study.py',
        'n_repetidos': N_REPETIDOS,
        'sets': sets,
        'ensayos': mapa,
    }
    (destino / 'assets' / 'sets.json').write_text(
        json.dumps(salida, ensure_ascii=False, separators=(',', ':')),
        encoding='utf-8')

    return {'sets': len(sets), 'ensayos': len(mapa),
            'por_set': len(next(iter(sets.values()))) + N_REPETIDOS}


def copiar_recortes(estudio: Path, destino: Path) -> int:
    """Copia los PNG tal cual. Se saltea los que ya están y coinciden en tamaño."""
    origen = estudio / 'img'
    carpeta = destino / 'img'
    carpeta.mkdir(parents=True, exist_ok=True)

    copiados = 0
    for png in sorted(origen.glob('*.png')):
        meta = carpeta / png.name
        if meta.exists() and meta.stat().st_size == png.stat().st_size:
            continue
        shutil.copy2(png, meta)
        copiados += 1
    return copiados


# figuras de la sección informativa

def _fuente(tam: int):
    """TrueType si hay alguna a mano; si no, la bitmap de PIL."""
    from PIL import ImageFont

    for ruta in ('C:/Windows/Fonts/segoeui.ttf', 'C:/Windows/Fonts/arial.ttf',
                 '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
                 '/System/Library/Fonts/Helvetica.ttc'):
        if Path(ruta).exists():
            return ImageFont.truetype(ruta, tam)
    try:
        return ImageFont.load_default(size=tam)
    except TypeError:
        return ImageFont.load_default()


def figura_degradaciones(destino: Path) -> str | None:
    """
    Reescala la figura de degradaciones del capítulo 4 para la web.

    Es la única figura con radiografías de la sección informativa: enseña qué le
    pasa a una imagen mal expuesta, sin enseñar cómo se ve una procesada.
    Mostrar ejemplos procesados sesgaría el juicio.
    """
    from PIL import Image

    origen = PROJECT_ROOT / 'docs/libro/Figures/capitulo4/degradaciones.png'
    if not origen.exists():
        return None

    img = Image.open(origen).convert('RGB')
    if img.width > ANCHO_WEB:
        alto = round(img.height * ANCHO_WEB / img.width)
        img = img.resize((ANCHO_WEB, alto), Image.LANCZOS)

    salida = destino / 'assets' / 'concepto' / 'degradaciones.jpg'
    img.save(salida, 'JPEG', quality=82, optimize=True, progressive=True)
    return salida.name


def _elegir_ilustracion(aprobados: set[str], preferida: str) -> str | None:
    """
    Panorámica para la figura de sextantes.

    Tiene que ser una que NO esté en las 50: mostrar una del estudio sería
    enseñarle al participante, antes de empezar, una imagen que después va a
    juzgar. `preferida` está elegida a ojo (dentición permanente completa, buena
    exposición); si no está, cae a la mejor puntuada del triaje que quede fuera
    de las 50, que puede tocar una dentición mixta pediátrica.
    """
    if preferida and (PROJECT_ROOT / 'data/original' / preferida).exists():
        if Path(preferida).stem not in aprobados:
            return preferida

    csv_path = PROJECT_ROOT / 'docs/evaluacion/screening/screening_scores.csv'
    if not csv_path.exists():
        return None

    with csv_path.open(encoding='utf-8') as fh:
        for fila in csv.DictReader(fh):
            if fila['image_id'] in aprobados:
                continue
            if fila['flag_metal'].strip().lower() == 'true':
                continue
            if (PROJECT_ROOT / 'data/original' / fila['file']).exists():
                return fila['file']
    return None


def figura_sextantes(destino: Path, aprobados: set[str],
                     preferida: str = ILUSTRACION) -> str | None:
    """
    Arma la figura que explica por qué se muestra un recorte y no la panorámica.

    Arriba la panorámica completa con los seis sextantes marcados y velo sobre
    los cinco que no interesan; abajo el sextante resaltado, al tamaño al que se
    ve en la evaluación.
    """
    from PIL import Image, ImageDraw

    archivo = _elegir_ilustracion(aprobados, preferida)
    if archivo is None:
        return None

    limpio = Image.open(PROJECT_ROOT / 'data/original' / archivo).convert('RGB')
    escala = ANCHO_WEB / limpio.width
    limpio = limpio.resize((ANCHO_WEB, round(limpio.height * escala)), Image.LANCZOS)

    # La banda de la arcada: la panorámica trae texto quemado arriba y abajo.
    x0, x1 = int(limpio.width * 0.06), int(limpio.width * 0.94)
    y0, y1 = int(limpio.height * 0.24), int(limpio.height * 0.81)
    ancho_celda, alto_celda = (x1 - x0) // 3, (y1 - y0) // 2

    etiquetas = [['Post. derecho sup.', 'Anterior sup.', 'Post. izquierdo sup.'],
                 ['Post. derecho inf.', 'Anterior inf.', 'Post. izquierdo inf.']]
    resaltado = (1, 1)                       # fila, columna: anterior inferior
    rx0 = x0 + resaltado[1] * ancho_celda
    ry0 = y0 + resaltado[0] * alto_celda
    caja_res = (rx0, ry0, rx0 + ancho_celda, ry0 + alto_celda)

    # Velo sobre todo menos el sextante que se explica.
    velo = Image.new('RGBA', limpio.size, (5, 9, 13, 120))
    marco = Image.alpha_composite(limpio.convert('RGBA'), velo).convert('RGB')
    marco.paste(limpio.crop(caja_res), (rx0, ry0))

    dib = ImageDraw.Draw(marco, 'RGBA')
    fuente = _fuente(19)

    for fila in range(2):
        for col in range(3):
            cx0 = x0 + col * ancho_celda
            cy0 = y0 + fila * alto_celda
            caja = (cx0, cy0, cx0 + ancho_celda, cy0 + alto_celda)
            activo = (fila, col) == resaltado
            color = (34, 197, 199, 255) if activo else (255, 255, 255, 120)
            dib.rectangle(caja, outline=color, width=4 if activo else 2)

            texto = etiquetas[fila][col]
            ancho = dib.textlength(texto, font=fuente)
            dib.rectangle((cx0 + 6, cy0 + 6, cx0 + 18 + ancho, cy0 + 36),
                          fill=(5, 9, 13, 170))
            dib.text((cx0 + 12, cy0 + 9), texto, font=fuente,
                     fill=(90, 226, 230, 255) if activo else (255, 255, 255, 190))

    # El recorte, sin las líneas de la grilla encima.
    recorte = limpio.crop(caja_res)
    ancho_rec = int(ANCHO_WEB * 0.52)
    recorte = recorte.resize(
        (ancho_rec, round(recorte.height * ancho_rec / recorte.width)), Image.LANCZOS)

    margen, hueco = 28, 54
    lienzo = Image.new(
        'RGB',
        (marco.width + 2 * margen,
         marco.height + hueco + recorte.height + 2 * margen + 34),
        (16, 20, 24))
    lienzo.paste(marco, (margen, margen))
    px = (lienzo.width - recorte.width) // 2
    py = margen + marco.height + hueco
    lienzo.paste(recorte, (px, py))

    dib = ImageDraw.Draw(lienzo)
    dib.rectangle((px - 2, py - 2, px + recorte.width + 1, py + recorte.height + 1),
                  outline=(34, 197, 199), width=3)
    pie = _fuente(21)
    texto = 'Lo que ves en cada pantalla: un sextante, al tamaño en que se puede juzgar'
    ancho_texto = dib.textlength(texto, font=pie)
    dib.text(((lienzo.width - ancho_texto) / 2, py + recorte.height + 12),
             texto, font=pie, fill=(150, 165, 175))

    salida = destino / 'assets' / 'concepto' / 'sextantes.jpg'
    lienzo.save(salida, 'JPEG', quality=84, optimize=True, progressive=True)
    return salida.name


def leer_aprobados() -> set[str]:
    ruta = PROJECT_ROOT / 'docs/evaluacion/ids_aprobados.txt'
    if not ruta.exists():
        return set()
    return {ln.strip() for ln in ruta.read_text(encoding='utf-8').splitlines()
            if ln.strip() and not ln.startswith('#')}



def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--study-dir', default='docs/evaluacion/estudio')
    parser.add_argument('--destino', default='webapp/public')
    parser.add_argument('--sin-figuras', action='store_true',
                        help='no regenerar las figuras de la sección informativa')
    args = parser.parse_args()

    estudio = PROJECT_ROOT / args.study_dir
    destino = PROJECT_ROOT / args.destino
    if not (estudio / 'trials.json').exists():
        print(f'No encuentro {estudio / "trials.json"}. '
              f'Corre antes scripts/build_study_materials.py', file=sys.stderr)
        return 1

    (destino / 'assets' / 'concepto').mkdir(parents=True, exist_ok=True)

    resumen = escribir_sets(estudio, destino)
    print(f'sets.json: {resumen["sets"]} sets x {resumen["por_set"]} pantallas '
          f'({resumen["por_set"] - N_REPETIDOS} propios + {N_REPETIDOS} repetidos), '
          f'{resumen["ensayos"]} ensayos, {resumen["ensayos"] * 2} referencias a imagen')

    copiados = copiar_recortes(estudio, destino)
    total = len(list((destino / 'img').glob('*.png')))
    peso = sum(p.stat().st_size for p in (destino / 'img').glob('*.png')) / 1e6
    print(f'img/: {total} recortes ({peso:.1f} MB), {copiados} copiados esta vez')

    if not args.sin_figuras:
        aprobados = leer_aprobados()
        for nombre in (figura_degradaciones(destino),
                       figura_sextantes(destino, aprobados)):
            print(f'concepto/: {nombre}' if nombre
                  else 'concepto/: figura omitida (falta el material de origen)')

    print(f'\nListo. Subir la carpeta {args.destino} completa al hosting.')
    print('Antes de subir: completar assets/config.js con la URL y la clave anon.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
