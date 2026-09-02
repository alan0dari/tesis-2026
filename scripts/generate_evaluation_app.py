"""
Fase 3: genera la aplicación web que usan los odontólogos.

Produce un HTML por evaluador dentro de `docs/evaluacion/estudio/`, que comparten
la carpeta `img/`. Se abren con doble clic, funcionan sin conexión y sin servidor,
y al terminar descargan un JSON con las respuestas.

Decisiones de diseño que vienen del protocolo (§4 y §6):

  - Sin rótulos, leyendas ni nombres de método. Los identificadores de ensayo son
    hashes: ni siquiera mirando el código fuente se infiere qué condición es cuál.
    La correspondencia vive sólo en `clave.csv`, que se queda con el equipo.
  - Izquierda/derecha ya vienen sorteadas por `build_study_materials.py`; acá se
    sortea además el orden de los ensayos, distinto para cada evaluador.
  - Zoom y desplazamiento sincronizados: mover una imagen mueve la otra. Sin eso
    la comparación de detalle fino es imposible, y es justamente lo que se pide.
  - No se puede volver atrás. Corregir respuestas anteriores introduce sesgo de
    consistencia: el evaluador tiende a alinear lo nuevo con lo ya contestado.
  - Se guarda el progreso en `localStorage`: la sesión se puede cortar y retomar.
  - Se registra el tiempo por ensayo, que sirve para detectar el punto donde el
    evaluador empieza a responder por responder.

Uso:
    python scripts/generate_evaluation_app.py
"""

import sys
import json
import argparse
from pathlib import Path

PROJECT_ROOT = str(Path(__file__).parent.parent)
sys.path.insert(0, PROJECT_ROOT)

import numpy as np

N_WARMUP = 5   # de calentamiento, se descartan en el análisis
N_REPEAT = 5   # repetidos al final, miden consistencia intra-evaluador


PAGE = """<!doctype html>
<html lang="es">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Evaluación de radiografías — Evaluador __RATER__</title>
<style>
  :root { --bg:#111; --panel:#1c1c1c; --ink:#eee; --dim:#999; --accent:#4a9eff; }
  * { box-sizing: border-box; }
  body { margin:0; background:var(--bg); color:var(--ink);
         font-family: system-ui, -apple-system, "Segoe UI", sans-serif; }
  header { padding:10px 18px; background:var(--panel); border-bottom:1px solid #333;
           display:flex; align-items:center; gap:18px; position:sticky; top:0; z-index:10; }
  #bar { flex:1; height:8px; background:#333; border-radius:4px; overflow:hidden; }
  #fill { height:100%; width:0%; background:var(--accent); transition:width .3s; }
  #count { font-variant-numeric: tabular-nums; color:var(--dim); font-size:14px;
           white-space:nowrap; }
  main { max-width:1500px; margin:0 auto; padding:16px; }
  .pair { display:grid; grid-template-columns:1fr 1fr; gap:14px; }
  .frame { position:relative; overflow:hidden; background:#000; border-radius:6px;
           aspect-ratio: 450 / 297; cursor:grab; }
  .frame.drag { cursor:grabbing; }
  .frame img { position:absolute; left:0; top:0; width:100%; height:100%;
               transform-origin:0 0; image-rendering:auto; user-select:none;
               -webkit-user-drag:none; }
  .tag { position:absolute; left:10px; top:8px; font-size:15px; font-weight:600;
         color:#fff; background:rgba(0,0,0,.55); padding:2px 10px; border-radius:4px;
         pointer-events:none; }
  .q { margin-top:18px; text-align:center; }
  .q p { margin:0 0 10px; font-size:17px; }
  .btns { display:flex; gap:10px; justify-content:center; flex-wrap:wrap; }
  button { background:#262626; color:var(--ink); border:1px solid #3a3a3a;
           padding:12px 26px; border-radius:6px; font-size:15px; cursor:pointer; }
  button:hover { background:#333; border-color:var(--accent); }
  button kbd { color:var(--dim); font-size:12px; margin-left:8px; }
  .hint { color:var(--dim); font-size:13px; margin-top:14px; text-align:center; }
  .card { max-width:760px; margin:6vh auto; background:var(--panel); padding:30px 34px;
          border-radius:10px; line-height:1.65; }
  .card h1 { margin-top:0; font-size:22px; }
  .card li { margin-bottom:8px; }
  .card button { margin-top:18px; background:var(--accent); border:none; color:#fff;
                 font-size:16px; padding:13px 30px; }
  .hidden { display:none; }
  .warm { color:#ffb454; font-size:13px; text-align:center; margin-bottom:8px; }
</style>
</head>
<body>

<div id="intro" class="card">
  <h1>Evaluación perceptual de radiografías panorámicas</h1>
  <p>Gracias por participar. Va a ver <b>pares de recortes</b> de una misma
     radiografía, mostrando la misma región en ambos. Cambia sólo el
     procesamiento de la imagen.</p>
  <p>En cada pantalla elija <b>en cuál de las dos evaluaría esa región con más
     confianza</b> para un diagnóstico.</p>
  <ul>
    <li>No hay respuesta correcta: interesa su criterio clínico.</li>
    <li><b>Muchos pares van a parecerle iguales o casi iguales.</b> Es esperable y
        forma parte de lo que se quiere medir. Elija igual la que prefiera y
        marque <i>“Estoy adivinando”</i>.</li>
    <li>Puede <b>acercar con la rueda del mouse</b> y <b>arrastrar</b> para
        desplazarse. Las dos imágenes se mueven juntas.</li>
    <li>No se puede volver atrás.</li>
    <li>Puede cerrar y retomar después: el progreso queda guardado.</li>
    <li>Son unas <b>__TOTAL__ pantallas</b>, entre 25 y 30 minutos. Tómese un
        descanso si lo necesita.</li>
  </ul>
  <p>Los primeros __WARMUP__ pares son de práctica y no se analizan.</p>
  <button onclick="start()">Comenzar</button>
</div>

<header id="hud" class="hidden">
  <strong>Evaluador __RATER__</strong>
  <div id="bar"><div id="fill"></div></div>
  <div id="count"></div>
</header>

<main id="task" class="hidden">
  <div id="warm" class="warm hidden">Par de práctica — no se analiza</div>
  <div class="pair">
    <div class="frame" id="fa"><img id="ia" alt=""><span class="tag">A</span></div>
    <div class="frame" id="fb"><img id="ib" alt=""><span class="tag">B</span></div>
  </div>

  <div class="q" id="q1">
    <p>¿En cuál de las dos evaluaría esta región con más confianza?</p>
    <div class="btns">
      <button onclick="choose('a')">Imagen A <kbd>A</kbd></button>
      <button onclick="choose('b')">Imagen B <kbd>B</kbd></button>
    </div>
  </div>

  <div class="q hidden" id="q2">
    <p>¿Qué tan seguro está de esa elección?</p>
    <div class="btns">
      <button onclick="confidence('seguro')">Seguro <kbd>1</kbd></button>
      <button onclick="confidence('algo')">Algo seguro <kbd>2</kbd></button>
      <button onclick="confidence('adivinando')">Estoy adivinando <kbd>3</kbd></button>
    </div>
  </div>

  <div class="hint">Rueda del mouse para acercar · arrastrar para desplazar ·
       doble clic para reiniciar la vista</div>
</main>

<div id="done" class="card hidden">
  <h1>Listo, muchas gracias</h1>
  <p>Se descargó el archivo con sus respuestas. Si la descarga no arrancó,
     use el botón.</p>
  <button onclick="save()">Descargar respuestas</button>
</div>

<script>
const RATER = "__RATER__";
const TRIALS = __TRIALS__;
const KEY = "eval_odont_" + RATER;

let i = 0, pick = null, t0 = 0, answers = [];
let zoom = 1, panX = 0, panY = 0;

const $ = id => document.getElementById(id);

// Se guarda en cada respuesta para poder cortar y retomar la sesión
try {
  const saved = JSON.parse(localStorage.getItem(KEY) || "null");
  if (saved && saved.answers) { answers = saved.answers; i = answers.length; }
} catch (e) {}

function start() {
  $("intro").classList.add("hidden");
  $("hud").classList.remove("hidden");
  $("task").classList.remove("hidden");
  show();
}

function show() {
  if (i >= TRIALS.length) return finish();
  const t = TRIALS[i];
  resetView();
  $("ia").src = "img/" + t.a;
  $("ib").src = "img/" + t.b;
  $("warm").classList.toggle("hidden", !t.warmup);
  $("q1").classList.remove("hidden");
  $("q2").classList.add("hidden");
  $("fill").style.width = (100 * i / TRIALS.length) + "%";
  $("count").textContent = (i + 1) + " / " + TRIALS.length;
  pick = null;
  t0 = performance.now();
}

function choose(side) {
  pick = side;
  $("q1").classList.add("hidden");
  $("q2").classList.remove("hidden");
}

function confidence(level) {
  const t = TRIALS[i];
  answers.push({
    trial_id: t.id, eleccion: pick, confianza: level,
    ms: Math.round(performance.now() - t0),
    warmup: !!t.warmup, repeticion: !!t.repeat, orden: i + 1
  });
  localStorage.setItem(KEY, JSON.stringify({ answers }));
  i++;
  show();
}

function finish() {
  $("hud").classList.add("hidden");
  $("task").classList.add("hidden");
  $("done").classList.remove("hidden");
  save();
}

function save() {
  const blob = new Blob(
    [JSON.stringify({ evaluador: RATER, fecha: new Date().toISOString(),
                      respuestas: answers }, null, 1)],
    { type: "application/json" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "respuestas_evaluador_" + RATER + ".json";
  a.click();
}

// --- zoom y desplazamiento, sincronizados entre las dos imágenes ---
function applyView() {
  const t = `translate(${panX}px, ${panY}px) scale(${zoom})`;
  $("ia").style.transform = t;
  $("ib").style.transform = t;
}
function resetView() { zoom = 1; panX = 0; panY = 0; applyView(); }

for (const id of ["fa", "fb"]) {
  const el = $(id);
  el.addEventListener("wheel", e => {
    e.preventDefault();
    const r = el.getBoundingClientRect();
    const mx = e.clientX - r.left, my = e.clientY - r.top;
    const next = Math.min(8, Math.max(1, zoom * (e.deltaY < 0 ? 1.15 : 1 / 1.15)));
    // Acercar hacia el puntero: el punto bajo el cursor se queda quieto
    panX = mx - (mx - panX) * (next / zoom);
    panY = my - (my - panY) * (next / zoom);
    zoom = next;
    if (zoom === 1) { panX = 0; panY = 0; }
    applyView();
  }, { passive: false });

  let dragging = false, sx = 0, sy = 0;
  el.addEventListener("mousedown", e => {
    dragging = true; sx = e.clientX - panX; sy = e.clientY - panY;
    el.classList.add("drag");
  });
  window.addEventListener("mousemove", e => {
    if (!dragging) return;
    panX = e.clientX - sx; panY = e.clientY - sy; applyView();
  });
  window.addEventListener("mouseup", () => {
    dragging = false; el.classList.remove("drag");
  });
  el.addEventListener("dblclick", resetView);
}

document.addEventListener("keydown", e => {
  if ($("task").classList.contains("hidden")) return;
  const k = e.key.toLowerCase();
  if (!$("q1").classList.contains("hidden")) {
    if (k === "a" || k === "arrowleft") choose("a");
    if (k === "b" || k === "arrowright") choose("b");
  } else if (!$("q2").classList.contains("hidden")) {
    if (k === "1") confidence("seguro");
    if (k === "2") confidence("algo");
    if (k === "3") confidence("adivinando");
  }
});
</script>
</body>
</html>
"""


def build_sequence(rater, assigned, all_ids, trials, rng):
    """
    Secuencia del evaluador: calentamiento, ensayos propios y repeticiones.

    El calentamiento sale de ensayos que NO le tocaron, así no contamina sus
    datos ni le adelanta pares que después va a juzgar en serio.
    """
    others = [t for t in all_ids if t not in set(assigned)]
    warmup = list(rng.choice(others, size=min(N_WARMUP, len(others)), replace=False))

    main = list(assigned)
    rng.shuffle(main)
    repeats = list(rng.choice(main, size=min(N_REPEAT, len(main)), replace=False))

    seq = []
    for tid in warmup:
        seq.append({'id': tid, 'warmup': True, **_files(trials, tid)})
    for tid in main:
        seq.append({'id': tid, **_files(trials, tid)})
    for tid in repeats:
        seq.append({'id': tid, 'repeat': True, **_files(trials, tid)})
    return seq


def _files(trials, tid):
    return {'a': trials[tid]['file_a'], 'b': trials[tid]['file_b']}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--study-dir', default='docs/evaluacion/estudio')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    study = Path(args.study_dir)
    data = json.loads((study / 'trials.json').read_text(encoding='utf-8'))
    trials, assignment = data['trials'], data['assignment']
    all_ids = sorted(trials)

    print(f'Ensayos: {len(all_ids)}   Evaluadores: {len(assignment)}')
    for rater, assigned in sorted(assignment.items()):
        rng = np.random.default_rng(args.seed + int(rater))
        seq = build_sequence(rater, assigned, all_ids, trials, rng)
        html = (PAGE
                .replace('__RATER__', rater)
                .replace('__TOTAL__', str(len(seq)))
                .replace('__WARMUP__', str(N_WARMUP))
                .replace('__TRIALS__', json.dumps(seq, separators=(',', ':'))))
        path = study / f'evaluador_{rater}.html'
        path.write_text(html, encoding='utf-8')
        print(f'  {path.name}: {len(seq)} pantallas '
              f'({N_WARMUP} práctica + {len(assigned)} + {N_REPEAT} repetidos)')

    print(f'\nEnviar a cada odontólogo la carpeta {study} completa '
          f'(con img/) e indicarle qué archivo abrir.')
    print(f'NO enviar clave.csv ni encuadre.csv.')


if __name__ == '__main__':
    main()
