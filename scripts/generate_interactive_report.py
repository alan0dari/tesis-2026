"""
Genera un reporte HTML interactivo y autocontenido para un experimento.

El reporte (report.html en el directorio del experimento) permite:
  - Elegir cualquier imagen procesada del experimento.
  - Explorar su Frente de Pareto 3D: rotar/zoom, activar/desactivar puntos,
    superficie, selecciones de cada método MCDM y consenso (vía leyenda),
    colorear los puntos por métrica o por parámetro CLAHE, y ocultar ejes.
  - Ver el trío original/degradada/mejorada (referenciadas por ruta relativa).
  - Ver la matriz de decisión completa con las selecciones de cada método
    y la solución de consenso resaltadas.
  - Ver el resumen agregado del experimento (figuras y estadísticas).

plotly.js va incrustado, por lo que el archivo funciona sin conexión;
las imágenes se referencian por ruta relativa dentro del experimento.

Uso:
    python scripts/generate_interactive_report.py [--experiment results/experiment_X]
"""

import sys
import json
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

METHODS = ['SMARTER', 'TOPSIS', 'BellmanZadeh', 'PROMETHEEII',
           'GRA', 'VIKOR', 'CODAS', 'MABAC']
METHOD_LABELS = {'SMARTER': 'SMARTER', 'TOPSIS': 'TOPSIS',
                 'BellmanZadeh': 'Bellman-Zadeh', 'PROMETHEEII': 'PROMETHEE II',
                 'GRA': 'GRA', 'VIKOR': 'VIKOR', 'CODAS': 'CODAS', 'MABAC': 'MABAC'}
DEGRADATION_LABELS = {
    'low_contrast': 'Bajo contraste', 'underexposure': 'Subexposición',
    'overexposure': 'Sobreexposición', 'poor_local_contrast': 'Bajo contraste local',
    'skewed_histogram': 'Histograma sesgado',
}


def find_latest_experiment(results_dir: Path) -> Path:
    candidates = sorted(results_dir.glob('experiment_*'))
    candidates = [c for c in candidates if (c / 'experiment_data.csv').exists()]
    if not candidates:
        raise FileNotFoundError(f"No hay experimentos en {results_dir}")
    return candidates[-1]


def collect_payload(exp_dir: Path) -> dict:
    """Reúne los datos de todas las imágenes en un JSON para el reporte."""
    payload = {'images': {}, 'order': []}
    images_dir = exp_dir / 'images'
    if not images_dir.exists():
        return payload

    for img_dir in sorted(images_dir.iterdir()):
        rj = img_dir / 'result.json'
        pc = img_dir / 'pareto.csv'
        if not (rj.exists() and pc.exists()):
            continue
        result = json.loads(rj.read_text(encoding='utf-8'))
        if result.get('status') != 'success':
            continue
        df = pd.read_csv(pc)

        mcdm = {}
        for name in METHODS:
            info = result.get('mcdm_results', {}).get(name, {})
            if 'best_index' in info:
                mcdm[name] = {
                    'best': info['best_index'],
                    'order': info.get('ranking_order', []),
                }

        image_id = result['image_id']
        payload['images'][image_id] = {
            'degradation': DEGRADATION_LABELS.get(result.get('degradation_type', ''),
                                                  result.get('degradation_type', '')),
            'time': round(result.get('processing_time', 0), 1),
            'pareto': {
                'rx': df['param_0'].round(0).tolist(),
                'ry': df['param_1'].round(0).tolist(),
                'clip': df['param_2'].round(3).tolist(),
                'H': df['objective_0'].round(4).tolist(),
                'SSIM': df['objective_1'].round(4).tolist(),
                'VIF': df['objective_2'].round(4).tolist(),
            },
            'mcdm': mcdm,
            'consensus': result.get('consensus', {}),
            'validation': {k: round(v, 4) for k, v in
                           result.get('validation_vs_original', {}).items()},
        }
        payload['order'].append(image_id)
    return payload


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="utf-8">
<title>Experimento SMPSO-CLAHE + MCDM</title>
<script>__PLOTLYJS__</script>
<style>
  :root { --teal: #20606e; --gold: #c8a415; --bg: #fafaf8; }
  body { font-family: 'Segoe UI', system-ui, sans-serif; margin: 0; background: var(--bg); color: #222; }
  header { background: var(--teal); color: white; padding: 14px 28px; }
  header h1 { margin: 0; font-size: 20px; font-weight: 600; }
  header p { margin: 4px 0 0; font-size: 13px; opacity: 0.85; }
  .wrap { max-width: 1400px; margin: 0 auto; padding: 18px 28px; }
  .controls { display: flex; gap: 18px; align-items: center; flex-wrap: wrap;
              background: white; border: 1px solid #ddd; border-radius: 8px; padding: 12px 16px; margin-bottom: 16px; }
  .controls label { font-size: 13px; font-weight: 600; }
  select { font-size: 14px; padding: 4px 8px; }
  .meta { font-size: 13px; color: #555; }
  .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
  .card { background: white; border: 1px solid #ddd; border-radius: 8px; padding: 14px; }
  .card h2 { margin: 0 0 10px; font-size: 15px; color: var(--teal); }
  .trio img { width: 100%; border: 1px solid #ccc; border-radius: 4px; }
  .trio figure { margin: 0 0 10px; }
  .trio figcaption { font-size: 12px; color: #555; margin-top: 2px; }
  table { border-collapse: collapse; font-size: 12px; width: 100%; }
  th, td { border: 1px solid #ddd; padding: 3px 7px; text-align: center; }
  th { background: var(--teal); color: white; position: sticky; top: 0; }
  tr.consensus { background: #f6ecc4; font-weight: 600; }
  tr.picked { background: #e7f0f2; }
  .badge { display: inline-block; font-size: 10px; padding: 1px 5px; border-radius: 8px;
           background: var(--teal); color: white; margin: 1px; }
  .badge.gold { background: var(--gold); }
  .scroll { max-height: 480px; overflow-y: auto; }
  .check { font-size: 13px; margin-right: 12px; user-select: none; }
  #pareto { width: 100%; height: 560px; }
  .full { grid-column: 1 / -1; }
  .aggfigs { display: grid; grid-template-columns: repeat(auto-fit, minmax(380px, 1fr)); gap: 12px; }
  .aggfigs img { width: 100%; border: 1px solid #ccc; border-radius: 4px; }
</style>
</head>
<body>
<header>
  <h1>Framework SMPSO-CLAHE + MCDM — Reporte interactivo del experimento</h1>
  <p>__SUBTITLE__</p>
</header>
<div class="wrap">
  <div class="controls">
    <label>Imagen: <select id="selImage"></select></label>
    <label>Color de puntos: <select id="selColor">
      <option value="uniform">Uniforme</option>
      <option value="H">Entropía (H)</option>
      <option value="SSIM">SSIM</option>
      <option value="VIF">VIF</option>
      <option value="rx">R_x</option>
      <option value="ry">R_y</option>
      <option value="clip">Clip limit (C)</option>
    </select></label>
    <label class="check"><input type="checkbox" id="chkSurface"> Superficie</label>
    <label class="check"><input type="checkbox" id="chkAxes" checked> Ejes</label>
    <span class="meta" id="imgMeta"></span>
  </div>

  <div class="grid">
    <div class="card full">
      <h2>Frente de Pareto 3D — clic en la leyenda para activar/desactivar cada elemento</h2>
      <div id="pareto"></div>
    </div>

    <div class="card">
      <h2>Imágenes: original → degradada → mejorada</h2>
      <div class="trio">
        <figure><img id="imgOriginal" alt="original"><figcaption>Original (referencia)</figcaption></figure>
        <figure><img id="imgDegraded" alt="degradada"><figcaption id="capDegraded">Degradada (entrada)</figcaption></figure>
        <figure><img id="imgEnhanced" alt="mejorada"><figcaption id="capEnhanced">Mejorada (consenso MCDM)</figcaption></figure>
      </div>
    </div>

    <div class="card">
      <h2>Matriz de decisión y selecciones MCDM</h2>
      <p class="meta">Fila dorada: solución de consenso. Filas sombreadas: elegidas por algún método.</p>
      <div class="scroll"><table id="tblMatrix"></table></div>
    </div>

    <div class="card full">
      <h2>Resumen agregado del experimento</h2>
      <div class="aggfigs">
        <img src="figures/acuerdo_mcdm.png" alt="acuerdo entre métodos" onerror="this.style.display='none'">
        <img src="figures/votos_consenso.png" alt="votos de consenso" onerror="this.style.display='none'">
        <img src="figures/recuperacion_vif.png" alt="recuperación VIF" onerror="this.style.display='none'">
        <img src="figures/recuperacion_por_degradacion.png" alt="recuperación por degradación" onerror="this.style.display='none'">
        <img src="figures/parametros_clahe.png" alt="parámetros CLAHE" onerror="this.style.display='none'">
      </div>
    </div>
  </div>
</div>

<script>
const DATA = __DATA__;
const METHODS = __METHODS__;
const METHOD_LABELS = __METHOD_LABELS__;
const MCDM_COLORS = ['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd','#8c564b','#e377c2','#7f7f7f'];
const MARKERS = ['circle','square','diamond','cross','x','circle-open','square-open','diamond-open'];

const sel = document.getElementById('selImage');
DATA.order.forEach(id => {
  const o = document.createElement('option');
  o.value = id; o.textContent = id;
  sel.appendChild(o);
});

function currentImage() { return DATA.images[sel.value]; }

function buildTraces(img) {
  const p = img.pareto;
  const colorBy = document.getElementById('selColor').value;
  const marker = {size: 4.5, opacity: 0.75};
  if (colorBy === 'uniform') {
    marker.color = '#20606e';
  } else {
    marker.color = p[colorBy];
    marker.colorscale = 'Viridis';
    marker.showscale = true;
    marker.colorbar = {title: {text: colorBy}, len: 0.5, x: 1.02};
  }
  const hover = p.H.map((h, i) =>
    `A${i}<br>Rx=${p.rx[i]} Ry=${p.ry[i]} C=${p.clip[i]}<br>` +
    `H=${p.H[i]} SSIM=${p.SSIM[i]} VIF=${p.VIF[i]}`);

  const traces = [{
    name: 'Frente de Pareto', type: 'scatter3d', mode: 'markers',
    x: p.H, y: p.SSIM, z: p.VIF, marker, text: hover, hoverinfo: 'text',
  }];

  if (document.getElementById('chkSurface').checked && p.H.length >= 4) {
    traces.push({
      name: 'Superficie', type: 'mesh3d',
      x: p.H, y: p.SSIM, z: p.VIF,
      alphahull: 0, opacity: 0.18, color: '#20606e', hoverinfo: 'skip',
    });
  }

  METHODS.forEach((m, k) => {
    const info = img.mcdm[m];
    if (!info) return;
    const i = info.best;
    traces.push({
      name: METHOD_LABELS[m], type: 'scatter3d', mode: 'markers',
      x: [p.H[i]], y: [p.SSIM[i]], z: [p.VIF[i]],
      marker: {size: 9, symbol: MARKERS[k], color: MCDM_COLORS[k],
               line: {color: '#000', width: 1}},
      text: [`${METHOD_LABELS[m]} → A${i}`], hoverinfo: 'text',
    });
  });

  const c = img.consensus.index;
  if (c !== undefined && c !== null) {
    traces.push({
      name: `Consenso (${img.consensus.votes} votos)`, type: 'scatter3d', mode: 'markers',
      x: [p.H[c]], y: [p.SSIM[c]], z: [p.VIF[c]],
      marker: {size: 13, symbol: 'diamond', color: '#c8a415',
               line: {color: '#000', width: 1.5}},
      text: [`Consenso → A${c}`], hoverinfo: 'text',
    });
  }
  return traces;
}

function layout() {
  const showAxes = document.getElementById('chkAxes').checked;
  const ax = (title) => ({title: showAxes ? {text: title} : {text: ''},
                          visible: showAxes, showgrid: showAxes,
                          zeroline: showAxes, showticklabels: showAxes});
  return {
    margin: {l: 0, r: 0, t: 8, b: 0},
    scene: {xaxis: ax('Entropía (H)'), yaxis: ax('SSIM'), zaxis: ax('VIF'),
            camera: {eye: {x: 1.6, y: -1.6, z: 0.7}}},
    legend: {orientation: 'v', x: 0, y: 1, font: {size: 11}},
    paper_bgcolor: 'rgba(0,0,0,0)',
  };
}

function fmtDur(s) {
  // horas si supera 90 min, minutos si supera 60 s, segundos en otro caso
  if (s > 90 * 60) return (s / 3600).toFixed(1) + ' h';
  if (s > 60) return (s / 60).toFixed(1) + ' min';
  return s.toFixed(1) + ' s';
}

function renderPareto() {
  Plotly.react('pareto', buildTraces(currentImage()), layout(),
               {displaylogo: false, responsive: true});
}

function renderImages() {
  const id = sel.value;
  const img = currentImage();
  document.getElementById('imgOriginal').src = `images/${id}/original.png`;
  document.getElementById('imgDegraded').src = `images/${id}/degraded.png`;
  document.getElementById('imgEnhanced').src = `images/${id}/enhanced.png`;
  const v = img.validation || {};
  document.getElementById('capDegraded').textContent =
    `Degradada (entrada) — ${img.degradation} · SSIM=${v.ssim_degraded ?? '?'} · VIF=${v.vif_degraded ?? '?'} (vs. original)`;
  document.getElementById('capEnhanced').textContent =
    `Mejorada (consenso) — SSIM=${v.ssim_enhanced ?? '?'} · VIF=${v.vif_enhanced ?? '?'} (vs. original)`;
  document.getElementById('imgMeta').textContent =
    `Degradación: ${img.degradation} · Frente: ${img.pareto.H.length} soluciones · ${fmtDur(img.time)}`;
}

function renderMatrix() {
  const img = currentImage();
  const p = img.pareto;
  const pickers = {};   // idx -> [métodos]
  METHODS.forEach(m => {
    const info = img.mcdm[m];
    if (info) (pickers[info.best] = pickers[info.best] || []).push(METHOD_LABELS[m]);
  });
  const c = img.consensus.index;
  let html = '<tr><th>A</th><th>Rx</th><th>Ry</th><th>C</th><th>H</th><th>SSIM</th><th>VIF</th><th>Elegida por</th></tr>';
  for (let i = 0; i < p.H.length; i++) {
    const cls = (i === c) ? 'consensus' : (pickers[i] ? 'picked' : '');
    const badges = (pickers[i] || []).map(m => `<span class="badge">${m}</span>`).join('')
                 + (i === c ? '<span class="badge gold">Consenso</span>' : '');
    html += `<tr class="${cls}"><td>${i}</td><td>${p.rx[i]}</td><td>${p.ry[i]}</td>` +
            `<td>${p.clip[i]}</td><td>${p.H[i]}</td><td>${p.SSIM[i]}</td><td>${p.VIF[i]}</td><td>${badges}</td></tr>`;
  }
  document.getElementById('tblMatrix').innerHTML = html;
}

function renderAll() { renderImages(); renderMatrix(); renderPareto(); }

sel.addEventListener('change', renderAll);
document.getElementById('selColor').addEventListener('change', renderPareto);
document.getElementById('chkSurface').addEventListener('change', renderPareto);
document.getElementById('chkAxes').addEventListener('change', renderPareto);
renderAll();
</script>
</body>
</html>
"""


def main():
    parser = argparse.ArgumentParser(description='Reporte HTML interactivo')
    parser.add_argument('--experiment', type=str, default=None)
    args = parser.parse_args()

    exp_dir = Path(args.experiment) if args.experiment \
        else find_latest_experiment(PROJECT_ROOT / 'results')
    print(f'Experimento: {exp_dir}')

    payload = collect_payload(exp_dir)
    if not payload['order']:
        raise SystemExit('No hay imágenes procesadas con éxito en el experimento')

    summary_file = exp_dir / 'experiment_summary.json'
    subtitle = exp_dir.name
    if summary_file.exists():
        s = json.loads(summary_file.read_text(encoding='utf-8'))
        cfg = s.get('config', {})
        subtitle = (f"{exp_dir.name} — {len(payload['order'])} imágenes · "
                    f"SMPSO {cfg.get('particles')}×{cfg.get('iterations')} · "
                    f"semilla {cfg.get('seed')} · pesos {cfg.get('weights')}")

    from plotly.offline import get_plotlyjs
    html = (HTML_TEMPLATE
            .replace('__PLOTLYJS__', get_plotlyjs())
            .replace('__SUBTITLE__', subtitle)
            .replace('__DATA__', json.dumps(payload, ensure_ascii=False))
            .replace('__METHODS__', json.dumps(METHODS))
            .replace('__METHOD_LABELS__', json.dumps(METHOD_LABELS, ensure_ascii=False)))

    out = exp_dir / 'report.html'
    out.write_text(html, encoding='utf-8')
    size_mb = out.stat().st_size / 1e6
    print(f'Reporte generado: {out} ({size_mb:.1f} MB)')


if __name__ == '__main__':
    main()
