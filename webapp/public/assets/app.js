/*
  Estudio perceptual con odontólogos - FP-UNA

  Cada respuesta va primero a una cola en localStorage y de ahí al servidor, así
  un corte de internet no interrumpe la evaluación. El token de sesión también
  vive en localStorage y se valida contra el servidor al cargar la página.

  guardar() es idempotente por (participante, posición), así que reintentar
  nunca duplica.

  El orden de los ensayos lo arma el navegador con la semilla que manda la base.
  Determinista: recargar no cambia la secuencia.
*/

(() => {
  'use strict';

  const CFG = window.CONFIG || {};
  const LLAVE_TOKEN     = 'estudio.token';
  const LLAVE_COLA      = 'estudio.cola';
  const LLAVE_CORREO    = 'estudio.correo';
  const REINTENTO_MS    = 15000;

  const $ = (id) => document.getElementById(id);

  const estado = {
    token: null, set: null, semilla: 0, correo: '', comentario: '',
    secuencia: [], indice: 0, total: 0,
    eleccion: null, t0: 0, procesando: false,
    datos: null,
    // Respuestas que ya tiene el servidor. La pantalla actual es esto + lo que
    // quede en la cola; mirar una sola de las dos da un número corto.
    respondidasServidor: 0
  };

  function mulberry32(a) {
    return function () {
      a |= 0; a = (a + 0x6D2B79F5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  function barajar(lista, rnd) {
    for (let i = lista.length - 1; i > 0; i--) {
      const j = Math.floor(rnd() * (i + 1));
      [lista[i], lista[j]] = [lista[j], lista[i]];
    }
    return lista;
  }

  const correoValido = (c) => /^[^@\s]+@[^@\s]+\.[^@\s]{2,}$/.test(String(c).trim());

  function leerCola() {
    try { return JSON.parse(localStorage.getItem(LLAVE_COLA) || '[]'); }
    catch { return []; }
  }
  const escribirCola = (c) => localStorage.setItem(LLAVE_COLA, JSON.stringify(c));

  function brindis(texto, ms = 3200) {
    const el = $('brindis');
    el.textContent = texto;
    el.classList.add('visible');
    clearTimeout(brindis._t);
    brindis._t = setTimeout(() => el.classList.remove('visible'), ms);
  }

  function avisar(titulo, texto, alCerrar) {
    $('aviso-titulo').textContent = titulo;
    $('aviso-texto').textContent = texto;
    $('modal-aviso').hidden = false;
    $('btn-aviso-ok').onclick = () => {
      $('modal-aviso').hidden = true;
      if (alCerrar) alCerrar();
    };
  }

  function mostrarVista(cual) {
    $('vista-inicio').hidden  = cual !== 'inicio';
    $('vista-tarea').hidden   = cual !== 'tarea';
    $('vista-gracias').hidden = cual !== 'gracias';
    document.body.classList.toggle('evaluando', cual === 'tarea');
    window.scrollTo(0, 0);
  }

  // Dos backends con la misma interfaz. El local sirve para recorrer el sitio
  // entero sin base montada.

  async function rpc(nombre, cuerpo) {
    const r = await fetch(`${CFG.SUPABASE_URL}/rest/v1/rpc/${nombre}`, {
      method: 'POST',
      headers: {
        'apikey': CFG.SUPABASE_ANON_KEY,
        'Authorization': `Bearer ${CFG.SUPABASE_ANON_KEY}`,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(cuerpo || {})
    });
    if (!r.ok) throw new Error(`${nombre}: ${r.status} ${await r.text()}`);
    return r.json();
  }

  const backendRemoto = {
    local: false,
    iniciar: (correo) => rpc('iniciar', { p_correo: correo, p_navegador: navigator.userAgent }),
    sesion: (token) => rpc('sesion', { p_token: token }),
    guardar: (token, r) => rpc('guardar', {
      p_token: token, p_ensayo: r.ensayo, p_posicion: r.posicion,
      p_eleccion: r.eleccion, p_confianza: r.confianza,
      p_ms: r.ms, p_repeticion: r.repeticion
    }),
    finalizar: (token, comentario) =>
      rpc('finalizar', { p_token: token, p_comentario: comentario ?? null })
  };

  // modo demostración: todo queda en este navegador
  const backendLocal = (() => {
    const LL = 'estudio.demo';
    const leer = () => { try { return JSON.parse(localStorage.getItem(LL) || '{}'); } catch { return {}; } };
    const escribir = (d) => localStorage.setItem(LL, JSON.stringify(d));

    function armar(d, correo) {
      const p = d.participantes[correo];
      return {
        token: 'demo-' + correo, set: p.set, semilla: p.semilla,
        respondidas: Object.keys(p.respuestas).length,
        terminado: !!p.terminado, comentario: p.comentario || '',
        correo, valida: true, ok: true
      };
    }
    return {
      local: true,
      async iniciar(correo) {
        const d = leer();
        d.participantes = d.participantes || {};
        d.orden = d.orden || 0;
        const nuevo = !d.participantes[correo];
        if (nuevo) {
          d.orden += 1;
          // mismo criterio que la base: el set menos usado, desempata el más chico
          const usos = {};
          for (let n = 1; n <= 8; n++) usos[n] = 0;
          for (const p of Object.values(d.participantes)) usos[Number(p.set)] += 1;
          const set = Object.keys(usos).sort((a, b) => usos[a] - usos[b] || a - b)[0];
          d.participantes[correo] = {
            set: String(set).padStart(2, '0'),
            semilla: Math.floor(Math.random() * 2147483000),
            respuestas: {}, terminado: false
          };
        }
        escribir(d);
        return { ...armar(d, correo), nuevo };
      },
      async sesion(token) {
        const correo = String(token).replace(/^demo-/, '');
        const d = leer();
        if (!d.participantes || !d.participantes[correo]) return { valida: false };
        return armar(d, correo);
      },
      async guardar(token, r) {
        const correo = String(token).replace(/^demo-/, '');
        const d = leer();
        if (!d.participantes || !d.participantes[correo]) return { ok: false, motivo: 'sesion' };
        d.participantes[correo].respuestas[r.posicion] = r;
        escribir(d);
        return armar(d, correo);
      },
      async finalizar(token, comentario) {
        const correo = String(token).replace(/^demo-/, '');
        const d = leer();
        if (!d.participantes || !d.participantes[correo]) return { ok: false, motivo: 'sesion' };
        d.participantes[correo].terminado = true;
        if (comentario) d.participantes[correo].comentario = comentario;
        escribir(d);
        return armar(d, correo);
      }
    };
  })();

  const backend = CFG.SUPABASE_URL ? backendRemoto : backendLocal;

  // En una esquina y no a lo ancho: en pantalla chica los botones de respuesta
  // llegan hasta el borde de abajo y una banda los taparía.
  function bandaDemo() {
    const b = document.createElement('div');
    b.textContent = 'Modo demo';
    b.title = 'No se guardan las respuestas';
    b.style.cssText = 'position:fixed;left:12px;bottom:12px;z-index:80;pointer-events:none;' +
      'background:#26365e;color:#fff;font:600 12px/1 system-ui;padding:7px 13px;' +
      'border-radius:999px;letter-spacing:.02em;opacity:.9';
    document.body.appendChild(b);
  }

  // --- cola de sincronización

  let sincronizando = false;

  function pintarEstado(cual) {
    const el = $('estado-guardado');
    if (!el) return;
    el.dataset.estado = cual;
    $('estado-texto').textContent =
      cual === 'ok' ? 'Guardado' : cual === 'pendiente' ? 'Guardando…' : '';
  }

  async function sincronizar() {
    if (sincronizando || !estado.token) return;
    const cola = leerCola();
    if (!cola.length) { pintarEstado('ok'); return; }

    sincronizando = true;
    pintarEstado('pendiente');
    try {
      while (cola.length) {
        const res = await backend.guardar(estado.token, cola[0]);
        if (res && res.ok === false) {
          if (res.motivo === 'sesion') { sesionVencida(); return; }
          throw new Error(res.motivo || 'rechazado');
        }
        if (res && typeof res.respondidas === 'number') {
          estado.respondidasServidor = res.respondidas;
        }
        cola.shift();
        escribirCola(cola);
      }
      pintarEstado('ok');
    } catch {
      pintarEstado('error');
      setTimeout(sincronizar, REINTENTO_MS);
    } finally {
      sincronizando = false;
    }
  }

  // Venció la sesión a mitad de camino. No se pierde nada: la cola queda en el
  // navegador y sale cuando vuelva a entrar con su correo.
  function sesionVencida() {
    localStorage.removeItem(LLAVE_TOKEN);
    estado.token = null;
    mostrarVista('inicio');
    $('correo').value = localStorage.getItem(LLAVE_CORREO) || '';
    avisar('Tu sesión expiró',
      'Ingresá otra vez con tu mismo correo para retomar donde lo dejaste.',
      () => $('correo').focus());
  }

  // --- arranque y sesión

  async function arrancar() {
    try {
      estado.datos = await (await fetch('assets/sets.json', { cache: 'no-cache' })).json();
    } catch {
      avisar('No se pudo cargar el contenido',
        'Revisá tu conexión y volvé a cargar la página.');
      return;
    }
    if (backend.local) bandaDemo();

    const token = localStorage.getItem(LLAVE_TOKEN);
    if (!token) return;

    try {
      const s = await backend.sesion(token);
      if (!s || !s.valida) { localStorage.removeItem(LLAVE_TOKEN); return; }

      aplicarSesion(token, s);
      if (s.terminado) { irAGracias(); return; }

      await sincronizar();
      estado.indice = estado.respondidasServidor + leerCola().length;
      entrarATarea();
      if (estado.indice > 0) {
        brindis(`Retomamos donde lo dejaste: evaluación ${estado.indice + 1} de ${estado.total}`, 4500);
      }
    } catch {
      /* sin conexión al arrancar: se queda en la presentación */
    }
  }

  function aplicarSesion(token, s) {
    estado.token = token;
    estado.respondidasServidor = s.respondidas || 0;
    estado.set = s.set;
    estado.semilla = Number(s.semilla) || 1;
    estado.correo = s.correo || localStorage.getItem(LLAVE_CORREO) || '';
    estado.comentario = s.comentario || '';
    localStorage.setItem(LLAVE_TOKEN, token);
    if (estado.correo) localStorage.setItem(LLAVE_CORREO, estado.correo);
    estado.secuencia = construirSecuencia();
    estado.total = estado.secuencia.length;
  }

  function construirSecuencia() {
    const d = estado.datos;
    const propios = d.sets[estado.set] || d.sets[Object.keys(d.sets)[0]];
    const rnd = mulberry32(estado.semilla >>> 0);
    const orden = barajar(propios.slice(), rnd);

    // Los 5 del final miden consistencia intra-evaluador. Salen de los propios,
    // así que no se distinguen del resto.
    const resto = orden.slice();
    const repetidos = [];
    for (let i = 0; i < (d.n_repetidos || 0) && resto.length; i++) {
      repetidos.push(resto.splice(Math.floor(rnd() * resto.length), 1)[0]);
    }
    return orden.map((id) => ({ id, r: false }))
      .concat(repetidos.map((id) => ({ id, r: true })));
  }

  async function registrar(ev) {
    ev.preventDefault();
    const correo = $('correo').value.trim().toLowerCase();
    const error = $('error-inicio');
    error.hidden = true;

    if (!correoValido(correo)) {
      error.textContent = 'Revisá el formato del correo. Algo no está bien.';
      error.hidden = false;
      $('correo').focus();
      return;
    }
    const btn = $('btn-empezar');
    const rotulo = btn.querySelector('span');
    btn.disabled = true;
    rotulo.textContent = 'Preparando…';

    try {
      const s = await backend.iniciar(correo);
      localStorage.setItem(LLAVE_CORREO, correo);
      aplicarSesion(s.token, s);

      if (s.terminado) {
        estado.indice = estado.total;
        irAGracias();
        brindis('Ya completaste la evaluación. Gracias!', 5000);
        return;
      }

      // Primero se manda lo que quedó colgado de antes; recién ahí se sabe en
      // qué pantalla va.
      if (leerCola().length) await sincronizar();
      const hechas = estado.respondidasServidor + leerCola().length;

      if (!s.nuevo && hechas > 0) {
        ofrecerReanudar(correo, hechas);
      } else {
        estado.indice = hechas;
        entrarATarea();
      }
    } catch (e) {
      error.textContent = String(e).includes('correo_invalido')
        ? 'Ese correo no parece válido.'
        : 'No pudimos conectarte. Revisá tu internet y probá de nuevo.';
      error.hidden = false;
    } finally {
      btn.disabled = false;
      rotulo.textContent = 'Comenzar';
    }
  }

  // mismo correo + sesión caída = ofrecer retomar
  function ofrecerReanudar(correo, hechas) {
    $('reanudar-texto').textContent =
      `Bien! Continuá desde donde lo dejaste.`;
    $('reanudar-detalle').innerHTML =
      `Llevás <b>${hechas}</b> de <b>${estado.total}</b> evaluaciones respondidas.`;
    $('reanudar-barra').style.width = `${Math.round(100 * hechas / estado.total)}%`;
    $('modal-reanudar').hidden = false;

    $('btn-retomar').onclick = () => {
      $('modal-reanudar').hidden = true;
      estado.indice = hechas;
      entrarATarea();
      sincronizar();
    };
    $('btn-otro-correo').onclick = () => {
      $('modal-reanudar').hidden = true;
      localStorage.removeItem(LLAVE_TOKEN);
      estado.token = null;
      $('correo').value = '';
      $('correo').focus();
    };
  }

  // --- la tarea

  function entrarATarea() {
    mostrarVista('tarea');
    pintarEstado(leerCola().length ? 'pendiente' : 'ok');
    mostrarEnsayo();
    sincronizar();          // por si quedó algo sin mandar de una sesión anterior
    // Se abre sola en el primer par, siempre. Después queda en el botón de la
    // barra. Atarlo al índice y no a una marca en localStorage hace que también
    // la vea quien arranca desde otra computadora.
    if (estado.indice === 0) abrirInstrucciones();
  }

  function abrirInstrucciones() { $('modal-instrucciones').hidden = false; }

  function cerrarInstrucciones() { $('modal-instrucciones').hidden = true; }

  function mostrarEnsayo() {
    if (estado.indice >= estado.total) return cerrar();

    const t = estado.secuencia[estado.indice];
    const archivos = estado.datos.ensayos[t.id];
    resetVista();
    $('img-a').src = 'img/' + archivos[0];
    $('img-b').src = 'img/' + archivos[1];

    $('pregunta-eleccion').hidden = false;
    $('pregunta-confianza').hidden = true;
    $('progreso-barra').style.width = `${100 * estado.indice / estado.total}%`;
    $('visor-cuenta').textContent = `${estado.indice + 1} / ${estado.total}`;

    precargar(estado.indice + 1);
    precargar(estado.indice + 2);

    estado.eleccion = null;
    estado.procesando = false;
    estado.t0 = performance.now();
  }

  function precargar(i) {
    if (i >= estado.total) return;
    const archivos = estado.datos.ensayos[estado.secuencia[i].id];
    for (const f of archivos) { const im = new Image(); im.src = 'img/' + f; }
  }

  function elegir(lado) {
    if (estado.procesando || $('pregunta-eleccion').hidden) return;
    estado.eleccion = lado;
    $('confianza-elegida').textContent = lado === 'a' ? 'Imagen 1' : 'Imagen 2';
    $('pregunta-eleccion').hidden = true;
    $('pregunta-confianza').hidden = false;
  }

  // Nada se guardó todavía: hasta que no responde la confianza, el ensayo no
  // existe. El cronómetro sigue corriendo a propósito, así el tiempo del ensayo
  // incluye lo que dudó.
  function volverAEleccion() {
    if (estado.procesando || $('pregunta-confianza').hidden) return;
    estado.eleccion = null;
    $('pregunta-confianza').hidden = true;
    $('pregunta-eleccion').hidden = false;
  }

  function responder(nivel) {
    if (estado.procesando || $('pregunta-confianza').hidden) return;
    estado.procesando = true;

    const t = estado.secuencia[estado.indice];
    const cola = leerCola();
    cola.push({
      ensayo: t.id,
      posicion: estado.indice + 1,
      eleccion: estado.eleccion,
      confianza: nivel,
      ms: Math.round(performance.now() - estado.t0),
      repeticion: !!t.r
    });
    escribirCola(cola);

    estado.indice += 1;
    mostrarEnsayo();
    sincronizar();
  }

  // --- cierre

  async function cerrar() {
    irAGracias();
    asegurarCierre();
  }

  async function asegurarCierre() {
    await sincronizar();
    if (leerCola().length) { setTimeout(asegurarCierre, REINTENTO_MS); return; }
    try {
      await backend.finalizar(estado.token, null);
    } catch {
      setTimeout(asegurarCierre, REINTENTO_MS);
    }
  }

  function irAGracias() {
    mostrarVista('gracias');
    pintarComentario(false);
  }

  // Ojo: la página de gracias se puede recargar. Si ya hay comentario guardado,
  // mostrar el campo vacío invita a escribir de nuevo y pisar lo anterior.
  function pintarComentario(editando) {
    const hay = !!(estado.comentario && estado.comentario.trim());
    const conCampo = editando || !hay;

    $('bloque-comentario').hidden = !conCampo;
    $('comentario-listo').hidden = conCampo;
    $('comentario-estado').hidden = true;

    const campo = $('comentario');
    campo.disabled = false;
    campo.value = hay ? estado.comentario : '';

    $('btn-comentario').disabled = false;
    $('btn-comentario').textContent = hay ? 'Guardar cambios' : 'Enviar';
    $('btn-cancelar-comentario').hidden = !hay;

    if (editando) campo.focus();
  }

  async function enviarComentario() {
    const texto = $('comentario').value.trim();
    const aviso = $('comentario-estado');
    const btn = $('btn-comentario');
    if (!texto) {
      aviso.textContent = 'El campo está vacío.';
      aviso.hidden = false;
      return;
    }
    btn.disabled = true;
    try {
      const r = await backend.finalizar(estado.token, texto);
      estado.comentario = (r && r.comentario) || texto;
      pintarComentario(false);
    } catch {
      aviso.textContent = 'No pudimos guardar tu comentario. Probá de nuevo en un momento.';
      aviso.hidden = false;
      btn.disabled = false;
    }
  }

  // --- zoom y desplazamiento, sincronizados entre las dos imágenes.
  // Sin esto no se puede comparar detalle fino, que es de lo que se trata.

  const vista = { zoom: 1, x: 0, y: 0 };

  function aplicarVista() {
    const t = `translate(${vista.x}px, ${vista.y}px) scale(${vista.zoom})`;
    $('img-a').style.transform = t;
    $('img-b').style.transform = t;
  }

  function resetVista() { vista.zoom = 1; vista.x = 0; vista.y = 0; aplicarVista(); }

  function acercar(marco, cx, cy, factor) {
    const r = marco.getBoundingClientRect();
    const mx = cx - r.left, my = cy - r.top;
    const nuevo = Math.min(8, Math.max(1, vista.zoom * factor));
    // el punto bajo el cursor se queda quieto
    vista.x = mx - (mx - vista.x) * (nuevo / vista.zoom);
    vista.y = my - (my - vista.y) * (nuevo / vista.zoom);
    vista.zoom = nuevo;
    if (vista.zoom <= 1.001) { vista.zoom = 1; vista.x = 0; vista.y = 0; }
    aplicarVista();
  }

  function conectarMarco(marco) {
    const punteros = new Map();
    let arrastre = null, pellizco = null;

    marco.addEventListener('wheel', (e) => {
      e.preventDefault();
      acercar(marco, e.clientX, e.clientY, e.deltaY < 0 ? 1.16 : 1 / 1.16);
    }, { passive: false });

    marco.addEventListener('pointerdown', (e) => {
      marco.setPointerCapture(e.pointerId);
      punteros.set(e.pointerId, e);
      marco.classList.add('arrastrando');
      if (punteros.size === 1) arrastre = { x: e.clientX - vista.x, y: e.clientY - vista.y };
      else { arrastre = null; pellizco = null; }
    });

    marco.addEventListener('pointermove', (e) => {
      if (!punteros.has(e.pointerId)) return;
      punteros.set(e.pointerId, e);

      if (punteros.size === 1 && arrastre) {
        vista.x = e.clientX - arrastre.x;
        vista.y = e.clientY - arrastre.y;
        aplicarVista();
      } else if (punteros.size === 2) {
        const [p1, p2] = [...punteros.values()];
        const d = Math.hypot(p1.clientX - p2.clientX, p1.clientY - p2.clientY);
        if (pellizco) {
          acercar(marco, (p1.clientX + p2.clientX) / 2, (p1.clientY + p2.clientY) / 2, d / pellizco);
        }
        pellizco = d;
      }
    });

    const soltar = (e) => {
      punteros.delete(e.pointerId);
      if (punteros.size < 2) pellizco = null;
      if (punteros.size === 1) {
        const p = [...punteros.values()][0];
        arrastre = { x: p.clientX - vista.x, y: p.clientY - vista.y };
      } else if (punteros.size === 0) {
        arrastre = null;
        marco.classList.remove('arrastrando');
      }
    };
    marco.addEventListener('pointerup', soltar);
    marco.addEventListener('pointercancel', soltar);
    marco.addEventListener('dblclick', resetVista);
  }

  // --- eventos

  function conectar() {
    $('form-inicio').addEventListener('submit', registrar);
    $('btn-comentario').addEventListener('click', enviarComentario);
    $('btn-editar-comentario').addEventListener('click', () => pintarComentario(true));
    $('btn-cancelar-comentario').addEventListener('click', () => pintarComentario(false));

    document.querySelectorAll('[data-eleccion]').forEach((b) =>
      b.addEventListener('click', () => elegir(b.dataset.eleccion)));
    document.querySelectorAll('[data-confianza]').forEach((b) =>
      b.addEventListener('click', () => responder(b.dataset.confianza)));
    $('btn-volver-eleccion').addEventListener('click', volverAEleccion);

    $('zoom-mas').addEventListener('click', () => acercarCentro(1.3));
    $('zoom-menos').addEventListener('click', () => acercarCentro(1 / 1.3));
    $('zoom-reset').addEventListener('click', resetVista);

    $('btn-ayuda').addEventListener('click', abrirInstrucciones);
    $('btn-entendido').addEventListener('click', cerrarInstrucciones);
    $('btn-cerrar-instrucciones').addEventListener('click', cerrarInstrucciones);
    $('modal-instrucciones').addEventListener('click', (e) => {
      if (e.target === $('modal-instrucciones')) cerrarInstrucciones();
    });

    conectarMarco($('marco-a'));
    conectarMarco($('marco-b'));

    document.addEventListener('keydown', (e) => {
      if (!$('modal-instrucciones').hidden) {
        if (e.key === 'Escape') cerrarInstrucciones();
        return;
      }
      if ($('vista-tarea').hidden || !$('modal-aviso').hidden) return;
      const k = e.key.toLowerCase();
      // 1/2/3 sirven en los dos pasos: nunca están visibles a la vez
      if (!$('pregunta-eleccion').hidden) {
        if (k === '1' || k === 'arrowleft') elegir('a');
        if (k === '2' || k === 'arrowright') elegir('b');
      } else if (!$('pregunta-confianza').hidden) {
        if (k === '1') responder('seguro');
        if (k === '2') responder('algo');
        if (k === '3') responder('adivinando');
        if (k === 'escape' || k === 'backspace') { e.preventDefault(); volverAEleccion(); }
      }
    });

    window.addEventListener('online', sincronizar);
    window.addEventListener('beforeunload', (e) => {
      if (leerCola().length && !$('vista-tarea').hidden) {
        e.preventDefault();
        e.returnValue = '';
      }
    });
  }

  function acercarCentro(factor) {
    const r = $('marco-a').getBoundingClientRect();
    acercar($('marco-a'), r.left + r.width / 2, r.top + r.height / 2, factor);
  }

  conectar();
  arrancar();
})();
