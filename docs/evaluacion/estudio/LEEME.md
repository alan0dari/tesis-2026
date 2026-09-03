# Material del estudio perceptual — instrucciones de uso

> **Esto describe la app offline, que quedó como respaldo.** La recolección se
> hace con el sitio web: un enlace único, sesiones con reanudación y respuestas
> que llegan solas a una base de datos. Ver [`webapp/README.md`](../../../webapp/README.md).
> Lo de acá sirve si hace falta evaluar sin conexión o si el sitio se cae.

Generado por `scripts/build_study_materials.py` + `scripts/generate_evaluation_app.py`
a partir de `results/experiment_20260803_203341`. Todo es determinista con
semilla 42: volver a correr los scripts reproduce exactamente estos archivos.

---

## Qué mandar a cada odontólogo

Mandar **toda esta carpeta comprimida**, EXCEPTO estos tres archivos:

| No enviar | Por qué |
|---|---|
| `clave.csv` | Dice qué condición es cada imagen. Revelaría el experimento. |
| `encuadre.csv` | Control de calidad interno. |
| `LEEME.md` | Este archivo. |

A cada evaluador se le indica **un solo archivo**: `evaluador_01.html`,
`evaluador_02.html`, …, hasta `evaluador_08.html`. **No repetir número entre
personas**: cada archivo trae un subconjunto distinto de comparaciones, y el
diseño de bloques se rompe si dos personas hacen el mismo.

Se abre con doble clic, en cualquier navegador. No necesita conexión ni servidor.

> Si es más cómodo, se puede subir la carpeta a un Drive compartido y pasarle a
> cada uno el enlace a su archivo. Lo importante es que `img/` viaje junto.

---

## Qué decirles

- Son unas **75 pantallas, entre 25 y 30 minutos**. Las instrucciones están en la
  primera pantalla del propio archivo; no hace falta explicar nada más.
- Pueden **cortar y retomar**: el progreso se guarda en el navegador. Eso sí,
  tienen que retomar **en la misma computadora y el mismo navegador**.
- Al terminar se descarga solo un archivo `respuestas_evaluador_NN.json`. **Ese
  archivo es el que hay que devolver.**
- Conviene anotar aparte, por cada evaluador: **años de ejercicio y
  especialidad**. Van como covariables al análisis.

Vale la pena avisarles de una cosa, porque si no genera desconcierto: **muchos
pares van a parecer idénticos**. Es esperable y es parte de lo que se mide. Para
eso está el botón "Estoy adivinando".

---

## Antes de convocar a los 8: hacer el piloto

Correr primero con **1 o 2 personas** (`evaluador_01`, `evaluador_02`) y mirar:

- Cuánto tardan de verdad por pantalla. La estimación de 20-25 s es una hipótesis
  sin verificar; el JSON trae el tiempo de cada ensayo en `ms`.
- Si las instrucciones se entienden sin explicación oral.
- Si el recorte del sextante alcanza o piden ver la radiografía completa.

**Los datos del piloto no entran en el análisis final.** Si hay que cambiar algo
del material, se regenera y se vuelve a empezar.

---

## Qué contiene la carpeta

| Archivo | Qué es |
|---|---|
| `evaluador_NN.html` | La aplicación, una por evaluador. 5 de práctica + 65 ensayos + 5 repetidos |
| `img/` | 193 recortes de sextante, PNG sin pérdida |
| `trials.json` | Los 130 ensayos y el reparto entre evaluadores |
| `clave.csv` | **Interno.** Condición de cada lado, sextante, degradación, ΔH/ΔSSIM/ΔVIF |
| `encuadre.csv` | **Interno.** Control de calidad del encuadre de los recortes |

Los 130 ensayos se reparten así (ver `docs/evaluacion/protocolo_evaluacion.md` §5):

| Bloque | N | Qué compara |
|---|---:|---|
| Q2 | 50 | original vs. consenso MCDM |
| Q4 | 45 | extremos mono-objetivo entre sí (mejor-H / mejor-SSIM / mejor-VIF) |
| Q3 | 20 | selecciones MCDM entre sí, sólo los pares más distinguibles |
| Q1 | 15 | degradada vs. consenso — control de atención |

Cada ensayo lo ven 4 evaluadores distintos: 130 × 4 = **520 juicios**.

---

## Sobre las imágenes

**No se modificaron.** Las radiografías de `data/evaluacion_50/` y las salidas del
experimento conservan intacta su anotación quemada (equipo, fecha, kV/mA, marcas
R/L). Lo que se muestra es un **recorte de un sextante**, por la razón clínica de
que el diagnóstico se hace por regiones, no porque se haya borrado nada.

El recorte es **idéntico en geometría para las dos imágenes de cada par**, así la
comparación queda exactamente pareada.

---

## Cuando vuelvan los JSON

Guardarlos todos juntos. El análisis (Fase 6, todavía por escribir) los cruza con
`clave.csv` por `trial_id`. El plan estadístico está fijado de antemano en
`docs/evaluacion/protocolo_evaluacion.md` §7 — **no cambiarlo después de ver los
datos**.

Primer paso del análisis, antes que cualquier otra cosa: excluir a quien falle
más del 20% de los ensayos de Q1, que son los de respuesta casi obvia.
