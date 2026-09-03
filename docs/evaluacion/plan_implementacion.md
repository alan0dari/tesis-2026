# Plan de implementación — estudio perceptual con odontólogos

Ruta desde el estado actual hasta los resultados analizados. Cada fase lista
entregables concretos y qué la bloquea.

Documentos hermanos: [`protocolo_evaluacion.md`](protocolo_evaluacion.md) (el
diseño y su justificación) y [`colapso_perceptual_mcdm.md`](colapso_perceptual_mcdm.md)
(el hallazgo que condiciona el diseño).

---

## Fase 0 — Selección del conjunto ✅ HECHO

| Entregable | Estado |
|---|---|
| `scripts/screen_evaluation_set.py` | ✅ |
| `docs/evaluacion/screening/screening_scores.csv` (598 imágenes puntuadas) | ✅ |
| `docs/evaluacion/screening/hojas_contacto/` (14 hojas) | ✅ |
| `docs/evaluacion/ids_aprobados.txt` (50 IDs, Tier A/B) | ✅ |
| `data/evaluacion_50/` (50 imágenes, **sin modificar**) | ✅ |

El triaje automático ordenó las 598 por completitud de dentición tras descartar
material radiopaco franco; sobre las 99 mejores se hizo revisión visual una por
una. 40 imágenes cumplen el criterio estricto y 10 entran bajo la tolerancia
explícita del tutor ("la menor modificación posible").

**Pendiente de validación humana:** conviene que el tutor revise el Tier B a
resolución completa. Son las 10 últimas de `ids_aprobados.txt`. Si descarta
alguna, se reemplaza por la siguiente candidata de las hojas 12-14, que no
llegaron a revisarse.

---

## Fase 1 — Experimento sobre las 50 ✅ HECHO

```bash
python scripts/run_experiment.py --data-dir data/evaluacion_50 \
    --sample-size 50 --particles 100 --iterations 100 \
    --workers 4 --seed 42 --balanced-degradation
```

Salida en `results/experiment_20260803_203341/`. 50/50 en 4.6 h con 4 workers.
Degradaciones perfectamente balanceadas (10 de cada tipo). VIF contra la original
prístina 0.735 → 0.977, mejora en **50/50** imágenes (Wilcoxon W=0, p=1.8e-15).

> **Corrección aplicada.** La primera corrida salió con los pesos MCDM
> preliminares (0.40, 0.35, 0.25): el default de argparse pisaba la constante
> `DEFAULT_WEIGHTS` con los ROC, y como SMARTER deriva sus pesos internamente y
> nunca recibe `weights`, los otros 7 métodos rankeaban con otro esquema. Se
> detectó porque **SMARTER vs. MABAC daba 30% de acuerdo en vez de 100%**. Se
> corrigió con `python scripts/rerun_mcdm.py --experiment
> results/experiment_20260803_203341 --scheme roc`, que recomputa sólo la etapa
> MCDM en segundos; las salidas erróneas quedaron en `_legacy_pesos_040_035_025/`.
> El default del CLI ya se deriva de `DEFAULT_WEIGHTS` y hay una advertencia si
> se pasan pesos distintos.
>
> **Chequeo de sanidad para cualquier experimento futuro:** SMARTER vs. MABAC en
> `mcdm_agreement_matrix.csv` debe dar exactamente 100%. Y ojo:
> `experiment_summary.json` **no** se actualiza al correr `rerun_mcdm`, así que
> no sirve para saber qué pesos tienen los datos vigentes.

`--balanced-degradation` es nuevo: reparte los 5 tipos de degradación en partes
iguales (10 cada uno) en lugar de sortear cada imagen por separado. Con n=50 el
sorteo libre deja grupos de tamaño muy dispar (DE ≈ 2.8 por tipo) justo donde el
estudio va a reportar resultados por tipo de degradación.

Por imagen quedan: `original.png`, `degraded.png`, `enhanced.png`, `pareto.csv`
(≈100 soluciones con sus parámetros y objetivos) y `result.json` con la selección
de cada uno de los 8 métodos MCDM y el consenso.

---

## Fase 2 — Materiales del estudio ✅ HECHO

`scripts/build_study_materials.py` → `docs/evaluacion/estudio/`. Determinista con
semilla 42. 130 ensayos, 193 recortes, 50 imágenes, 8 evaluadores × 65 ensayos,
4 evaluadores por ensayo = **520 juicios**, cero ensayos sin cubrir.

> **La geometría del recorte costó cinco iteraciones y ninguna regla simple
> funcionó.** Vale dejarlo escrito porque es el tipo de error que no se ve en los
> números y sí en las imágenes:
>
> 1. Trazar el plano oclusal por mínimo de intensidad por columna: falla en los
>    sextantes posteriores, donde el fondo negro fuera del maxilar y el seno son
>    más oscuros que el espacio interoclusal. Recortes sobre fondo y sobre la
>    banda de texto.
> 2. Anclar en los dos lóbulos más altos del top-hat dental: en el sector anterior
>    el paladar duro y la espina nasal responden más que los incisivos, así que la
>    mandíbula no se detectaba. 3 de 9 recortes anteriores superiores sobre paladar.
> 3. Tomar el lóbulo más bajo: agarra el borde cortical de la mandíbula, que
>    también es brillante y compacto. Recortes sobre hueso.
> 4. Lóbulo más bajo **con ancho de pieza dental** (`find_peaks` con `width`),
>    banda de búsqueda acotada por arriba del borde cortical, y un ajuste fino
>    que elegía el desplazamiento vertical que **dejaba más estructura dental en
>    cuadro**. Con esto se dejó de recortar sobre fondo, paladar y hueso.
> 5. **Y ese ajuste fino era el error siguiente.** Los odontólogos que revisaron
>    el material lo marcaron: faltaban ápices, terminaciones radiculares y tejido
>    de alrededor, sobre todo en los posteriores. Tiene sentido visto en
>    retrospectiva — el esmalte responde al top-hat mucho más que el hueso
>    periapical, así que maximizar contenido dental **centra la caja en las
>    coronas y empuja los ápices fuera de cuadro**. La métrica de calidad estaba
>    premiando justo lo que había que evitar.
>
>    Además el anclaje del maxilar era inestable: el seno y el malar responden
>    tanto como los molares, y según la imagen el recorte superior se iba al seno
>    o caía sobre el propio plano oclusal.
>
> **Lo que corre hoy** (ver `SEXTANT_SPAN` en `build_study_materials.py`): anclar
> en el **plano oclusal**, que es el mínimo de intensidad entre las dos arcadas y
> se detecta parejo en las 50, y extenderse asimétricamente hacia donde van las
> raíces de la arcada pedida — 27 % de la altura hacia arriba y 7 % hacia abajo en
> el maxilar, 7 % y 35 % en la mandíbula. Así entran corona, raíz completa y el
> hueso de alrededor: el piso del seno en los posteriores superiores, la zona del
> conducto en los inferiores. El recorte pasó de 449×296 a 449×347 (superiores) y
> 449×429 (inferiores).
>
> **El control de calidad también cambió**, porque el viejo medía lo que dejamos
> de querer. `encuadre.csv` ahora trae `oclusal_rel`: dónde cae el plano oclusal
> dentro del recorte. Tiene que dar ~0.79 en superiores y ~0.17 en inferiores;
> alejarse de ahí delata una detección fallada. En la corrida vigente los 26
> superiores dan 0.80 y los inferiores 0.17, con dos anteriores en 0.28-0.33 que
> toparon contra la banda de anotación y se revisaron a ojo sin problema.
>
> `relativo_al_mejor` sigue estando pero **ya no se lee contra un umbral fijo**:
> el encuadre nuevo incluye hueso a propósito, así que bajó para todos.
>
> Regenerar los recortes **no toca el diseño experimental**. El emparejamiento, la
> deduplicación perceptual y la selección de Q3 se calculan sobre imágenes
> completas, y el recorte se aplica recién al escribir los PNG; los nombres de
> archivo son `sha1(image_id, condición)`. Verificado tras la regeneración: los
> 130 ensayos, sus pares de archivos y el reparto entre los 8 evaluadores salen
> idénticos a los de `webapp/public/assets/sets.json`.

### 2.1 Deduplicación perceptual

Por cada imagen, regenerar las soluciones candidatas y agrupar por equivalencia
perceptual (SSIM ≥ 0.98, union-find). Cubre el requisito de no duplicar las
soluciones de consenso, y evita que dos tercios de las pantallas sean pares
indistinguibles. Salida: `study_conditions.csv` con una fila por
(imagen, clase perceptual, métodos que la eligieron).

### 2.2 Asignación de regiones

Un sextante por imagen, aleatorización balanceada con semilla sobre los 6
sextantes (~8 imágenes cada uno). Las coordenadas del recorte se derivan de la
línea oclusal ya detectada en `screen_evaluation_set.py:trace_occlusal_line`,
para que el recorte siga la curva de la arcada y no sea un rectángulo ciego.

**El recorte es de visualización, no de edición**: las imágenes de
`data/evaluacion_50/` y las salidas del experimento quedan intactas, con su
anotación quemada. El visor puede ofrecer un botón "ver imagen completa".

### 2.3 Generación de pares

Las 130 comparaciones de la §5 del protocolo: 50 de Q2, 45 de Q4, 20 de Q3,
15 de Q1. Con el recorte idéntico en ambos miembros del par, izquierda/derecha
al azar, y sin ningún rótulo. Salida: `trials.json`.

### 2.4 Diseño de bloques incompletos balanceados

Reparto de los 130 ensayos entre R evaluadores de modo que cada comparación
reciba el mismo número de juicios. Salida: `assignments/<evaluador>.json`,
determinado por semilla.

---

## Fase 3 — Aplicación de evaluación ✅ HECHO

`scripts/generate_evaluation_app.py` → `evaluador_01.html` … `evaluador_08.html`,
75 pantallas cada uno (5 de práctica + 65 ensayos + 5 repetidos). Se abren con
doble clic; no necesitan conexión ni servidor. Instrucciones de uso en
`docs/evaluacion/estudio/LEEME.md`.

Probado en navegador de punta a punta: carga de imágenes por ruta relativa
`file://`, flujo elección → confianza → avance, registro de tiempo por ensayo,
zoom y desplazamiento sincronizados entre las dos imágenes, persistencia en
`localStorage`, y las 75 pantallas hasta la descarga del JSON. Verificado que los
5 de práctica **no** solapan con los 65 propios (salen de ensayos asignados a
otros evaluadores, así no contaminan ni adelantan pares) y que los 5 repetidos sí
son un subconjunto de los propios.

Las imágenes van como archivos en `img/` y no embebidas en base64: en PNG sin
pérdida, embeberlas daba ~40 MB por archivo × 8. Se usa PNG y no JPEG a propósito
-- recomprimir con pérdida introduciría artefactos justo en un estudio que mide
percepción de calidad.

Los identificadores de ensayo son hashes: la condición de cada imagen no se puede
inferir ni mirando el código fuente. La correspondencia vive sólo en `clave.csv`.

### 3.bis Versión web ✅ HECHO — **es la que se usa**

`webapp/` reemplaza al reparto de archivos por Drive: un enlace único, sesiones
con reanudación y respuestas que llegan solas a una base de datos. Guía de
despliegue paso a paso en [`webapp/README.md`](../../webapp/README.md); los
materiales estáticos los arma `scripts/build_web_study.py` a partir de
`trials.json`.

El sitio **no manda correo**. El correo del participante se pide sólo para
asignarle su set y para que pueda retomar; quién terminó sale de la vista
`vw_progreso` si hace falta agradecer a mano.

Diferencias con la app offline, que hay que tener presentes al analizar:

- **70 pantallas, no 75: se sacaron los 5 ensayos de práctica.** El efecto de
  aprendizaje de las primeras pantallas ahora cae sobre ensayos que sí entran al
  análisis. Como el orden es aleatorio por participante, el ruido se reparte
  entre los 130 ensayos en vez de concentrarse. Los 5 repetidos se mantienen.
- **El set se asigna solo**, por orden de registro (el menos usado, desempatando
  por el más chico). El 9no participante vuelve al set 01 con otro orden.
- **El orden de los ensayos sale de una semilla por participante**, no por
  evaluador: dos personas con el mismo set no ven la misma secuencia.
- **No se piden años de ejercicio ni especialidad.** Si el análisis por
  experiencia va a hacer falta, hay que recolectarlos aparte.
- **Las imágenes del par se rotulan 1 y 2**, no A y B, y el tercer nivel de
  confianza pasó de «Estoy adivinando» a **«Se ven iguales»**, que se entiende
  sin explicación y apunta directo a lo que interesa medir. **El valor que se
  guarda sigue siendo `adivinando`**: el plan de análisis no cambia.
- **La identidad visual sigue el Manual de Identidad Institucional FP-UNA**
  (azul `#384f87`, Titillium y Lato, escudo oficial).

La lógica de sesión, guardado y asignación está en `webapp/supabase/esquema.sql`;
el navegador no puede tocar las tablas, sólo llamar a esas funciones.

---

## Fase 3.ter — Puesta en línea ⬜ POR HACER — **acá es donde sigue**

El código está listo y probado de punta a punta en local. Falta el trámite de
cuentas, que son ~30 minutos y no requiere tocar código. Guía clic por clic en
[`webapp/README.md`](../../webapp/README.md); el resumen:

1. **Supabase** — crear el proyecto (región São Paulo), pegar
   `webapp/supabase/esquema.sql` en el SQL Editor y correrlo, copiar *Project
   URL* y clave *anon*.
2. **`webapp/public/assets/config.js`** — pegar esos dos valores. Va antes de
   subir: Netlify publica los archivos tal como estén.
3. **Netlify** — arrastrar `webapp/public` a <https://app.netlify.com/drop>,
   crear la cuenta gratuita para que el sitio no expire y ponerle un nombre.
4. **Probar y limpiar** — registrarse con un correo propio, verificar en
   `vw_progreso`, y después `delete from participantes;` para que el reparto de
   sets arranque en el 01.

Recién con la URL en mano se puede convocar.

---

## Fase 4 — Piloto ⬜ POR HACER

**Con 1-2 odontólogos, antes de convocar al resto.** Objetivo: detectar
problemas de instrucciones, tiempos y usabilidad mientras corregirlos todavía es
barato. Se mide:

- Tiempo real por ensayo (la estimación de 20-25 s es una hipótesis).
- Si las instrucciones se entienden sin explicación oral.
- Si el recorte de sextante es suficiente o piden ver la imagen completa.
- Si el modal «Cómo se responde» alcanza, o hace falta explicar algo por
  teléfono igual.
- Si «Se ven iguales» se entiende como lo que es, o alguien lo lee como
  «no quiero elegir».

Los datos del piloto **no** entran en el análisis final.

---

## Fase 5 — Recolección ⬜ POR HACER

- **8-10 odontólogos** (mínimo 5, ver §5 del protocolo).
- Sesión de ≤30 min por evaluador.
- **Un correo distinto por persona**: de ahí sale el set que le toca.
- Las covariables (años de ejercicio, especialidad) **el sitio no las pide**. Si
  van a hacer falta, hay que anotarlas aparte al convocar.
- Consentimiento informado; **consultar antes si hace falta aval del comité de
  ética de la facultad**.
- Seguimiento en vivo desde `vw_progreso`, en el panel de Supabase.

---

## Fase 6 — Análisis ⬜ POR HACER

Entregable: `scripts/analyze_evaluation.py`. **El plan se fija antes de ver los
datos** (§7 del protocolo).

1. Exclusión de evaluadores que fallen >20% de los ensayos trampa de Q1.
2. Q1/Q2: proporción de preferencia con IC 95% por bootstrap agrupado por
   evaluador. Q2 planteado como **no inferioridad** con margen δ prefijado.
3. Q4: qué criterio gobierna la preferencia, con contraste máximo.
4. Regresión `P(prefiere A) ~ β_H·ΔH + β_SSIM·ΔSSIM + β_VIF·ΔVIF + (1|evaluador)
   + (1|imagen)` → **pesos empíricos** contra los ROC (0.611, 0.278, 0.111).
5. Ranking de métodos MCDM por alineación con la utilidad empírica, sobre las 50
   imágenes (§2.bis del protocolo — la vía indirecta).
6. Q3: Bradley-Terry sobre los pares de alto contraste, sólo descriptivo si hay
   pocos evaluadores.
7. Umbral de discriminación: tasa de acierto vs. SSIM del par. Calibra
   empíricamente el corte de 0.98 que hoy es hipótesis.
8. Fiabilidad: κ de Fleiss entre evaluadores, test-retest intra-evaluador.

---

## Riesgos

| Riesgo | Mitigación |
|---|---|
| No se consiguen 8 odontólogos | El diseño degrada con gracia: con 5 se reportan Q1, Q2 y Q4; Q3 pasa a descriptivo |
| Q3 sale sin significancia | Es un resultado, no un fracaso: es la predicción de `colapso_perceptual_mcdm.md`. Se reporta como equivalencia (TOST), no como ausencia de efecto |
| Fatiga pese a los controles | El tiempo por ensayo se mide desde el piloto; si sube, se recorta Q3 primero |
| El Tier B contamina | Está etiquetado en `ids_aprobados.txt`: se puede repetir el análisis sin esas 10 imágenes como control de sensibilidad |
| El sextante asignado no tiene hallazgos interesantes | La aleatorización balanceada lo reparte; además Q2 y Q4 no dependen de que haya patología |
