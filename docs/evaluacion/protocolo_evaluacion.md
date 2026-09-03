# Protocolo de evaluación perceptual con odontólogos

Borrador para discutir con el tutor. Versión 2026-08-03.

Estudio de validación perceptual del framework SMPSO-CLAHE + MCDM sobre
ortopantomografías. Todo lo cuantitativo de acá sale del experimento `n=83`
(`results/experiment_20260709_215441`) y es reproducible con
`scripts/perceptual_equivalence.py`.

---

## 1. El hallazgo que condiciona todo el diseño

Antes de decidir el formato hay que mirar un número que cambia el problema.

Los 8 métodos MCDM eligen filas distintas del frente de Pareto casi siempre: el
acuerdo exacto medio entre pares de métodos es 17.3%, y por imagen eligen en
promedio **5.07 soluciones distintas**. Con eso, enfrentar todos los pares de
candidatos daría 541 comparaciones para 50 imágenes (1098 si además se meten la
original y la degradada). Inviable.

Pero "fila distinta de la matriz de decisión" no es lo mismo que "imagen distinta
para el ojo". Al regenerar las imágenes realzadas de cada candidato y comparar
todos los pares (898 pares sobre 83 imágenes):

| SSIM entre candidatos | Pares | % | MAE medio (niveles de gris) |
|---|---:|---:|---:|
| [0.00, 0.90) | 48 | 5.3% | 18.34 |
| [0.90, 0.95) | 67 | 7.5% | 16.40 |
| [0.95, 0.98) | 185 | 20.6% | 10.61 |
| [0.98, 0.99) | 251 | 28.0% | 6.93 |
| [0.99, 1.00] | 347 | 38.6% | 3.14 |

**Dos tercios de los pares difieren en menos de 7 niveles de gris sobre 255.**
Agrupando por equivalencia perceptual, la cantidad de candidatos realmente
distintos por imagen se desploma:

| Corte de equivalencia | Candidatos distintos por imagen | Imágenes con un solo candidato | Comparaciones en 50 imágenes |
|---|---:|---:|---:|
| sin fusionar | 5.07 | 0/83 | 541 |
| SSIM ≥ 0.99 | 2.37 | 15/83 | 104 |
| SSIM ≥ 0.98 | 1.60 | 42/83 | 36 |
| SSIM ≥ 0.97 | 1.34 | 57/83 | 18 |
| SSIM ≥ 0.95 | 1.17 | 69/83 | 8 |

Con el corte operativo de 0.98, **en la mitad de las imágenes los 8 métodos
colapsan a una sola imagen perceptualmente distinta**. La conclusión no depende
de dónde se ponga exactamente el corte: en todo el rango razonable las
comparaciones MCDM-vs-MCDM caen de 541 a entre 18 y 104.

**Replicado sobre las 50 imágenes que usa este estudio**
(`results/experiment_20260803_203341`), que son un conjunto distinto — dentición
completa, sin trabajos, degradaciones balanceadas: 70.6% de pares con SSIM ≥ 0.98,
1.54 candidatos distintos por imagen, **29/50 imágenes (58%) colapsan a una
sola**, y quedan 33 comparaciones MCDM-vs-MCDM. Las cifras de arriba, medidas
sobre n=83, se sostienen.

Esto tiene tres consecuencias:

1. **El estudio es viable.** El problema combinatorio se disuelve solo.
2. **El núcleo del estudio no puede ser MCDM-vs-MCDM.** No hay suficiente
   señal perceptual: en la mitad de los casos no hay nada que comparar.
3. **Es un resultado publicable por sí mismo**, y bastante fuerte: los métodos
   MCDM discrepan numéricamente pero convergen perceptualmente. Para el usuario
   final, la elección del método MCDM es en buena medida indiferente. Eso
   refuerza la propuesta de consenso en lugar de debilitarla.

> El corte de 0.98 es una hipótesis, no un hecho. El propio estudio la calibra:
> se incluyen pares de distintas bandas de SSIM y se estima a partir de qué
> distancia los odontólogos discriminan por encima del azar (§7).

---

## 2. Qué se compara: respuesta a las tres alternativas planteadas

Las tres opciones eran: (1) original vs. MCDM, (2) degradada vs. MCDM,
(3) original + degradada vs. MCDM.

**Recomendación: ninguna tal cual, sino la 3 reformulada.** En lugar de un panel
de tres imágenes, meter la original y la degradada en el *mismo pool anónimo*
que los candidatos MCDM y comparar de a pares. Razones:

- Un panel de tres con la degradada presente desperdicia atención: la degradada
  es obviamente la peor y ancla el juicio. El odontólogo gasta su atención en el
  descarte fácil en lugar de en la comparación informativa.
- Comparando de a pares con un modelo de Bradley-Terry, todas las condiciones
  quedan en **una misma escala latente de calidad**. Se obtiene lo mismo que
  daría el panel de tres, y además se puede repartir el presupuesto de ensayos
  donde está la información.
- El cegado sale gratis: ninguna imagen lleva rótulo, y la original no se
  presenta como "referencia". Esto último importa: si se la rotula como
  referencia, el odontólogo tiende a preferir lo que más se le parezca, lo que
  sesga *en contra* de encontrar que la realzada supera a la original. Como
  competidora anónima, si igual gana la realzada, el resultado es limpio.

Con eso, las tres preguntas se responden en una sola tarea:

| | Pregunta | Comparación | Efecto esperado |
|---|---|---|---|
| **Q1** | ¿El realce sirve? | degradada vs. realzada | Enorme; sirve de control de atención |
| **Q2** | ¿El realce alcanza a la original prístina? | original vs. realzada | Moderado; **es la pregunta interesante** |
| **Q3** | ¿Qué método MCDM elige mejor? | realzada_i vs. realzada_j | Pequeño y sólo en ~36 pares |
| **Q4** | ¿Qué criterio gobierna el juicio experto? | extremos mono-objetivo del frente | **El mayor del estudio** (ver abajo) |

**Q4 es un agregado que sale del análisis de mecanismo** (`docs/evaluacion/colapso_perceptual_mcdm.md`
§2). Las soluciones que maximizan un solo objetivo — mejor-H, mejor-SSIM,
mejor-VIF — son lo más distinguible que produce el framework: SSIM medio ~0.91 y
sólo 17% de pares indistinguibles sobre las 50 imágenes del estudio, contra 70%
entre las selecciones MCDM. Enfrentarlas mide con contraste máximo cuál de los
tres criterios prefiere el odontólogo, que es exactamente lo que Q3 quiere saber
y no puede. Cuestan 3 comparaciones por imagen y son las de mayor rendimiento por
pantalla de todo el estudio.

> No son *garantizadamente* distinguibles: en el 17% de los casos los óptimos de
> dos objetivos caen muy cerca. Esos pares hay que filtrarlos igual que los
> MCDM, con el mismo criterio de SSIM, al generar los ensayos en la Fase 2.

Q2 es donde debe ir el grueso del presupuesto. Q1 ya está demostrado
cuantitativamente (VIF 0.682 → 0.950, Wilcoxon p<1e-14) y perceptualmente es
trivial; sirve como trampa para detectar evaluadores que responden sin mirar.

---

## 2.bis ¿Se puede asociar un método MCDM concreto al criterio de los odontólogos?

Es la pregunta correcta y la respuesta tiene dos mitades.

### Directamente, en buena medida no

Y por dos razones que conviene no mezclar, porque una no se arregla con más
evaluadores y la otra sí.

**Razón estructural (no tiene arreglo).** SMARTER y MABAC eligen la misma
solución en el **100%** de las imágenes (SSIM medio 1.000 exacto). No es que
cuesten distinguir: es la equivalencia ordinal ya demostrada algebraicamente
para criterios de beneficio con normalización max-min, donde el puntaje de MABAC
es la utilidad aditiva de SMARTER más una constante. Ningún experimento, con
ningún número de odontólogos, puede separarlos. Lo mismo vale, imagen por
imagen, para cualquier par de métodos que haya elegido la misma alternativa.

**Razón empírica (es cuestión de potencia, pero la potencia no alcanza).**
Entre los métodos que sí eligen distinto, la mayoría de los pares son
perceptualmente equivalentes casi siempre:

| Método | SSIM medio contra los demás | % de pares indistinguibles (≥0.98) |
|---|---:|---:|
| PROMETHEE II | 0.963 | 52.8% |
| Bellman-Zadeh | 0.974 | 58.9% |
| TOPSIS | 0.983 | 80.2% |
| CODAS | 0.984 | 79.2% |
| GRA | 0.986 | 80.9% |
| VIKOR | 0.987 | 82.1% |
| SMARTER | 0.988 | 84.2% |
| MABAC | 0.988 | 84.2% |

Un ranking perceptual de los 8 métodos estaría estimado, para la mayoría de los
pares, sobre comparaciones donde el evaluador está adivinando. Se puede calcular,
pero los intervalos de confianza lo dejarían sin valor.

**Dónde sí hay señal:** PROMETHEE II es el único que se despega de verdad, y el
par con más contraste es **Bellman-Zadeh vs. PROMETHEE II** — eligen la misma
solución sólo el 3.6% de las veces, con SSIM medio 0.951. Son los dos extremos
del espacio de métodos (el maximin no compensatorio contra el de flujos de
sobreclasificación). Si los odontólogos no distinguen *esos dos*, la pregunta
queda cerrada con un experimento mucho más barato que rankear los ocho.

### Indirectamente sí, y es mejor diseño

La asociación no corre por la **etiqueta del método** sino por los **criterios**.
Cada comparación mostrada tiene conocidos ΔH, ΔSSIM y ΔVIF entre las dos
imágenes. Con eso se ajusta

```
P(prefiere A)  ~  β_H·ΔH + β_SSIM·ΔSSIM + β_VIF·ΔVIF + (1|evaluador) + (1|imagen)
```

y los β estimados **son pesos empíricos**, directamente comparables con los pesos
ROC que hoy usa el framework (0.611, 0.278, 0.111 para VIF > H > SSIM). Esto:

1. **Usa todos los juicios** (~500), no sólo los de las imágenes donde los
   métodos discrepan, porque los predictores son continuos y no la identidad del
   método. Las imágenes donde todo colapsa aportan ΔH ≈ ΔSSIM ≈ ΔVIF ≈ 0 y
   preferencia al azar, que es información válida sobre la pendiente.
2. **Recupera el ranking de métodos**: una vez estimada la función de utilidad
   empírica, se puntúa cada método por cuán cerca están sus selecciones del
   óptimo bajo esa utilidad, **sobre las 50 imágenes**, incluidas las colapsadas.
   Se pasa de "¿cuál gana el enfrentamiento?" (imposible) a "¿cuál elige lo que
   los odontólogos valoran?" (bien potenciado).
3. Responde la pregunta que de verdad importa para la tesis: **valida o refuta
   el orden VIF > H > SSIM**, que hoy se sostiene sólo por argumento.

En resumen: no se puede coronar un método MCDM por enfrentamiento perceptual, y
además el hallazgo de §1 dice que esa corona no significaría gran cosa. Lo que sí
se puede es medir qué criterio objetivo gobierna el juicio experto, y ordenar los
métodos por su alineación con él.

---

## 3. La unidad de evaluación: región, no imagen completa

El tutor tiene razón en que el diagnóstico se hace por regiones, y la objeción
de que enumerar regiones dispara el test también es correcta: 50 imágenes × 6
sextantes = 300 pantallas.

**Solución: muestrear regiones, no enumerarlas.** Cada imagen aporta **una sola
región**, asignada por aleatorización balanceada sobre los 6 sextantes
(anterior / posterior derecho / posterior izquierdo × superior / inferior).

- Se respeta el criterio clínico: se juzga una región a magnificación
  diagnóstica, no una panorámica entera miniaturizada.
- El test sigue teniendo 50 pantallas, no 300.
- Con 50 imágenes cada sextante cae ~8 veces, suficiente para reportar
  resultados por región.
- Resuelve además un problema práctico: la panorámica es 2041×1024 (~2:1). Dos
  panorámicas completas lado a lado no entran a resolución útil en ninguna
  pantalla. Dos recortes de sextante, sí.

El recorte es **idéntico en ambas imágenes del par** (mismas coordenadas), así
la comparación queda exactamente pareada.

> Si el tutor prefiere más regiones por imagen, el canje es directo a igual
> carga: 25 imágenes × 2 regiones, o 17 × 3. Se pierde diversidad de pacientes y
> se gana cobertura regional. Mi recomendación es 50 × 1, porque la variabilidad
> entre pacientes es mayor que entre regiones del mismo paciente.

---

## 4. La tarea, pantalla por pantalla

Sobre cada pantalla: dos recortes de la misma región de la misma radiografía,
sin rótulos, leyendas, bordes ni nombres de método; izquierda/derecha asignadas
al azar en cada ensayo.

1. **Elección forzada:** *"¿En cuál de las dos evaluaría esta región con más
   confianza?"* — A / B, sin empate.
2. **Confianza:** *"Seguro / Algo seguro / Estoy adivinando"*.

La opción "estoy adivinando" es la que mide indistinguibilidad sin romper el
modelo de Bradley-Terry (un empate explícito sí lo rompería). Dado el hallazgo
de §1, se espera que sea la respuesta más frecuente en los pares MCDM-vs-MCDM, y
esa frecuencia es en sí un resultado.

Opcionalmente, una tercera pregunta anclada a una estructura anatómica de la
región mostrada (unión amelocementaria, lámina dura, espacio del ligamento
periodontal, trabeculado óseo, cámara pulpar). Esto convierte el juicio de
estético en **diagnóstico** y sigue la lógica del *Visual Grading Analysis*, que
es la metodología establecida para estudios de calidad de imagen radiográfica.
**La lista final de estructuras la deben fijar el tutor y los odontólogos**; acá
sólo se propone el mecanismo.

---

## 5. Carga: los números

Comparaciones únicas necesarias sobre las 50 imágenes:

| Bloque | Comparaciones | Criterio |
|---|---:|---|
| Q2 — original vs. consenso | 50 | una por imagen |
| Q4 — extremos mono-objetivo | 45 | 15 imágenes × 3 pares (H/SSIM, H/VIF, SSIM/VIF) |
| Q3 — MCDM vs. MCDM | 20 | sólo los pares de mayor contraste (PROMETHEE II vs. Bellman-Zadeh y similares, SSIM < 0.95) |
| Q1 — degradada vs. realzada | 15 | control de atención, intercalado |
| **Total** | **130** | |

Q3 se recorta de ~36 a 20 a propósito: es el bloque con menos señal por ensayo,
y el presupuesto rinde mucho más en Q4, que mide lo mismo (qué criterio prefiere
el experto) con contraste garantizado en lugar de marginal.

Presupuesto por evaluador:

- ~20-25 s por ensayo para un odontólogo que mira con cuidado.
- Sesión de **≤30 min**, en línea con la práctica habitual en estudios de
  calidad subjetiva de imagen (ITU-R BT.500).
- **65 ensayos efectivos + 5 de calentamiento (descartados) + 5 repetidos**
  (fiabilidad test-retest) = 75 pantallas, ~25-30 min.

Con 130 comparaciones únicas y 4 evaluadores por comparación → 520 juicios →
**8 odontólogos**. Con 10 se llega a 5 evaluadores por comparación. Con un
mínimo de 5 odontólogos quedan ~2.5 por comparación: alcanza para Q1, Q2 y Q4,
y obliga a reportar Q3 sólo de forma descriptiva.

El reparto es un **diseño de bloques incompletos balanceados**: ningún evaluador
ve las 101, y cada comparación recibe la misma cantidad de evaluadores.

---

## 6. Control de sesgo y de fatiga

Sesgo:

- Sin captions, leyendas, nombres de método ni identificadores visibles.
- Izquierda/derecha aleatoria por ensayo; orden de ensayos aleatorio por
  evaluador.
- La original **no** se presenta como referencia: compite anónima.
- Deduplicación previa: si dos métodos eligieron la misma solución, o dos
  soluciones perceptualmente equivalentes, el par no se presenta. Esto cubre el
  requisito de no duplicar las soluciones de consenso.
- **Las imágenes no se modifican.** Los archivos de `data/evaluacion_50/`
  conservan intactas las bandas de anotación quemada (equipo, fecha, kV/mA,
  marcas R/L); no se borra ni se altera nada. Las imágenes pertenecen al mismo
  grupo evaluador, así que no hay problema de privacidad que resolver.
  La vista de evaluación muestra un recorte del sextante por la razón clínica
  de §3 -- se juzga una región a magnificación diagnóstica --, no por censura:
  el archivo completo queda disponible y se puede ofrecer un botón para verlo
  entero. Como efecto secundario el recorte evita un sesgo real: el texto
  quemado está saturado a 255, y la degradación y CLAHE lo renderizan distinto
  en cada condición, así que en la imagen completa su aspecto podría delatar
  qué versión es cuál.

Fatiga:

- Barra de progreso y pausa/reanudación (el evaluador puede cortar y seguir).
- Descanso sugerido cada 20 ensayos.
- **Ensayos trampa:** los 15 de Q1 (degradada vs. realzada) tienen respuesta
  casi obvia. Un evaluador que falle más de ~20% de ellos responde sin mirar y
  se excluye. Criterio de exclusión **pre-registrado**, antes de ver los datos.
- **Ensayos repetidos:** 5 pares se repiten al final para medir consistencia
  intra-evaluador.
- Registro del tiempo de respuesta por ensayo: una caída sistemática delata el
  punto donde empieza a responder por responder.

---

## 7. Plan de análisis

Fijado **antes** de recoger datos.

- **Q1 y Q2:** proporción de preferencia con IC 95% por bootstrap agrupado por
  evaluador (o GLMM con efectos aleatorios de evaluador y de imagen), contra
  0.5. Para Q2 el planteo correcto es de **no inferioridad**: la afirmación
  interesante no es "la realzada gana" sino "la realzada no es peor que la
  original prístina". Hay que fijar el margen δ de antemano (propuesta: 0.10).
- **Q3:** modelo de Bradley-Terry sobre los métodos MCDM → escala latente de
  calidad con intervalos de confianza. Más la matriz cruda de tasas de victoria.
- **Fiabilidad:** κ de Fleiss o α de Krippendorff entre evaluadores; acuerdo
  test-retest intra-evaluador sobre los 5 repetidos.
- **Umbral de discriminación perceptual:** tasa de acierto vs. SSIM del par.
  Estima empíricamente a partir de qué distancia dos soluciones son
  distinguibles, y convierte el corte arbitrario de 0.98 en un resultado medido.
- **El análisis que más aporta a la tesis:** correlacionar la preferencia de los
  odontólogos con ΔH, ΔSSIM y ΔVIF entre las dos imágenes mostradas. Responde
  *cuál de las tres métricas objetivas predice mejor el juicio experto*, y con
  eso **valida o refuta empíricamente el orden de importancia VIF > H > SSIM**
  que hoy sostiene los pesos ROC. Hoy ese orden se justifica por argumento; con
  esto pasaría a estar respaldado por datos.

---

## 8. Plataforma

Recomendación: **una página HTML autocontenida**, en la línea del `report.html`
que ya genera `scripts/generate_interactive_report.py`. Se abre en el navegador,
funciona sin conexión y al terminar descarga un JSON con las respuestas.

Frente a Google Forms: control de la aleatorización por evaluador, imágenes a
resolución diagnóstica con zoom, registro de tiempos de respuesta, imposibilidad
de volver atrás a cambiar respuestas, y reproducibilidad (la asignación de
ensayos queda determinada por una semilla). Google Forms no da nada de eso y
además muestra las imágenes reescaladas.

---

## 9. Ética y datos

- Es un estudio con sujetos humanos (los evaluadores), aunque sea de percepción.
  Corresponde consentimiento informado y probablemente aval del comité de la
  facultad. **Hay que consultarlo antes de empezar.**
- Las radiografías pertenecen al mismo grupo evaluador y no salen de él, así que
  no se modifican ni se les quita la anotación quemada (ver §6).
- Registrar el nivel de experiencia de cada evaluador (años de ejercicio,
  especialidad) como covariable.

---

## 10. Lo que falta decidir

1. **Lista de estructuras anatómicas** para la pregunta de grading (clínico —
   tutor y odontólogos).
2. **Cuántos odontólogos** se consiguen realmente. De eso depende si Q3 entra o
   se reporta sólo descriptivamente.
3. **Sextantes vs. otra partición** de regiones.
4. **Margen de no inferioridad δ** para Q2.
5. Si se aplica o no el criterio de **dentición permanente adulta**: varias de
   las mejores candidatas del triaje son de pacientes pediátricos con dentición
   mixta (§ selección de imágenes).
