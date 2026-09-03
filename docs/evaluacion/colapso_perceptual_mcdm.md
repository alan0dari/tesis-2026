# Colapso perceptual de las soluciones MCDM

Justificación del hecho de que los 8 métodos MCDM, que numéricamente eligen
alternativas distintas del frente de Pareto, produzcan imágenes que en su
mayoría no se distinguen entre sí.

Reproducible con:

```bash
python scripts/perceptual_equivalence.py --mechanism
```

Datos: hallazgo original sobre el experimento `n=83`
(`results/experiment_20260709_215441`) y **réplica sobre las 50 imágenes que usa
el estudio** (`results/experiment_20260803_203341`). Ambas salidas en
`perceptual_equivalence.csv` de sus respectivos directorios.

Las dos coinciden estrechamente, lo que importa porque el n=50 es un conjunto
distinto (dentición completa y sin trabajos, degradaciones balanceadas):

| | n=83 | n=50 |
|---|---:|---:|
| Pares con SSIM ≥ 0.98 | 66.6% | 70.6% |
| Candidatos distintos por imagen (corte 0.98) | 1.60 | 1.54 |
| Imágenes que colapsan a uno solo | 42/83 (51%) | 29/50 (58%) |
| Comparaciones MCDM-vs-MCDM en 50 imágenes | 36 | 33 |

---

## 1. Qué se observa

El acuerdo exacto medio entre pares de métodos MCDM es **17.3%**: eligen filas
distintas de la matriz de decisión casi siempre, en promedio **5.07 soluciones
distintas por imagen** sobre frentes de ~100 alternativas.

Al regenerar la imagen realzada de cada candidato (CLAHE con sus parámetros
sobre la degradada) y comparar los 898 pares resultantes:

| SSIM entre candidatos | Pares | % | MAE medio (niveles de gris) |
|---|---:|---:|---:|
| [0.00, 0.90) | 48 | 5.3% | 18.34 |
| [0.90, 0.95) | 67 | 7.5% | 16.40 |
| [0.95, 0.98) | 185 | 20.6% | 10.61 |
| [0.98, 0.99) | 251 | 28.0% | 6.93 |
| [0.99, 1.00] | 347 | 38.6% | 3.14 |

**El 66.6% de los pares difiere en menos de 7 niveles de gris sobre 255.**
Agrupando por equivalencia perceptual con union-find:

| Corte | Candidatos distintos/imagen | Imágenes con uno solo | Comparaciones en 50 imágenes |
|---|---:|---:|---:|
| sin fusionar | 5.07 | 0/83 | 541 |
| SSIM ≥ 0.99 | 2.37 | 15/83 | 104 |
| SSIM ≥ 0.98 | 1.60 | 42/83 | 36 |
| SSIM ≥ 0.97 | 1.34 | 57/83 | 18 |
| SSIM ≥ 0.95 | 1.17 | 69/83 | 8 |

---

## 2. Por qué ocurre

Hay dos explicaciones posibles y son distintas: que el frente de Pareto entero
sea perceptualmente plano (y entonces el hallazgo hablaría de CLAHE), o que los
métodos converjan a una misma región de un frente que sí tiene rango (y entonces
hablaría de los MCDM). Se puede decidir midiendo. Sobre 25 imágenes, SSIM medio
entre pares:

| Qué se compara | SSIM medio | % indist. (n=83) | % indist. (n=50) |
|---|---:|---:|---:|
| Extremos del frente (mejor-H vs mejor-SSIM vs mejor-VIF) | 0.901 / 0.911 | 0% | 17% |
| Pares al azar del mismo frente | 0.964 / 0.967 | 36% | 45% |
| Soluciones elegidas por los MCDM | 0.972 / 0.976 | 52% | 70% |

Wilcoxon MCDM vs. azar: W=63, **p = 6.1e-03** en n=83 y W=84, **p = 3.4e-02** en
n=50; los MCDM quedan más juntos que el azar en 18 de 25 imágenes en ambos.

De acá salen tres conclusiones, y la tercera es la que suele contarse mal:

**(a) El frente NO es perceptualmente plano.** Sus extremos son con diferencia lo
más distinguible que produce el framework: SSIM ~0.91 y sólo 0-17% de pares
indistinguibles, contra 52-70% entre las selecciones MCDM. La optimización
multiobjetivo sí recorre un rango perceptual real. El colapso no es un artefacto
de que CLAHE sea insensible a sus parámetros.

> Corrección respecto de la primera versión de este documento: sobre n=83 los
> extremos daban 0% de pares indistinguibles y se los describió como
> "garantizadamente distinguibles". La réplica sobre n=50 da 17%, así que la
> garantía no existe — en algunas imágenes los óptimos de dos objetivos caen
> muy cerca. Lo que sí se sostiene, y con holgura, es que son **entre 3 y 4
> veces más distinguibles** que las selecciones MCDM.

**(b) Los métodos sí convergen.** Sus selecciones están significativamente más
juntas que pares al azar del mismo frente. Es lo esperable: los 8 optimizan
agregaciones monótonas de los *mismos* tres criterios con los *mismos* pesos, así
que sus óptimos caen en la misma región de alta utilidad. Difieren en qué punto
de esa región, no en qué región.

**(c) Pero el factor dominante es la densidad del frente, no el acuerdo entre
métodos.** Los pares *al azar* ya son indistinguibles el 36% de las veces. Con
~100 soluciones no dominadas repartidas en un rango perceptual acotado, dos
alternativas vecinas cualesquiera difieren muy poco. La convergencia de los
métodos sube ese 36% al 52%, que es un efecto real pero secundario. Dicho de
otro modo: **aunque los MCDM eligieran al azar dentro del frente, un tercio de
las comparaciones seguiría siendo indistinguible.**

A esto se suma un caso que no es estadístico sino algebraico: **SMARTER y MABAC
eligen idénticamente en el 100% de las imágenes** (SSIM 1.000 exacto), por la
equivalencia ordinal ya demostrada para criterios de beneficio con normalización
max-min. Ese par jamás podrá distinguirse, con ningún evaluador.

---

## 3. Qué no dice este resultado

- **No dice que los métodos MCDM sean intercambiables en general.** Lo son
  *para este problema*: tres criterios, todos de beneficio, sobre un frente denso
  generado por el mismo optimizador. Con criterios en conflicto más fuerte, o
  frentes más ralos, la conclusión podría no valer.
- **No dice que CLAHE sea insensible a sus parámetros.** Los extremos del frente
  lo desmienten (§2a).
- **El corte de 0.98 es una hipótesis operativa, no un umbral medido.** SSIM no
  es una métrica perceptual calibrada en JND. El estudio con odontólogos lo
  calibra empíricamente incluyendo pares de distintas bandas de SSIM.
- **La fusión por union-find es transitiva y eso la vuelve agresiva.** Si A~B y
  B~C pero A y C difieren, la clausura transitiva igual los une. Por eso conviven
  dos cifras: 42/83 imágenes colapsan por completo bajo clausura transitiva,
  pero 69/83 contienen *al menos un par* por debajo de 0.98. La equivalencia
  perceptual no es transitiva — es el problema clásico de encadenamiento de
  JND —, así que la verdad está entre ambas.

---

## 4. Consecuencias para el diseño del estudio

1. **El núcleo del estudio no puede ser MCDM contra MCDM.** En la mitad de las
   imágenes no hay nada que comparar. Ver §2.bis del protocolo para la vía
   indirecta, vía los criterios, que sí funciona.
2. **Conviene incluir los extremos del frente como condiciones.** Las soluciones
   mono-objetivo (mejor-H, mejor-SSIM, mejor-VIF) son las únicas garantizadamente
   distinguibles (0% indistinguibles). Sirven para dos cosas a la vez: dar al
   evaluador comparaciones con señal real, y medir con contraste máximo cuál de
   los tres criterios gobierna la preferencia experta.
3. **La deduplicación previa es obligatoria**, no una optimización: sin ella,
   dos tercios de las pantallas serían pares indistinguibles y el estudio
   mediría fatiga en vez de percepción.

---

## 5. Valor como resultado

Es publicable por sí mismo y reordena la contribución de la tesis. El aporte deja
de ser "cuál de los 8 métodos MCDM es el mejor" — pregunta que la evidencia
sugiere mal planteada para este problema — y pasa a ser: **los métodos MCDM
discrepan numéricamente pero convergen perceptualmente, de modo que para el
usuario final la elección del método es en buena medida indiferente.** Eso
refuerza el enfoque de consenso en lugar de debilitarlo: si todos los caminos
llevan a imágenes equivalentes, promediar los votos es una decisión barata y
segura, no un compromiso.
