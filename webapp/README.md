# Sitio de evaluación con odontólogos

La versión web del estudio perceptual. Reemplaza a los ocho `evaluador_NN.html`
que había que mandar por Drive y que devolvían un JSON a mano: ahora es un enlace
único, las respuestas llegan solas a una base de datos y el participante puede
cortar y retomar desde donde quiera.

El diseño experimental **no cambia**. Los 130 ensayos, el reparto en bloques
incompletos balanceados y los recortes son exactamente los de
`docs/evaluacion/estudio/`. Lo que cambia respecto de la app offline:

| | App offline | Sitio web |
|---|---|---|
| Pantallas por evaluador | 75 (5 práctica + 65 + 5 repetidos) | **70** (65 + 5 repetidos) |
| Ensayos de práctica | 5 | **ninguno** |
| Asignación de set | a dedo, al mandar el archivo | automática, por orden de registro |
| Orden de los ensayos | fijo por semilla del evaluador | por semilla propia de cada participante |
| Persistencia | `localStorage`, misma máquina | base de datos, cualquier máquina |
| Devolución de datos | JSON por correo | automática |
| Covariables | anotadas aparte | no se piden |
| Rótulo de las imágenes | A / B | **1 / 2** |
| Tercer nivel de confianza | «Estoy adivinando» | **«Se ven iguales»** |

> El rótulo cambió; **el valor que se guarda sigue siendo `adivinando`**, así que
> el plan de análisis y `clave.csv` no se tocan. La etiqueta nueva mide lo mismo
> y se entiende sin explicación: lo que interesa de ese botón es la
> indistinguibilidad, no la introspección sobre si estaba adivinando.

> Sacar los ensayos de práctica tiene un costo conocido: el efecto de aprendizaje
> de las primeras pantallas ahora cae sobre ensayos que sí entran al análisis.
> Como el orden es aleatorio por participante, ese ruido se reparte parejo entre
> los 130 ensayos en vez de concentrarse; no lo elimina, lo diluye.

---

## Qué hay acá

```
webapp/
├── public/                     ← esto es lo que se sube al hosting
│   ├── index.html              presentación + evaluación + gracias
│   ├── info.html               la sección informativa del botón ⓘ
│   ├── img/                    generado (193 recortes)
│   └── assets/
│       ├── estilo.css
│       ├── app.js
│       ├── config.ejemplo.js   ← copiar a config.js y completar
│       ├── config.js           (fuera de git: lleva tus claves)
│       ├── sets.json           generado
│       ├── concepto/           generado (figuras de info.html)
│       └── marca/              escudo e isotipo oficiales de la FP-UNA
├── supabase/
│   └── esquema.sql             tablas + funciones + permisos
└── mantener-despierto.yml      opcional, ver §6
```

`public/img/` y `public/assets/concepto/` no están en git: los genera
`scripts/build_web_study.py` a partir del material del estudio.

La identidad visual sigue el **Manual de Identidad Institucional FP-UNA**
(actualizado set. 2025, <https://www.pol.una.py>): azul institucional `#384f87`,
Titillium para titulares y Lato para texto corrido. El escudo y el isotipo de
`assets/marca/` salieron de las páginas 5 y 11 de ese manual, en las versiones
aprobadas; el isotipo solo se usa como favicon, que es el uso que el manual le
reserva para tamaños chicos.

```bash
.venv/Scripts/python scripts/build_web_study.py
```

---

## El orden, de un vistazo

| | Dónde | Qué pasa si se saltea |
|---|---|---|
| 1. Crear el proyecto y correr `esquema.sql` | Supabase | — |
| 2. Copiar URL y clave anon a `config.js` | tu editor | El sitio queda en modo demostración y no guarda nada |
| 3. Arrastrar `webapp/public` | Netlify | — |
| 4. Probar de punta a punta y **borrar las pruebas** | los dos | El reparto de sets arranca corrido |

El paso 2 va **antes** del 3: Netlify sube los archivos tal como estén. Si
editás `config.js` después, hay que volver a arrastrar la carpeta.

---

## 0. Probarlo antes de montar nada

Con `config.js` vacío el sitio arranca en **modo demostración**: funciona entero
—registro, las 70 pantallas, reanudación, página de gracias— pero guarda solo en
el navegador. Sirve para recorrerlo y para mostrárselo al tutor.

```bash
.venv/Scripts/python -m http.server 8123 --directory webapp/public
```

Y abrir <http://localhost:8123>. Una pastilla azul abajo a la izquierda avisa
que está en modo demostración.

---

## 1. La base de datos (Supabase, 10 minutos)

Supabase es Postgres administrado, de código abierto, con un plan gratuito que
sobra para ocho participantes.

1. Entrar a <https://supabase.com> → **Start your project** → iniciar sesión con
   GitHub.
2. **New project**. Nombre: `estudio-radiografias`. Elegir una contraseña de base
   de datos y **guardarla**. Región: **South America (São Paulo)**, que es la más
   cercana. Crear y esperar un par de minutos.
3. En el menú lateral: **SQL Editor** → **New query**. Pegar el contenido entero
   de `webapp/supabase/esquema.sql` y darle **Run**. Tiene que decir
   *Success. No rows returned*.
4. **Project Settings** → **API**. Copiar dos cosas:
   - **Project URL** — algo como `https://abcdefgh.supabase.co`
   - la clave **anon / public**. Según la antigüedad del proyecto viene en uno de
     dos formatos, los dos válidos: `sb_publishable_...` (el nuevo, que Supabase
     rotula *Publishable key*) o un texto largo que arranca con `eyJ...`
5. Copiar `webapp/public/assets/config.ejemplo.js` a `config.js` en la misma
   carpeta y pegar ahí los dos valores:

```js
window.CONFIG = {
  SUPABASE_URL: "https://abcdefgh.supabase.co",
  SUPABASE_ANON_KEY: "sb_publishable_..."
};
```

> **La clave anon es pública y está bien que lo sea**: viaja en el sitio, como en
> cualquier aplicación de Supabase. Lo que protege los datos es que las tablas
> tienen RLS sin políticas —por la API no se pueden leer ni escribir— y que lo
> único expuesto son las funciones del esquema, que validan el token de sesión.
> **La clave `service_role` no se pone acá nunca.**

---

## 2. Publicar el sitio

Es HTML estático: no necesita build ni servidor. La forma más rápida:

**Netlify Drop** — <https://app.netlify.com/drop>

1. Arrastrar la carpeta `webapp/public` entera a esa página.
2. En segundos queda en línea, con una URL tipo
   `https://celebrated-marzipan-1a2b3c.netlify.app`.
3. Crear la cuenta gratuita cuando lo pida (si no, el sitio expira).
4. **Site configuration → Change site name** para dejar algo decente:
   `estudio-radiografias-fpuna.netlify.app`.
5. Para actualizar: **Deploys** → arrastrar la carpeta de nuevo.

Alternativa equivalente: **Cloudflare Pages** (*Workers & Pages → Create →
Pages → Upload assets*). Las dos son gratuitas y sirven igual.

> Son ~11 MB de imágenes. La primera carga de cada participante baja unos pocos
> cientos de KB y el resto va llegando a medida que avanza; el sitio precarga las
> dos pantallas siguientes.

---

## 3. Probar de punta a punta antes de convocar

Con el sitio ya publicado:

1. Abrirlo, registrarse con un correo tuyo, responder 3 o 4 pantallas.
2. En Supabase → **Table Editor** → `vw_progreso`: tenés que aparecer con tu
   set y la cantidad de respuestas.
3. Recargar la página: tiene que retomar sola en la pantalla siguiente.
4. Borrar el token para simular una sesión vencida (F12 → *Application* →
   *Local Storage* → borrar `estudio.token`) y volver a entrar con el mismo
   correo: tiene que aparecer el pop-up de reanudación.
5. Antes de convocar de verdad, **borrar las pruebas** para que el reparto de
   sets arranque en el 01:

   ```sql
   delete from participantes;   -- arrastra sesiones y respuestas
   ```

### Si algo no anda

| Síntoma | Causa casi siempre |
|---|---|
| *No pudimos conectarnos* al apretar Comenzar | `config.js` quedó vacío o con la URL mal copiada. Si está bien, en Supabase: **Project Settings → API → Reload schema cache** (PostgREST tarda unos segundos en ver funciones recién creadas). |
| Sigue apareciendo la pastilla *Modo demostración* | `SUPABASE_URL` quedó vacío en el `config.js` que subiste. Editalo y volvé a arrastrar la carpeta. |
| Las imágenes no cargan | Faltó subir la carpeta `img/`. Se sube `webapp/public` **entera**, no solo los HTML. |

---

## 4. Durante el estudio

**Ver cómo va:** Table Editor → `vw_progreso`. Una fila por participante, con
set asignado, respuestas, segundos promedio por pantalla, si terminó y qué
escribió en el campo opcional del final.

El sitio **no manda ningún correo**. Si querés agradecerles, la lista de quiénes
terminaron sale de esa misma vista y se les escribe a mano.

**Bajar los datos:** SQL Editor →

```sql
select * from vw_respuestas;
```

→ botón **Download CSV**. Ese archivo se cruza con
`docs/evaluacion/estudio/clave.csv` por `ensayo` = `trial_id`, igual que estaba
previsto en el plan de análisis.

**A quién le tocó qué set:** `vw_progreso` lo muestra. Al registrarse, cada uno
recibe el **set menos usado** (desempatando por el más chico). Sin registros de
más eso es exactamente 01, 02, … 08, y el 9no vuelve al 01 pero con otro orden de
pantallas. La ventaja sobre un contador ciclico es que si hay que borrar una
prueba, el hueco lo vuelve a tomar el próximo que entre en vez de quedar sin
cubrir — y un set sin cubrir rompe el balance del diseño.

**Si alguien pide que borren sus datos:**

```sql
delete from participantes where correo = 'quien@sea.com';
```

---

## 4.bis Qué decirles al convocar

Reemplaza al "Qué decirles" de `docs/evaluacion/estudio/LEEME.md`, que describía
el reparto por Drive.

- Mandar **el enlace del sitio**, nada más. No hay archivos que bajar ni devolver.
- **Cada uno tiene que entrar con su propio correo.** De ahí sale el set que le
  toca, y si dos entran con el mismo, el segundo continúa la evaluación del
  primero en vez de empezar la suya.
- Son **70 pantallas, entre 20 y 25 minutos**. Conviene notebook o tablet, no
  celular.
- **Pueden cortar y seguir después**, incluso desde otra computadora: entran con
  el mismo correo y retoman donde quedaron.
- Vale la pena avisarlo porque si no genera desconcierto: **muchos pares van a
  parecer idénticos**. Es esperable y es parte de lo que se mide; para eso está
  el botón *Se ven iguales*.
- Las instrucciones completas están en la primera pantalla del sitio: no hace
  falta explicar nada por teléfono.

---

## 5. Lo que conviene saber

- **Un enlace, un correo por persona.** El set se asigna por orden de registro.
  Si dos personas entran con el mismo correo, el sistema cree que son la misma y
  la segunda retoma la evaluación de la primera. Hay que decirlo al convocar.
- **La sesión dura 24 h y se renueva sola** con cada respuesta. Si vence, no se
  pierde nada: se entra otra vez con el mismo correo.
- **Sin internet la evaluación sigue.** Las respuestas se acumulan en el
  navegador y se mandan cuando vuelve la conexión; el indicador de la barra
  superior pasa a *Sin conexión* y vuelve a *Guardado*.
- **No se puede volver atrás**, igual que en la app offline y por la misma razón.
- **Los identificadores son hashes.** Ni mirando el código fuente se puede saber
  qué condición es cada imagen: la correspondencia vive solo en `clave.csv`, que
  no se sube a ningún lado.

---

## 6. El proyecto gratuito se pausa si nadie lo usa

Los proyectos gratuitos de Supabase se pausan tras ~7 días sin actividad, y hay
que despertarlos a mano desde el panel. Si eso pasa justo cuando un odontólogo
abre el enlace, se encuentra con un error.

Dos formas de evitarlo:

- Entrar al panel de Supabase una vez por semana. Alcanza.
- O activar el latido automático: copiar `webapp/mantener-despierto.yml` a
  `.github/workflows/mantener-despierto.yml`, y en GitHub (*Settings → Secrets
  and variables → Actions*) crear los secretos `SUPABASE_URL` y
  `SUPABASE_ANON_KEY`. Llama a la función `ping()` cada tres días.

---

## 7. Volver a generar los materiales

Si se rehace el experimento o cambia el reparto de ensayos:

```bash
.venv/Scripts/python scripts/build_study_materials.py    # regenera trials.json e img/
.venv/Scripts/python scripts/build_web_study.py          # regenera sets.json y copia todo
```

Y volver a subir `webapp/public`. **Ojo:** si cambian los `trial_id`, las
respuestas ya recogidas dejan de cruzar con la clave nueva. No se regenera nada
en medio de una recolección.
