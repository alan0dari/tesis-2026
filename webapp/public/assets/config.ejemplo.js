/*
  Plantilla de configuración. Copiar este archivo como `config.js` (que está en
  .gitignore) y completar los dos valores despues de crear el proyecto en
  Supabase; el paso 1 del README dice de donde salen.

  Los dos son públicos por diseño: la clave anon viaja al navegador. Lo que
  protege los datos es que las tablas tienen RLS sin políticas y que la única
  superficie expuesta son las funciones de supabase/esquema.sql. La clave de
  servicio no va acá nunca.

  Con SUPABASE_URL vacío el sitio arranca en modo demostración: funciona entero
  pero guarda solo en el navegador.
*/

window.CONFIG = {
  SUPABASE_URL: "",
  SUPABASE_ANON_KEY: ""
};
