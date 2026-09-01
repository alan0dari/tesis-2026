-- Estudio perceptual con odontologos - base de datos
-- FP-UNA
--
-- Pegar entero en el SQL Editor de Supabase y ejecutar una vez. Es idempotente.
--
-- Seguridad: la clave anon del navegador es publica, asi que las tablas van con
-- RLS activo y sin ninguna politica; por la API REST no se pueden tocar. Todo
-- pasa por las funciones de abajo (SECURITY DEFINER, search_path fijo), y cada
-- una valida el token de sesion antes de escribir.
--
-- Lo que nunca viaja al navegador: que condicion experimental es cada imagen.
-- Eso vive solo en docs/evaluacion/estudio/clave.csv.


-- tablas

create table if not exists public.participantes (
  id                 uuid primary key default gen_random_uuid(),
  correo             text        not null unique,
  orden              int         not null unique,       -- orden de registro, sin huecos
  set_numero         int         not null check (set_numero between 1 and 8),
  semilla            bigint      not null,              -- barajado determinista en el cliente
  creado_en          timestamptz not null default now(),
  visto_en           timestamptz not null default now(),
  terminado_en       timestamptz,
  comentario         text,                              -- "que tuviste en cuenta..."
  navegador          text
);

comment on column public.participantes.set_numero is
  'Set 1..8: al registrarse se toma el menos usado. Sin registros de mas eso da
   01, 02, ... 08, y el 9no participante vuelve al set 01.';
comment on column public.participantes.semilla is
  'Semilla del barajado. Dos participantes con el mismo set no ven el mismo orden.';

create table if not exists public.sesiones (
  token          uuid primary key default gen_random_uuid(),
  participante   uuid        not null references public.participantes(id) on delete cascade,
  creada_en      timestamptz not null default now(),
  expira_en      timestamptz not null
);

create index if not exists sesiones_participante_idx
  on public.sesiones (participante);

create table if not exists public.respuestas (
  id            bigserial primary key,
  participante  uuid        not null references public.participantes(id) on delete cascade,
  ensayo        text        not null,
  posicion      int         not null,                   -- 1..70, orden en que se vio
  eleccion      text        not null check (eleccion in ('a', 'b')),
  confianza     text        not null check (confianza in ('seguro', 'algo', 'adivinando')),
  ms            int         not null,
  repeticion    boolean     not null default false,
  creada_en     timestamptz not null default now(),
  unique (participante, posicion)                       -- reenviar una respuesta no duplica
);

create index if not exists respuestas_participante_idx
  on public.respuestas (participante);

alter table public.participantes enable row level security;
alter table public.sesiones      enable row level security;
alter table public.respuestas    enable row level security;


-- constantes del estudio

create or replace function public.duracion_sesion()
returns interval language sql immutable as $$ select interval '24 hours' $$;

create or replace function public.total_sets()
returns int language sql immutable as $$ select 8 $$;


-- estado de un participante (helper interno)

create or replace function public.estado(p_id uuid, p_token uuid)
returns jsonb
language plpgsql
security definer
set search_path = public, pg_temp
as $$
declare
  v_p          public.participantes%rowtype;
  v_respondidas int;
begin
  select * into v_p from public.participantes where id = p_id;
  select count(*) into v_respondidas from public.respuestas where participante = p_id;

  return jsonb_build_object(
    'token',       p_token,
    'set',         lpad(v_p.set_numero::text, 2, '0'),
    'semilla',     v_p.semilla,
    'respondidas', v_respondidas,
    'terminado',   v_p.terminado_en is not null,
    'comentario',  v_p.comentario,
    'correo',      v_p.correo
  );
end;
$$;


-- alta / reanudacion por correo
-- Devuelve `nuevo = false` cuando el correo ya estaba registrado. El sitio usa
-- ese dato para mostrar el pop-up de "te reconocimos, seguimos donde lo dejaste".

create or replace function public.iniciar(p_correo text, p_navegador text default null)
returns jsonb
language plpgsql
security definer
set search_path = public, pg_temp
as $$
declare
  v_correo text := lower(trim(p_correo));
  v_p      public.participantes%rowtype;
  v_nuevo  boolean := false;
  v_orden  int;
  v_set    int;
  v_token  uuid;
begin
  if v_correo !~ '^[^@\s]+@[^@\s]+\.[^@\s]{2,}$' then
    raise exception 'correo_invalido' using errcode = '22023';
  end if;

  select * into v_p from public.participantes where correo = v_correo;

  if not found then
    -- Serializa el alta: dos personas registrandose a la vez no pueden llevarse
    -- el mismo set.
    perform pg_advisory_xact_lock(918273645);

    select coalesce(max(orden), 0) + 1 into v_orden from public.participantes;

    -- El set menos usado, desempatando por el mas chico. Sin registros basura da
    -- exactamente 01, 02, ... 08, 01, ... igual que un contador ciclico; la
    -- diferencia es que si hay que borrar una prueba, el hueco lo vuelve a tomar
    -- el proximo que se registre en vez de quedar sin cubrir. Un set sin cubrir
    -- desbalancearia el diseno de bloques incompletos.
    select t.n into v_set
      from (select g.n as n, count(p.id) as usos
              from generate_series(1, public.total_sets()) as g(n)
              left join public.participantes p on p.set_numero = g.n
             group by g.n) t
     order by t.usos, t.n
     limit 1;

    insert into public.participantes (correo, orden, set_numero, semilla, navegador)
    values (v_correo,
            v_orden,
            v_set,
            (random() * 2147483000)::bigint,
            left(p_navegador, 300))
    returning * into v_p;

    v_nuevo := true;
  else
    update public.participantes
       set visto_en  = now(),
           navegador = coalesce(left(p_navegador, 300), navegador)
     where id = v_p.id;
  end if;

  insert into public.sesiones (participante, expira_en)
  values (v_p.id, now() + public.duracion_sesion())
  returning token into v_token;

  return public.estado(v_p.id, v_token) || jsonb_build_object('nuevo', v_nuevo);
end;
$$;


-- validacion de sesion (la usa el sitio al recargar)
-- Expiracion deslizante: cada llamada corre el vencimiento 24 h mas adelante,
-- asi nadie pierde la sesion en medio de la evaluacion.

create or replace function public.sesion(p_token uuid)
returns jsonb
language plpgsql
security definer
set search_path = public, pg_temp
as $$
declare
  v_id uuid;
begin
  update public.sesiones
     set expira_en = now() + public.duracion_sesion()
   where token = p_token
     and expira_en > now()
  returning participante into v_id;

  if v_id is null then
    return jsonb_build_object('valida', false);
  end if;

  update public.participantes set visto_en = now() where id = v_id;

  return public.estado(v_id, p_token) || jsonb_build_object('valida', true);
end;
$$;


-- guardar una respuesta
-- Idempotente por (participante, posicion): si el navegador reintenta despues
-- de un corte de red, no se duplica ni se pisa lo ya guardado.

create or replace function public.guardar(
  p_token      uuid,
  p_ensayo     text,
  p_posicion   int,
  p_eleccion   text,
  p_confianza  text,
  p_ms         int,
  p_repeticion boolean default false
)
returns jsonb
language plpgsql
security definer
set search_path = public, pg_temp
as $$
declare
  v_id uuid;
begin
  update public.sesiones
     set expira_en = now() + public.duracion_sesion()
   where token = p_token
     and expira_en > now()
  returning participante into v_id;

  if v_id is null then
    return jsonb_build_object('ok', false, 'motivo', 'sesion');
  end if;

  insert into public.respuestas
    (participante, ensayo, posicion, eleccion, confianza, ms, repeticion)
  values
    (v_id, p_ensayo, p_posicion, p_eleccion, p_confianza,
     greatest(p_ms, 0), coalesce(p_repeticion, false))
  on conflict (participante, posicion) do nothing;

  return public.estado(v_id, p_token) || jsonb_build_object('ok', true);
end;
$$;


-- cierre
-- Se llama dos veces: al responder la ultima pantalla (sin comentario) y otra
-- vez si el participante manda el campo opcional de la pagina de gracias.
--
-- A proposito NO exige que la sesion siga vigente: cerrar es idempotente e
-- inofensivo, y seria absurdo que alguien pierda el cierre por vencer justo en
-- la ultima pantalla. Tambien deja que alguien vuelva mas tarde a escribir el
-- comentario opcional.

create or replace function public.finalizar(p_token uuid, p_comentario text default null)
returns jsonb
language plpgsql
security definer
set search_path = public, pg_temp
as $$
declare
  v_id uuid;
begin
  select participante into v_id
    from public.sesiones
   where token = p_token;

  if v_id is null then
    return jsonb_build_object('ok', false, 'motivo', 'sesion');
  end if;

  update public.participantes
     set terminado_en = coalesce(terminado_en, now()),
         comentario   = coalesce(nullif(trim(coalesce(p_comentario, '')), ''), comentario),
         visto_en     = now()
   where id = v_id;

  return public.estado(v_id, p_token) || jsonb_build_object('ok', true);
end;
$$;


-- latido, para que el proyecto gratuito no se pause por inactividad

create or replace function public.ping()
returns text language sql security definer set search_path = public, pg_temp
as $$ select 'ok' $$;


-- vistas de seguimiento (para el panel de supabase, no para el navegador)

create or replace view public.vw_progreso as
select p.orden,
       p.correo,
       lpad(p.set_numero::text, 2, '0')                as set_nro,
       count(r.id)                                     as respondidas,
       count(r.id) filter (where r.repeticion)         as repetidas,
       round(avg(r.ms) / 1000.0, 1)                    as seg_por_pantalla,
       p.terminado_en is not null                      as termino,
       p.creado_en,
       p.visto_en,
       p.comentario
  from public.participantes p
  left join public.respuestas r on r.participante = p.id
 group by p.id
 order by p.orden;

create or replace view public.vw_respuestas as
select p.orden           as participante,
       lpad(p.set_numero::text, 2, '0') as set_nro,
       r.posicion,
       r.ensayo,
       r.eleccion,
       r.confianza,
       r.ms,
       r.repeticion,
       r.creada_en
  from public.respuestas r
  join public.participantes p on p.id = r.participante
 order by p.orden, r.posicion;


-- permisos

revoke all on public.participantes, public.sesiones, public.respuestas
  from anon, authenticated;
revoke all on public.vw_progreso, public.vw_respuestas
  from anon, authenticated;

-- Lo que el navegador puede llamar, y nada mas.
grant execute on function public.iniciar(text, text)                        to anon, authenticated;
grant execute on function public.sesion(uuid)                               to anon, authenticated;
grant execute on function public.guardar(uuid, text, int, text, text, int, boolean)
                                                                            to anon, authenticated;
grant execute on function public.finalizar(uuid, text)                      to anon, authenticated;
grant execute on function public.ping()                                     to anon, authenticated;

-- El helper interno no: no tiene por que estar expuesto en la API.
revoke execute on function public.estado(uuid, uuid) from public, anon, authenticated;
