create table public.lithops_run_leases (
    run_id uuid primary key references public.lithops_runs(id) on delete cascade,
    owner_id text not null,
    token uuid not null unique,
    acquired_at timestamptz not null,
    heartbeat_at timestamptz not null,
    expires_at timestamptz not null,
    check (expires_at > heartbeat_at),
    check (heartbeat_at >= acquired_at)
);

create index lithops_run_leases_expiration_idx
    on public.lithops_run_leases (expires_at);

create function public.lithops_claim_run_lease(
    p_run_id uuid,
    p_owner_id text,
    p_token uuid,
    p_now timestamptz,
    p_expires_at timestamptz
)
returns setof public.lithops_run_leases
language sql
set search_path = ''
as $$
    insert into public.lithops_run_leases (
        run_id,
        owner_id,
        token,
        acquired_at,
        heartbeat_at,
        expires_at
    ) values (
        p_run_id,
        p_owner_id,
        p_token,
        p_now,
        p_now,
        p_expires_at
    )
    on conflict (run_id) do update
    set owner_id = excluded.owner_id,
        token = excluded.token,
        acquired_at = excluded.acquired_at,
        heartbeat_at = excluded.heartbeat_at,
        expires_at = excluded.expires_at
    where public.lithops_run_leases.expires_at <= p_now
       or public.lithops_run_leases.owner_id = p_owner_id
    returning *;
$$;

create function public.lithops_renew_run_lease(
    p_run_id uuid,
    p_token uuid,
    p_now timestamptz,
    p_expires_at timestamptz
)
returns setof public.lithops_run_leases
language sql
set search_path = ''
as $$
    update public.lithops_run_leases
    set heartbeat_at = p_now,
        expires_at = p_expires_at
    where run_id = p_run_id
      and token = p_token
      and expires_at > p_now
    returning *;
$$;

alter table public.lithops_run_leases enable row level security;
revoke all on table public.lithops_run_leases from anon, authenticated;

create policy "deny direct client access"
    on public.lithops_run_leases
    for all to anon, authenticated using (false) with check (false);

revoke all on function public.lithops_claim_run_lease(
    uuid, text, uuid, timestamptz, timestamptz
) from public, anon, authenticated;
revoke all on function public.lithops_renew_run_lease(
    uuid, uuid, timestamptz, timestamptz
) from public, anon, authenticated;
grant execute on function public.lithops_claim_run_lease(
    uuid, text, uuid, timestamptz, timestamptz
) to service_role;
grant execute on function public.lithops_renew_run_lease(
    uuid, uuid, timestamptz, timestamptz
) to service_role;
