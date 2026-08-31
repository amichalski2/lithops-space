create table public.lithops_runs (
    id uuid primary key,
    status text not null,
    current_day integer not null default 0 check (current_day >= 0),
    benchmark_session_id text unique,
    version integer not null default 0 check (version >= 0),
    payload jsonb not null,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

create table public.lithops_decisions (
    id uuid primary key,
    run_id uuid not null references public.lithops_runs(id) on delete cascade,
    week integer not null check (week >= 0),
    status text not null,
    payload jsonb not null,
    created_at timestamptz not null default now(),
    unique (run_id, week)
);

create table public.lithops_action_receipts (
    id uuid primary key,
    run_id uuid not null references public.lithops_runs(id) on delete cascade,
    decision_id uuid not null references public.lithops_decisions(id) on delete cascade,
    idempotency_key text not null,
    status text not null,
    payload jsonb not null,
    created_at timestamptz not null default now(),
    unique (run_id, idempotency_key)
);

create table public.lithops_operations (
    id uuid primary key,
    run_id uuid not null references public.lithops_runs(id) on delete cascade,
    request_id text not null,
    status text not null,
    payload jsonb not null,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    unique (run_id, request_id)
);

create table public.lithops_events (
    id bigint generated always as identity primary key,
    event_id uuid not null unique,
    run_id uuid not null references public.lithops_runs(id) on delete cascade,
    type text not null,
    payload jsonb not null,
    created_at timestamptz not null default now()
);

create index lithops_decisions_run_id_idx
    on public.lithops_decisions (run_id);
create index lithops_action_receipts_decision_id_idx
    on public.lithops_action_receipts (decision_id);
create index lithops_operations_run_id_idx
    on public.lithops_operations (run_id);
create index lithops_events_run_id_id_idx
    on public.lithops_events (run_id, id);

alter table public.lithops_runs enable row level security;
alter table public.lithops_decisions enable row level security;
alter table public.lithops_action_receipts enable row level security;
alter table public.lithops_operations enable row level security;
alter table public.lithops_events enable row level security;

revoke all on table public.lithops_runs from anon, authenticated;
revoke all on table public.lithops_decisions from anon, authenticated;
revoke all on table public.lithops_action_receipts from anon, authenticated;
revoke all on table public.lithops_operations from anon, authenticated;
revoke all on table public.lithops_events from anon, authenticated;
