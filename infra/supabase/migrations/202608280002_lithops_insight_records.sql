-- Purchased market information as durable, uncertainty-aware evidence.
--
-- Identity is deterministic per (run, week, request), so a replayed week appends
-- the same row instead of buying the same answer twice.

create table public.lithops_insight_records (
    id uuid primary key,
    run_id uuid not null references public.lithops_runs(id) on delete cascade,
    week integer not null check (week >= 0),
    tool text not null,
    target_group text,
    parse_status text not null check (parse_status in (
        'succeeded', 'partial', 'failed', 'pending'
    )),
    payload jsonb not null,
    created_at timestamptz not null default now()
);

create index lithops_insight_records_run_week_idx
    on public.lithops_insight_records (run_id, week);
create index lithops_insight_records_group_idx
    on public.lithops_insight_records (run_id, target_group);

alter table public.lithops_insight_records enable row level security;
revoke all on table public.lithops_insight_records from anon, authenticated;

create policy "deny direct client access"
    on public.lithops_insight_records
    for all to anon, authenticated using (false) with check (false);
