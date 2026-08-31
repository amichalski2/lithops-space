create table public.lithops_model_challenges (
    id uuid primary key,
    run_id uuid not null references public.lithops_runs(id) on delete cascade,
    health_signal_id uuid not null references public.lithops_model_health_signals(id),
    base_model_version_id uuid not null references public.lithops_world_models(id),
    status text not null check (
        status in ('triggered', 'building', 'backtesting', 'awaiting_executive', 'completed', 'failed')
    ),
    payload jsonb not null,
    created_at timestamptz not null,
    updated_at timestamptz not null
);

create table public.lithops_model_challenge_packages (
    challenge_id uuid primary key references public.lithops_model_challenges(id) on delete cascade,
    run_id uuid not null references public.lithops_runs(id) on delete cascade,
    payload jsonb not null,
    created_at timestamptz not null
);

create table public.lithops_model_builder_proposals (
    id uuid primary key,
    challenge_id uuid not null references public.lithops_model_challenges(id) on delete cascade,
    builder_name text not null,
    payload jsonb not null,
    created_at timestamptz not null,
    unique (challenge_id, builder_name)
);

create table public.lithops_hypothesis_backtests (
    id uuid primary key,
    challenge_id uuid not null references public.lithops_model_challenges(id) on delete cascade,
    proposal_id uuid not null references public.lithops_model_builder_proposals(id) on delete cascade,
    supported boolean not null,
    penalized_improvement double precision not null,
    payload jsonb not null,
    created_at timestamptz not null,
    unique (proposal_id)
);

create table public.lithops_model_builder_calls (
    id uuid primary key,
    challenge_id uuid not null references public.lithops_model_challenges(id) on delete cascade,
    builder_name text not null,
    attempt smallint not null check (attempt between 1 and 2),
    status text not null check (status in ('completed', 'timed_out', 'invalid_output', 'failed')),
    input_hash text not null check (input_hash ~ '^[0-9a-f]{64}$'),
    payload jsonb not null,
    created_at timestamptz not null,
    unique (challenge_id, builder_name, attempt)
);

create table public.lithops_model_challenge_decisions (
    id uuid primary key,
    challenge_id uuid not null unique references public.lithops_model_challenges(id) on delete cascade,
    resolution text not null check (
        resolution in ('accepted', 'merged', 'executive_rejected', 'no_supported_winner')
    ),
    activated_model_version_id uuid references public.lithops_world_models(id),
    payload jsonb not null,
    created_at timestamptz not null
);

create index lithops_model_challenges_run_created_idx
    on public.lithops_model_challenges (run_id, created_at desc);
create index lithops_model_builder_proposals_challenge_idx
    on public.lithops_model_builder_proposals (challenge_id, builder_name);
create index lithops_hypothesis_backtests_challenge_score_idx
    on public.lithops_hypothesis_backtests (challenge_id, penalized_improvement desc);
create index lithops_model_builder_calls_challenge_idx
    on public.lithops_model_builder_calls (challenge_id, builder_name, attempt);

alter table public.lithops_model_challenges enable row level security;
alter table public.lithops_model_challenge_packages enable row level security;
alter table public.lithops_model_builder_proposals enable row level security;
alter table public.lithops_hypothesis_backtests enable row level security;
alter table public.lithops_model_builder_calls enable row level security;
alter table public.lithops_model_challenge_decisions enable row level security;

revoke all on table public.lithops_model_challenges from anon, authenticated;
revoke all on table public.lithops_model_challenge_packages from anon, authenticated;
revoke all on table public.lithops_model_builder_proposals from anon, authenticated;
revoke all on table public.lithops_hypothesis_backtests from anon, authenticated;
revoke all on table public.lithops_model_builder_calls from anon, authenticated;
revoke all on table public.lithops_model_challenge_decisions from anon, authenticated;

create policy "deny direct client access"
    on public.lithops_model_challenges
    for all to anon, authenticated using (false) with check (false);
create policy "deny direct client access"
    on public.lithops_model_challenge_packages
    for all to anon, authenticated using (false) with check (false);
create policy "deny direct client access"
    on public.lithops_model_builder_proposals
    for all to anon, authenticated using (false) with check (false);
create policy "deny direct client access"
    on public.lithops_hypothesis_backtests
    for all to anon, authenticated using (false) with check (false);
create policy "deny direct client access"
    on public.lithops_model_builder_calls
    for all to anon, authenticated using (false) with check (false);
create policy "deny direct client access"
    on public.lithops_model_challenge_decisions
    for all to anon, authenticated using (false) with check (false);
