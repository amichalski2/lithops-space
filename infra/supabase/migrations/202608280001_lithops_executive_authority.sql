-- Durable strategic-authority records for executive_authority_v2.
--
-- These tables are replay-reconstructible projections of the decision history:
-- portfolio revisions form a hash chain, experiment outcomes and reviews carry
-- deterministic identities, and one evaluation set / choice exists per week.

create table public.lithops_strategic_portfolio_revisions (
    id uuid primary key,
    run_id uuid not null references public.lithops_runs(id) on delete cascade,
    decision_id uuid references public.lithops_decisions(id) on delete cascade,
    week integer not null check (week >= 0),
    revision integer not null check (revision >= 1),
    portfolio_hash text not null check (portfolio_hash ~ '^[0-9a-f]{64}$'),
    prior_portfolio_hash text check (prior_portfolio_hash ~ '^[0-9a-f]{64}$'),
    payload jsonb not null,
    created_at timestamptz not null default now(),
    unique (run_id, revision),
    unique (run_id, portfolio_hash),
    check ((revision = 1) = (prior_portfolio_hash is null))
);

create table public.lithops_experiment_outcomes (
    id uuid primary key,
    run_id uuid not null references public.lithops_runs(id) on delete cascade,
    commitment_id text not null,
    hypothesis_id text,
    outcome_status text not null check (outcome_status in (
        'immature', 'valid_exposure', 'no_exposure', 'exposed_zero_conversion',
        'positive_conversion', 'censored', 'invalid_execution', 'stopped_for_safety'
    )),
    started_week integer not null check (started_week >= 0),
    measured_week integer check (measured_week >= 0),
    payload jsonb not null,
    created_at timestamptz not null default now()
);

create table public.lithops_commitment_reviews (
    id uuid primary key,
    run_id uuid not null references public.lithops_runs(id) on delete cascade,
    commitment_id text not null,
    week integer not null check (week >= 0),
    verdict text not null check (verdict in (
        'continue', 'stop_for_safety', 'falsified', 'mature_and_probe', 'revert'
    )),
    payload jsonb not null,
    created_at timestamptz not null default now(),
    unique (run_id, commitment_id, week)
);

create table public.lithops_candidate_evaluation_sets (
    id uuid primary key,
    run_id uuid not null references public.lithops_runs(id) on delete cascade,
    week integer not null check (week >= 0),
    set_hash text not null check (set_hash ~ '^[0-9a-f]{64}$'),
    payload jsonb not null,
    created_at timestamptz not null default now(),
    unique (run_id, week)
);

create table public.lithops_executive_choices (
    id uuid primary key,
    run_id uuid not null references public.lithops_runs(id) on delete cascade,
    week integer not null check (week >= 0),
    evaluation_set_id uuid not null
        references public.lithops_candidate_evaluation_sets(id) on delete cascade,
    selected_candidate_id text not null,
    payload jsonb not null,
    created_at timestamptz not null default now(),
    unique (run_id, week)
);

create index lithops_strategic_portfolio_revisions_run_idx
    on public.lithops_strategic_portfolio_revisions (run_id, revision desc);
create index lithops_experiment_outcomes_run_commitment_idx
    on public.lithops_experiment_outcomes (run_id, commitment_id, started_week);
create index lithops_commitment_reviews_run_commitment_idx
    on public.lithops_commitment_reviews (run_id, commitment_id, week);
create index lithops_candidate_evaluation_sets_run_week_idx
    on public.lithops_candidate_evaluation_sets (run_id, week);
create index lithops_executive_choices_run_week_idx
    on public.lithops_executive_choices (run_id, week);
create index lithops_executive_choices_evaluation_set_idx
    on public.lithops_executive_choices (evaluation_set_id);

alter table public.lithops_strategic_portfolio_revisions enable row level security;
alter table public.lithops_experiment_outcomes enable row level security;
alter table public.lithops_commitment_reviews enable row level security;
alter table public.lithops_candidate_evaluation_sets enable row level security;
alter table public.lithops_executive_choices enable row level security;

revoke all on table public.lithops_strategic_portfolio_revisions from anon, authenticated;
revoke all on table public.lithops_experiment_outcomes from anon, authenticated;
revoke all on table public.lithops_commitment_reviews from anon, authenticated;
revoke all on table public.lithops_candidate_evaluation_sets from anon, authenticated;
revoke all on table public.lithops_executive_choices from anon, authenticated;

create policy "deny direct client access"
    on public.lithops_strategic_portfolio_revisions
    for all to anon, authenticated using (false) with check (false);
create policy "deny direct client access"
    on public.lithops_experiment_outcomes
    for all to anon, authenticated using (false) with check (false);
create policy "deny direct client access"
    on public.lithops_commitment_reviews
    for all to anon, authenticated using (false) with check (false);
create policy "deny direct client access"
    on public.lithops_candidate_evaluation_sets
    for all to anon, authenticated using (false) with check (false);
create policy "deny direct client access"
    on public.lithops_executive_choices
    for all to anon, authenticated using (false) with check (false);
