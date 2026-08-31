create table public.lithops_world_models (
    id uuid primary key,
    run_id uuid not null references public.lithops_runs(id) on delete cascade,
    version integer not null check (version >= 1),
    source_observation_day integer not null check (source_observation_day >= 0),
    based_on_version_id uuid references public.lithops_world_models(id),
    update_method text not null,
    payload jsonb not null,
    created_at timestamptz not null default now(),
    unique (run_id, version)
);

create table public.lithops_world_model_relationships (
    world_model_id uuid not null references public.lithops_world_models(id) on delete cascade,
    run_id uuid not null references public.lithops_runs(id) on delete cascade,
    relationship_key text not null,
    cause text not null,
    effect text not null,
    payload jsonb not null,
    created_at timestamptz not null default now(),
    primary key (world_model_id, relationship_key)
);

create table public.lithops_predictions (
    id uuid primary key,
    run_id uuid not null references public.lithops_runs(id) on delete cascade,
    decision_id uuid not null references public.lithops_decisions(id) on delete cascade,
    model_version_id uuid not null references public.lithops_world_models(id),
    issued_day integer not null check (issued_day >= 0),
    payload jsonb not null,
    created_at timestamptz not null default now(),
    unique (decision_id)
);

create table public.lithops_prediction_outcomes (
    id uuid primary key,
    run_id uuid not null references public.lithops_runs(id) on delete cascade,
    ledger_entry_id uuid not null references public.lithops_predictions(id) on delete cascade,
    target_id uuid not null unique,
    observed_day integer not null check (observed_day >= 0),
    payload jsonb not null,
    created_at timestamptz not null default now()
);

create table public.lithops_candidate_simulations (
    decision_id uuid not null references public.lithops_decisions(id) on delete cascade,
    run_id uuid not null references public.lithops_runs(id) on delete cascade,
    strategy text not null,
    selected boolean not null,
    robust_utility double precision not null,
    payload jsonb not null,
    created_at timestamptz not null default now(),
    primary key (decision_id, strategy)
);

create table public.lithops_model_health_signals (
    id uuid primary key,
    run_id uuid not null references public.lithops_runs(id) on delete cascade,
    model_version_id uuid not null references public.lithops_world_models(id),
    evaluated_day integer not null check (evaluated_day >= 0),
    status text not null check (status in ('healthy', 'watching', 'degraded')),
    rebuild_recommended boolean not null,
    payload jsonb not null,
    created_at timestamptz not null default now()
);

create index lithops_world_models_run_version_idx
    on public.lithops_world_models (run_id, version desc);
create index lithops_world_model_relationships_run_idx
    on public.lithops_world_model_relationships (run_id, world_model_id);
create index lithops_predictions_run_day_idx
    on public.lithops_predictions (run_id, issued_day);
create index lithops_prediction_outcomes_run_day_idx
    on public.lithops_prediction_outcomes (run_id, observed_day);
create index lithops_candidate_simulations_run_idx
    on public.lithops_candidate_simulations (run_id, decision_id);
create index lithops_model_health_signals_run_day_idx
    on public.lithops_model_health_signals (run_id, evaluated_day);

create function public.lithops_materialize_world_model_relationships()
returns trigger
language plpgsql
set search_path = ''
as $$
begin
    insert into public.lithops_world_model_relationships (
        world_model_id,
        run_id,
        relationship_key,
        cause,
        effect,
        payload,
        created_at
    )
    select
        new.id,
        new.run_id,
        relationship ->> 'key',
        relationship ->> 'cause',
        relationship ->> 'effect',
        relationship,
        new.created_at
    from jsonb_array_elements(new.payload -> 'relationships') as relationship;
    return new;
end;
$$;

create trigger lithops_world_models_materialize_relationships
after insert on public.lithops_world_models
for each row execute function public.lithops_materialize_world_model_relationships();

create function public.lithops_materialize_candidate_simulations()
returns trigger
language plpgsql
set search_path = ''
as $$
begin
    delete from public.lithops_candidate_simulations
    where decision_id = new.id;

    insert into public.lithops_candidate_simulations (
        decision_id,
        run_id,
        strategy,
        selected,
        robust_utility,
        payload,
        created_at
    )
    select
        new.id,
        new.run_id,
        candidate ->> 'strategy',
        candidate ->> 'strategy' = new.payload #>> '{action_plan,strategy_family}',
        (candidate ->> 'robust_utility')::double precision,
        candidate,
        new.created_at
    from jsonb_array_elements(new.payload -> 'candidate_evaluations') as candidate;
    return new;
end;
$$;

create trigger lithops_decisions_materialize_candidates
after insert or update of payload on public.lithops_decisions
for each row execute function public.lithops_materialize_candidate_simulations();

insert into public.lithops_candidate_simulations (
    decision_id,
    run_id,
    strategy,
    selected,
    robust_utility,
    payload,
    created_at
)
select
    decision.id,
    decision.run_id,
    candidate ->> 'strategy',
    candidate ->> 'strategy' = decision.payload #>> '{action_plan,strategy_family}',
    (candidate ->> 'robust_utility')::double precision,
    candidate,
    decision.created_at
from public.lithops_decisions as decision
cross join lateral jsonb_array_elements(decision.payload -> 'candidate_evaluations') as candidate
on conflict (decision_id, strategy) do nothing;

alter table public.lithops_world_models enable row level security;
alter table public.lithops_world_model_relationships enable row level security;
alter table public.lithops_predictions enable row level security;
alter table public.lithops_prediction_outcomes enable row level security;
alter table public.lithops_candidate_simulations enable row level security;
alter table public.lithops_model_health_signals enable row level security;

revoke all on table public.lithops_world_models from anon, authenticated;
revoke all on table public.lithops_world_model_relationships from anon, authenticated;
revoke all on table public.lithops_predictions from anon, authenticated;
revoke all on table public.lithops_prediction_outcomes from anon, authenticated;
revoke all on table public.lithops_candidate_simulations from anon, authenticated;
revoke all on table public.lithops_model_health_signals from anon, authenticated;

create policy "deny direct client access"
    on public.lithops_world_models
    for all to anon, authenticated using (false) with check (false);
create policy "deny direct client access"
    on public.lithops_world_model_relationships
    for all to anon, authenticated using (false) with check (false);
create policy "deny direct client access"
    on public.lithops_predictions
    for all to anon, authenticated using (false) with check (false);
create policy "deny direct client access"
    on public.lithops_prediction_outcomes
    for all to anon, authenticated using (false) with check (false);
create policy "deny direct client access"
    on public.lithops_candidate_simulations
    for all to anon, authenticated using (false) with check (false);
create policy "deny direct client access"
    on public.lithops_model_health_signals
    for all to anon, authenticated using (false) with check (false);

revoke all on function public.lithops_materialize_world_model_relationships()
    from public, anon, authenticated;
revoke all on function public.lithops_materialize_candidate_simulations()
    from public, anon, authenticated;
