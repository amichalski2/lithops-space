create table public.lithops_model_artifacts (
    id uuid primary key,
    content_hash text not null unique check (content_hash ~ '^[0-9a-f]{64}$'),
    runtime_kind text not null check (runtime_kind in ('sandboxed_python', 'trusted_baseline')),
    parent_artifact_id uuid references public.lithops_model_artifacts(id),
    payload jsonb not null,
    created_at timestamptz not null
);

create table public.lithops_model_artifact_authoring_receipts (
    id uuid primary key,
    challenge_id uuid not null,
    run_id uuid not null references public.lithops_runs(id) on delete cascade,
    author_key text not null,
    artifact_id uuid not null references public.lithops_model_artifacts(id),
    artifact_hash text not null check (artifact_hash ~ '^[0-9a-f]{64}$'),
    input_hash text not null check (input_hash ~ '^[0-9a-f]{64}$'),
    payload jsonb not null,
    created_at timestamptz not null,
    unique (run_id, challenge_id, author_key)
);

create table public.lithops_fitted_models (
    id uuid not null,
    run_id uuid not null references public.lithops_runs(id) on delete cascade,
    artifact_id uuid not null references public.lithops_model_artifacts(id),
    artifact_hash text not null check (artifact_hash ~ '^[0-9a-f]{64}$'),
    state_hash text not null check (state_hash ~ '^[0-9a-f]{64}$'),
    training_start_day integer not null check (training_start_day >= 0),
    training_end_day integer not null check (training_end_day >= training_start_day),
    payload jsonb not null,
    created_at timestamptz not null,
    primary key (run_id, id),
    unique (run_id, state_hash)
);

create table public.lithops_sandbox_executions (
    id uuid primary key,
    run_id uuid not null references public.lithops_runs(id) on delete cascade,
    idempotency_key text not null,
    artifact_id uuid not null references public.lithops_model_artifacts(id),
    fitted_model_id uuid,
    operation text not null check (
        operation in ('validate', 'test', 'fit', 'predict', 'diagnostics')
    ),
    status text not null check (status in ('completed', 'denied', 'failed', 'timed_out')),
    input_hash text not null check (input_hash ~ '^[0-9a-f]{64}$'),
    output_hash text check (output_hash ~ '^[0-9a-f]{64}$'),
    payload jsonb not null,
    created_at timestamptz not null,
    unique (run_id, idempotency_key),
    foreign key (run_id, fitted_model_id)
        references public.lithops_fitted_models(run_id, id)
);

create table public.lithops_temporal_evaluation_folds (
    id uuid primary key,
    run_id uuid not null references public.lithops_runs(id) on delete cascade,
    artifact_id uuid not null references public.lithops_model_artifacts(id),
    fitted_model_id uuid not null,
    fold_index integer not null check (fold_index >= 0),
    total_score double precision not null check (total_score >= 0),
    invariant_gate_passed boolean not null,
    payload jsonb not null,
    created_at timestamptz not null,
    unique (run_id, id),
    unique (run_id, artifact_id, fitted_model_id, fold_index),
    foreign key (run_id, fitted_model_id)
        references public.lithops_fitted_models(run_id, id)
);

create table public.lithops_model_promotion_decisions (
    id uuid primary key,
    challenge_id uuid not null,
    run_id uuid not null references public.lithops_runs(id) on delete cascade,
    decision_day integer not null check (decision_day >= 0),
    champion_artifact_id uuid not null references public.lithops_model_artifacts(id),
    champion_fitted_model_id uuid not null,
    candidate_artifact_id uuid references public.lithops_model_artifacts(id),
    candidate_fitted_model_id uuid,
    disposition text not null check (disposition in ('promoted', 'rejected', 'no_update')),
    payload jsonb not null,
    created_at timestamptz not null,
    check (candidate_fitted_model_id is null or candidate_artifact_id is not null),
    check (
        disposition <> 'promoted'
        or (candidate_artifact_id is not null and candidate_fitted_model_id is not null)
    ),
    foreign key (run_id, champion_fitted_model_id)
        references public.lithops_fitted_models(run_id, id),
    foreign key (run_id, candidate_fitted_model_id)
        references public.lithops_fitted_models(run_id, id),
    unique (run_id, challenge_id)
);

create table public.lithops_model_promotion_evaluation_folds (
    promotion_decision_id uuid not null
        references public.lithops_model_promotion_decisions(id) on delete cascade,
    run_id uuid not null references public.lithops_runs(id) on delete cascade,
    evaluation_fold_id uuid not null,
    primary key (promotion_decision_id, evaluation_fold_id),
    foreign key (run_id, evaluation_fold_id)
        references public.lithops_temporal_evaluation_folds(run_id, id)
);

create function public.lithops_materialize_promotion_evaluation_folds()
returns trigger
language plpgsql
set search_path = ''
as $$
begin
    insert into public.lithops_model_promotion_evaluation_folds (
        promotion_decision_id,
        run_id,
        evaluation_fold_id
    )
    select
        new.id,
        new.run_id,
        (fold_id #>> '{}')::uuid
    from jsonb_array_elements(new.payload -> 'evaluation_fold_ids') as fold_id;
    return new;
end;
$$;

create trigger lithops_model_promotion_materialize_evaluation_folds
after insert on public.lithops_model_promotion_decisions
for each row execute function public.lithops_materialize_promotion_evaluation_folds();

create table public.lithops_active_model_assignments (
    id uuid primary key,
    run_id uuid not null references public.lithops_runs(id) on delete cascade,
    sequence integer not null check (sequence >= 1),
    artifact_id uuid not null references public.lithops_model_artifacts(id),
    fitted_model_id uuid not null,
    promotion_decision_id uuid not null unique
        references public.lithops_model_promotion_decisions(id),
    payload jsonb not null,
    created_at timestamptz not null,
    unique (run_id, sequence),
    foreign key (run_id, fitted_model_id)
        references public.lithops_fitted_models(run_id, id)
);

create index lithops_fitted_models_run_created_idx
    on public.lithops_fitted_models (run_id, created_at desc);
create index lithops_model_artifact_authoring_challenge_idx
    on public.lithops_model_artifact_authoring_receipts (run_id, challenge_id);
create index lithops_sandbox_executions_run_created_idx
    on public.lithops_sandbox_executions (run_id, created_at);
create index lithops_temporal_evaluation_folds_run_artifact_idx
    on public.lithops_temporal_evaluation_folds (run_id, artifact_id, fold_index);
create index lithops_model_promotion_decisions_run_day_idx
    on public.lithops_model_promotion_decisions (run_id, decision_day);
create index lithops_active_model_assignments_run_sequence_idx
    on public.lithops_active_model_assignments (run_id, sequence desc);

alter table public.lithops_model_artifacts enable row level security;
alter table public.lithops_model_artifact_authoring_receipts enable row level security;
alter table public.lithops_fitted_models enable row level security;
alter table public.lithops_sandbox_executions enable row level security;
alter table public.lithops_temporal_evaluation_folds enable row level security;
alter table public.lithops_model_promotion_decisions enable row level security;
alter table public.lithops_model_promotion_evaluation_folds enable row level security;
alter table public.lithops_active_model_assignments enable row level security;

revoke all on table public.lithops_model_artifacts from anon, authenticated;
revoke all on table public.lithops_model_artifact_authoring_receipts from anon, authenticated;
revoke all on table public.lithops_fitted_models from anon, authenticated;
revoke all on table public.lithops_sandbox_executions from anon, authenticated;
revoke all on table public.lithops_temporal_evaluation_folds from anon, authenticated;
revoke all on table public.lithops_model_promotion_decisions from anon, authenticated;
revoke all on table public.lithops_model_promotion_evaluation_folds from anon, authenticated;
revoke all on table public.lithops_active_model_assignments from anon, authenticated;

create policy "deny direct client access"
    on public.lithops_model_artifacts
    for all to anon, authenticated using (false) with check (false);
create policy "deny direct client access"
    on public.lithops_model_artifact_authoring_receipts
    for all to anon, authenticated using (false) with check (false);
create policy "deny direct client access"
    on public.lithops_fitted_models
    for all to anon, authenticated using (false) with check (false);
create policy "deny direct client access"
    on public.lithops_sandbox_executions
    for all to anon, authenticated using (false) with check (false);
create policy "deny direct client access"
    on public.lithops_temporal_evaluation_folds
    for all to anon, authenticated using (false) with check (false);
create policy "deny direct client access"
    on public.lithops_model_promotion_decisions
    for all to anon, authenticated using (false) with check (false);
create policy "deny direct client access"
    on public.lithops_model_promotion_evaluation_folds
    for all to anon, authenticated using (false) with check (false);
create policy "deny direct client access"
    on public.lithops_active_model_assignments
    for all to anon, authenticated using (false) with check (false);

revoke all on function public.lithops_materialize_promotion_evaluation_folds()
    from public, anon, authenticated;
