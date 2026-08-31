-- Scope temporal evaluation folds by the challenge and seed that produced them.
--
-- Fold identity previously covered only (run, artifact, fitted model, fold index).
-- Two legitimate evaluations can share all four: a later challenge re-scoring the
-- unchanged fixed baseline, and two candidates evaluated side by side inside one
-- challenge on deliberately different seeds. Both then collided on one immutable
-- row and failed the week. The domain identity now carries the challenge and the
-- evaluation seed, and this constraint mirrors it.

alter table public.lithops_temporal_evaluation_folds
    add column if not exists challenge_id uuid,
    add column if not exists evaluation_seed bigint;

do $$
declare
    constraint_name text;
begin
    select conname
      into constraint_name
      from pg_constraint
     where conrelid = 'public.lithops_temporal_evaluation_folds'::regclass
       and contype = 'u'
       and pg_get_constraintdef(oid)
           = 'UNIQUE (run_id, artifact_id, fitted_model_id, fold_index)';
    if constraint_name is not null then
        execute format(
            'alter table public.lithops_temporal_evaluation_folds drop constraint %I',
            constraint_name
        );
    end if;
end $$;

alter table public.lithops_temporal_evaluation_folds
    drop constraint if exists lithops_temporal_eval_folds_challenge_scope_key;

alter table public.lithops_temporal_evaluation_folds
    add constraint lithops_temporal_eval_folds_challenge_scope_key
    unique (run_id, challenge_id, artifact_id, fitted_model_id, fold_index, evaluation_seed);

create index if not exists lithops_temporal_evaluation_folds_run_challenge_idx
    on public.lithops_temporal_evaluation_folds (run_id, challenge_id);
