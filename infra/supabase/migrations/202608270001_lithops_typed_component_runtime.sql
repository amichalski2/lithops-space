alter table public.lithops_model_artifacts
    drop constraint if exists lithops_model_artifacts_runtime_kind_check;

alter table public.lithops_model_artifacts
    add constraint lithops_model_artifacts_runtime_kind_check
    check (
        runtime_kind in (
            'sandboxed_python',
            'trusted_baseline',
            'typed_component_assembly'
        )
    );
