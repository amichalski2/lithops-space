create index lithops_world_models_parent_idx
    on public.lithops_world_models (based_on_version_id);
create index lithops_predictions_model_version_idx
    on public.lithops_predictions (model_version_id);
create index lithops_prediction_outcomes_ledger_idx
    on public.lithops_prediction_outcomes (ledger_entry_id);
create index lithops_model_health_signals_model_version_idx
    on public.lithops_model_health_signals (model_version_id);
