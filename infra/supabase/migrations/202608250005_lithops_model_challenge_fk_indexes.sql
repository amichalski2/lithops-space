create index lithops_model_challenges_health_signal_idx
    on public.lithops_model_challenges (health_signal_id);
create index lithops_model_challenges_base_model_idx
    on public.lithops_model_challenges (base_model_version_id);
create index lithops_model_challenge_packages_run_idx
    on public.lithops_model_challenge_packages (run_id);
create index lithops_model_challenge_decisions_activated_model_idx
    on public.lithops_model_challenge_decisions (activated_model_version_id);
