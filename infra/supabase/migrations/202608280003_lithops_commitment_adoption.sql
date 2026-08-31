-- A matured treatment can now be adopted as the new operating baseline, and a
-- commitment can be abandoned before its window closes. Both are decisions the
-- Executive makes, so both need to be recordable weekly verdicts.

alter table public.lithops_commitment_reviews
    drop constraint if exists lithops_commitment_reviews_verdict_check;

alter table public.lithops_commitment_reviews
    add constraint lithops_commitment_reviews_verdict_check
    check (verdict in (
        'continue', 'stop_for_safety', 'falsified', 'mature_and_probe',
        'revert', 'adopted', 'abandoned'
    ));
