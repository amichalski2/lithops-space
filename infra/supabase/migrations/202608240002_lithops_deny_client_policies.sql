create policy "deny direct client access"
    on public.lithops_runs
    for all
    to anon, authenticated
    using (false)
    with check (false);

create policy "deny direct client access"
    on public.lithops_decisions
    for all
    to anon, authenticated
    using (false)
    with check (false);

create policy "deny direct client access"
    on public.lithops_action_receipts
    for all
    to anon, authenticated
    using (false)
    with check (false);

create policy "deny direct client access"
    on public.lithops_operations
    for all
    to anon, authenticated
    using (false)
    with check (false);

create policy "deny direct client access"
    on public.lithops_events
    for all
    to anon, authenticated
    using (false)
    with check (false);

revoke all on sequence public.lithops_events_id_seq from anon, authenticated;
