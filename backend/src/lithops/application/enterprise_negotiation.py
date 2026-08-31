"""Deterministic execution of an authorized enterprise negotiation policy.

The Executive authorizes an envelope — engage or not, a target and a floor price
per seat, a seat ceiling — as part of the candidate that was simulated. This
module only executes inside that envelope: it walks the open threads, makes the
offers, and stops. It never chooses a price the envelope does not already allow,
and it never invents a thread.

Answering every open thread in the week it arrives keeps the company inside the
counterparty's response window without encoding what that window is.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from uuid import UUID

from lithops.domain.models import ActionCommand, ActionPlan, ActionReceipt

# The counterparty only replies as benchmark days advance, so a second offer in
# the same week would answer nothing. One offer per thread per week is therefore
# the fastest honest cadence; a counter-offer is answered in the next week's
# envelope, which still lands inside the thread's response window.
MAX_OFFERS_PER_THREAD_PER_WEEK = 1
MAX_OFFERS_PER_WEEK = 12

ExecuteAction = Callable[[ActionCommand], Awaitable[ActionReceipt]]
EmitEvent = Callable[[str, dict], Awaitable[object]]


@dataclass(frozen=True, slots=True)
class EnterpriseThread:
    """One open negotiation thread as reported by the observation inbox."""

    customer_id: int
    seat_count: int
    opened_day: int


@dataclass(frozen=True, slots=True)
class NegotiationOutcome:
    receipts: tuple[ActionReceipt, ...]
    offers_made: int
    threads_seen: int
    skipped: tuple[str, ...]


def parse_enterprise_inbox(inbox: str) -> tuple[EnterpriseThread, ...]:
    """Read the compact ``customer:seats:day`` inbox, skipping malformed entries."""

    threads: dict[int, EnterpriseThread] = {}
    for entry in inbox.split(","):
        parts = entry.split(":")
        if len(parts) < 3:
            continue
        try:
            customer_id = int(parts[0])
            seat_count = max(1, int(float(parts[1])))
            opened_day = int(float(parts[2]))
        except ValueError:
            continue
        if customer_id > 0:
            threads[customer_id] = EnterpriseThread(
                customer_id=customer_id,
                seat_count=seat_count,
                opened_day=opened_day,
            )
    return tuple(threads.values())


def offer_price_per_seat(
    plan: ActionPlan,
    *,
    offer_index: int,
    total_offers: int,
) -> float:
    """Walk from the authorized target toward the authorized floor.

    A mechanical concession schedule inside an interval the Executive chose; it
    is execution of a policy, not a pricing decision of its own.
    """

    target = plan.enterprise_target_price_per_seat or 0.0
    floor = plan.enterprise_floor_price_per_seat or target
    if total_offers <= 1 or target <= floor:
        return target
    step = (target - floor) / (total_offers - 1)
    return max(floor, target - step * offer_index)


def seats_within_ceiling(
    threads: tuple[EnterpriseThread, ...],
    *,
    ceiling: float | None,
) -> tuple[EnterpriseThread, ...]:
    """Take threads in arrival order until the authorized seat ceiling is met."""

    if ceiling is None:
        return threads
    admitted: list[EnterpriseThread] = []
    committed = 0.0
    for thread in sorted(threads, key=lambda item: (item.opened_day, item.customer_id)):
        if committed + thread.seat_count > ceiling:
            continue
        admitted.append(thread)
        committed += thread.seat_count
    return tuple(admitted)


async def negotiate_open_threads(
    *,
    run_id: UUID,
    week: int,
    plan: ActionPlan,
    inbox: str,
    variable_cost_per_seat_weekly: float,
    execute_action: ExecuteAction,
    emit_event: EmitEvent,
) -> NegotiationOutcome:
    """Answer every open thread once, inside the authorized envelope."""

    threads = parse_enterprise_inbox(inbox)
    if not threads:
        return NegotiationOutcome((), 0, 0, ())
    if not plan.enterprise_engage or not plan.enterprise_target_price_per_seat:
        await emit_event(
            "enterprise.threads_not_engaged",
            {"week": week, "open_threads": len(threads)},
        )
        return NegotiationOutcome((), 0, len(threads), ("not_engaged",))

    admitted = seats_within_ceiling(threads, ceiling=plan.enterprise_max_new_seats)
    skipped: list[str] = []
    if len(admitted) < len(threads):
        skipped.append("seat_ceiling_reached")

    receipts: list[ActionReceipt] = []
    offers = 0
    for index, thread in enumerate(admitted):
        if offers >= MAX_OFFERS_PER_WEEK:
            skipped.append("weekly_offer_cap_reached")
            break
        price = offer_price_per_seat(
            plan, offer_index=index, total_offers=len(admitted)
        )
        # A monthly per-seat price must still cover the weekly cost of serving it.
        if price * 7.0 / 30.0 < variable_cost_per_seat_weekly:
            skipped.append(f"below_cost:{thread.customer_id}")
            await emit_event(
                "enterprise.offer_below_cost",
                {
                    "week": week,
                    "customer_id": thread.customer_id,
                    "price_per_seat": price,
                },
            )
            continue
        command = ActionCommand(
            tool="send_enterprise_deal",
            arguments={"deals": [[thread.customer_id, [["A", price]]]]},
            idempotency_key=(
                f"{run_id}:week-{week}:enterprise-{thread.customer_id}-offer-{index}"
            ),
        )
        try:
            receipt = await execute_action(command)
        except Exception as error:
            skipped.append(f"offer_failed:{thread.customer_id}")
            await emit_event(
                "enterprise.offer_failed",
                {
                    "week": week,
                    "customer_id": thread.customer_id,
                    "error": str(error)[:500],
                },
            )
            continue
        receipts.append(receipt)
        offers += 1
        await emit_event(
            "enterprise.turn_sent",
            {
                "week": week,
                "customer_id": thread.customer_id,
                "seat_count": thread.seat_count,
                "price_per_seat": price,
            },
        )
    return NegotiationOutcome(
        receipts=tuple(receipts),
        offers_made=offers,
        threads_seen=len(threads),
        skipped=tuple(skipped),
    )
