"""Formatting helpers for routing call records.

After a ``with LLMTracker(router=...) as tracker`` block exits,
``tracker.records`` holds the list of :class:`agentopt.CallRecord`s
produced under that session.  :func:`print_routing_summary` formats
that list into the three things a user usually wants to see at a glance:

* the model used for each call, in order;
* total input/output tokens per model;
* total wall latency.

:meth:`agentopt.LLMTracker.print_summary` is a thin shortcut for
``print_routing_summary(tracker.records)``.
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Dict, List, Sequence

if TYPE_CHECKING:
    from agentopt.proxy.models import CallRecord


def format_routing_summary(records: Sequence["CallRecord"]) -> str:
    """Return a multi-line summary string for *records*.

    When records span multiple distinct ``data_id``s, the sequence
    section is grouped per datapoint so you can see what each one
    routed to.  Otherwise it falls back to a flat list.

    Use this when you want to log or otherwise capture the summary
    instead of printing it; :func:`print_routing_summary` is the
    one-liner most callers want.
    """
    if not records:
        return "(no records — was the router active?)"

    lines: List[str] = []
    lines.append("=" * 60)
    lines.append("Routing summary")
    lines.append("=" * 60)

    # Group by data_id if there's meaningful per-datapoint attribution.
    data_ids = [r.data_id for r in records if r.data_id]
    distinct_data_ids = list(dict.fromkeys(data_ids))  # preserve order
    group_by_dp = len(distinct_data_ids) > 1

    lines.append("")
    if group_by_dp:
        lines.append("Model usage by datapoint:")
        groups: Dict[str, List["CallRecord"]] = defaultdict(list)
        for r in records:
            groups[r.data_id or "<unattributed>"].append(r)
        for dp in distinct_data_ids:
            grp = groups.get(dp, [])
            total = sum(r.latency_seconds for r in grp)
            lines.append(f"  [{dp}]  {len(grp)} call(s), {total:.2f}s")
            for r in grp:
                marker = " (cached)" if r.cached else ""
                lines.append(f"      {r.model:<30} {r.latency_seconds:>6.2f}s{marker}")
    else:
        lines.append("Model usage sequence:")
        for i, r in enumerate(records, 1):
            marker = " (cached)" if r.cached else ""
            lines.append(f"  {i:>2}. {r.model:<30} {r.latency_seconds:>6.2f}s{marker}")

    per_model: Dict[str, List[int]] = defaultdict(lambda: [0, 0])
    for r in records:
        per_model[r.model][0] += r.prompt_tokens
        per_model[r.model][1] += r.completion_tokens

    lines.append("")
    lines.append("Tokens per model:")
    width = max(len(m) for m in per_model)
    for model, (pt, ct) in sorted(per_model.items()):
        lines.append(
            f"  {model:<{width}}   prompt={pt:>6}   completion={ct:>6}   total={pt + ct:>6}"
        )

    total_latency = sum(r.latency_seconds for r in records)
    lines.append("")
    lines.append(f"Total latency: {total_latency:.2f}s across {len(records)} call(s)")
    return "\n".join(lines)


def print_routing_summary(records: Sequence["CallRecord"]) -> None:
    """Print model sequence / tokens-per-model / total latency for *records*."""
    print(format_routing_summary(records))
