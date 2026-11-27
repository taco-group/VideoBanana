"""Entry-checkpoint selection for a turn.

Every node of the conversation tree is a candidate, not only the previous
output. A node is feasible when it shows no edit the goal no longer wants.
Among feasible nodes the selector keeps the output within ``max_depth``
generations of the source, since quality decays with each regeneration.

A node that already satisfies the goal is reused as is. Otherwise the selector
takes the node with the shallowest output whose call stays within
``merge_max_instructions``, redoing earlier edits together with the new ones in
one call; when no such node exists it prefers the node that leaves the least
to execute.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .contracts import ConversationState
from .roles import Roles, compute_roles


@dataclass(frozen=True)
class EntryChoice:
    node_id: str
    roles: Roles
    reason: str


def expected_generations(execute_count: int, max_instructions_per_call: int) -> int:
    """Editor calls needed from an entry: one per call-sized group of instructions."""
    if execute_count == 0:
        return 0
    return math.ceil(execute_count / max_instructions_per_call)


def select_entry(state: ConversationState, exhausted: list[str]) -> EntryChoice:
    config = state.config
    options = []
    for node in state.nodes.values():
        roles = compute_roles(state, node.node_id, exhausted)
        if roles.unwanted:
            continue
        generations = expected_generations(len(roles.execute), config.max_instructions_per_call)
        options.append((node, roles, generations))
    # The source never shows an edit, so at least one option exists.
    within = [
        item
        for item in options
        if item[0].depth + item[2] <= config.max_depth
        and len(item[1].execute) <= config.max_instructions_per_call
    ]
    reusable = [item for item in within if not item[1].execute]
    if reusable:
        node, roles, _ = min(reusable, key=lambda item: (item[0].depth, -item[0].turn))
        return EntryChoice(node.node_id, roles, "entry already satisfies the goal")
    mergeable = [item for item in within if len(item[1].execute) <= config.merge_max_instructions]
    if mergeable:
        node, roles, _ = min(
            mergeable,
            key=lambda item: (item[0].depth + item[2], len(item[1].execute), -item[0].turn),
        )
        return EntryChoice(node.node_id, roles, "shallowest output within the merge limit")
    if within:
        node, roles, _ = min(within, key=lambda item: (len(item[1].execute), item[0].depth, -item[0].turn))
        return EntryChoice(node.node_id, roles, "fewest instructions within depth and call limits")
    fitting = [item for item in options if len(item[1].execute) <= config.max_instructions_per_call] or options
    node, roles, _ = min(fitting, key=lambda item: (item[0].depth + item[2], len(item[1].execute), -item[0].turn))
    return EntryChoice(node.node_id, roles, "no entry within the depth limit; shallowest output")
