"""Efficient LangGraph message state for durable Remy conversations."""

from langgraph.channels.delta import DeltaChannel
from langgraph.graph.message import _messages_delta_reducer

# Frequent snapshots keep replay bounded while delta writes avoid serialising
# the complete conversation after every tool/model node. LangGraph is pinned in
# pyproject.toml, so this private reducer is covered by Remy's upgrade tests.
MESSAGE_CHANNEL = DeltaChannel(
    _messages_delta_reducer,
    list,
    snapshot_frequency=50,
)
