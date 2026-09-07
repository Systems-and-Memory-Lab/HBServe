"""Minimal, semantic-free client for the external HBFSim executable."""

from hbfsim_client.simulation_session import (
    BatchResult,
    ResolvedSystemConfig,
    SimulationSession,
    SimulationSessionError,
)
from hbfsim_client.transaction_protocol import (
    HbfGeometry,
    NO_UPSTREAM_DIGEST,
    TRANSACTION_TARGETS,
    Transaction,
    TransactionBatch,
    TransactionProtocolError,
    hbf_dense_mapping_pages,
    hbf_link_bytes_by_stack,
)

__all__ = [
    "BatchResult",
    "HbfGeometry",
    "NO_UPSTREAM_DIGEST",
    "ResolvedSystemConfig",
    "SimulationSession",
    "SimulationSessionError",
    "TRANSACTION_TARGETS",
    "Transaction",
    "TransactionBatch",
    "TransactionProtocolError",
    "hbf_dense_mapping_pages",
    "hbf_link_bytes_by_stack",
]
