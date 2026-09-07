"""HBServe's deterministic fixed-window generators and matrix executor.

These memory-only controls deliberately fix logical work across topologies;
they do not use the closed-loop request scheduler or produce serving latency.
"""
