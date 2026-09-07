# Configuration provenance

The checked-in system configurations are exploratory simulator profiles, not
vendor performance guarantees. They are included so users can exercise
placement and protocol behavior with a compatible HBFSim build.

`4hbm-4hbf.cfg` and its `-miniquick` variant combine a four-stack HBM domain
with four HBF stacks. Link rates, queue depths, controller timing, flash-media
timing, overprovisioning, and thermal parameters are modeling assumptions. The
mini profile reduces capacity and host work for fast integration tests; it is
not a smaller hardware product claim.

`eight-stack-baseline.cfg`, `simulation-session-mini.cfg`, `cxl-memory.cfg`,
and `nvme-ssd.cfg` exercise the baseline HBM domain, persistent session
protocol, and external backing alternatives.

For publishable results, freeze the complete config set, hash it in the run
receipt, cite a source or calibration artifact for every physical parameter,
and validate against a holdout workload. Parameters without such evidence must
remain labeled assumptions or sensitivity variables.
