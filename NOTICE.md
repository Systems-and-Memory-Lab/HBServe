# Origin notice

The initial HBServe codebase was extracted from the MIT-licensed HBFSim working
tree based on commit `b44a7960ea44e20252aa6c8e758e8f9ed2d7a4b5`. It was renamed,
made independently installable, and stripped of production evidence and local
experiment history for this repository.

HBServe retains the original HBFSim copyright notice in `LICENSE`.

The unified fixed-window frontend and shared client incorporate HBFSim commit
`70c3fd5` (serving/window frontend unification). This standalone port preserves
HBServe's package name, public model schema, and external-simulator interface.
The 70B descriptor preserves the original per-output-channel W8 scale and
embedding storage assumptions through the public model ledger. Paper-specific
analysis runners, production evidence, and local experiment history are not
included.

The reference-only frontend imports the documented reference workload/cache
implementation from local release snapshot `e42490f` (2026-09-12), with
per-file identities in `hbserve/traces/SOURCES.json`. Historical simple,
its activation supplement, and the private historical simulator client are
not part of this branch. The optional C++ component models a reference
GPU cache; it is not an HBFSim core modification.

Current-client/system-profile compatibility and the parameter-provenance
ledger follow HBFSim `dad246f007ab47f2864358cafc87a4951ce7b699`.
The reference source catalog contains exact retained workload metadata and
compact address templates, not model weights or a universal hardware trace.
SHA-256 identities, scope and availability are recorded per artifact.
