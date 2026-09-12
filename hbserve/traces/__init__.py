"""Capture-bound reference generation; native coarse remains under hbserve run."""

ROUTES = {
    "reference": {
        "role": "capture-driven detailed reference",
        "requires_capture": True,
        "boundary": "generated addresses through one named reference GPU cache",
        "not_claimed": ["hardware post-L2 capture", "exact GPU issue timing",
                        "arbitrary model/context fidelity"],
    },
}
