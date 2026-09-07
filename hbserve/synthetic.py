#!/usr/bin/env python3
"""Deterministic synthetic request and MoE-router sensitivity workloads."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
import hashlib
import json
import math
import random
from typing import Mapping, Sequence

from hbserve.contracts import (
    ModelSpec,
    RequestSpec,
    RequestTrace,
    HBServeError,
    TraceProvenance,
    canonical_sha256,
)


@dataclass(frozen=True)
class SyntheticRequestConfig:
    request_count: int
    arrival_rate_per_second: float
    prompt_lognormal_mean_tokens: float
    prompt_lognormal_sigma: float
    output_lognormal_mean_tokens: float
    output_lognormal_sigma: float
    model_probabilities: Mapping[str, float]
    seed: int
    first_arrival_policy: str = "poisson_gap"

    def __post_init__(self) -> None:
        if (
            isinstance(self.request_count, bool)
            or not isinstance(self.request_count, int)
            or self.request_count <= 0
        ):
            raise HBServeError("synthetic request_count must be > 0")
        for name, value, strictly_positive in (
            ("arrival_rate_per_second", self.arrival_rate_per_second, True),
            (
                "prompt_lognormal_mean_tokens",
                self.prompt_lognormal_mean_tokens,
                True,
            ),
            ("prompt_lognormal_sigma", self.prompt_lognormal_sigma, False),
            (
                "output_lognormal_mean_tokens",
                self.output_lognormal_mean_tokens,
                True,
            ),
            ("output_lognormal_sigma", self.output_lognormal_sigma, False),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or (float(value) <= 0 if strictly_positive else float(value) < 0)
            ):
                comparator = "> 0" if strictly_positive else ">= 0"
                raise HBServeError(
                    f"synthetic {name} must be finite and {comparator}"
                )
        if not self.model_probabilities:
            raise HBServeError(
                "synthetic model_probabilities must not be empty"
            )
        total = 0.0
        for model_id, probability in self.model_probabilities.items():
            if not model_id or (
                isinstance(probability, bool)
                or not isinstance(probability, (int, float))
                or not math.isfinite(float(probability))
                or probability <= 0
            ):
                raise HBServeError(
                    "synthetic model probabilities must be finite and positive"
                )
            total += float(probability)
        if not math.isfinite(total) or total <= 0:
            raise HBServeError(
                "synthetic model probability sum is invalid"
            )
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise HBServeError("synthetic seed must be an integer")
        if self.first_arrival_policy not in {"zero", "poisson_gap"}:
            raise HBServeError(
                "first_arrival_policy must be zero or poisson_gap"
            )

    def canonical(self) -> dict[str, object]:
        return {
            "request_count": self.request_count,
            "arrival_rate_per_second": self.arrival_rate_per_second,
            "prompt_lognormal_mean_tokens": (
                self.prompt_lognormal_mean_tokens
            ),
            "prompt_lognormal_sigma": self.prompt_lognormal_sigma,
            "output_lognormal_mean_tokens": (
                self.output_lognormal_mean_tokens
            ),
            "output_lognormal_sigma": self.output_lognormal_sigma,
            "model_probabilities": dict(sorted(self.model_probabilities.items())),
            "seed": self.seed,
            "first_arrival_policy": self.first_arrival_policy,
            "arrival_process": (
                "poisson_exponential_interarrival_rounded_nanoseconds_v1"
            ),
            "length_process": "mean_parameterized_lognormal_rounded_v1",
        }


def _lognormal_integer(rng: random.Random, mean: float, sigma: float) -> int:
    # E[X] = exp(mu + sigma^2 / 2).  Taking the requested value as mu would
    # silently make the configured arithmetic mean wrong.
    mu = math.log(mean) - sigma * sigma / 2.0
    return max(1, int(round(rng.lognormvariate(mu, sigma))))


def _weighted_choice(
    rng: random.Random, weighted: Sequence[tuple[str, float]]
) -> str:
    total = sum(weight for _, weight in weighted)
    needle = rng.random() * total
    cumulative = 0.0
    for value, weight in weighted:
        cumulative += weight
        if needle < cumulative:
            return value
    return weighted[-1][0]


def generate_requests(config: SyntheticRequestConfig) -> RequestTrace:
    """Generate a reproducible Poisson/lognormal multi-model request trace."""

    rng = random.Random(config.seed)
    weighted_models = tuple(sorted(config.model_probabilities.items()))
    arrival_ns = 0.0
    requests: list[RequestSpec] = []
    for index in range(config.request_count):
        if index or config.first_arrival_policy == "poisson_gap":
            arrival_ns += (
                rng.expovariate(config.arrival_rate_per_second) * 1e9
            )
        request = RequestSpec(
            request_id=f"req{index:08d}",
            arrival_ns=float(round(arrival_ns)),
            model_id=_weighted_choice(rng, weighted_models),
            prompt_tokens=_lognormal_integer(
                rng,
                config.prompt_lognormal_mean_tokens,
                config.prompt_lognormal_sigma,
            ),
            output_tokens=_lognormal_integer(
                rng,
                config.output_lognormal_mean_tokens,
                config.output_lognormal_sigma,
            ),
        )
        requests.append(request)
    canonical_parameters = config.canonical()
    return RequestTrace(
        provenance=TraceProvenance(
            kind="synthetic_sensitivity",
            source=(
                "inline:"
                + json.dumps(
                    canonical_parameters,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
            ),
        ),
        requests=tuple(requests),
    )


@dataclass(frozen=True)
class HotsetZipfRouter:
    """Stateless weighted-without-replacement synthetic MoE routing.

    The configured ``hot_mass`` is the sum of the *sampling weights* assigned
    to the hot set.  Top-k sampling without replacement means the observed
    fraction is not mathematically forced to equal that mass; studies must
    report the observed route census.
    """

    seed: int
    hot_experts: int
    hot_mass: float
    alpha: float

    def __post_init__(self) -> None:
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise HBServeError("router seed must be an integer")
        if (
            isinstance(self.hot_experts, bool)
            or not isinstance(self.hot_experts, int)
            or self.hot_experts <= 0
        ):
            raise HBServeError("hot_experts must be an integer > 0")
        if (
            isinstance(self.hot_mass, bool)
            or not isinstance(self.hot_mass, (int, float))
            or not math.isfinite(float(self.hot_mass))
            or not 0.0 < float(self.hot_mass) <= 1.0
        ):
            raise HBServeError("hot_mass must be in (0, 1]")
        if (
            isinstance(self.alpha, bool)
            or not isinstance(self.alpha, (int, float))
            or not math.isfinite(float(self.alpha))
            or float(self.alpha) < 0.0
        ):
            raise HBServeError("router alpha must be finite and >= 0")

    @cached_property
    def provenance(self) -> TraceProvenance:
        return TraceProvenance(
            kind="synthetic_sensitivity",
            source=(
                "hotset_zipf_weighted_without_replacement_v1:"
                f"seed={self.seed},hot_experts={self.hot_experts},"
                f"hot_mass={self.hot_mass:.17g},alpha={self.alpha:.17g}"
            ),
        )

    @cached_property
    def digest(self) -> str:
        return canonical_sha256(
            {
                "algorithm": "exponential_race_weighted_without_replacement_v1",
                "seed": self.seed,
                "hot_experts": self.hot_experts,
                "hot_mass": self.hot_mass,
                "alpha": self.alpha,
                "provenance": self.provenance.canonical(),
            }
        )

    def _weights(self, expert_count: int) -> tuple[float, ...]:
        if self.hot_experts > expert_count:
            raise HBServeError(
                "synthetic hot expert count exceeds a model layer"
            )
        cold_count = expert_count - self.hot_experts
        if cold_count == 0 and not math.isclose(
            self.hot_mass, 1.0, rel_tol=0.0, abs_tol=1e-15
        ):
            raise HBServeError(
                "hot_mass must be one when every expert is in the hot set"
            )
        if cold_count and self.hot_mass == 1.0:
            raise HBServeError(
                "hot_mass=1 leaves cold experts with zero sampling weight"
            )
        hot_ranks = tuple(
            1.0 / ((index + 1) ** self.alpha)
            for index in range(self.hot_experts)
        )
        hot_total = sum(hot_ranks)
        hot = tuple(self.hot_mass * value / hot_total for value in hot_ranks)
        if cold_count == 0:
            return hot
        cold_ranks = tuple(
            1.0 / ((index + 1) ** self.alpha) for index in range(cold_count)
        )
        cold_total = sum(cold_ranks)
        cold = tuple(
            (1.0 - self.hot_mass) * value / cold_total
            for value in cold_ranks
        )
        return hot + cold

    def experts_for(
        self,
        *,
        request: RequestSpec,
        token_index: int,
        layer: int,
        model: ModelSpec,
    ) -> tuple[int, ...]:
        if layer < 0 or layer >= model.num_layers:
            raise HBServeError("router layer is outside the model")
        layer_spec = model.layers[layer]
        if not layer_spec.is_moe:
            raise HBServeError("synthetic router targets a dense layer")
        if token_index < 0 or token_index >= request.processed_input_tokens:
            raise HBServeError("synthetic router token is out of range")
        weights = self._weights(len(layer_spec.expert_weight_bytes))
        if layer_spec.top_k > sum(weight > 0 for weight in weights):
            raise HBServeError(
                "router distribution has fewer positive-weight experts than top_k"
            )
        priorities: list[tuple[float, int]] = []
        for expert, weight in enumerate(weights):
            if weight == 0:
                priorities.append((math.inf, expert))
                continue
            key = (
                f"{self.seed}\0{model.model_id}\0{request.request_id}\0"
                f"{token_index}\0{layer}\0{expert}"
            ).encode("utf-8")
            raw = int.from_bytes(hashlib.sha256(key).digest()[:8], "big")
            # Strictly inside (0, 1), avoiding log(0) and endpoint bias.
            uniform = (raw + 0.5) / 2**64
            priorities.append((-math.log(uniform) / weight, expert))
        priorities.sort()
        selected = tuple(
            sorted(expert for _, expert in priorities[: layer_spec.top_k])
        )
        if len(selected) != layer_spec.top_k or len(set(selected)) != len(selected):
            raise HBServeError("synthetic router selection is malformed")
        return selected

    def validate_complete(
        self,
        *,
        requests: Sequence[RequestSpec],
        models: Mapping[str, ModelSpec],
    ) -> None:
        for request in requests:
            try:
                model = models[request.model_id]
            except KeyError as error:
                raise HBServeError(
                    f"request names unknown model {request.model_id}"
                ) from error
            for layer_spec in model.layers:
                if layer_spec.is_moe:
                    weights = self._weights(
                        len(layer_spec.expert_weight_bytes)
                    )
                    if layer_spec.top_k > sum(weight > 0 for weight in weights):
                        raise HBServeError(
                            "router distribution has fewer positive-weight experts "
                            "than top_k"
                        )
