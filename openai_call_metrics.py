"""
Track OpenAI Chat Completions cost (from API usage) and wall-clock time per call.

Cost uses official list prices where configured (per 1M input/output tokens, USD).
Adjust MODEL_USD_PER_1M when OpenAI updates pricing, or use register_pricing() at runtime.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional

# USD per 1M input tokens, USD per 1M output tokens (standard, non-batch).
# Source: https://platform.openai.com/docs/pricing (verify periodically).
_MODEL_USD_PER_1M: dict[str, tuple[float, float]] = {
    "gpt-5-nano": (0.05, 0.40),
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-3.5-turbo": (0.50, 1.50),
}

_registered: dict[str, tuple[float, float]] = {}


def register_pricing(model: str, usd_per_1m_input: float, usd_per_1m_output: float) -> None:
    """Register or override (input, output) USD price per 1M tokens for a model id or family prefix."""
    _registered[model] = (usd_per_1m_input, usd_per_1m_output)


def _resolve_pricing(model: str) -> Optional[tuple[float, float]]:
    m = (model or "").lower().strip()
    if m in _registered:
        return _registered[m]
    if m in _MODEL_USD_PER_1M:
        return _MODEL_USD_PER_1M[m]
    for key, rates in _registered.items():
        if m.startswith(key.lower()):
            return rates
    for key, rates in _MODEL_USD_PER_1M.items():
        if m.startswith(key.lower()):
            return rates
    return None


def estimate_call_cost_usd(
    model: str,
    *,
    input_tokens: int,
    output_tokens: int,
) -> Optional[float]:
    """
    Estimated USD for one completion from token counts. Returns None if model pricing is unknown.
    """
    rates = _resolve_pricing(model)
    if rates is None:
        return None
    inp_rate, out_rate = rates
    return (input_tokens * inp_rate + output_tokens * out_rate) / 1_000_000


@dataclass
class CallMetric:
    """One Chat Completions request: usage, time, and estimated cost."""

    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: Optional[float]
    elapsed_s: float


@dataclass
class OpenAIMetricsSession:
    """Collects per-call metrics and can summarize total cost and latency."""

    calls: list[CallMetric] = field(default_factory=list)

    def add(self, metric: CallMetric) -> None:
        self.calls.append(metric)

    @property
    def total_input_tokens(self) -> int:
        return sum(c.input_tokens for c in self.calls)

    @property
    def total_output_tokens(self) -> int:
        return sum(c.output_tokens for c in self.calls)

    @property
    def total_elapsed_s(self) -> float:
        return sum(c.elapsed_s for c in self.calls)

    @property
    def total_cost_usd(self) -> Optional[float]:
        if not self.calls:
            return 0.0
        costs = [c.cost_usd for c in self.calls]
        if all(c is None for c in costs):
            return None
        return sum(c for c in costs if c is not None)

    def format_summary(self) -> str:
        n = len(self.calls)
        lines = [
            f"OpenAI chat.completions calls: {n}",
            f"Total wall time: {self.total_elapsed_s:.3f} s (sum of per-call latency)",
            f"Total tokens: {self.total_input_tokens} in + {self.total_output_tokens} out = "
            f"{self.total_input_tokens + self.total_output_tokens}",
        ]
        tc = self.total_cost_usd
        if any(c.cost_usd is None for c in self.calls):
            known_sum = sum(c.cost_usd for c in self.calls if c.cost_usd is not None)
            lines.append(
                f"Estimated cost (known pricing only): ${known_sum:.6f} USD "
                f"(some models had no pricing; see per-call or register_pricing())"
            )
        else:
            lines.append(f"Estimated total cost: ${(tc or 0.0):.6f} USD")
        if n and n <= 20:
            for i, c in enumerate(self.calls, 1):
                cost = f"${c.cost_usd:.6f}" if c.cost_usd is not None else "n/a"
                lines.append(
                    f"  #{i} {c.model} | in={c.input_tokens} out={c.output_tokens} | "
                    f"cost~{cost} | {c.elapsed_s:.3f}s"
                )
        return "\n".join(lines)

    def format_batch_totals(self) -> str:
        """Short block: total time and estimated cost. Delegates to :func:`format_openai_session_batch_totals`."""
        return format_openai_session_batch_totals(self)


def format_openai_session_batch_totals(session: OpenAIMetricsSession) -> str:
    """
    Text summary: call count, total wall time, estimated total cost.
    Exposed as a function so the notebook can use it even when the kernel has an old
    ``OpenAIMetricsSession`` class (it still works on the same session object).
    """
    n = len(session.calls)
    t = session.total_elapsed_s
    lines = [
        "========== Batch totals (this run) ==========",
        f"  Calls:              {n}",
        f"  Total wall time:   {t:.3f} s  (sum of per-request round-trip time)",
    ]
    if any(c.cost_usd is None for c in session.calls):
        known = sum(c.cost_usd for c in session.calls if c.cost_usd is not None)
        lines.append(
            f"  Est. total cost:   ${known:.6f} USD  (known pricing only; some calls omitted)"
        )
    else:
        lines.append(
            f"  Est. total cost:   ${(session.total_cost_usd or 0.0):.6f} USD  (usage × list price; verify on bill)"
        )
    return "\n".join(lines)


def chat_completions_create_with_metrics(
    client: Any,
    *,
    session: Optional[OpenAIMetricsSession] = None,
    **kwargs: Any,
) -> Any:
    """
    Calls client.chat.completions.create(**kwargs), measures time, reads response.usage, estimates cost,
    and optionally appends a CallMetric to `session`.
    Returns the ChatCompletion response (same as the API).
    """
    t0 = time.perf_counter()
    response = client.chat.completions.create(**kwargs)
    elapsed = time.perf_counter() - t0

    usage = getattr(response, "usage", None)
    if usage is None:
        raise ValueError("API response has no .usage; cannot compute tokens/cost (avoid stream-only without usage).")

    model = getattr(response, "model", None) or kwargs.get("model") or "unknown"
    in_tok = int(getattr(usage, "prompt_tokens", 0) or 0)
    out_tok = int(getattr(usage, "completion_tokens", 0) or 0)
    cost = estimate_call_cost_usd(str(model), input_tokens=in_tok, output_tokens=out_tok)
    metric = CallMetric(
        model=str(model),
        input_tokens=in_tok,
        output_tokens=out_tok,
        cost_usd=cost,
        elapsed_s=elapsed,
    )
    if session is not None:
        session.add(metric)
    return response
