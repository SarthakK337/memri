"""LLM cost calculation per model."""

# Cost per token in USD (input, output)
MODEL_COSTS: dict[str, dict[str, float]] = {
    # ── Anthropic ────────────────────────────────────────────────────
    "claude-haiku-4-5-20251001": {"input": 0.80 / 1_000_000, "output": 4.00 / 1_000_000},
    "claude-haiku-4-5":          {"input": 0.80 / 1_000_000, "output": 4.00 / 1_000_000},
    "claude-sonnet-4-6":         {"input": 3.00 / 1_000_000, "output": 15.00 / 1_000_000},
    "claude-opus-4-7":           {"input": 15.00 / 1_000_000, "output": 75.00 / 1_000_000},
    # ── Google Gemini ────────────────────────────────────────────────
    "gemini-2.5-flash":          {"input": 0.075 / 1_000_000, "output": 0.30 / 1_000_000},
    "gemini-2.5-flash-preview":  {"input": 0.075 / 1_000_000, "output": 0.30 / 1_000_000},
    "gemini-2.5-pro":            {"input": 1.25 / 1_000_000, "output": 10.00 / 1_000_000},
    "gemini-2.5-pro-preview":    {"input": 1.25 / 1_000_000, "output": 10.00 / 1_000_000},
    "gemini-2.0-flash":          {"input": 0.10 / 1_000_000, "output": 0.40 / 1_000_000},
    "gemini-2.0-flash-lite":     {"input": 0.075 / 1_000_000, "output": 0.30 / 1_000_000},
    "gemini-1.5-flash":          {"input": 0.075 / 1_000_000, "output": 0.30 / 1_000_000},
    "gemini-1.5-flash-8b":       {"input": 0.0375 / 1_000_000, "output": 0.15 / 1_000_000},
    "gemini-1.5-pro":            {"input": 1.25 / 1_000_000, "output": 5.00 / 1_000_000},
    # ── OpenAI ───────────────────────────────────────────────────────
    "gpt-4o":                    {"input": 2.50 / 1_000_000, "output": 10.00 / 1_000_000},
    "gpt-4o-mini":               {"input": 0.15 / 1_000_000, "output": 0.60 / 1_000_000},
    "gpt-4-turbo":               {"input": 10.00 / 1_000_000, "output": 30.00 / 1_000_000},
    "o1":                        {"input": 15.00 / 1_000_000, "output": 60.00 / 1_000_000},
    "o1-mini":                   {"input": 1.10 / 1_000_000, "output": 4.40 / 1_000_000},
    "o3-mini":                   {"input": 1.10 / 1_000_000, "output": 4.40 / 1_000_000},
    # ── Groq ─────────────────────────────────────────────────────────
    "llama-3.1-8b-instant":      {"input": 0.05 / 1_000_000, "output": 0.08 / 1_000_000},
    "llama-3.3-70b-versatile":   {"input": 0.59 / 1_000_000, "output": 0.79 / 1_000_000},
    "llama3-70b-8192":           {"input": 0.59 / 1_000_000, "output": 0.79 / 1_000_000},
    "mixtral-8x7b-32768":        {"input": 0.24 / 1_000_000, "output": 0.24 / 1_000_000},
    # ── Mistral ──────────────────────────────────────────────────────
    "mistral-small-latest":      {"input": 0.20 / 1_000_000, "output": 0.60 / 1_000_000},
    "mistral-medium-latest":     {"input": 2.70 / 1_000_000, "output": 8.10 / 1_000_000},
    "mistral-large-latest":      {"input": 2.00 / 1_000_000, "output": 6.00 / 1_000_000},
    # ── Local / self-hosted (free) ───────────────────────────────────
    "ollama":                    {"input": 0.0, "output": 0.0},
}

# Fallback cost for unknown models (conservative estimate)
_DEFAULT_COST = {"input": 1.00 / 1_000_000, "output": 3.00 / 1_000_000}


def _prefix_match(model: str) -> dict[str, float]:
    """Match cost by model name prefix for versioned model IDs like gemini-2.5-flash-001."""
    for key, cost in MODEL_COSTS.items():
        if model.startswith(key):
            return cost
    return _DEFAULT_COST

# Estimated cost per 1K tokens in a full context window (for computing savings)
# Assumes typical model used by the coding agent itself
AGENT_MODEL_COST_PER_1K_INPUT = 3.00 / 1_000  # ~claude-sonnet pricing


def calculate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Return USD cost for a single LLM call."""
    costs = MODEL_COSTS.get(model) or _prefix_match(model)
    return costs["input"] * input_tokens + costs["output"] * output_tokens


def estimate_savings(raw_tokens: int, compressed_tokens: int) -> float:
    """Estimate USD saved by not sending raw_tokens - compressed_tokens to the agent."""
    saved_tokens = max(0, raw_tokens - compressed_tokens)
    return saved_tokens * AGENT_MODEL_COST_PER_1K_INPUT / 1_000


def format_cost(usd: float) -> str:
    if usd < 0.001:
        return f"${usd * 1000:.4f}m"
    return f"${usd:.4f}"
