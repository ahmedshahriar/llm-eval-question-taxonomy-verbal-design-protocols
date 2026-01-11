import types


def _mk_usage(
    *,
    input_tokens: int,
    cache_creation_input_tokens: int,
    cache_read_input_tokens: int,
    output_tokens: int,
    cache_creation_obj=None,
):
    u = types.SimpleNamespace()
    u.input_tokens = input_tokens
    u.cache_creation_input_tokens = cache_creation_input_tokens
    u.cache_read_input_tokens = cache_read_input_tokens
    u.output_tokens = output_tokens
    u.cache_creation = cache_creation_obj
    return u


def compute_like_adapter(usage, cache_ttl: str):
    # Mirrors current AnthropicAdapter logic exactly (token math only)
    uncached_after_breakpoint = int(getattr(usage, "input_tokens", 0) or 0)
    cache_creation_total = int(getattr(usage, "cache_creation_input_tokens", 0) or 0)
    cache_read_tokens = int(getattr(usage, "cache_read_input_tokens", 0) or 0)
    output_tokens = int(getattr(usage, "output_tokens", 0) or 0)

    cache_creation_obj = getattr(usage, "cache_creation", None)
    created_5m = 0
    created_1h = 0
    if isinstance(cache_creation_obj, dict):
        created_5m = int(cache_creation_obj.get("ephemeral_5m_input_tokens", 0) or 0)
        created_1h = int(cache_creation_obj.get("ephemeral_1h_input_tokens", 0) or 0)
    elif cache_creation_obj is not None:
        created_5m = int(getattr(cache_creation_obj, "ephemeral_5m_input_tokens", 0) or 0)
        created_1h = int(getattr(cache_creation_obj, "ephemeral_1h_input_tokens", 0) or 0)

    breakdown_sum = created_5m + created_1h
    if breakdown_sum > 0:
        # adapter reconciles total to breakdown_sum
        cache_creation_total = breakdown_sum
    else:
        if cache_creation_total > 0:
            if cache_ttl == "1h":
                created_1h = cache_creation_total
            else:
                created_5m = cache_creation_total

    input_tokens = uncached_after_breakpoint + cache_creation_total + cache_read_tokens
    total_tokens = input_tokens + output_tokens

    return {
        "uncached": uncached_after_breakpoint,
        "cache_creation_total": cache_creation_total,
        "cache_read": cache_read_tokens,
        "created_5m": created_5m,
        "created_1h": created_1h,
        "input_tokens": input_tokens,
        "total_tokens": total_tokens,
    }


def test_breakdown_wins_over_total():
    usage = _mk_usage(
        input_tokens=10,
        cache_creation_input_tokens=999,  # conflicting total
        cache_read_input_tokens=3,
        output_tokens=7,
        cache_creation_obj={"ephemeral_5m_input_tokens": 4, "ephemeral_1h_input_tokens": 5},
    )
    out = compute_like_adapter(usage, cache_ttl="5m")
    assert out["cache_creation_total"] == 9
    assert out["created_5m"] == 4
    assert out["created_1h"] == 5
    assert out["input_tokens"] == 10 + 9 + 3


def test_ttl_fallback_attributes_all_to_1h():
    usage = _mk_usage(
        input_tokens=10,
        cache_creation_input_tokens=6,
        cache_read_input_tokens=2,
        output_tokens=1,
        cache_creation_obj=None,  # no breakdown
    )
    out = compute_like_adapter(usage, cache_ttl="1h")
    assert out["created_1h"] == 6
    assert out["created_5m"] == 0
    assert out["input_tokens"] == 10 + 6 + 2


def test_ttl_fallback_attributes_all_to_5m():
    usage = _mk_usage(
        input_tokens=10,
        cache_creation_input_tokens=6,
        cache_read_input_tokens=2,
        output_tokens=1,
        cache_creation_obj=None,  # no breakdown
    )
    out = compute_like_adapter(usage, cache_ttl="5m")
    assert out["created_5m"] == 6
    assert out["created_1h"] == 0
    assert out["input_tokens"] == 10 + 6 + 2
