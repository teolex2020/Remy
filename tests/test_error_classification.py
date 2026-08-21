from remy.core.error_classification import classify_llm_error


def test_invalid_openai_request_is_configuration_error_without_retry():
    result = classify_llm_error(
        "Error code: 400 - invalid_request_error: function tools are not supported"
    )

    assert result["error_class"] == "configuration"
    assert result["retryable"] is False
    assert "Retry will not help" in result["message"]


def test_unknown_error_does_not_trigger_false_limited_mode():
    result = classify_llm_error("unexpected provider payload")

    assert result["error_class"] == "unknown"
    assert result["retryable"] is False


def test_rate_limit_remains_retryable():
    result = classify_llm_error("HTTP 429 RESOURCE_EXHAUSTED")

    assert result["error_class"] == "rate_limit"
    assert result["retryable"] is True


def test_generic_provider_capacity_error_is_retryable():
    result = classify_llm_error("Provider returned error")

    assert result["error_class"] == "provider_capacity"
    assert result["retryable"] is True
