from langchain_core.messages import AIMessage

from remy.core.model_trace import extract_token_usage, model_call_event


def test_extract_token_usage_from_standard_message_attribute():
    message = AIMessage(
        content="ok",
        usage_metadata={"input_tokens": 120, "output_tokens": 30, "total_tokens": 150},
    )

    assert extract_token_usage(message) == {
        "input_tokens": 120,
        "output_tokens": 30,
        "total_tokens": 150,
    }


def test_model_call_event_normalizes_openai_usage_fields():
    message = AIMessage(
        content="ok",
        response_metadata={
            "model_name": "example-model",
            "token_usage": {"prompt_tokens": 40, "completion_tokens": 12},
        },
    )

    event = model_call_event(message, purpose="agent", channel="desktop")

    assert event["token_usage"] == {
        "input_tokens": 40,
        "output_tokens": 12,
        "total_tokens": 52,
    }


def test_missing_usage_is_not_presented_as_reported():
    message = AIMessage(content="no provider usage")

    assert extract_token_usage(message) is None
    assert "token_usage" not in model_call_event(message, purpose="agent")
