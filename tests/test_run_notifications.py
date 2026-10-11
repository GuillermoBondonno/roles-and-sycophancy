from __future__ import annotations

import pytest

import run


@pytest.fixture(autouse=True)
def notification_topic(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(run.config, "NTFY_TOPIC", "test-topic")


def test_completed_stage_sends_notification(monkeypatch: pytest.MonkeyPatch) -> None:
    messages: list[tuple[str, str]] = []
    monkeypatch.setattr(
        run,
        "send_ntfy_message",
        lambda message, topic: messages.append((message, topic)),
    )
    monkeypatch.setattr(run.time, "monotonic", lambda: 12.5)

    run.log_stage_done("Model load", 10.0)

    assert messages == [
        (
            "Stage completed: Model load (2.5s)",
            "test-topic",
        )
    ]


def test_main_sends_error_notification_and_reraises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    messages: list[tuple[str, str]] = []
    monkeypatch.setattr(
        run,
        "send_ntfy_message",
        lambda message, topic: messages.append((message, topic)),
    )

    def fail() -> None:
        raise ValueError("broken stage")

    monkeypatch.setattr(run, "_run", fail)

    with pytest.raises(ValueError, match="broken stage"):
        run.main()

    assert messages == [
        (
            "run.py failed with ValueError: broken stage",
            "test-topic",
        )
    ]


def test_notification_failure_does_not_mask_stage_completion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_notification(message: str, topic: str) -> None:
        raise RuntimeError("ntfy unavailable")

    monkeypatch.setattr(run, "send_ntfy_message", fail_notification)
    monkeypatch.setattr(run.time, "monotonic", lambda: 12.5)

    run.log_stage_done("Model load", 10.0)


def test_notifications_are_disabled_without_topic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(run.config, "NTFY_TOPIC", "")

    def unexpected_notification(*args, **kwargs) -> None:
        pytest.fail("A notification was sent without a configured topic")

    monkeypatch.setattr(run, "send_ntfy_message", unexpected_notification)
    run.notify("stage completed")
