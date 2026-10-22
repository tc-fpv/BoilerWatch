from boiler_reader.alert_delay import AlertDelayTracker


def test_issue_timer_starts_and_becomes_eligible_after_limit() -> None:
    tracker = AlertDelayTracker(delay_seconds=300)

    initial = tracker.update("low_pressure", now=100.0)
    before_limit = tracker.update("low_pressure", now=399.0)
    at_limit = tracker.update("low_pressure", now=400.0)

    assert initial.elapsed_seconds == 0
    assert initial.remaining_seconds == 300
    assert not initial.eligible
    assert before_limit.elapsed_seconds == 299
    assert before_limit.remaining_seconds == 1
    assert not before_limit.eligible
    assert at_limit.elapsed_seconds == 300
    assert at_limit.remaining_seconds == 0
    assert at_limit.eligible


def test_switch_between_low_and_error_keeps_continuous_timer() -> None:
    tracker = AlertDelayTracker(delay_seconds=300)

    tracker.update("low_pressure", now=100.0)
    error = tracker.update("error", now=250.0)

    assert error.issue == "error"
    assert error.elapsed_seconds == 150
    assert error.remaining_seconds == 150


def test_recovery_resets_timer() -> None:
    tracker = AlertDelayTracker(delay_seconds=300)

    tracker.update("low_pressure", now=100.0)
    recovered = tracker.update(None, now=250.0)
    restarted = tracker.update("error", now=260.0)

    assert recovered.issue is None
    assert recovered.elapsed_seconds == 0
    assert not recovered.eligible
    assert restarted.issue == "error"
    assert restarted.elapsed_seconds == 0
    assert restarted.remaining_seconds == 300


def test_due_notification_stays_due_until_acknowledged_and_resets_on_recovery() -> None:
    tracker = AlertDelayTracker(delay_seconds=10)

    tracker.update("error", now=10.0)
    due = tracker.update("error", now=20.0)
    assert due.eligible
    assert due.notification_due

    tracker.mark_notified()
    repeated = tracker.update("error", now=21.0)
    assert repeated.eligible
    assert not repeated.notification_due

    tracker.update(None, now=22.0)
    restarted = tracker.update("error", now=23.0)
    assert not restarted.eligible
    assert not restarted.notification_due