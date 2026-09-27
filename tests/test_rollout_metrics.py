"""The wedge branch of `train_policy.rollout_metrics`."""

from train_policy import rollout_metrics


def test_wedge_end_phase_fractions() -> None:
    metrics = rollout_metrics(
        "wedge",
        max_coverage=[1.0, 0.0, 0.0, 0.0],
        final_coverage=[0.8, 0.0, 0.0, 0.0],
        end_phases=["SUCCESS", "FAILED", "TIMEOUT", "TIMEOUT"],
    )
    assert metrics is not None
    assert metrics["end success"] == 0.25
    assert metrics["end failed"] == 0.25
    assert metrics["end timeout"] == 0.5
    assert metrics["success rate"] == 0.25 and metrics["p_score mean"] == 0.2


def test_wedge_logs_every_end_phase_even_when_absent() -> None:
    metrics = rollout_metrics("wedge", [0.0], [0.0], ["TIMEOUT"])
    assert metrics is not None
    assert metrics["end success"] == 0.0 and metrics["end failed"] == 0.0


def test_other_envs_unchanged() -> None:
    metrics = rollout_metrics("cube", [1.0, 0.0], [0.0, 1.0])
    assert metrics is not None and "entered mean" in metrics and "end success" not in metrics
