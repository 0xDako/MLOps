from quality_gate import THRESHOLDS, check_metrics


def test_check_metrics_pass():
    metrics = {name: {"accuracy": threshold} for name, threshold in THRESHOLDS.items()}
    assert check_metrics(metrics) == []


def test_check_metrics_fail():
    metrics = {name: {"accuracy": 0.0} for name in THRESHOLDS}
    assert check_metrics(metrics) == list(THRESHOLDS)


def test_check_metrics_partial_fail():
    metrics = {name: {"accuracy": 1.0} for name in THRESHOLDS}
    metrics["cnn_small"]["accuracy"] = 0.0
    assert check_metrics(metrics) == ["cnn_small"]
