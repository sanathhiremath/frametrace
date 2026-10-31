import json
import numpy as np
import pytest
from perception_replay.fixtures import SCENARIOS, ZONE, background, render
from perception_replay.detector import detect, in_zone
from perception_replay.replay import replay, score
from perception_replay.cli import demo

def test_detector_reads_pixels_without_annotation_argument():
    blank = background()
    assert detect(blank, blank) == []
    pixels = render(SCENARIOS[0], 2)
    detections = detect(pixels, blank)
    assert len(detections) == 1 and detections[0]["area"] == 27 * 48
    assert in_zone(detections[0], ZONE)
    pixels[65:113, 194:221] = blank[65:113, 194:221]
    assert detect(pixels, blank) == []

def test_detector_rejects_invalid_images():
    with pytest.raises(ValueError):
        detect(np.zeros((3, 3)), background())
    with pytest.raises(ValueError):
        detect(background().astype(float), background().astype(float))

def frames(indices, dropped=()):
    return [{"index": i, "time_ms": i * 100, "dropped": i in dropped,
             "detections": [{"x": 190, "y": 60, "width": 20, "height": 30}]} for i in indices]

def test_gap_recovery_is_deterministic_without_changing_observations():
    observations = frames(range(2, 7), (3, 5))
    before = json.dumps(observations, sort_keys=True)
    assert not replay(observations, ZONE, "consecutive")["triggered"]
    result = replay(observations, ZONE, "window")
    assert result["triggered"] and result["trigger_time_ms"] == 600
    assert replay(observations, ZONE, "window") == result
    assert json.dumps(observations, sort_keys=True) == before

def test_window_expires_and_ignores_dropped_detections():
    assert not replay(frames((0, 4, 8)), ZONE, "window")["triggered"]
    assert not replay(frames(range(3), (1,)), ZONE, "window")["triggered"]

@pytest.mark.parametrize("observations", [frames((1, 0)), frames((1, 1)), frames((-1,))])
def test_replay_rejects_invalid_order(observations):
    with pytest.raises(ValueError):
        replay(observations, ZONE, "window")

def test_metric_denominators_and_false_positives():
    assert score([True, True, False], [True, False, True]) == {"true_positives": 1, "false_positives": 1, "false_negatives": 1, "precision": .5, "recall": .5}
    assert score([False], [False])["precision"] is None
    with pytest.raises(ValueError):
        score([True], [])

def test_end_to_end_records_pixel_provenance_and_actual_tradeoff(tmp_path):
    report = demo(tmp_path)
    assert len(list((tmp_path / "frames").glob("*.png"))) == 48
    assert report["summary"]["consecutive"]["false_negatives"] == 1
    assert report["summary"]["window"]["false_negatives"] == 0
    assert report["summary"]["window"]["false_positives"] == 1
    assert demo(tmp_path) == report
