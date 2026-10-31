"""Two versioned policies consume exactly the same recorded observations."""
from .detector import in_zone

def replay(frames: list[dict], zone: dict, policy: str) -> dict:
    if policy not in {"consecutive", "window"}:
        raise ValueError("unknown policy")
    previous_time, previous_index, streak = -1, -1, 0
    hit_times, decisions = [], []
    trigger_time = None
    previous_seen_index = -1
    for frame in frames:
        time, index = frame["time_ms"], frame["index"]
        if time < 0 or time <= previous_time or index <= previous_seen_index:
            raise ValueError("frames require nonnegative strictly increasing times and indices")
        previous_time, previous_seen_index = time, index
        if frame.get("dropped", False):
            decisions.append(False)
            continue
        occupied = any(in_zone(item, zone) for item in frame["detections"])
        if policy == "consecutive":
            streak = (streak + 1 if index == previous_index + 1 else 1) if occupied else 0
            active = streak >= 3
        else:
            hit_times = [hit for hit in hit_times if time - hit <= 500]
            if occupied:
                hit_times.append(time)
            active = occupied and len(hit_times) >= 3
        previous_index = index
        if active and trigger_time is None:
            trigger_time = time
        decisions.append(active)
    return {"triggered": trigger_time is not None, "trigger_time_ms": trigger_time, "decisions": decisions}

def score(truth: list[bool], predictions: list[bool]) -> dict:
    if len(truth) != len(predictions) or not truth:
        raise ValueError("nonempty aligned truth and predictions required")
    tp = sum(actual and predicted for actual, predicted in zip(truth, predictions))
    fp = sum(not actual and predicted for actual, predicted in zip(truth, predictions))
    fn = sum(actual and not predicted for actual, predicted in zip(truth, predictions))
    return {"true_positives": tp, "false_positives": fp, "false_negatives": fn,
            "precision": tp / (tp + fp) if tp + fp else None, "recall": tp / (tp + fn) if tp + fn else None}
