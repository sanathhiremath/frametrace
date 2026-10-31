import argparse
import hashlib
import json
from pathlib import Path
import cv2
from .fixtures import SCENARIOS, WIDTH, HEIGHT, ZONE, background, render
from .detector import detect
from .replay import replay, score

def demo(output: Path) -> dict:
    images = output / "frames"
    images.mkdir(parents=True, exist_ok=True)
    scenarios = []
    pixel_hash, observation_hash = hashlib.sha256(), hashlib.sha256()
    for scenario in SCENARIOS:
        frames = []
        for index in range(12):
            pixels = render(scenario, index)
            pixel_hash.update(pixels.tobytes())
            name = f"{scenario.id}-{index:02d}.png"
            if not cv2.imwrite(str(images / name), pixels):
                raise OSError(f"Could not write {name}")
            dropped = index in scenario.dropped
            frames.append({"index": index, "time_ms": index * 100, "image": f"frames/{name}", "dropped": dropped,
                           "detections": [] if dropped else detect(pixels, background())})
        observation_hash.update(json.dumps(frames, sort_keys=True).encode())
        policies = {name: replay(frames, ZONE, name) for name in ("consecutive", "window")}
        for index, frame in enumerate(frames):
            frame["policies"] = {name: result["decisions"][index] for name, result in policies.items()}
        for result in policies.values():
            del result["decisions"]
        scenarios.append({"id": scenario.id, "title": scenario.title, "truth": scenario.truth, "frames": frames, "policies": policies})
    summary = {name: score([s["truth"] for s in scenarios], [s["policies"][name]["triggered"] for s in scenarios])
               for name in ("consecutive", "window")}
    report = {"version": "0.1", "frame_size": {"width": WIDTH, "height": HEIGHT}, "zone": ZONE,
              "scenarios": scenarios, "summary": summary,
              "provenance": {"source": "Owned deterministic synthetic pixel fixtures; no external footage", "fixture_version": "0.1",
                             "detector": "uint8 background difference + connected components", "opencv": cv2.__version__,
                             "detector_settings": {"threshold": 45, "min_area": 40}, "pixels_sha256": pixel_hash.hexdigest(),
                             "observations_sha256": observation_hash.hexdigest(),
                             "policies": {"consecutive": "v1: 3 consecutive source frames", "window": "v2: 3 observed zone hits within 500ms"},
                             "ground_truth": "Independent authored event labels, excluded from detector input"},
              "limitations": ["Synthetic fixtures demonstrate replay mechanics, not real-world vision accuracy.",
                              "The pixel detector has no semantic object understanding; flashes can cause false alarms.",
                              "Window tolerance recovers a dropped-frame event but introduces a false positive in this suite.",
                              "Four authored scenarios are not an independent statistical benchmark or safety validation.",
                              "Dropped frames are marked and not processed; replay does not model production queueing."]}
    (output / "demo.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report

def main() -> int:
    parser = argparse.ArgumentParser(description="Replay synthetic pixel-based perception incidents")
    parser.add_argument("command", choices=["demo", "prepare-real", "real-demo", "inspect-real"])
    parser.add_argument("--real", action="store_true", help="Run the licensed real-footage model comparison")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[2] / "artifacts")
    args = parser.parse_args()
    if args.command != "demo" or args.real:
        from . import real
        data = Path(__file__).resolve().parents[2] / "data"
        if args.command == "prepare-real":
            real.prepare(data)
            return 0
        if args.command == "inspect-real":
            real.contact_sheet(data / "people-crossing.mp4", args.output / "annotation-review")
            return 0
        report = real.real_demo(data, args.output)
        print(json.dumps({"artifact": str(args.output / "real-demo.json"), "summary": report["summary"]}, indent=2))
        return 0
    report = demo(args.output)
    print(json.dumps({"artifact": str(args.output / "demo.json"), "summary": report["summary"]}, indent=2))
    return 0
