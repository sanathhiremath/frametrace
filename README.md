# FrameTrace

Version 0.1.

A runnable pixel-processing and event-replay demo. In the controlled suite, dropped-frame tolerance can recover a missed event **and** introduce a false alarm. Its four clips are owned, generated synthetic fixtures; that controlled baseline is not a learned detector or a real-world vision benchmark.

The separate real-footage experiment now compares two frozen pretrained TorchVision detectors on a licensed Pexels pedestrian video. It preserves the controlled fixtures above rather than replacing them or mixing their metrics.

## Real-footage comparison

Install the optional dependencies with `pip install -e '.[real,dev]'` in your project environment. On Windows use double quotes around `.[real,dev]`. Use compatible PyTorch/TorchVision packages for your platform; this reference run used PyTorch2.9 and TorchVision0.24 on CPU.

```text
python run.py prepare-real
python run.py demo --real
```

Preparation explicitly downloads one16.7MB Pexels clip and two official pretrained weight artifacts. Downloads and model weights stay under ignored `data/` files. The local `data/annotations.json` contains independently authored source-review labels. See [source attribution](ATTRIBUTION.md) for the video license, modifications, software licenses, and model provenance.

The run writes `artifacts/real-demo.json` and40 resized JPEGs under `artifacts/real-frames/`. The portfolio can inspect recorded detections without starting model inference. Both Faster R-CNN MobileNetV3 Large320FPN and SSDLite320 MobileNetV3 Large use frozen COCO_V1 pretrained weights; neither is fine-tuned here.

Confidence thresholds are selected on four manually labeled development frames at0–3seconds and frozen before evaluation on four labeled frames at10–13seconds. Remaining frames are unscored replay observations. At IoU0.5, greedy one-to-one matching scores visible people of at least45pixels height in a declared ROI; partial/heavily occluded regions are ignored under the recorded protocol. Raw model predictions remain available so the scoring filters can be inspected.

This is a tiny, single-annotator, same-camera diagnostic set. It does not establish general pedestrian accuracy or an independent-camera test. Labels were authored from viewed source frames before model execution, not from model outputs. Local warmed CPU timing excludes video decoding, HTTP, queueing, and concurrent load. `parameter_bytes` is model tensor storage, not measured peak process memory.

To inspect the source frames used for annotation:

```text
python run.py inspect-real
```

The six real-evaluation unit tests validate one-to-one matching, ignored regions, nonmutation of inference outputs, development/test separation, annotation bounds, and invalid timestamps. They do not require model downloads. The actual reference inference run is recorded separately.

## Run on Windows or Ubuntu

Python 3.11+: `python -m venv .venv`.
Windows installation: `.venv\Scripts\python -m pip install -e ".[dev]"`.
Ubuntu installation: `.venv/bin/python -m pip install -e '.[dev]'`.
Activate the environment or use its Python for:

```text
python run.py demo
python -m pytest -q
```

The command writes `artifacts/demo.json` and 48 PNG frames. No server, credentials, external footage, or downloaded model weights are required.

## What actually runs

`fixtures.py` renders 320×180 pixels and separately defines event truth. `detector.py` processes image arrays using background difference and connected components; it never receives truth labels. Dropped frames are not passed to the detector. `replay.py` evaluates identical recorded detections using v1 (three consecutive source frames) and v2 (three observed zone hits in 500ms).

The suite includes an interrupted crossing, a sustained crossing, a brief non-event, and reflected flashes. An event means a sustained physical-object intrusion; bright reflections can fool the deliberately simple pixel detector. Results are computed from replay decisions rather than inserted into the report.

The report preserves pixel/output hashes, detector settings/version, event-policy versions, per-frame detections, dropped-frame markers, activation times, event precision/recall, and limitations. The window policy is intentionally not presented as universally better.

## Scope

This implements replay and inspection mechanics. It does not contain real camera footage, a pretrained AI detector, production streaming/backpressure, or evidence of deployment safety. A real-video extension needs a licensed dataset, semantic detection, independent labeled splits, and deployment-representative timing. The browser portfolio is a recorded artifact viewer, not live camera inference.

## Repository

Source: https://github.com/sanathhiremath/frametrace


This repository contains a local working prototype. Reproduction commands, measured scope, and limitations are documented above.
