"""Real-video replay with independently authored boxes and pretrained detectors."""
from pathlib import Path
import hashlib
import json
import time
import urllib.request
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
VIDEO_URL = "https://videos.pexels.com/video-files/5365843/5365843-hd_1920_1080_30fps.mp4"
WEIGHTS_URL = "https://download.pytorch.org/models/ssdlite320_mobilenet_v3_large_coco-a79551df.pth"
FASTER_URL = "https://download.pytorch.org/models/fasterrcnn_mobilenet_v3_large_320_fpn-907ea3f9.pth"
SOURCE_URL = "https://www.pexels.com/video/people-walking-on-a-street-5365843/"


def prepare(data: Path):
    data.mkdir(parents=True, exist_ok=True)
    for filename, url in (("people-crossing.mp4", VIDEO_URL), ("ssdlite320.pth", WEIGHTS_URL), ("fasterrcnn320.pth", FASTER_URL)):
        target = data / filename
        if not target.exists():
            temporary = target.with_suffix(target.suffix + ".partial")
            with urllib.request.urlopen(url, timeout=60) as response, temporary.open("wb") as output:
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
            temporary.replace(target)


def frame_at(video: Path, time_ms: int):
    if time_ms < 0:
        raise ValueError('Frame timestamp must be nonnegative')
    capture = cv2.VideoCapture(str(video))
    capture.set(cv2.CAP_PROP_POS_MSEC, time_ms)
    ok, pixels = capture.read()
    capture.release()
    if not ok:
        raise ValueError(f"Cannot decode video frame at {time_ms}ms")
    return cv2.resize(pixels, (640, 360), interpolation=cv2.INTER_AREA)


def contact_sheet(video: Path, output: Path):
    times = [0, 1000, 2000, 3000, 10000, 11000, 12000, 13000]
    frames = []
    output.mkdir(parents=True, exist_ok=True)
    for milliseconds in times:
        pixels = frame_at(video, milliseconds)
        cv2.imwrite(str(output / f"annotation-{milliseconds:05d}.jpg"), pixels)
        labeled = pixels.copy()
        cv2.putText(labeled, f"{milliseconds} ms", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, .8, (0, 0, 255), 2)
        frames.append(labeled)
    sheet = np.vstack([np.hstack(frames[i:i+2]) for i in range(0, len(frames), 2)])
    cv2.imwrite(str(output / "contact-sheet.jpg"), sheet)


def iou(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0, x1 - x0) * max(0, y1 - y0)
    union = (a[2]-a[0]) * (a[3]-a[1]) + (b[2]-b[0]) * (b[3]-b[1]) - intersection
    return intersection / union if union > 0 else 0


def validate_annotations(config):
    times = set()
    for annotation in config['frames']:
        stamp = annotation['time_ms']
        if stamp < 0 or stamp in times or annotation['split'] not in ('development','test'):
            raise ValueError('Annotations need unique nonnegative timestamps and explicit splits')
        times.add(stamp)
        for box in annotation['boxes'] + annotation['ignored_regions']:
            if len(box)!=4 or not (0 <= box[0] < box[2] <= config['frame_size']['width'] and 0 <= box[1] < box[3] <= config['frame_size']['height']):
                raise ValueError('Invalid annotation box')


def detection_metrics(truth, predictions, threshold=.5):
    matched = set()
    tp = 0
    for prediction in sorted(predictions, key=lambda p:p["score"], reverse=True):
        candidates = [(iou(prediction["box"], box), index) for index, box in enumerate(truth) if index not in matched]
        overlap, index = max(candidates, default=(0, -1))
        if overlap >= threshold:
            matched.add(index)
            tp += 1
    fp, fn = len(predictions) - tp, len(truth) - tp
    return {"tp": tp, "fp": fp, "fn": fn}


class PeopleDetectors:
    def __init__(self, weights: Path):
        import torch
        from torchvision.models.detection import ssdlite320_mobilenet_v3_large, fasterrcnn_mobilenet_v3_large_320_fpn
        self.torch = torch
        torch.set_num_threads(2)
        self.model = ssdlite320_mobilenet_v3_large(weights=None, weights_backbone=None).eval()
        self.model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
        self.faster = fasterrcnn_mobilenet_v3_large_320_fpn(weights=None, weights_backbone=None).eval()
        self.faster.load_state_dict(torch.load(weights.parent / 'fasterrcnn320.pth', map_location='cpu', weights_only=True))

    def predict(self, pixels, name):
        start = time.perf_counter()
        if name in ("ssdlite", "fasterrcnn"):
            tensor = self.torch.from_numpy(cv2.cvtColor(pixels, cv2.COLOR_BGR2RGB)).permute(2,0,1).float()/255
            with self.torch.inference_mode():
                prediction = (self.model if name == "ssdlite" else self.faster)([tensor])[0]
            result = [{"box":[round(float(v),2) for v in box],"score":float(score)} for box,score,label in zip(prediction["boxes"],prediction["scores"],prediction["labels"]) if int(label)==1]
        else:
            raise ValueError("Unknown detector")
        return result, (time.perf_counter()-start)*1000


def evaluation_predictions(predictions, annotation, config, threshold):
    """Annotations only filter scoring regions; they never enter model inference."""
    roi = config["evaluation_roi"]
    result = []
    for prediction in predictions:
        box = prediction["box"]
        cx, cy = (box[0]+box[2])/2, (box[1]+box[3])/2
        if prediction["score"] < threshold or box[3]-box[1] < config["min_height"]:
            continue
        if not (roi["x"] <= cx < roi["x"]+roi["width"] and roi["y"] <= cy < roi["y"]+roi["height"]):
            continue
        # Ground-truth match takes precedence when an ignored region overlaps it.
        matches_truth = any(iou(box, target) >= .5 for target in annotation["boxes"])
        if not matches_truth and any(iou(box, ignored) >= .3 for ignored in annotation["ignored_regions"]):
            continue
        result.append(prediction)
    return result


def summarize(rows, annotations, config, model, threshold, split):
    total = {"tp":0,"fp":0,"fn":0}
    for row in rows:
        annotation = annotations.get(row["time_ms"])
        if not annotation or annotation["split"] != split:
            continue
        values = detection_metrics(annotation["boxes"], evaluation_predictions(row["models"][model]["detections"], annotation, config, threshold))
        for key, value in values.items():
            total[key] += value
    total["precision"] = total["tp"]/(total["tp"]+total["fp"]) if total["tp"]+total["fp"] else 0
    total["recall"] = total["tp"]/(total["tp"]+total["fn"]) if total["tp"]+total["fn"] else 0
    total["f1"] = 2*total["precision"]*total["recall"]/(total["precision"]+total["recall"]) if total["precision"]+total["recall"] else 0
    return total


def real_demo(data: Path, output: Path):
    config = json.loads((data/"annotations.json").read_text(encoding="utf-8"))
    validate_annotations(config)
    annotations = {item["time_ms"]:item for item in config["frames"]}
    if {item["time_ms"] for item in config["frames"] if item["split"]=="development"} & {item["time_ms"] for item in config["frames"] if item["split"]=="test"}:
        raise ValueError("Development/test overlap")
    output.mkdir(parents=True, exist_ok=True)
    images = output/"real-frames"
    images.mkdir(exist_ok=True)
    video, weights = data/"people-crossing.mp4", data/"ssdlite320.pth"
    detectors = PeopleDetectors(weights)
    warmup = frame_at(video,0)
    for name in ("fasterrcnn","ssdlite"):
        detectors.predict(warmup,name)
    rows = []
    for index, milliseconds in enumerate(range(0,20000,500)):
        pixels = frame_at(video,milliseconds)
        filename = f"frame-{milliseconds:05d}.jpg"
        if not cv2.imwrite(str(images/filename), pixels, [cv2.IMWRITE_JPEG_QUALITY,86]):
            raise OSError("Unable to write replay frame")
        annotation = annotations.get(milliseconds)
        row = {"index":index,"time_ms":milliseconds,"split":annotation["split"] if annotation else "unscored",
               "image":f"real-frames/{filename}","ground_truth":annotation["boxes"] if annotation else None,
               "ignored_regions":annotation["ignored_regions"] if annotation else [],"models":{}}
        for name in ("fasterrcnn","ssdlite"):
            predictions, latency = detectors.predict(pixels,name)
            row["models"][name] = {"detections":predictions,"latency_ms":latency}
        rows.append(row)
    selection, summary = {}, {}
    for name, candidates in (("fasterrcnn",(.3,.5,.7)),("ssdlite",(.3,.5,.7))):
        trials = [{"threshold":threshold,**summarize(rows,annotations,config,name,threshold,"development")} for threshold in candidates]
        winner = max(trials,key=lambda item:(item["f1"],item["threshold"]))
        selection[name] = {"selected_threshold":winner["threshold"],"development_trials":trials}
        result = summarize(rows,annotations,config,name,winner["threshold"],"test")
        times = [row["models"][name]["latency_ms"] for row in rows]
        model = detectors.model if name=='ssdlite' else detectors.faster
        summary[name] = {**result,"threshold":winner["threshold"],"mean_latency_ms":float(np.mean(times)),"p95_latency_ms":float(np.percentile(times,95)),"parameter_bytes":sum(parameter.numel()*parameter.element_size() for parameter in model.parameters())}
    report = {"version":"0.1","frame_size":config["frame_size"],"evaluation_roi":config["evaluation_roi"],"frames":rows,
              "summary":summary,"selection":selection,
              "provenance":{"source":"People Walking on a Street (Pexels5365843)","author":"CityXcape","source_url":SOURCE_URL,"license":"Pexels License","license_url":"https://www.pexels.com/license/",
                            "video_sha256":hashlib.sha256(video.read_bytes()).hexdigest(),"weights_sha256":hashlib.sha256(weights.read_bytes()).hexdigest(),"annotations_sha256":hashlib.sha256((data/"annotations.json").read_bytes()).hexdigest(),
                            "annotation_method":config["method"],"development_times_ms":[0,1000,2000,3000],"test_times_ms":[10000,11000,12000,13000],"test_instances":sum(len(item["boxes"]) for item in config["frames"] if item["split"]=="test"),
                            "models":{"fasterrcnn":"TorchVision FasterRCNN MobileNetV3 Large320FPN COCO_V1; frozen pretrained weights","ssdlite":"TorchVision SSDLite320 MobileNetV3 Large COCO_V1; frozen pretrained weights"},"weights_url":WEIGHTS_URL,"faster_weights_url":FASTER_URL,"faster_weights_sha256":hashlib.sha256((data/'fasterrcnn320.pth').read_bytes()).hexdigest(),"opencv":cv2.__version__,"torch":detectors.torch.__version__,"device":"CPU","cpu_threads":2,
                            "evaluation":"Greedy score-ordered one-to-one IoU>=0.5 matching inside fixedROI; minimum boxheight45pixels. Ignore partial/heavily occluded regions atIoU>=0.3 unless matched to annotatedperson. Threshold selected on developmentframes only; unscoredframes are replayonly."},
              "limitations":["One23-second Pexels clip and four sparse test frames are not a general pedestrian benchmark.","Single-annotator boxes were authored from viewed source frames before model inference; no independent second annotation review.","Development and test windows are separated in time but share a scene/camera and remain correlated.","Only the stated region and visible people at least45pixels tall are scored; ignored regions and label uncertainty affect precision/recall.","This compares two frozen pretrained detectors with development-selected confidence thresholds; no model fine-tuning was performed.","Local warmed CPU latency is measured once per sampled frame and excludes decoding, HTTP, queues, and concurrent load.","Footage is resized/JPEG-encoded for this annotated replay; no endorsement by depicted people or brands is implied."]}
    (output/"real-demo.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    return report
