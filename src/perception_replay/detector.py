"""Pixel foreground baseline, not a semantic or learned object detector."""
import cv2
import numpy as np

def detect(pixels: np.ndarray, reference: np.ndarray, *, threshold: int = 45, min_area: int = 40) -> list[dict]:
    if pixels.shape != reference.shape or pixels.ndim != 3 or pixels.shape[2] != 3:
        raise ValueError("frames must have matching HxWx3 shapes")
    if pixels.dtype != np.uint8 or reference.dtype != np.uint8:
        raise ValueError("frames must be uint8")
    if not 0 <= threshold <= 255 or min_area <= 0:
        raise ValueError("invalid detector settings")
    mask = (np.max(cv2.absdiff(pixels, reference), axis=2) > threshold).astype(np.uint8)
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    return [dict(x=int(x), y=int(y), width=int(w), height=int(h), area=int(area))
            for x, y, w, h, area in stats[1:count] if area >= min_area]

def in_zone(detection: dict, zone: dict) -> bool:
    cx, cy = detection["x"] + detection["width"] / 2, detection["y"] + detection["height"] / 2
    return zone["x"] <= cx < zone["x"] + zone["width"] and zone["y"] <= cy < zone["y"] + zone["height"]
