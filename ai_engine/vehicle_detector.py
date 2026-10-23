"""
AI & Computer Vision Engine - Vehicle Detector
------------------------------------------------
Uses OpenCV background subtraction + contour analysis as a lightweight,
dependency-free stand-in for a YOLOv8 detector. Each detected blob is
classified into car / bike / heavy_vehicle based on its bounding-box
area and aspect ratio, mimicking a YOLO class-head decision without
requiring GPU inference or pretrained weights - ideal for an offline
hackathon demo.

To upgrade to a real YOLOv8 model later, only `_detect_raw_boxes` needs
to change (swap in `ultralytics.YOLO(...).predict(frame)`); everything
downstream (classification, counting, queue length) stays the same.
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict


VEHICLE_CLASSES = {
    "bike": {"min_area": 400, "max_area": 2500, "aspect_ratio_range": (0.4, 1.4)},
    "car": {"min_area": 2500, "max_area": 9000, "aspect_ratio_range": (1.0, 2.2)},
    "heavy_vehicle": {"min_area": 9000, "max_area": 100000, "aspect_ratio_range": (1.2, 3.5)},
}


@dataclass
class Detection:
    vehicle_type: str
    bbox: tuple
    confidence: float
    area: float


@dataclass
class FrameAnalysis:
    detections: List[Detection] = field(default_factory=list)
    counts: Dict[str, int] = field(default_factory=dict)
    queue_length: int = 0
    total_vehicles: int = 0


class VehicleDetector:
    """
    Lightweight vehicle detector using MOG2 background subtraction.
    """

    def __init__(self, history: int = 300, var_threshold: int = 40, detect_shadows: bool = True):
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=history, varThreshold=var_threshold, detectShadows=detect_shadows
        )
        self.frame_count = 0

    def _detect_raw_boxes(self, frame: np.ndarray) -> List[tuple]:
        """Runs background subtraction + contour extraction to get raw bounding boxes."""
        fg_mask = self.bg_subtractor.apply(frame)
        _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        fg_mask = cv2.dilate(fg_mask, np.ones((5, 5), np.uint8), iterations=2)

        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 300:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            boxes.append((x, y, w, h))
        return boxes

    def _classify_box(self, w: int, h: int) -> Detection:
        area = w * h
        aspect_ratio = w / float(h) if h > 0 else 0

        best_class = "car"
        best_score = -1
        for cls_name, rule in VEHICLE_CLASSES.items():
            if rule["min_area"] <= area <= rule["max_area"]:
                ar_min, ar_max = rule["aspect_ratio_range"]
                if ar_min <= aspect_ratio <= ar_max:
                    score = 1.0 - abs(area - (rule["min_area"] + rule["max_area"]) / 2) / rule["max_area"]
                    if score > best_score:
                        best_score = score
                        best_class = cls_name

        confidence = round(min(0.99, 0.55 + best_score * 0.4), 2) if best_score > 0 else 0.5
        return Detection(vehicle_type=best_class, bbox=(w, h), confidence=confidence, area=area)

    def analyze_frame(self, frame: np.ndarray, queue_zone_ratio: float = 0.6) -> FrameAnalysis:
        """
        Runs the full detection + classification pipeline on a single frame.
        `queue_zone_ratio` is the fraction of frame height (measured from the
        bottom) treated as the "stop line queue zone" for queue-length estimation.
        """
        self.frame_count += 1
        boxes = self._detect_raw_boxes(frame)

        detections: List[Detection] = []
        counts = {"bike": 0, "car": 0, "heavy_vehicle": 0}
        queue_length = 0
        frame_h = frame.shape[0]
        queue_zone_y = frame_h * (1 - queue_zone_ratio)

        for (x, y, w, h) in boxes:
            det = self._classify_box(w, h)
            det.bbox = (x, y, w, h)
            detections.append(det)
            counts[det.vehicle_type] += 1
            if y >= queue_zone_y:
                queue_length += 1

        return FrameAnalysis(
            detections=detections,
            counts=counts,
            queue_length=queue_length,
            total_vehicles=sum(counts.values()),
        )

    @staticmethod
    def generate_synthetic_frame(width: int = 640, height: int = 480, num_vehicles: int = 10,
                                  seed: int = None) -> np.ndarray:
        """
        Generates a synthetic 'traffic camera' frame with randomly placed
        rectangles simulating vehicles, for demo purposes when a live
        camera / video file is not available.
        """
        rng = np.random.default_rng(seed)
        frame = np.full((height, width, 3), 40, dtype=np.uint8)  # dark road background
        cv2.rectangle(frame, (0, int(height * 0.4)), (width, height), (60, 60, 60), -1)  # road surface

        vehicle_size_ranges = {
            "bike": ((20, 35), (35, 50)),
            "car": ((40, 60), (60, 90)),
            "heavy_vehicle": ((70, 100), (100, 150)),
        }

        for _ in range(num_vehicles):
            v_type = rng.choice(list(vehicle_size_ranges.keys()), p=[0.35, 0.5, 0.15])
            (w_min, w_max), (h_min, h_max) = vehicle_size_ranges[v_type]
            w = int(rng.integers(w_min, w_max))
            h = int(rng.integers(h_min, h_max))
            x = int(rng.integers(0, max(1, width - w)))
            y = int(rng.integers(int(height * 0.42), max(int(height * 0.42) + 1, height - h)))
            color = tuple(int(c) for c in rng.integers(80, 220, size=3))
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, -1)

        return frame
