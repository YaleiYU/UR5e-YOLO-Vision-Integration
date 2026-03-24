"""
yolo_detector.py
----------------
Wrapper around YOLOv8 (Ultralytics) that provides a simple API for detecting
objects in RGB frames and annotating images with bounding-box overlays.

Dependencies
~~~~~~~~~~~~
- ultralytics >= 8.0  (pip install ultralytics)
- opencv-python >= 4.8
- numpy >= 1.24

Usage
~~~~~
    from src.yolo_detector import YOLODetector
    import cv2

    detector = YOLODetector(model_path="yolov8n.pt", confidence=0.5)

    frame = cv2.imread("scene.jpg")
    detections = detector.detect(frame)
    for d in detections:
        print(d["class"], d["confidence"], d["centroid"])

    annotated = detector.annotate(frame, detections)
    cv2.imshow("result", annotated)
    cv2.waitKey(0)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import cv2

try:
    from ultralytics import YOLO as _YOLO
    _ULTRALYTICS_AVAILABLE = True
except ImportError:  # pragma: no cover
    _ULTRALYTICS_AVAILABLE = False


# Type alias for a single detection result dictionary.
Detection = Dict[str, Any]


class YOLODetector:
    """Run YOLOv8 inference on BGR images.

    Parameters
    ----------
    model_path:
        Path to a YOLOv8 ``.pt`` weights file (e.g. ``"yolov8n.pt"``).
        When a pre-trained COCO model name is given (e.g. ``"yolov8n.pt"``),
        Ultralytics will automatically download it on first use.
    confidence:
        Minimum confidence score (0–1) for a detection to be returned.
    device:
        Inference device, e.g. ``"cpu"``, ``"cuda:0"``, or ``""`` (auto).
    """

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        confidence: float = 0.5,
        device: str = "",
    ) -> None:
        if not _ULTRALYTICS_AVAILABLE:
            raise ImportError(
                "ultralytics is not installed.  Run: pip install ultralytics"
            )
        self.confidence = confidence
        self._device = device
        print(f"[YOLODetector] Loading model: {model_path}")
        self._model = _YOLO(model_path)
        print("[YOLODetector] Model loaded.")

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def detect(
        self,
        image: np.ndarray,
        target_classes: Optional[List[str]] = None,
    ) -> List[Detection]:
        """Run inference on *image* and return a list of detection dicts.

        Parameters
        ----------
        image:
            BGR image as a ``uint8`` numpy array (H, W, 3).
        target_classes:
            Optional allow-list of class names.  Only detections whose
            class label is in this list are returned.  Pass ``None`` to
            return all detected classes.

        Returns
        -------
        List of dicts, each containing:

        ``"class"``     – class label string  
        ``"confidence"``– detection confidence (float 0–1)  
        ``"bbox"``      – bounding box [x1, y1, x2, y2] in pixels  
        ``"centroid"``  – (cx, cy) centroid in pixels  
        """
        results = self._model.predict(
            source=image,
            conf=self.confidence,
            device=self._device,
            verbose=False,
        )

        detections: List[Detection] = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                cls_id = int(box.cls[0])
                cls_name = self._model.names[cls_id]
                if target_classes is not None and cls_name not in target_classes:
                    continue
                conf = float(box.conf[0])
                x1, y1, x2, y2 = (int(v) for v in box.xyxy[0])
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                detections.append(
                    {
                        "class": cls_name,
                        "confidence": conf,
                        "bbox": [x1, y1, x2, y2],
                        "centroid": (cx, cy),
                    }
                )

        return detections

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def annotate(
        self,
        image: np.ndarray,
        detections: List[Detection],
    ) -> np.ndarray:
        """Draw bounding boxes and labels onto a copy of *image*.

        Parameters
        ----------
        image:
            Original BGR image (not modified in place).
        detections:
            List of detection dicts returned by :meth:`detect`.

        Returns
        -------
        A new BGR image with annotations drawn.
        """
        annotated = image.copy()
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            cx, cy = det["centroid"]
            label = f"{det['class']} {det['confidence']:.2f}"

            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(annotated, (cx, cy), 5, (0, 0, 255), -1)

            label_size, baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            label_y = max(y1 - 5, label_size[1])
            cv2.rectangle(
                annotated,
                (x1, label_y - label_size[1] - baseline),
                (x1 + label_size[0], label_y + baseline),
                (0, 255, 0),
                thickness=cv2.FILLED,
            )
            cv2.putText(
                annotated,
                label,
                (x1, label_y - baseline),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1,
            )

        return annotated

    # ------------------------------------------------------------------
    # Convenience selector
    # ------------------------------------------------------------------

    def best_detection(
        self,
        detections: List[Detection],
        target_class: Optional[str] = None,
    ) -> Optional[Detection]:
        """Return the highest-confidence detection (optionally filtered by class).

        Parameters
        ----------
        detections:
            List returned by :meth:`detect`.
        target_class:
            If provided, only detections of this class are considered.

        Returns
        -------
        The detection dict with the highest confidence, or ``None`` if no
        matching detections exist.
        """
        candidates = [
            d for d in detections
            if target_class is None or d["class"] == target_class
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda d: d["confidence"])
