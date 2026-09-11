"""
Colour-based detection of the three carom balls.

On a fixed overhead camera the balls are the only saturated red, yellow and
white blobs on the cloth, so plain HSV segmentation identifies each ball by
colour directly. That matters more than raw detector accuracy: a generic
detector gives anonymous boxes, and a shot is meaningless until we know which
ball is the cue ball.

Hue ranges are derived from the cloth hue found in the frame rather than
hardcoded, because cloth colour varies between venues (blue at 99-104 in the
two broadcasts sampled so far) and the red ball's hue wraps around 180.
"""

import cv2
import numpy as np

from src.physics.table_calibration import Calibration, detect_cloth_quad

BALL_DIAMETER_MM = 61.5  # ref/carom_spec.txt: 61-61.5 mm for carom

# Hue windows, sat/val floors. Red is listed as two windows because its hue
# wraps around the end of the OpenCV 0-180 scale.
COLOUR_RANGES = {
    "red": ([(0, 10), (160, 180)], 110, 90),
    "yellow": ([(18, 32)], 120, 120),
}
WHITE_SAT_MAX = 60
WHITE_VAL_MIN = 190


class BallDetector:
    """Finds the white, yellow and red balls in an overhead frame."""

    def __init__(self, calibration=None, area_tolerance=(0.35, 2.0), min_circularity=0.62):
        self.calibration = calibration
        self.area_tolerance = area_tolerance
        self.min_circularity = min_circularity
        self._table_mask = None
        self._expected_radius_px = None

    def _prepare(self, frame):
        """Build the search mask and the expected ball size once per camera setup."""
        if self._table_mask is not None and self._table_mask.shape == frame.shape[:2]:
            return True

        if isinstance(self.calibration, Calibration):
            quad = self.calibration.corners
            radius = (BALL_DIAMETER_MM / self.calibration.mm_per_px) / 2.0
        else:
            quad = detect_cloth_quad(frame)
            if quad is None:
                return False
            # Fall back to the cloth edge: the long side is the table length.
            long_px = max(
                np.hypot(*(quad[1] - quad[0])),
                np.hypot(*(quad[2] - quad[3])),
            )
            radius = (BALL_DIAMETER_MM / (2844.0 / long_px)) / 2.0

        mask = np.zeros(frame.shape[:2], np.uint8)
        cv2.fillPoly(mask, [quad.astype(np.int32)], 255)
        # Pull in slightly: a ball frozen against the cushion still has its
        # centre inside, and this keeps the dark cushion line out of the mask.
        self._table_mask = cv2.erode(mask, np.ones((5, 5), np.uint8))
        self._expected_radius_px = radius
        return True

    def _colour_mask(self, hsv, colour):
        if colour == "white":
            return cv2.inRange(
                hsv,
                np.array([0, 0, WHITE_VAL_MIN]),
                np.array([180, WHITE_SAT_MAX, 255]),
            )
        windows, sat_min, val_min = COLOUR_RANGES[colour]
        mask = None
        for lo_h, hi_h in windows:
            part = cv2.inRange(hsv, np.array([lo_h, sat_min, val_min]), np.array([hi_h, 255, 255]))
            mask = part if mask is None else cv2.bitwise_or(mask, part)
        return mask

    def _best_blob(self, mask):
        """Pick the blob that looks most like a ball, or None."""
        mask = cv2.bitwise_and(mask, self._table_mask)
        # Open first to drop speckle, then close so the red spots printed on the
        # cue balls do not split a ball into fragments.
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))

        expected_area = np.pi * self._expected_radius_px ** 2
        lo, hi = expected_area * self.area_tolerance[0], expected_area * self.area_tolerance[1]

        best, best_score = None, None
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if not (lo < area < hi):
                continue
            perimeter = cv2.arcLength(contour, True)
            if perimeter <= 0:
                continue
            # Circularity separates a ball from the cue shaft, an arm, or a
            # stripe of cloth reflection, all of which are elongated.
            circularity = 4.0 * np.pi * area / (perimeter ** 2)
            if circularity < self.min_circularity:
                continue
            m = cv2.moments(contour)
            if m["m00"] == 0:
                continue
            cx, cy = m["m10"] / m["m00"], m["m01"] / m["m00"]
            score = circularity - abs(np.log(area / expected_area))
            if best_score is None or score > best_score:
                best_score = score
                best = (cx, cy, float(np.sqrt(area / np.pi)), float(circularity))
        return best

    def detect(self, frame):
        """Detect the three balls.

        Returns {colour: {"px": (x, y), "radius_px": r, "circularity": c,
        "mm": (x, y) or None}}. A colour that is hidden or ambiguous is absent
        rather than guessed at.
        """
        if not self._prepare(frame):
            return {}

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        result = {}
        for colour in ("red", "yellow", "white"):
            blob = self._best_blob(self._colour_mask(hsv, colour))
            if blob is None:
                continue
            cx, cy, radius, circularity = blob
            entry = {"px": (cx, cy), "radius_px": radius, "circularity": circularity, "mm": None}
            if isinstance(self.calibration, Calibration):
                x, y = self.calibration.to_table((cx, cy))
                entry["mm"] = (float(x), float(y))
            result[colour] = entry
        return result

    def annotate(self, frame, detections):
        """Draw detections for visual verification."""
        colours = {"red": (0, 0, 255), "yellow": (0, 210, 255), "white": (255, 255, 255)}
        out = frame.copy()
        if self._table_mask is not None:
            quad = (
                self.calibration.corners
                if isinstance(self.calibration, Calibration)
                else detect_cloth_quad(frame)
            )
            if quad is not None:
                cv2.polylines(out, [quad.astype(np.int32)], True, (0, 255, 255), 2)
        for name, det in detections.items():
            x, y = det["px"]
            cv2.circle(out, (int(round(x)), int(round(y))), int(det["radius_px"]) + 7, colours[name], 2)
            label = name if det["mm"] is None else f"{name} {det['mm'][0]:.0f},{det['mm'][1]:.0f}"
            cv2.putText(out, label, (int(x) - 40, int(y) - 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, colours[name], 1, cv2.LINE_AA)
        return out
