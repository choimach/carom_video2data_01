"""
Detecting the broadcast's drawn shot path.

This production draws the cue ball's path over the table when it replays a
point - and, as far as the match reviewed shows, only when a point was made.
That makes the graphic a direct statement of success from the broadcast itself,
independent of the scoreboard's timing and unaffected by a play being shown
twice.

It matters most exactly where the geometry cannot help. A shot the director cut
away from leaves a trajectory that stops before the second object ball, so the
path alone cannot say whether the point was made; the replay that follows draws
the whole thing and settles it.

What separates the graphic from ordinary play is shape, not brightness. The
balls are bright too, but they are small and round; the drawn path is a handful
of long strokes. Counting only components far longer than a ball keeps the two
apart.
"""

import cv2
import numpy as np

from src.physics.table_calibration import Calibration

# A ball is about 25 px across at broadcast scale; nothing else in normal play
# is both bright and this long.
MIN_LINE_LENGTH_PX = 120
LINE_VALUE_MIN = 150
LINE_SATURATION_MAX = 80


class OverlayDetector:
    """Measures how much of the table is covered by drawn path lines."""

    def __init__(self, calibration, min_line_length_px=MIN_LINE_LENGTH_PX):
        if not isinstance(calibration, Calibration):
            raise TypeError("a Calibration is required to know where the table is")
        self.calibration = calibration
        self.min_line_length_px = min_line_length_px
        self._mask = None
        self._area = None
        self._shape = None

    def _table_mask(self, frame):
        if self._mask is None or self._shape != frame.shape[:2]:
            mask = np.zeros(frame.shape[:2], np.uint8)
            cv2.fillPoly(mask, [self.calibration.corners.astype(np.int32)], 255)
            # Pull in from the cushions: the rail highlights are bright and long
            # and would read as drawn lines.
            self._mask = cv2.erode(mask, np.ones((21, 21), np.uint8))
            self._area = max(1, int(self._mask.sum() // 255))
            self._shape = frame.shape[:2]
        return self._mask

    def score(self, frame):
        """Fraction of the table covered by line-like bright structures."""
        mask = self._table_mask(frame)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        bright = cv2.inRange(
            hsv,
            np.array([0, 0, LINE_VALUE_MIN]),
            np.array([180, LINE_SATURATION_MAX, 255]),
        )
        bright = cv2.bitwise_and(bright, mask)
        # Close small gaps so a path crossing a ball stays one component.
        bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

        count, _, stats, _ = cv2.connectedComponentsWithStats(bright, connectivity=8)
        covered = 0
        for index in range(1, count):
            x, y, w, h, area = stats[index]
            if max(w, h) >= self.min_line_length_px:
                covered += area
        return covered / self._area

    def is_overlay(self, frame, threshold=0.004):
        return self.score(frame) >= threshold


def span_overlay_fraction(detector, frames):
    """Share of the given frames that carry the drawn path.

    Judged over a span rather than a single frame: the graphic is drawn
    progressively, so the first frames of a replay look like ordinary play.
    """
    scores = [detector.score(f) for f in frames]
    if not scores:
        return 0.0, []
    return float(np.mean([s >= 0.004 for s in scores])), scores


def overlay_mask(detector, frame):
    """Just the drawn path: bright line-like structures inside the table."""
    mask = detector._table_mask(frame)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    bright = cv2.inRange(
        hsv, np.array([0, 0, LINE_VALUE_MIN]), np.array([180, LINE_SATURATION_MAX, 255])
    )
    bright = cv2.bitwise_and(bright, mask)
    bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    count, labels, stats, _ = cv2.connectedComponentsWithStats(bright, connectivity=8)
    keep = np.zeros_like(bright)
    for index in range(1, count):
        x, y, w, h, area = stats[index]
        if max(w, h) >= detector.min_line_length_px:
            keep[labels == index] = 255
    return keep


def path_overlap(detector, frame, cue_xy_mm, tolerance_px=14):
    """How much of a tracked cue path lies on the path drawn in this frame.

    Attribution by time does not work: three-cushion turns run about forty
    seconds, so a replay that follows a missed shot falls inside the window of
    that shot as readily as of the one it actually shows. The drawn line *is*
    the earlier shot's path, so the two can be matched on shape instead, which
    is the same reasoning that made replay detection by opening layout exact.
    """
    drawn = overlay_mask(detector, frame)
    if not drawn.any():
        return 0.0
    drawn = cv2.dilate(drawn, np.ones((tolerance_px, tolerance_px), np.uint8))

    inverse = np.linalg.inv(detector.calibration.matrix)
    points = cue_xy_mm[np.isfinite(cue_xy_mm[:, 0]) & np.isfinite(cue_xy_mm[:, 1])]
    if len(points) < 10:
        return 0.0
    pixels = cv2.perspectiveTransform(
        points.astype(np.float32).reshape(-1, 1, 2), inverse
    ).reshape(-1, 2)

    height, width = frame.shape[:2]
    inside = (
        (pixels[:, 0] >= 0) & (pixels[:, 0] < width)
        & (pixels[:, 1] >= 0) & (pixels[:, 1] < height)
    )
    pixels = pixels[inside].astype(int)
    if len(pixels) < 10:
        return 0.0
    hits = drawn[pixels[:, 1], pixels[:, 0]] > 0
    return float(hits.mean())
