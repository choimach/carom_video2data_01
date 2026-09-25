"""
Colour-based detection of the three carom balls.

On a fixed overhead camera the balls are the only red, yellow and white blobs
on the cloth, so colour segmentation identifies each ball by name directly.
That matters more than raw detector accuracy: a generic detector gives
anonymous boxes, and a shot is meaningless until we know which ball is the cue
ball.

★**Absolute colour windows cannot work, and that is measured, not assumed.**
On 2026-09-25 eight of sixty-one matches produced 570 detected plays and one
usable one. Cropping the balls out of the video and measuring them in Lab:

    (L, chroma, hue angle)      red            yellow          white
    soop_163210563_AKR2025      150, 28,  +2°  230, 23, +101°  252,  2, -130°
    soop_188224321_T3WC2026      91, 57, +32°  141, 70,  +60°  205, 42, +104°

**T3WC's white ball sits exactly where AKR's yellow ball sits** (+104 vs +101).
No fixed threshold separates those two, in any colour space. One broadcast's
yellow is another's white: T3WC's set is an orange yellow and a cream white,
and the whole picture is darker (low-saturation pixels have V 54 against AKR's
207). The old HSV constants - yellow S>=120, white V>=190 - missed every ball
in both matches.

What *is* stable is the **order within one frame**, and it is stable because of
what Lab measures rather than because it was tuned:

* the white ball is the one with the most lightness for the least chroma,
* red is dominated by +a and yellow by +b, so red's hue angle is the smaller.

So the three balls are named relatively, the first few hundred frames of a
match teach a **palette**, and from then on each blob is matched to the nearest
learned colour - which keeps working when a player's arm hides one ball.

The cloth's own colour is measured from the frame and everything is referred to
it, which is what the first version of this file claimed to do and did not.

`COLOUR_RANGES` below is kept as the fallback for frames seen before the
palette exists and for frames showing fewer than three balls; it is the only
place absolute numbers survive.
"""

import itertools
import math

import cv2
import numpy as np

from src.physics.table_calibration import Calibration, detect_cloth_quad

BALL_DIAMETER_MM = 61.5  # ref/carom_spec.txt: 61-61.5 mm for carom

# Hue windows, sat/val floors. Red is listed as two windows because its hue
# wraps around the end of the OpenCV 0-180 scale.
# ⚠️ Fallback only - see the module docstring for why these cannot be the
# primary test.
COLOUR_RANGES = {
    "red": ([(0, 10), (160, 180)], 110, 90),
    "yellow": ([(18, 32)], 120, 120),
}
WHITE_SAT_MAX = 60
WHITE_VAL_MIN = 190

# How far from the cloth's own Lab colour a pixel must sit to be a candidate.
# The balls in the two hardest broadcasts measured are 60-190 away; the cloth's
# own shading spreads about 20.
CLOTH_GAP = 40.0
# A mark or a shadow on the cloth is the cloth's own colour at another
# brightness, so it is the *chromaticity* - (a, b) with lightness dropped -
# that gives it away. Measured distance from the cloth's own (a, b):
#
#     T3WC's false blob (a mark on the cloth)     11.7
#     AKR's white ball                            37     <- nearly neutral
#     AKR red / yellow                            52 / 59
#     T3WC white / red / yellow                  101 / 98 / 123
#
# ⚠️ The first version of this tested the hue *angle* instead, and that threw
# the white ball away: a ball with chroma 2 is grey, so its angle is noise, and
# AKR's landed 26 degrees from the cloth's. 43 frames with three good blobs
# became 6. An angle is the wrong question to ask a neutral colour.
CLOTH_CHROMA_GAP = 20.0
# Frames of three clean candidates to learn the palette from. Thirty is about
# half a second of play and every match reaches it in the first rally.
PALETTE_FRAMES = 30


class BallDetector:
    """Finds the white, yellow and red balls in an overhead frame."""

    def __init__(self, calibration=None, area_tolerance=(0.35, 2.0), min_solidity=0.60,
                 max_aspect=3.2):
        self.calibration = calibration
        self.area_tolerance = area_tolerance
        self.min_solidity = min_solidity
        self.max_aspect = max_aspect
        self._table_mask = None
        self._expected_radius_px = None
        self._roi = None  # (x, y, w, h) bounding box of the table within the frame
        self._frame_shape = None
        self._cloth_mask = None  # pulled further in, for measuring the cloth
        self._palette = None  # {colour: (a, b)} once learned
        self._samples = []  # (colour, a, b) triples while learning

    def _prepare(self, frame):
        """Build the search mask and the expected ball size once per camera setup."""
        if self._table_mask is not None and self._frame_shape == frame.shape[:2]:
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
        mask = cv2.erode(mask, np.ones((5, 5), np.uint8))

        # Everything downstream only looks inside the table, and the table is
        # about a third of a broadcast frame, so cropping to it before the
        # colour conversion is the difference between processing a match in
        # hours and in minutes.
        x, y, w, h = cv2.boundingRect(mask)
        self._roi = (x, y, w, h)
        self._table_mask = mask[y:y + h, x:x + w]
        # Measuring the cloth wants to be nowhere near the rails, which are
        # wood in some venues and would drag the median off the cloth. Four
        # ball-widths in is still almost the whole table.
        inset = max(9, int(round(radius * 4)))
        self._cloth_mask = cv2.erode(self._table_mask, np.ones((inset, inset), np.uint8))
        if not self._cloth_mask.any():
            self._cloth_mask = self._table_mask
        self._frame_shape = frame.shape[:2]
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

    def _blobs(self, mask):
        """Every blob in `mask` that passes the ball tests, as dicts.

        The shape tests are shared by both naming paths, so a ball that the
        relative path finds is the same ball the fallback would have found.
        """
        mask = cv2.bitwise_and(mask, self._table_mask)
        # Open first to drop speckle, then close so the red spots printed on the
        # cue balls do not split a ball into fragments.
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))

        expected_area = np.pi * self._expected_radius_px ** 2
        lo, hi = expected_area * self.area_tolerance[0], expected_area * self.area_tolerance[1]

        found = []
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if not (lo < area < hi):
                continue
            hull_area = cv2.contourArea(cv2.convexHull(contour))
            if hull_area <= 0:
                continue
            # Solidity, not perimeter circularity. A ball crossing the cloth at
            # speed is smeared by the shutter into a smooth ellipse: still
            # convex, but elongated enough that circularity collapses to ~0.5
            # and drops it. That single test was losing a tenth of the white
            # ball's frames - the fastest tenth, which is exactly the frames a
            # shot is recognised from. Solidity stays near 0.8 through the blur
            # while still rejecting ragged reflections.
            solidity = area / hull_area
            if solidity < self.min_solidity:
                continue
            # Solidity alone would accept the cue shaft, which is perfectly
            # convex, so cap how elongated the blob may be.
            (_, (w_rect, h_rect), _) = cv2.minAreaRect(contour)
            short, long_ = sorted((w_rect, h_rect))
            if short <= 0 or long_ / short > self.max_aspect:
                continue
            m = cv2.moments(contour)
            if m["m00"] == 0:
                continue
            cx, cy = m["m10"] / m["m00"], m["m01"] / m["m00"]
            found.append({
                "px": (cx, cy),
                "radius_px": float(np.sqrt(area / np.pi)),
                "solidity": float(solidity),
                "score": float(solidity - abs(np.log(area / expected_area))),
                "contour": contour,
            })
        return found

    def _best_blob(self, mask):
        """Pick the blob that looks most like a ball, or None. (fallback path)"""
        found = self._blobs(mask)
        if not found:
            return None
        best = max(found, key=lambda b: b["score"])
        return (best["px"][0], best["px"][1], best["radius_px"], best["solidity"])

    def _cloth_colour(self, lab):
        """The cloth's own Lab, measured from this frame.

        Strided: a tenth of the table is tens of thousands of pixels, which is
        more than enough for a median and keeps this off the hot path.
        """
        sample = lab[::4, ::4][self._cloth_mask[::4, ::4] > 0]
        if len(sample) < 50:
            return None
        return np.median(sample, axis=0)

    def _candidates(self, lab, cloth):
        """Ball-shaped blobs that are not the cloth, with their colours."""
        gap = np.linalg.norm(lab.astype(np.float32) - cloth, axis=2)
        mask = (gap > CLOTH_GAP).astype(np.uint8) * 255
        cloth_a, cloth_b = float(cloth[1]) - 128.0, float(cloth[2]) - 128.0
        out = []
        for blob in self._blobs(mask):
            filled = np.zeros(lab.shape[:2], np.uint8)
            cv2.drawContours(filled, [blob["contour"]], -1, 255, -1)
            pixels = lab[filled > 0]
            if len(pixels) == 0:
                continue
            L, a, b = (float(v) for v in np.median(pixels, axis=0))
            a, b = a - 128.0, b - 128.0
            angle = math.degrees(math.atan2(b, a))
            if math.hypot(a - cloth_a, b - cloth_b) < CLOTH_CHROMA_GAP:
                continue
            blob.update({"L": L, "a": a, "b": b, "angle": angle,
                         "chroma": math.hypot(a, b)})
            out.append(blob)
        return out

    @staticmethod
    def _name_by_order(three):
        """Name three candidates by their order, with no absolute thresholds.

        White is the one carrying the most lightness for the least colour; of
        the remaining two, Lab puts red along +a and yellow along +b, so red is
        the one whose hue angle is the smaller. Both hold in every broadcast
        measured, including the two where every absolute window failed.
        """
        white = max(three, key=lambda c: c["L"] - 2.0 * c["chroma"])
        rest = sorted((c for c in three if c is not white), key=lambda c: c["angle"])
        return {"white": white, "red": rest[0], "yellow": rest[1]}

    def _name_by_palette(self, candidates):
        """Match blobs to the learned palette, one colour to one blob."""
        names = list(self._palette)
        best, best_cost = None, None
        for chosen in itertools.permutations(range(len(candidates)), min(len(names), len(candidates))):
            for subset in itertools.permutations(names, len(chosen)):
                cost = sum(
                    math.hypot(candidates[i]["a"] - self._palette[name][0],
                               candidates[i]["b"] - self._palette[name][1])
                    for i, name in zip(chosen, subset)
                )
                if best_cost is None or cost < best_cost:
                    best_cost = cost
                    best = {name: candidates[i] for i, name in zip(chosen, subset)}
        return best or {}

    def _learn(self, named):
        """Remember one frame's colours; freeze the palette once there are enough."""
        for colour, blob in named.items():
            self._samples.append((colour, blob["a"], blob["b"]))
        if len(self._samples) < PALETTE_FRAMES * 3:
            return
        palette = {}
        for colour in ("red", "yellow", "white"):
            got = [(a, b) for name, a, b in self._samples if name == colour]
            if len(got) < PALETTE_FRAMES // 2:
                self._samples = []  # too ragged to trust; start the count over
                return
            palette[colour] = tuple(np.median(np.array(got), axis=0))
        self._palette = palette
        self._samples = []

    def detect(self, frame):
        """Detect the three balls.

        Returns {colour: {"px": (x, y), "radius_px": r, "solidity": s,
        "mm": (x, y) or None}}. A colour that is hidden or ambiguous is absent
        rather than guessed at.
        """
        if not self._prepare(frame):
            return {}

        x0, y0, w, h = self._roi
        crop = frame[y0:y0 + h, x0:x0 + w]

        named = None
        lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
        cloth = self._cloth_colour(lab)
        if cloth is not None:
            found = self._candidates(lab, cloth)
            if self._palette is not None:
                named = self._name_by_palette(found)
            elif len(found) >= 3:
                # More than three means something else on the cloth passed the
                # shape tests; the three that look most like balls are the balls.
                three = sorted(found, key=lambda c: -c["score"])[:3]
                named = self._name_by_order(three)
                self._learn(named)

        if not named:
            # Before the palette exists, and on frames showing fewer than three
            # balls, fall back to the absolute windows. They are wrong for some
            # broadcasts, but a wrong guess here only costs the frames until the
            # first rally teaches the palette.
            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            named = {}
            for colour in ("red", "yellow", "white"):
                blob = self._best_blob(self._colour_mask(hsv, colour))
                if blob is not None:
                    cx, cy, radius, solidity = blob
                    named[colour] = {"px": (cx, cy), "radius_px": radius,
                                     "solidity": solidity}

        result = {}
        for colour, blob in named.items():
            cx, cy = blob["px"][0] + x0, blob["px"][1] + y0
            entry = {"px": (cx, cy), "radius_px": blob["radius_px"],
                     "solidity": blob["solidity"], "mm": None}
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
