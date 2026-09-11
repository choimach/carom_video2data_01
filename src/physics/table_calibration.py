"""
Diamond-based table calibration.

The rail diamonds are the only reference on a carom table whose positions are
fixed by the rules, so they - not the visible edge of the cloth - define the
mapping from pixels to millimetres. Calibrating on the cloth outline instead
overstates the table by the width of the cushion cloth, about 83 mm per side,
which is a 3% scale error that would propagate straight into every speed.

On a match table the diamonds are spaced 355.5 mm apart along the cushion nose
line (2844/8 on the long rails, 1422/4 on the short rails), giving 9 marks per
long rail and 5 per short rail, corner marks included. The markers themselves
sit on the rail, a constant distance outside the nose line; that offset is
recovered during calibration instead of being assumed.
"""

import cv2
import numpy as np

# Official match table ("대대") interior, from ref/carom_spec.txt
TABLE_LENGTH_MM = 2844.0
TABLE_WIDTH_MM = 1422.0
DIAMOND_SPACING_MM = 355.5  # 2844/8 == 1422/4

LONG_RAIL_DIAMONDS = 9  # indices 0..8
SHORT_RAIL_DIAMONDS = 5  # indices 0..4

RAILS = ("top", "right", "bottom", "left")
LONG_RAILS = ("top", "bottom")


class CalibrationError(Exception):
    """Raised when a frame cannot be calibrated."""


class Calibration:
    """Result of a successful calibration."""

    def __init__(self, matrix, corners, mm_per_px, rail_offset_mm, reprojection_error, diamonds):
        self.matrix = matrix
        self.corners = corners  # nose-line corners in pixels: TL, TR, BR, BL
        self.mm_per_px = mm_per_px
        self.rail_offset_mm = rail_offset_mm  # (long_rails, short_rails)
        self.reprojection_error = reprojection_error  # mean, in mm
        self.diamonds = diamonds  # {rail: [(index, x, y), ...]}

    @property
    def diamond_count(self):
        return sum(len(v) for v in self.diamonds.values())

    def to_table(self, points):
        """Map pixel points to table millimetres. Accepts a single (x, y) or a sequence."""
        pts = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
        out = cv2.perspectiveTransform(pts, self.matrix).reshape(-1, 2)
        return out[0] if np.ndim(points) == 1 else out

    def __repr__(self):
        return (
            f"<Calibration {self.mm_per_px:.4f} mm/px, {self.diamond_count} diamonds, "
            f"err={self.reprojection_error:.1f} mm>"
        )


def dominant_cloth_hue(hsv, sat_min=90, val_min=140):
    """Most common hue among saturated, bright pixels - the cloth, whatever its colour."""
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    sel = (s > sat_min) & (v > val_min)
    if sel.sum() < 1000:
        return None
    return int(np.bincount(h[sel], minlength=180).argmax())


def cloth_mask(hsv, hue=None, hue_tol=13, sat_min=90, val_min=140):
    """Binary mask of the playing surface.

    val_min matters: the cloth is brightly lit, while a dark blue carpet or
    backdrop around the table shares its hue and will otherwise be merged in.
    """
    if hue is None:
        hue = dominant_cloth_hue(hsv, sat_min, val_min)
        if hue is None:
            return None
    lo = np.array([max(0, hue - hue_tol), sat_min, val_min])
    hi = np.array([min(180, hue + hue_tol), 255, 255])
    mask = cv2.inRange(hsv, lo, hi)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))


def _order_corners(pts):
    """Order four points as top-left, top-right, bottom-right, bottom-left."""
    pts = np.asarray(pts, dtype=np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    return np.array([pts[s.argmin()], pts[d.argmin()], pts[s.argmax()], pts[d.argmax()]], np.float32)


def detect_cloth_quad(frame, min_area_ratio=0.15):
    """Rough quadrilateral around the cloth. Used only to locate the rails."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cloth_mask(hsv)
    if mask is None:
        return None
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < frame.shape[0] * frame.shape[1] * min_area_ratio:
        return None
    approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
    quad = approx.reshape(-1, 2) if len(approx) == 4 else cv2.boxPoints(cv2.minAreaRect(contour))
    quad = _order_corners(quad)
    # Reject anything too degenerate for the rail bands to make sense.
    sides = [np.hypot(*(quad[i] - quad[(i + 1) % 4])) for i in range(4)]
    if min(sides) < 50:
        return None
    return quad


def rail_edges(quad):
    """Each rail as an oriented segment. Orientation fixes which diamond is index 0."""
    return {
        "top": (quad[0], quad[1]),
        "right": (quad[1], quad[2]),
        "bottom": (quad[3], quad[2]),
        "left": (quad[0], quad[3]),
    }


def detect_diamonds(frame, quad, band_outer=61, band_inner=11, area_range=(15, 400),
                    offset_tolerance_px=10.0):
    """Find diamond markers in the rail band just outside the cloth.

    Returns {rail: [(x, y), ...]} ordered along the rail. Markers on one rail
    all sit the same distance from the cloth edge, so a blob whose offset
    disagrees with its neighbours - a chalk cube, a shirt, a reflection - is
    dropped before it can corrupt the index assignment.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    quad_i = quad.astype(np.int32)

    filled = np.zeros(frame.shape[:2], np.uint8)
    cv2.fillPoly(filled, [quad_i], 255)
    band = cv2.subtract(
        cv2.dilate(filled, np.ones((band_outer, band_outer), np.uint8)),
        cv2.dilate(filled, np.ones((band_inner, band_inner), np.uint8)),
    )

    bright = cv2.inRange(hsv, np.array([0, 0, 150]), np.array([180, 90, 255]))
    candidates = cv2.morphologyEx(cv2.bitwise_and(bright, band), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    contours, _ = cv2.findContours(candidates, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    edges = rail_edges(quad)

    grouped = {rail: [] for rail in RAILS}
    for contour in contours:
        area = cv2.contourArea(contour)
        if not (area_range[0] < area < area_range[1]):
            continue
        m = cv2.moments(contour)
        if m["m00"] == 0:
            continue
        x, y = m["m10"] / m["m00"], m["m01"] / m["m00"]

        best = None
        for rail, (p, q) in edges.items():
            pq = q - p
            denom = float(np.dot(pq, pq))
            if denom <= 0:
                continue
            t = float(np.clip(np.dot([x - p[0], y - p[1]], pq) / denom, 0.0, 1.0))
            proj = p + t * pq
            dist = float(np.hypot(x - proj[0], y - proj[1]))
            if best is None or dist < best[0]:
                best = (dist, rail, t)
        if best is None:
            continue
        dist, rail, t = best
        grouped[rail].append((t, x, y, dist))

    result = {}
    for rail, items in grouped.items():
        if len(items) >= 3:
            median_offset = float(np.median([d for _, _, _, d in items]))
            items = [it for it in items if abs(it[3] - median_offset) <= offset_tolerance_px]
        result[rail] = [(x, y) for _, x, y, _ in sorted(items)]
    return result


def _index_along_rail(points, rail_vector, expected_count, max_residual_ratio=0.25):
    """Assign diamond indices to the points detected on one rail.

    Points are projected onto a fitted line and spaced by the median gap, so a
    marker hidden behind a player or a cue leaves a gap rather than shifting
    every index after it.
    """
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < 3:
        raise CalibrationError(f"only {len(pts)} diamonds")

    vx, vy, x0, y0 = cv2.fitLine(pts.astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01).ravel()
    direction = np.array([vx, vy], dtype=np.float64)
    # fitLine's direction has an arbitrary sign; without pinning it to the rail's
    # own orientation, index 0 lands at whichever end the fit happened to pick.
    if float(np.dot(direction, rail_vector)) < 0:
        direction = -direction

    t = (pts - np.array([x0, y0])) @ direction
    order = np.argsort(t)
    t, pts = t[order], pts[order]

    gaps = np.diff(t)
    if len(gaps) == 0 or gaps.min() <= 0:
        raise CalibrationError("degenerate rail")
    spacing = float(np.median(gaps))
    # A gap that spans a missing marker is a multiple of the spacing; dividing
    # through by that multiple recovers the true spacing from every gap.
    multiples = np.maximum(1, np.round(gaps / spacing))
    spacing = float(np.median(gaps / multiples))
    if spacing <= 0:
        raise CalibrationError("non-positive diamond spacing")

    indices = np.round((t - t[0]) / spacing).astype(int)
    residual = np.abs(t - (t[0] + indices * spacing))

    # Two blobs landing on the same index means one of them is not a diamond;
    # keep whichever sits closest to the regular grid.
    keep = {}
    for i, idx in enumerate(indices):
        if idx not in keep or residual[i] < residual[keep[idx]]:
            keep[idx] = i
    sel = sorted(keep.values())
    indices, pts, residual = indices[sel], pts[sel], residual[sel]

    sel = residual <= spacing * max_residual_ratio
    indices, pts = indices[sel], pts[sel]
    if len(indices) < 3:
        raise CalibrationError("too few diamonds fit a regular spacing")
    if indices[-1] - indices[0] > expected_count - 1:
        raise CalibrationError(f"diamond span {indices[-1] - indices[0] + 1} exceeds {expected_count}")
    return indices, pts, spacing


def _anchor_indices(indices, points, expected_count, quad_start, quad_end):
    """Shift indices so index 0 is the diamond on the rail's starting corner."""
    missing = expected_count - 1 - (indices[-1] - indices[0])
    if missing == 0:
        return indices - indices[0]
    # Some markers are missing; decide whether they fell off the start or the end
    # by comparing the detected ends against the corners they should sit on.
    d_start = float(np.hypot(*(points[0] - quad_start)))
    d_end = float(np.hypot(*(points[-1] - quad_end)))
    shift = 0 if d_start <= d_end else missing
    return indices - indices[0] + shift


def calibrate(frame, min_diamonds=14, min_rails=3, max_reprojection_mm=12.0):
    """Calibrate a frame from its rail diamonds.

    Rails are handled independently, so a cue or a player hiding one rail costs
    that rail's points rather than the whole frame. Raises CalibrationError if
    the frame does not show a full table with enough visible diamonds.
    """
    quad = detect_cloth_quad(frame)
    if quad is None:
        raise CalibrationError("no table-sized cloth region found")

    detected = detect_diamonds(frame, quad)
    edges = rail_edges(quad)

    rails, problems = {}, {}
    for rail in RAILS:
        expected = LONG_RAIL_DIAMONDS if rail in LONG_RAILS else SHORT_RAIL_DIAMONDS
        start, end = edges[rail]
        try:
            indices, pts, spacing = _index_along_rail(detected[rail], end - start, expected)
            rails[rail] = (_anchor_indices(indices, pts, expected, start, end), pts, spacing)
        except CalibrationError as exc:
            problems[rail] = str(exc)

    if len(rails) < min_rails:
        raise CalibrationError(f"only {len(rails)} usable rails ({problems})")
    total = sum(len(v[0]) for v in rails.values())
    if total < min_diamonds:
        raise CalibrationError(f"only {total} diamonds detected")

    # Scale comes from the longest available baseline on each rail, which is far
    # less noisy than any single gap.
    scales = []
    for rail, (indices, pts, _) in rails.items():
        span_idx = int(indices[-1] - indices[0])
        span_px = float(np.hypot(*(pts[-1] - pts[0])))
        if span_idx > 0 and span_px > 0:
            scales.append(span_idx * DIAMOND_SPACING_MM / span_px)
    if not scales:
        raise CalibrationError("could not establish scale")
    mm_per_px = float(np.median(scales))

    # The markers sit outside the nose line by a constant distance. Recover it
    # from the separation between opposite rails and the known table size.
    def opposite_offset(rail_a, rail_b, true_mm):
        if rail_a not in rails or rail_b not in rails:
            return None
        a = rails[rail_a][1].mean(axis=0)
        b = rails[rail_b][1].mean(axis=0)
        return (float(np.hypot(*(a - b))) * mm_per_px - true_mm) / 2.0

    offset_long = opposite_offset("top", "bottom", TABLE_WIDTH_MM)
    offset_short = opposite_offset("left", "right", TABLE_LENGTH_MM)
    # The rails are built alike, so the two offsets agree to a couple of
    # millimetres; either one stands in for the other when a rail is hidden.
    if offset_long is None:
        offset_long = offset_short
    if offset_short is None:
        offset_short = offset_long
    if offset_long is None:
        raise CalibrationError("no opposite rail pair visible")
    if not (0 <= offset_long < 300) or not (0 <= offset_short < 300):
        raise CalibrationError(f"implausible rail offsets ({offset_long:.0f}, {offset_short:.0f} mm)")

    # Ideal table-millimetre position of every detected marker, including its
    # outward offset, so the homography is fitted to consistent data.
    ideal = {
        "top": lambda k: (k * DIAMOND_SPACING_MM, -offset_long),
        "bottom": lambda k: (k * DIAMOND_SPACING_MM, TABLE_WIDTH_MM + offset_long),
        "left": lambda k: (-offset_short, k * DIAMOND_SPACING_MM),
        "right": lambda k: (TABLE_LENGTH_MM + offset_short, k * DIAMOND_SPACING_MM),
    }

    src, dst, diamonds = [], [], {}
    for rail, (indices, pts, _) in rails.items():
        diamonds[rail] = [(int(k), float(p[0]), float(p[1])) for k, p in zip(indices, pts)]
        for k, p in zip(indices, pts):
            src.append(p)
            dst.append(ideal[rail](int(k)))

    src = np.array(src, np.float32)
    dst = np.array(dst, np.float32)
    matrix, _ = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    if matrix is None:
        raise CalibrationError("homography fit failed")

    projected = cv2.perspectiveTransform(src.reshape(-1, 1, 2), matrix).reshape(-1, 2)
    error = float(np.mean(np.linalg.norm(projected - dst, axis=1)))
    if error > max_reprojection_mm:
        raise CalibrationError(f"reprojection error {error:.1f} mm too large")

    nose = np.array(
        [[0, 0], [TABLE_LENGTH_MM, 0], [TABLE_LENGTH_MM, TABLE_WIDTH_MM], [0, TABLE_WIDTH_MM]],
        np.float32,
    )
    corners = cv2.perspectiveTransform(nose.reshape(-1, 1, 2), np.linalg.inv(matrix)).reshape(-1, 2)

    return Calibration(
        matrix=matrix,
        corners=corners,
        mm_per_px=mm_per_px,
        rail_offset_mm=(offset_long, offset_short),
        reprojection_error=error,
        diamonds=diamonds,
    )
