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
TABLE_ASPECT = TABLE_LENGTH_MM / TABLE_WIDTH_MM  # exactly 2

LONG_RAIL_DIAMONDS = 9  # indices 0..8
SHORT_RAIL_DIAMONDS = 5  # indices 0..4

# Long edge of the cloth quad, in pixels, on the broadcast the detector's
# pixel sizes below were measured against.
REFERENCE_TABLE_PX = 1181.0

RAILS = ("top", "right", "bottom", "left")
LONG_RAILS = ("top", "bottom")


class CalibrationError(Exception):
    """Raised when a frame cannot be calibrated."""


class Calibration:
    """Result of a successful calibration."""

    def __init__(self, matrix, corners, mm_per_px, rail_offset_mm, reprojection_error, diamonds,
                 cloth_hue=None, polarity="bright", frame_size=None):
        self.matrix = matrix
        self.corners = corners  # nose-line corners in pixels: TL, TR, BR, BL
        self.mm_per_px = mm_per_px
        self.rail_offset_mm = rail_offset_mm  # (long_rails, short_rails)
        self.reprojection_error = reprojection_error  # mean, in mm
        self.diamonds = diamonds  # {rail: [(index, x, y), ...]}
        self.cloth_hue = cloth_hue
        self.polarity = polarity
        self.frame_size = frame_size  # (height, width), for clipping the probes
        self.rail_value = None  # brightness of bare rail, learned from the frame
        self.diamond_value = None
        self.rail_reference = None  # what each rail probe looked like, one per pixel
        self.diamond_reference = None
        self._probes = None

    @property
    def diamond_count(self):
        return sum(len(v) for v in self.diamonds.values())

    def _build_probes(self):
        """Pixel positions of a handful of points whose appearance identifies this table."""
        inverse = np.linalg.inv(self.matrix)
        offset = self.rail_offset_mm[0]

        cloth_mm, rail_mm, diamond_mm = [], [], []
        for u in (0.2, 0.4, 0.6, 0.8):
            for w in (0.25, 0.75):
                cloth_mm.append((u * TABLE_LENGTH_MM, w * TABLE_WIDTH_MM))
        for k in range(8):  # between diamonds, so these land on bare rail
            x = (k + 0.5) * DIAMOND_SPACING_MM
            rail_mm += [(x, -offset), (x, TABLE_WIDTH_MM + offset)]
        for k in (0, 2, 4, 6, 8):
            diamond_mm += [(k * DIAMOND_SPACING_MM, -offset),
                           (k * DIAMOND_SPACING_MM, TABLE_WIDTH_MM + offset)]

        height, width = self.frame_size or (10 ** 6, 10 ** 6)

        def to_px(points):
            pts = np.array(points, np.float32).reshape(-1, 1, 2)
            px = cv2.perspectiveTransform(pts, inverse).reshape(-1, 2).astype(int)
            # Dropping the out-of-frame points here, once, is what lets a later
            # sample line up one-to-one with the reference taken at calibration.
            inside = (px[:, 0] >= 0) & (px[:, 0] < width) & (px[:, 1] >= 0) & (px[:, 1] < height)
            return px[inside]

        self._probes = (to_px(cloth_mm), to_px(rail_mm), to_px(diamond_mm))
        return self._probes

    def learn_appearance(self, frame):
        """Record how each probe pixel looks on the frame that calibrated.

        `is_table_visible` recognises the table again thousands of times a scan
        by reading fixed pixels, and what it is really asking is whether they
        still look the way they did here. One grey level for the whole rail
        cannot answer that: the Pohang rail carries printing, so a third of the
        points between its diamonds are nothing like the bare timber either
        side of them, and comparing all sixteen against a single median failed
        the test on a table that was in plain view. Each probe is its own
        reference instead.
        """
        _, rail_px, diamond_px = self._probes or self._build_probes()
        value = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[:, :, 2]

        def values_at(points):
            return value[points[:, 1], points[:, 0]].astype(np.int16) if len(points) else None

        self.rail_reference = values_at(rail_px)
        self.diamond_reference = values_at(diamond_px)
        if self.rail_reference is None or self.diamond_reference is None:
            self.rail_reference = self.diamond_reference = None
        else:
            self.rail_value = float(np.median(self.rail_reference))
            self.diamond_value = float(np.median(self.diamond_reference))
        return self

    def is_table_visible(self, frame, hue_tol=13, min_ratio=0.6, rail_tol=45, diamond_tol=60):
        """Is this frame still showing the calibrated table?

        Testing for cloth-coloured pixels alone is not enough: a full-screen
        sponsor graphic on a blue field passes that test, and if it happens to
        carry red, white and yellow circles the ball detector will happily
        report three balls sitting perfectly still. So this also requires the
        dark rail between the diamonds, and the diamonds themselves - a
        combination a graphic will not reproduce at these exact pixels.
        """
        if self.cloth_hue is None:
            return False
        cloth_px, rail_px, diamond_px = self._probes or self._build_probes()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, w = frame.shape[:2]

        def sample(points):
            return hsv[points[:, 1], points[:, 0]] if len(points) else np.empty((0, 3))

        cloth = sample(cloth_px)
        rail = sample(rail_px)
        diamonds = sample(diamond_px)
        if len(cloth) == 0 or len(rail) == 0 or len(diamonds) == 0:
            return False

        hue_delta = np.abs(cloth[:, 0].astype(int) - self.cloth_hue)
        is_cloth = (hue_delta <= hue_tol) & (cloth[:, 1] > 90) & (cloth[:, 2] > 140)
        if self.rail_reference is None:
            is_rail = rail[:, 2] < 120  # rails are dark timber in shadow
            is_diamond = (diamonds[:, 2] > 140) & (diamonds[:, 1] < 110)
        else:
            # Each probe against what that same pixel looked like when the table
            # was calibrated: "dark rail, bright diamond" describes only some of
            # the tables in use, and one grey level describes only a rail with
            # nothing printed on it.
            is_rail = np.abs(rail[:, 2].astype(np.int16) - self.rail_reference) <= rail_tol
            is_diamond = (np.abs(diamonds[:, 2].astype(np.int16) - self.diamond_reference)
                          <= diamond_tol)

        return (
            is_cloth.mean() >= min_ratio
            and is_rail.mean() >= min_ratio
            and is_diamond.mean() >= 0.4
        )

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
    if min(sides) < 50 * frame.shape[1] / 1920:
        return None
    return quad


def looks_like_a_table(quad, tolerance=0.15):
    """Is this quad the right shape for a table seen from above?

    A blue quad is not by itself a table. Half the Antalya broadcast is lit in
    blue from the rig above the arena, and the cloth mask happily returns a
    table-sized region of empty hall; a close-up of one corner returns a long
    thin sliver of cloth. Both were being counted as "the table is in shot but
    would not calibrate", which is the one verdict that buys a match a second,
    more expensive look. Seen from anywhere near overhead the nose line keeps
    its 2:1 proportions, and neither of those does.
    """
    sides = [float(np.hypot(*(quad[i] - quad[(i + 1) % 4]))) for i in range(4)]
    shortest = min(sides)
    if shortest <= 0:
        return False
    aspect = max(sides) / shortest
    return abs(aspect - TABLE_ASPECT) <= TABLE_ASPECT * tolerance


def rail_edges(quad):
    """Each rail as an oriented segment. Orientation fixes which diamond is index 0."""
    return {
        "top": (quad[0], quad[1]),
        "right": (quad[1], quad[2]),
        "bottom": (quad[3], quad[2]),
        "left": (quad[0], quad[3]),
    }


POLARITIES = ("bright", "dark")


def _contrast_mask(value, band, scale, polarity):
    """Blobs inside the rail band that stand out from the rail around them.

    A top-hat keeps what is brighter than its surroundings and a black-hat what
    is darker, so the same code finds white markers on a dark rail and the
    inlaid dark dots that a pale wooden rail carries instead. The threshold is
    taken from the band's own contrast rather than an absolute grey level,
    because the rail's brightness varies with the venue lighting.
    """
    size = max(5, int(25 * scale) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    operation = cv2.MORPH_TOPHAT if polarity == "bright" else cv2.MORPH_BLACKHAT
    hat = cv2.morphologyEx(value, operation, kernel)
    inside = hat[band > 0]
    if inside.size == 0:
        return np.zeros_like(band)
    strongest = float(np.percentile(inside, 99.0))
    threshold = max(12.0, strongest / 2.0)
    return cv2.bitwise_and(cv2.inRange(hat, threshold, 255.0), band)


def table_scale(frame, quad):
    """How big this table is against the one the pixel constants were measured on.

    Using the frame width instead is wrong whenever the camera is pulled back:
    a marker's size in pixels follows the table, not the picture around it. The
    Ankara final is shot from further away than the SOOP world cup coverage, so
    its table fills little more than half the frame and its diamonds are half
    the size the frame width would predict.
    """
    sides = [float(np.hypot(*(quad[i] - quad[(i + 1) % 4]))) for i in range(4)]
    return max(sides) / REFERENCE_TABLE_PX


def detect_diamonds(frame, quad, band_outer=None, band_inner=None, area_range=None,
                    offset_tolerance_px=None, polarity="bright"):
    """Find diamond markers in the rail band just outside the cloth.

    Returns {rail: [(x, y), ...]} ordered along the rail. Markers on one rail
    all sit the same distance from the cloth edge, so a blob whose offset
    disagrees with its neighbours - a chalk cube, a shirt, a reflection - is
    dropped before it can corrupt the index assignment.

    `polarity` says whether the markers are lighter or darker than the rail
    they sit on; both kinds are in use and a table shows only one of them.
    """
    # Every size here was measured on one broadcast, and they are areas and
    # distances in pixels, so they do not survive a change of scale: at half
    # size a diamond covers a quarter of the area and falls straight through
    # the lower bound. Scaling them against the table is what lets the same
    # code judge a 540p preview and a full-size scan.
    scale = table_scale(frame, quad)
    band_outer = band_outer or max(9, int(61 * scale) | 1)
    band_inner = band_inner or max(3, int(11 * scale) | 1)
    offset_tolerance_px = offset_tolerance_px or 10.0 * scale
    if area_range is None:
        area_range = (max(4.0, 15 * scale * scale), 400 * scale * scale)

    quad_i = quad.astype(np.int32)
    filled = np.zeros(frame.shape[:2], np.uint8)
    cv2.fillPoly(filled, [quad_i], 255)
    band = cv2.subtract(
        cv2.dilate(filled, np.ones((band_outer, band_outer), np.uint8)),
        cv2.dilate(filled, np.ones((band_inner, band_inner), np.uint8)),
    )

    value = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[:, :, 2]
    candidates = _contrast_mask(value, band, scale, polarity)
    if scale > 0.75:
        # Opening clears speckle at full size, where a diamond covers some forty
        # pixels. On a half-size frame it covers eight, and a 3x3 open erases
        # half of them - which is how a 540p preview came to show no diamonds at
        # all and ten perfectly good matches were screened out.
        candidates = cv2.morphologyEx(candidates, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

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


def _index_along_rail(points, rail_vector, expected_count, max_residual_ratio=0.25,
                      ratio_range=(0.85, 1.15), steps=31):
    """Assign diamond indices to the points detected on one rail.

    Points are projected onto a fitted line and matched against a regular grid,
    so a marker hidden behind a player or a cue leaves a gap rather than
    shifting every index after it.

    The grid is searched for rather than read off the gaps between neighbours.
    Reading it off is only right when nearly every point is a real marker: one
    stray blob at the end of a rail shifts the phase by a quarter of a spacing
    and throws out the whole rail, and a rail whose wood grain contributes more
    blobs than the markers do never recovers a sensible spacing at all. The
    search starts from the length of the cloth edge, which spans exactly
    `expected_count - 1` gaps, and keeps whichever grid the most points fall on.
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

    prior = float(np.hypot(*rail_vector)) / (expected_count - 1)
    if prior <= 0:
        raise CalibrationError("degenerate rail")

    best = None
    for spacing in prior * np.linspace(ratio_range[0], ratio_range[1], steps):
        for anchor in t:
            indices = np.round((t - anchor) / spacing)
            residual = np.abs(t - (anchor + indices * spacing))
            inlier = residual <= spacing * max_residual_ratio
            if inlier.sum() < 3:
                continue
            kept = indices[inlier]
            if kept.max() - kept.min() > expected_count - 1:
                continue
            # More markers on the grid wins; among equals, the tighter fit.
            score = (int(inlier.sum()), -float(residual[inlier].sum() / spacing))
            if best is None or score > best[0]:
                best = (score, spacing, indices, residual, inlier)

    if best is None:
        raise CalibrationError("no regular spacing fits these points")
    _, spacing, indices, residual, inlier = best
    indices, pts, residual = indices[inlier].astype(int), pts[inlier], residual[inlier]

    # Two blobs landing on the same index means one of them is not a diamond;
    # keep whichever sits closest to the grid.
    keep = {}
    for i, idx in enumerate(indices):
        if idx not in keep or residual[i] < residual[keep[idx]]:
            keep[idx] = i
    sel = sorted(keep.values())
    indices, pts = indices[sel], pts[sel]
    if len(indices) < 3:
        raise CalibrationError("too few diamonds fit a regular spacing")
    return indices - indices[0], pts, spacing


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


def _calibrate_one(frame, quad, hue, polarity, min_diamonds, min_rails, max_reprojection_mm):
    """One attempt at a calibration, for markers of a single polarity.

    Rails are handled independently, so a cue or a player hiding one rail costs
    that rail's points rather than the whole frame. Raises CalibrationError if
    the frame does not show a full table with enough visible diamonds.
    """
    detected = detect_diamonds(frame, quad, polarity=polarity)
    edges = rail_edges(quad)

    def fit_rails(ratio_range, steps):
        rails, problems = {}, {}
        for rail in RAILS:
            expected = LONG_RAIL_DIAMONDS if rail in LONG_RAILS else SHORT_RAIL_DIAMONDS
            start, end = edges[rail]
            try:
                indices, pts, spacing = _index_along_rail(
                    detected[rail], end - start, expected, ratio_range=ratio_range, steps=steps)
                rails[rail] = (_anchor_indices(indices, pts, expected, start, end), pts, spacing,
                               spacing / (float(np.hypot(*(end - start))) / (expected - 1)))
            except CalibrationError as exc:
                problems[rail] = str(exc)
        return rails, problems

    # The cloth quad overstates the nose line by the width of the cushion cloth,
    # by the same few per cent on every rail, so the four rails agree on the
    # ratio between their diamond spacing and their cloth edge. Fitting each
    # rail alone does not know that, and a rail carrying more wood grain than
    # markers settles on a grid several per cent off - which costs nothing in
    # that rail's own residuals but skews the homography for the whole table.
    # So the rails are fitted loosely once, and then again with the spacing
    # pinned to the ratio the majority of them agreed on.
    rails, problems = fit_rails((0.85, 1.15), 31)
    if len(rails) >= 2:
        agreed = float(np.median([r[3] for r in rails.values()]))
        tightened, tight_problems = fit_rails((agreed * 0.98, agreed * 1.02), 9)
        if len(tightened) >= len(rails):
            rails, problems = tightened, tight_problems

    if len(rails) < min_rails:
        raise CalibrationError(f"only {len(rails)} usable rails ({problems})")
    total = sum(len(v[0]) for v in rails.values())
    if total < min_diamonds:
        raise CalibrationError(f"only {total} diamonds detected")

    # Scale comes from the longest available baseline on each rail, which is far
    # less noisy than any single gap.
    scales = []
    for rail, (indices, pts, _, _) in rails.items():
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
        """Separation of two opposite marker rows, less the table's true width.

        Only markers that both rails have in common may be compared. Averaging
        each rail's points and measuring between the two averages is right only
        while both rails are fully visible: with all nine markers on one rail
        and the last four on the other, the two averages sit diagonally apart,
        and the length of that diagonal gets booked as rail offset. On the
        Pohang table that returned 197 mm where the rail stands 84 mm out, and
        put every ball 57 mm from where it was.
        """
        if rail_a not in rails or rail_b not in rails:
            return None
        a = {int(k): p for k, p in zip(rails[rail_a][0], rails[rail_a][1])}
        b = {int(k): p for k, p in zip(rails[rail_b][0], rails[rail_b][1])}
        shared = sorted(set(a) & set(b))
        if len(shared) < 2:
            return None
        spans = [float(np.hypot(*(a[k] - b[k]))) * mm_per_px for k in shared]
        return (float(np.median(spans)) - true_mm) / 2.0

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

    src, dst, owner = [], [], []
    for rail, (indices, pts, _, _) in rails.items():
        for k, p in zip(indices, pts):
            src.append(p)
            dst.append(ideal[rail](int(k)))
            owner.append((rail, int(k)))

    src = np.array(src, np.float32)
    dst = np.array(dst, np.float32)
    matrix, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    if matrix is None:
        raise CalibrationError("homography fit failed")

    # RANSAC has already decided which markers it believes; scoring the fit
    # against the ones it threw out measures the blobs, not the calibration. A
    # short rail carries only five markers, so a couple of knots in the timber
    # can fit a grid well enough to be indexed, and on the Ankara table they
    # put 25 mm on a homography whose own rails came out at 2. What the count
    # has to guarantee is that enough real markers survived.
    keep = mask.ravel().astype(bool) if mask is not None else np.ones(len(src), bool)
    if int(keep.sum()) < min_diamonds:
        raise CalibrationError(f"only {int(keep.sum())} diamonds fit one homography")

    projected = cv2.perspectiveTransform(src.reshape(-1, 1, 2), matrix).reshape(-1, 2)
    error = float(np.mean(np.linalg.norm(projected[keep] - dst[keep], axis=1)))
    if error > max_reprojection_mm:
        raise CalibrationError(f"reprojection error {error:.1f} mm too large")

    diamonds = {}
    for (rail, k), point, kept in zip(owner, src, keep):
        if kept:
            diamonds.setdefault(rail, []).append((k, float(point[0]), float(point[1])))

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
        cloth_hue=hue,
        polarity=polarity,
        frame_size=frame.shape[:2],
    )


def calibrate(frame, min_diamonds=14, min_rails=3, max_reprojection_mm=12.0):
    """Calibrate a frame from its rail diamonds.

    Both marker polarities are tried because both are in use: the tables under
    the SOOP world cup lighting carry white inlays on a dark rail, while the
    Ankara final is played on a pale wooden rail with dark dots. Assuming the
    first kind is what made an overhead broadcast look like it had no overhead
    camera at all. Whichever polarity indexes more diamonds wins, and ties go
    to the lower reprojection error - a rail's wood grain never produces a row
    of blobs at 355.5 mm intervals, so the wrong polarity loses on count.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hue = dominant_cloth_hue(hsv)
    quad = detect_cloth_quad(frame)
    if quad is None:
        raise CalibrationError("no table-sized cloth region found")

    best, failures = None, []
    for polarity in POLARITIES:
        try:
            found = _calibrate_one(frame, quad, hue, polarity,
                                   min_diamonds, min_rails, max_reprojection_mm)
        except CalibrationError as exc:
            failures.append(f"{polarity}: {exc}")
            continue
        rank = (found.diamond_count, -found.reprojection_error)
        if best is None or rank > best[0]:
            best = (rank, found)
    if best is None:
        raise CalibrationError("; ".join(failures))
    calibration = best[1]
    calibration.learn_appearance(frame)
    return calibration
